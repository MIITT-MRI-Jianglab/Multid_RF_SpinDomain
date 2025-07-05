'''spin-domain pulse optimization'''
import os
import numpy as np
import torch
import scipy.io as spio
import matplotlib.pyplot as plt
from .. import mri
from ..mrsim import SpinDomain
# from ..mrsim import Bloch
from ..PulseOpti import PulseOptimization, losslib, parameterlib


def compute_weighting(cube:mri.SpinArray,roi_inside_idx,roi_idx,transition_idx,roi_weight_scaling):
    ''''''
    # Weighting function for the loss 
    lossweight = torch.ones_like(cube.df)
    if cube.mask == None: 
        mask = torch.ones_like(cube.df)
    else:
        mask = cube.mask
    num_obj = torch.sum(mask).item()
    num_roi_inner = len(roi_inside_idx)
    num_transi = len(transition_idx)
    roiweighting = (num_obj-num_roi_inner-num_transi)/num_roi_inner*roi_weight_scaling
    lossweight[roi_idx] = roiweighting
    lossweight[transition_idx] = 0.
    lossweight = lossweight*mask
    return lossweight


def optimize_excitation(pulse_init:mri.Pulse, cube:mri.SpinGrid,
                        # --- design info ---
                        opti_roi=[6,6,5], opti_roi_offset=[0.,0.,0.], opti_transition_width=[0.,0.,0.], 
                        opti_roi_flip=10, opti_roi_phase=0,
                        # --- system constraints ---
                        rfmax=0.014, grmax=24, slewrate=120,
                        # --- optimization info ---
                        niter=10,rf_niter=5,gr_niter=0,lam=1e4,
                        terminate_roi_error=0.03,
                        early_stopping=False,
                        cost_option='alphabeta+beta',
                        roi_weight_scaling=1,
                        # --- ouput info ---
                        save_results_folder = 'excitationDesign',
                        save_pulse_name = 'excitation_pulse.mat',
                        save_simulation_results_name = 'simulation', 
                        run_optimization=False, 
                        run_evaluation=False,
                        device=torch.device('cpu')):
    '''optimize an excitation pulse based on spin-domain representations
    
    Args:
        pulse_init:      initial rf pulse
        cube:            spins object for simulation 
        opti_roi:                (cm)
        opti_roi_offset:         (cm)
        opti_transition_width:   (cm)
        opti_roi_flip:           0-180 (deg)
        opti_roi_phase:          0-360 (deg)
        rf_max:    (mT)
        gr_max:    (mT/m)
        slewrate:  (mT/m/ms)
        niter:       number of big iteration loops
        rf_niter:    number of rf updates
        gr_niter:    number of gr updates
        lam:         regularization parameter
        terminate_roi_error:        if terminate the optimization eariler 
        early_stopping:             whether choose to terminate the optimization earlier
        cost_option:                choice of cost function for excitation 
        save_results_folder:        where to save the results
        save_pulse_name:            name of the pulse
        save_simulation_results_name:    
        run_optimization: 
        run_evaluation:
        device:             torch.device
    '''
    fov = cube.fov
    dim = cube.dim

    # get the index for different design region
    roi_idx = cube.get_index_roi(opti_roi,opti_roi_offset)
    roi_inside_idx = cube.get_index_roi_inside(opti_roi,opti_transition_width,opti_roi_offset)
    transition_idx = cube.get_index_roi_transition(opti_roi,opti_transition_width,opti_roi_offset)
    stopband_idx = cube.get_index_roi_stopband(opti_roi,opti_transition_width,opti_roi_offset)


    # information about the spins cube 
    print(''.center(40,"*"))
    print('dim={}   fov={}    roi={}'.format(cube.dim, cube.fov, opti_roi))
    print('ROI B0 std: {}'.format(torch.std(cube.df[roi_inside_idx])) + '      '
          'ROI B1 std: {}'.format(torch.std(cube.kappa[roi_inside_idx])))
    

    # Weighting function for the loss
    # -------------------------------------
    # lossweight = torch.ones_like(cube.df)
    # if cube.mask == None: 
    #     mask = torch.ones_like(cube.df)
    # else:
    #     mask = cube.mask
    # num_obj = torch.sum(mask).item()
    # num_roi_inner = len(roi_inside_idx)
    # num_transi = len(transition_idx)
    # roiweighting = (num_obj-num_roi_inner-num_transi)/num_roi_inner*roi_weight_scaling
    # roiweighting = roiweighting*0.75 # try to increase a little more on outer-vol
    # lossweight[roi_idx] = roiweighting
    # lossweight[transition_idx] = 0.
    # lossweight = lossweight*mask
    lossweight = compute_weighting(cube,roi_inside_idx,roi_idx,transition_idx,roi_weight_scaling)

    # ------------------------------------------------
    # functions for measure the performance
    # (excitation case)
    # ---------------------------------------------------    
    target_M = cube.calculate_target_excited_M(flip=opti_roi_flip, phase=opti_roi_phase, roi=opti_roi, roi_offset=opti_roi_offset)
    target_Mxy = cube.calculate_Mxy(target_M)
    # print('target:',torch.mean(target_Mxy[roi_inside_idx]))
    # target_alphaconj_beta,beta_mag = mri.Spin.calculate_target_spindomain_excitation(flip=opti_roi_flip,phase=opti_roi_phase)
    # print(target_alphaconj_beta, beta_mag)

    # save some information
    save_design_setups = False
    if save_design_setups:
        # -----------------------------------------
        def get_mask(idx):
            mask = torch.zeros_like(cube.df)
            mask[idx] = 1
            return mask.reshape(cube.dim)
        collector = {
            # setups
            # 'mask': cube.mask.reshape(dim).cpu().numpy(),
            'b0map': cube.df.reshape(dim).cpu().numpy(),
            'b1map': cube.kappa.reshape(dim).cpu().numpy(),
            'target': target_Mxy.reshape(dim).cpu().numpy(),
            'lossweight': lossweight.reshape(dim).cpu().numpy(),
            'roi_mask': get_mask(roi_idx).cpu().numpy(),
            'roi_inside_mask': get_mask(roi_inside_idx).cpu().numpy(),
            'transition_mask': get_mask(transition_idx).cpu().numpy(),
            'stopband_mask': get_mask(stopband_idx).cpu().numpy(),
        }
        spio.savemat(os.path.join(save_results_folder,'design_setup.mat'),collector)


    print(''.center(40,'*'))
    # ***************************************
    # 
    # ***************************************


    if run_optimization:
        # print info related to optimization -------------
        print('assigned weight for ROI: {}'.format(lossweight[roi_inside_idx[0]]))
        print('early stop:       ---{}'.format(early_stopping))
        print('#iterations:      ---{} x [{}(rf) + {}(gr)]'.format(niter,rf_niter,gr_niter))


        # folder for save the results
        if not os.path.exists(save_results_folder):
            print('create: '+save_results_folder)
            os.mkdir(save_results_folder)

        print('optimization'.center(40,'*'))
        
        
        # Get transformed target for later optimization ----------
        # ---
        if cost_option=='beta-only':
            spindomain_target,spindomain_para_fn = parameterlib.get_target_spindomain_parameters_list(
                cube, opti_roi, 
                pulse_function='excitation',
                phase=opti_roi_phase,
                flip=opti_roi_flip, 
                roi_offset=opti_roi_offset
            )
            loss_fn = losslib.loss_2term
        elif cost_option=='alphabeta':
            spindomain_target,spindomain_para_fn = parameterlib.get_target_spindomain_parameters_list(
                cube, opti_roi, 
                pulse_function='excitation_alphabeta',
                phase=opti_roi_phase,
                flip=opti_roi_flip, 
                roi_offset=opti_roi_offset
            )
            loss_fn = losslib.loss_2term
        elif cost_option=='alphabeta+beta':
            spindomain_target,spindomain_para_fn = parameterlib.get_target_spindomain_parameters_list(
                cube, opti_roi, 
                pulse_function='excitation_v2',
                phase=opti_roi_phase,
                flip=opti_roi_flip, 
                roi_offset=opti_roi_offset
            )
            loss_fn = losslib.loss_3term
        else:
            raise BaseException
        # ----------------------------------
        # para1 = spindomain_target[0]
        # para2 = spindomain_target[1]
        # para3 = spindomain_target[2]
        # print(para1[roi_inside_idx].mean())
        # print(para2[roi_inside_idx].mean())
        # print(para3[roi_inside_idx].mean())
        # return
        

        def eval_fn_roimxy(alpha,beta):
            '''Return performance for each spin (transverse magnetization).'''
            mxy = 2*torch.conj(alpha)*beta
            return torch.mean(mxy[roi_inside_idx])
        def eval_fn_outvolmxy(alpha,beta):
            '''Return performance for each spin (transverse magnetization).'''
            mxy = 2*torch.conj(alpha)*beta
            return torch.mean(mxy[stopband_idx])
        def eval_fn_roi_mse(alpha,beta):
            mxy = 2*torch.conj(alpha)*beta
            errsq = torch.abs(mxy - target_Mxy)**2
            return torch.mean(errsq[roi_inside_idx])
        def eval_fn_outvol_mse(alpha,beta):
            mxy = 2*torch.conj(alpha)*beta
            errsq = torch.abs(mxy - target_Mxy)**2
            return torch.mean(errsq[stopband_idx])
        
        # def eval_termination_fn(alpha,beta):
        #     stop_thre = 0.9
        #     mxy = 2*torch.conj(alpha)*beta
        #     if torch.mean(mxy[roi_inside_idx]).abs() > stop_thre:
        #         return True
        #     else:
        #         return False
        def eval_termination_fn(alpha,beta):
            if early_stopping:
                # stop_thre = 0.03
                mxy = 2*torch.conj(alpha)*beta
                mxyov = torch.mean(mxy[stopband_idx]).abs()
                if mxyov < terminate_roi_error:
                    return True
            else:
                return False
        # print('target threshold: {}'.format(terminate_roi_error))
        
        # Plot of some basic info
        # --------------------------------------------
        plot_opt_setup = False
        if plot_opt_setup:
            print('(plot subject info)')
            figname = os.path.join(save_results_folder,'fig-optimization-setup.png')
            fig = plt.figure(figsize=(15,10))
            pltm,pltn = 2,2
            subplot = lambda t: plt.subplot(pltm,pltn,t)

            ax = subplot(2)
            cube.plot(torch.log10(lossweight+1e-16),title='weighting',ax=ax)
            ax.set_title('log(weighting)')

            ax = subplot(1)
            cube.plot(target_Mxy.abs(),ax=ax)
            ax.set_title('target')

            ax = subplot(3)
            cube.plot(cube.df,title='B0 map',ax=ax)

            ax = subplot(4)
            cube.plot(cube.kappa,title='B1 map',ax=ax,cmap='hot')

            plt.suptitle('excitation pulse optimization')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close(fig)


        optimizer = PulseOptimization.Spindomain_opt_solver()
        try:
            pulse_opt = optimizer.optimize(
                cube, pulse_init,
                spindomain_target, spindomain_para_fn, loss_fn, lossweight,
                rfmax, grmax, slewrate,
                niter=niter, rf_niter=rf_niter, gr_niter=gr_niter, lam=lam,
                # -------------------------------------------
                eval_fn_list=[eval_fn_roimxy,eval_fn_outvolmxy,eval_fn_roi_mse,eval_fn_outvol_mse],
                eval_savename_list=['roi_Mxy','outvol_Mxy','roi_mse','outvol_mse'],
                eval_terminate_fn=eval_termination_fn,
                # --------------------------------
                results_folder=save_results_folder,
                save=True,
                savetmppulse_name=save_pulse_name,
                pulse_function='excitation',
                details=False
            )
        except KeyboardInterrupt:
            print('[key interrupted !]')

        # print(optimizer.optinfos)
        pulse_init.rf.detach_()
        pulse_init.gr.detach_()

        # Save of the results
        print('save solutions ...')
        try:
            print('logging #update = {}'.format(len(optimizer.optinfos['time_hist'])))
            # save optimization details
            spio.savemat(os.path.join(save_results_folder,'optimization.mat'),optimizer.optinfos)
            optimizer.savelog(os.path.join(save_results_folder,'opti_logs.txt'))
            # save optimized pulse
            pulse_opt.save(os.path.join(save_results_folder,save_pulse_name))
        except:
            pass


    # run_evaluation = False
    if run_evaluation:
        # ------------------------------------------
        # Get the optimized pulse
        # ------------------------------------------
        try:
            pulse_opt = mri.Pulse.load(os.path.join(save_results_folder,save_pulse_name),device=device)
        except:
            print('load pulse failed !')
            return        
        # pulse_opt.show_info()

        # simulation
        # ----------------------------------------
        print('(simulation)')
        # either Bloch simulation or spin-domain simulation for evaluation ...
        with torch.no_grad():
            # M = Bloch.blochsim(cube,pulse_init.Nt,pulse_init.dt,pulse_init.rf,pulse_init.gr,device=device)
            # perf_ref_init = mri.SpinArray.calculate_Mxy(M)
            alpha,beta = SpinDomain.spinorsim_c(
                cube,pulse_init.Nt,pulse_init.dt,pulse_init.rf,pulse_init.gr,device=device)
            perf_ref_init = 2*alpha.conj()*beta
        with torch.no_grad():
            # M = Bloch.blochsim(
            #     cube,pulse_opt.Nt,pulse_opt.dt,pulse_opt.rf,pulse_opt.gr,device=device)
            # perf_ref_opt = mri.SpinArray.calculate_Mxy(M)
            alpha,beta = SpinDomain.spinorsim_c(cube,pulse_opt.Nt,pulse_opt.dt,pulse_opt.rf,pulse_opt.gr,device=device)
            perf_ref_opt = 2*alpha.conj()*beta

        save_simulation_results = True
        if save_simulation_results:

            obj_mask = cube.mask
            if not isinstance(obj_mask,torch.Tensor):
                obj_mask = torch.ones_like(cube.df)

            def get_mask(idx):
                mask = torch.zeros_like(cube.df)
                mask[idx] = 1
                return mask.reshape(cube.dim)
            # save some results for plot
            # calculate slew-rate, k-space
            print('roi excitation (initial): {}'.format(perf_ref_init[roi_inside_idx].mean()),
                perf_ref_init[roi_inside_idx].abs().mean())
            print('roi excitation (final): {}'.format(perf_ref_opt[roi_inside_idx].mean()),
                perf_ref_opt[roi_inside_idx].abs().mean())
            
            collector = {
                # setups
                'mask': obj_mask.reshape(dim).cpu().numpy(),
                'b0map': cube.df.reshape(dim).cpu().numpy(),
                'b1map': cube.kappa.reshape(dim).cpu().numpy(),
                'target': target_Mxy.reshape(dim).cpu().numpy(),
                'lossweight': lossweight.reshape(dim).cpu().numpy(),
                # -------------------------------------------------------
                'roi_mask': get_mask(roi_idx).cpu().numpy(),
                'roi_inside_mask': get_mask(roi_inside_idx).cpu().numpy(),
                'transition_mask': get_mask(transition_idx).cpu().numpy(),
                'stopband_mask': get_mask(stopband_idx).cpu().numpy(),
                # --------------------------------------------------------------
                'performance_init': perf_ref_init.reshape(dim).cpu().numpy(),
                'performance_opti': perf_ref_opt.reshape(dim).cpu().numpy(),
                # pulses -------------------------------
                'rf_init': pulse_init.rf.cpu().numpy(),
                'gr_init': pulse_init.gr.cpu().numpy(),
                'slewrate_init': pulse_init.get_slewrate().cpu().numpy(),
                'kspace_init': pulse_init.get_kspace(case='excitation').cpu().numpy(),
                'rf_opti': pulse_opt.rf.cpu().numpy(),
                'gr_opti': pulse_opt.gr.cpu().numpy(),
                'slewrate_opti': pulse_opt.get_slewrate().cpu().numpy(),
                'kspace_opti': pulse_opt.get_kspace(case='excitation').cpu().numpy(),
            }
            try:
                if save_simulation_results_name[-4] == '.mat': 
                    savematname = save_simulation_results_name
                else:
                    savematname = save_simulation_results_name + '.mat'
            except:
                savematname = save_simulation_results_name + '.mat'
            print('simulation output:',os.path.join(save_results_folder,savematname))
            spio.savemat(os.path.join(save_results_folder,savematname),collector)


        plot_simulation_results = False
        if plot_simulation_results:
            print('(plot simulation results)')
            # figname = os.path.join(output_dir,'fig-simulation.png')
            figname = os.path.join(save_results_folder,save_simulation_results_name+'_pulse.png')
            fig = plt.figure(figsize=(15,10))
            pltm,pltn = 4,3
            subplot = lambda t: plt.subplot(pltm,pltn,t)
            # ----------------------------
            ax = subplot(1)
            pulse_init.plot_rf_magnitude(ax=ax)
            ax.set_ylim([0,rfmax])
            ax.set_title('initial rf amplitude')
            ax = subplot(2)
            pulse_init.plot_rf_phase(ax=ax,title='initial rf phase')
            ax.set_ylim([-np.pi, np.pi])
            ax = subplot(3)
            pulse_init.plot_gradients(ax=ax,title='initial gradient')
            # ax = subplot(4)
            # pulse_init.plot_slewrate(ax=ax,title='initial slew-rate')

            ax = subplot(4)
            pulse_opt.plot_rf_magnitude(ax=ax)
            ax.set_ylim([0,rfmax])
            ax.set_title('final rf amplitude')
            ax = subplot(5)
            pulse_opt.plot_rf_phase(ax=ax,title='final rf phase')
            ax.set_ylim([-np.pi, np.pi])
            ax = subplot(6)
            pulse_opt.plot_gradients(ax=ax,title='final gradient')
            # ax = subplot(8)
            # pulse_opt.plot_slewrate(ax=ax,title='final slew-rate')


            ax = plt.subplot(pltm,pltn,(2*pltn+1,3*pltn+1),projection='3d')
            pulse_init.plot_kspace3d(ax=ax)
            ax.set_title('initial k-space')

            ax = plt.subplot(pltm,pltn,(2*pltn+2,3*pltn+2),projection='3d')
            pulse_opt.plot_kspace3d(ax=ax)
            ax.set_title('final k-space')

            # ax = plt.subplot(pltm,pltn,(11,15))
            # cube.plot(perf_ref_init.abs(),ax=ax,title='initial Mxy')

            # ax = plt.subplot(pltm,pltn,(12,16))
            # cube.plot(perf_ref_opt.abs(),ax=ax,title='final Mxy')

            plt.suptitle('excitation pulse deisng simulation')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close(fig)

        plot_eval_profile = False
        if plot_eval_profile:
            ''''''
            figname = os.path.join(save_results_folder,save_simulation_results_name+'_profile.png')
            fig = plt.figure(figsize=(15,10))
            pltm,pltn = 2,2
            subplot = lambda t: plt.subplot(pltm,pltn,t)
            # ----------------------------

            ax = plt.subplot(pltm,pltn,1)
            cube.plot(perf_ref_init.abs(),ax=ax,title='initial Mxy',
                    #   cmap='gray',
                      vrange=[0,1],
                      )

            ax = plt.subplot(pltm,pltn,2)
            cube.plot(perf_ref_opt.abs(),ax=ax,title='final Mxy',
                    #   cmap='gray',
                      vrange=[0,1],
                      )

            ax = plt.subplot(pltm,pltn,3)
            cube.plot(perf_ref_init.angle(),ax=ax,title='initial Mxy')

            ax = plt.subplot(pltm,pltn,4)
            cube.plot(perf_ref_opt.angle(),ax=ax,title='final Mxy')

            plt.suptitle('excitation pulse deisng simulation')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close(fig)
            print(figname)
    
    
    return

def optimize_refocusing(pulse_init:mri.Pulse, cube:mri.SpinGrid,
                        # --- design info -----
                        opti_roi=[6,6,5], 
                        opti_roi_offset=[0.,0.,0.], 
                        opti_transition_width=[0.,0.,0.], 
                        opti_roi_phase=0,
                        # --- design constraints ---
                        rfmax=0.014, grmax=24, slewrate=120,
                        # --- parameters for optimization ----
                        niter=10,rf_niter=5,gr_niter=0,
                        lam=1e4,
                        terminate_roi_efficiency=0.9,
                        early_stopping=False,
                        roi_weight_scaling=1,
                        # --- output info ----
                        save_results_folder = 'refocusingDesign',
                        save_pulse_name = 'refocusing_pulse.mat',
                        save_simulation_results_name = 'simulation', 
                        run_optimization=False, 
                        run_evaluation=False,
                        device=torch.device('cpu')): # TODO
    '''Optimize a refocusing pulse using spin domatin representation.
    
    Args:
        pulse_init:      initial rf pulse
        cube:            spins object for simulation 
        opti_roi:                (cm)
        opti_roi_offset:         (cm)
        opti_transition_width:   (cm)
        opti_roi_phase:          0-360 (deg)
        rf_max:    (mT)
        gr_max:    (mT/m)
        slewrate:  (mT/m/ms)
        niter:       number of big iteration loops
        rf_niter:    number of rf updates
        gr_niter:    number of gr updates
        lam:         regularization parameter
        terminate_roi_efficiency:        if terminate the optimization eariler 
        early_stopping:                  whether choose to terminate the optimization earlier
        save_results_folder:           where to save the results
        save_pulse_name:               name of the pulse
        save_simulation_results_name:    
        run_optimization: 
        run_evaluation:
        device:             torch.device
    '''
    # print('(optimizing refocusing pulse)')
    fov = cube.fov
    dim = cube.dim

    # get the index for different design region
    roi_idx        = cube.get_index_roi(opti_roi,opti_roi_offset)
    roi_inside_idx = cube.get_index_roi_inside(opti_roi,opti_transition_width,opti_roi_offset)
    transition_idx = cube.get_index_roi_transition(opti_roi,opti_transition_width,opti_roi_offset)
    stopband_idx   = cube.get_index_roi_stopband(opti_roi,opti_transition_width,opti_roi_offset)

    # Display information about the spins cube 
    print(''.center(60,"*"))
    print('dim={}   fov={}    roi={}'.format(cube.dim, cube.fov, opti_roi))
    if len(roi_inside_idx)>1:
        b0_std = torch.std(cube.df[roi_inside_idx])
        b1_std = torch.std(cube.kappa[roi_inside_idx])
        print('ROI B0 std: {}'.format(b0_std) + '      '
            'ROI B1 std: {}'.format(b1_std))

    # Weighting function for the loss 
    # -------------------------------
    # lossweight = torch.ones_like(cube.df)
    # if cube.mask == None: 
    #     mask = torch.ones_like(cube.df)
    # else:
    #     mask = cube.mask
    # num_obj = torch.sum(mask).item()
    # num_roi_inner = len(roi_inside_idx)
    # num_transi = len(transition_idx)
    # roiweighting = (num_obj-num_roi_inner-num_transi)/num_roi_inner*roi_weight_scaling
    # lossweight[roi_idx] = roiweighting
    # lossweight[transition_idx] = 0.
    # lossweight = lossweight*mask
    lossweight = compute_weighting(cube,roi_inside_idx,roi_idx,transition_idx,roi_weight_scaling)

    # ------------------------------------------------
    # functions for measure the error
    # (refocusing case)
    # ---------------------------------------------------
    target_betasquare = cube.calculate_target_spindomain_refocusing(
        phase=opti_roi_phase, roi=opti_roi, roi_offset=opti_roi_offset
    )

    print(''.center(60,'*'))
    # ******************************************
    # 
    # ******************************************
    
    # run_optimization = True
    if run_optimization:
        # print info related to optimization -------------
        print('assigned weight for ROI: {}'.format(lossweight[roi_inside_idx[0]]))
        print('early stop:       ---{}'.format(early_stopping))
        print('#iterations:      ---{} x [{}(rf) + {}(gr)]'.format(niter,rf_niter,gr_niter))
        # folder for save the results
        if not os.path.exists(save_results_folder):
            print('results folder:      '+save_results_folder+'     ---create')
            os.mkdir(save_results_folder)
        else:
            print('results folder:      '+save_results_folder+'     ---exists')


        # --------------------------------------------------
        # Define some functions measuring the performance 
        def eval_fn_roi(alpha,beta):
            '''Return error for each spin.'''
            para = -beta**2
            return torch.mean(para[roi_inside_idx])
        def eval_fn_outvol(alpha,beta):
            para = -beta**2
            return torch.mean(para[stopband_idx])
        def eval_fn_roi_mse(alpha,beta):
            para = beta**2
            errsq = torch.abs(para - target_betasquare)**2
            return torch.mean(errsq[roi_inside_idx])
        def eval_fn_outvol_mse(alpha,beta):
            para = beta**2
            errsq = torch.abs(para - target_betasquare)**2
            return torch.mean(errsq[stopband_idx])
        def eval_termination_fn(alpha,beta):
            if early_stopping:
                para = -beta**2
                ref_perf = torch.mean(para[roi_inside_idx]).abs()
                if ref_perf>terminate_roi_efficiency:
                    return True
            else:
                return False
        # print('target refocusing threshold: roi efficiency={}'.format(terminate_roi_efficiency))
            
        # Get transformed target for later optimization 
        spindomain_target,spindomain_para_fn = parameterlib.get_target_spindomain_parameters_list(
            cube,opti_roi,pulse_function='refocusing',phase=opti_roi_phase, 
            flip=180,roi_offset=opti_roi_offset
        )
        loss_fn = losslib.loss_2term

        # Plot of some basic info
        # --------------------------------------------
        plot_opt_setup = False
        if plot_opt_setup:
            print('(plot subject info)')
            figname = os.path.join(save_results_folder,'fig-optimization-setup.png')
            # -----
            fig = plt.figure(figsize=(15,10))
            pltm,pltn = 2,2
            subplot = lambda t: plt.subplot(pltm,pltn,t)

            ax = subplot(2)
            cube.plot(torch.log10(lossweight+1e-16),title='weighting',ax=ax)
            ax.set_title('log(weighting)')

            ax = subplot(1)
            t = spindomain_target[0]**2 + spindomain_target[1]**2
            cube.plot(t,ax=ax)
            # cube.plot(target_betasquare.abs(),ax=ax)
            ax.set_title('target')

            ax = subplot(3)
            cube.plot(cube.df,title='B0 map',ax=ax)

            ax = subplot(4)
            cube.plot(cube.kappa.abs(),title='B1 map',ax=ax,cmap='hot',vrange=[0,1.4])

            # ax = subplot(5)
            # cube.plot(cube.mask,title='mask',ax=ax)
            # ax.set_title('mask')

            plt.suptitle('refocusing pulse optimization')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close(fig)
        
        # Optimization
        # ---------------------------------------------------
        print('optimize refocusing'.center(60,'*'))
        optimizer = PulseOptimization.Spindomain_opt_solver()
        try:
            pulse_opt = optimizer.optimize(
                cube,pulse_init,
                spindomain_target,
                spindomain_para_fn,
                loss_fn,
                lossweight,
                # 
                rfmax, grmax, slewrate,
                niter=niter, rf_niter=rf_niter, gr_niter=gr_niter,
                lam=lam,
                # 
                eval_fn_list=[eval_fn_roi,eval_fn_outvol,eval_fn_roi_mse,eval_fn_outvol_mse],
                eval_savename_list=['roi_effi','outvol_effi','roi_mse','outvol_mse'],
                eval_terminate_fn=eval_termination_fn,
                # 
                save=True,
                results_folder=save_results_folder,
                savetmppulse_name=save_pulse_name,
                pulse_function='refocusing',
                details=False
            )
        except KeyboardInterrupt:
            print('[key interrupted !]')

        # print(optimizer.optinfos)
        pulse_init.rf.detach_()
        pulse_init.gr.detach_()

        # Save of the results
        print('save solutions ...')
        try:
            # print(optimizer.optinfos.keys())
            # save optimization details
            print('logging #states = {}'.format(len(optimizer.optinfos['time_hist'])))
            spio.savemat(os.path.join(save_results_folder,'optimization.mat'), optimizer.optinfos)
            optimizer.savelog(os.path.join(save_results_folder,'opti_logs.txt'))
            # save optimized pulse
            pulse_opt.save(os.path.join(save_results_folder,save_pulse_name))
        except:
            pass
    
    # run_evaluation = True
    if run_evaluation:
        # ------------------------------------------
        # Get the initial pulse (which if from the input)
        # ------------------------------------------
        
        # ------------------------------------------
        # Get the optimized pulse (by read existing pulse)
        # ------------------------------------------
        try:
            pulse_opt = mri.Pulse.load(os.path.join(save_results_folder,save_pulse_name),device=device)
        except:
            print('load pulse failed !')
            return
        # pulse_opt.show_info()

        # simulation
        # ----------------------------------------
        print('(simulation)')
        # initial refocusing performance
        with torch.no_grad():
            a,b = SpinDomain.spinorsim_c(cube,pulse_init.Nt,pulse_init.dt,pulse_init.rf,pulse_init.gr,device=device)
            perf_ref_init = b**2
        # opt refocusing performance
        with torch.no_grad():
            a,b = SpinDomain.spinorsim_c(
                cube,pulse_opt.Nt,pulse_opt.dt,pulse_opt.rf,pulse_opt.gr,device=device)
            perf_ref_opt = b**2

        save_simulation_results = True
        if save_simulation_results:
            obj_mask = cube.mask
            if not isinstance(obj_mask,torch.Tensor):
                obj_mask = torch.ones_like(cube.df)

            def get_mask(idx):
                mask = torch.zeros_like(cube.df)
                mask[idx] = 1
                return mask.reshape(cube.dim)
            # save some results for plot
            # calculate slew-rate, k-space

            refocusing_eff_roi_init = torch.mean(perf_ref_init[roi_inside_idx])
            refocusing_eff_roi_opt = torch.mean(perf_ref_opt[roi_inside_idx])
            print('roi refocusing (initial): {}, {}'.format(refocusing_eff_roi_init,refocusing_eff_roi_init.abs()))
            print('roi refocusing (final): {}, {}'.format(refocusing_eff_roi_opt, refocusing_eff_roi_opt.abs()))
            collector = {
                # setups
                'mask': obj_mask.reshape(dim).cpu().numpy(),
                'b0map': cube.df.reshape(dim).cpu().numpy(),
                'b1map': cube.kappa.reshape(dim).cpu().numpy(),
                'target': target_betasquare.reshape(dim).cpu().numpy(),
                'lossweight': lossweight.reshape(dim).cpu().numpy(),
                # -------------------------------------------------------
                'roi_mask': get_mask(roi_idx).cpu().numpy(),
                'roi_inside_mask': get_mask(roi_inside_idx).cpu().numpy(),
                'transition_mask': get_mask(transition_idx).cpu().numpy(),
                'stopband_mask': get_mask(stopband_idx).cpu().numpy(),
                # --------------------------------------------------------------
                'performance_init': perf_ref_init.reshape(dim).cpu().numpy(),
                'performance_opti': perf_ref_opt.reshape(dim).cpu().numpy(),
                'refocusing_eff_roi_init': refocusing_eff_roi_init.item(),
                'refocusing_eff_roi_opt': refocusing_eff_roi_opt.item(),
                # pulses -------------------------------
                'rf_init': pulse_init.rf.cpu().numpy(),
                'gr_init': pulse_init.gr.cpu().numpy(),
                'slewrate_init': pulse_init.get_slewrate().cpu().numpy(),
                'kspace_init': pulse_init.get_kspace(case='excitation').cpu().numpy(),
                'rf_opti': pulse_opt.rf.cpu().numpy(),
                'gr_opti': pulse_opt.gr.cpu().numpy(),
                'slewrate_opti': pulse_opt.get_slewrate().cpu().numpy(),
                'kspace_opti': pulse_opt.get_kspace(case='excitation').cpu().numpy(),
            }
            try:
                if save_simulation_results_name[-4] == '.mat': 
                    savematname = save_simulation_results_name
                else:
                    savematname = save_simulation_results_name + '.mat'
            except:
                savematname = save_simulation_results_name + '.mat'
            print('simulation output:\n'+os.path.join(save_results_folder,savematname))
            spio.savemat(os.path.join(save_results_folder,savematname),collector)

        plot_eval_pulses = False
        if plot_eval_pulses:
            # print('(plot simulation results)')
            # figname = os.path.join(output_dir,'fig-simulation.png')
            figname = os.path.join(save_results_folder,save_simulation_results_name+'_pulse.png')
            fig = plt.figure(figsize=(15,10))
            pltm,pltn = 4,3
            subplot = lambda t: plt.subplot(pltm,pltn,t)
            # ----------------------------
            ax = subplot(1)
            pulse_init.plot_rf_magnitude(ax=ax)
            ax.set_ylim([0,rfmax])
            ax.set_title('initial rf amplitude')
            ax = subplot(2)
            pulse_init.plot_rf_phase(ax=ax,title='initial rf phase')
            ax.set_ylim([-np.pi, np.pi])
            ax = subplot(3)
            pulse_init.plot_gradients(ax=ax,title='initial gradient')
            # ax = subplot(4)
            # pulse_init.plot_slewrate(ax=ax,title='initial slew-rate')

            ax = subplot(4)
            pulse_opt.plot_rf_magnitude(ax=ax)
            ax.set_ylim([0,rfmax])
            ax.set_title('final rf amplitude')
            ax = subplot(5)
            pulse_opt.plot_rf_phase(ax=ax,title='final rf phase')
            ax.set_ylim([-np.pi, np.pi])
            ax = subplot(6)
            pulse_opt.plot_gradients(ax=ax,title='final gradient')
            # ax = subplot(8)
            # pulse_opt.plot_slewrate(ax=ax,title='final slew-rate')

            ax = plt.subplot(pltm,pltn,(2*pltn+1,3*pltn+1),projection='3d')
            pulse_init.plot_kspace3d(ax=ax)
            ax.set_title('initial k-space')

            ax = plt.subplot(pltm,pltn,(2*pltn+2,3*pltn+2),projection='3d')
            pulse_opt.plot_kspace3d(ax=ax)
            ax.set_title('final k-space')

            # ax = plt.subplot(pltm,pltn,(11,15))
            # cube.plot(perf_ref_init.abs(),ax=ax,title='initial refocusing')

            # ax = plt.subplot(pltm,pltn,(12,16))
            # cube.plot(perf_ref_opt.abs(),ax=ax,title='final refocusing')

            plt.suptitle('refocusing pulse deisng simulation')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close(fig)
            print(figname)

        plot_eval_profile = False
        if plot_eval_profile:
            ''''''
            figname = os.path.join(save_results_folder,save_simulation_results_name+'_profile.png')
            fig = plt.figure(figsize=(15,10))
            pltm,pltn = 2,2
            subplot = lambda t: plt.subplot(pltm,pltn,t)
            # ----------------------------

            ax = plt.subplot(pltm,pltn,1)
            cube.plot(perf_ref_init.abs(),ax=ax,title='initial refocusing',
                    #   cmap='gray',
                      vrange=[0,1],
                      )

            ax = plt.subplot(pltm,pltn,2)
            cube.plot(perf_ref_opt.abs(),ax=ax,title='final refocusing',
                    #   cmap='gray',
                      vrange=[0,1],
                      )

            ax = plt.subplot(pltm,pltn,3)
            cube.plot(perf_ref_init.angle(),ax=ax,title='initial refocusing')

            ax = plt.subplot(pltm,pltn,4)
            cube.plot(perf_ref_opt.angle(),ax=ax,title='final refocusing')

            plt.suptitle('refocusing pulse deisng simulation')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close(fig)
            print(figname)
    
    return
