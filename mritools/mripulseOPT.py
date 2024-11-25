'''more compact functions for rf pulse design using spin-domain optimization or bloach simulation'''
import os
import torch
import scipy.io as spio
import matplotlib.pyplot as plt
from mritools import mri
# from mritools import mripulse
from mritools.Opti import PulseOptimization
from mritools.Opti import losslib
from mritools.Opti import parameterlib


def optimize_excitation(pulse_init:mri.Pulse, cube:mri.SpinGrid,
                        # --- design info ---
                        opti_roi=[6,6,5], opti_roi_offset=[0.,0.,0.], opti_transition_width=[0.,0.,0.], 
                        opti_roi_flip=10, opti_roi_phase=0,
                        # --- system constraints ---
                        rfmax=0.014, grmax=24, slewrate=120,
                        # --- optimization info ---
                        niter=10,rf_niter=5,gr_niter=0,lam=1e4,
                        # --- ouput info ---
                        output_dir = '',
                        output_pulsename='opti_exc_pulse.mat',
                        output_simulation_name='simulation.mat',
                        run_optimization=True, run_evaluation=True,
                        device=torch.device('cpu')):
    '''optimize an excitation pulse based on spin-domain representations
    
    Args:
        pulse_init: initial rf pulse
        cube: optimization target 
        opti_roi: 
        opti_roi_offset:
        opti_transition_width:
        opti_roi_flip:
        opti_roi_phase:
        rf_max:
        gr_max:
        slewrate:
        niter:
    '''
    fov = cube.fov
    dim = cube.dim

    # get the index for different design region
    roi_idx = cube.get_index_roi(opti_roi,opti_roi_offset)
    roi_inside_idx = cube.get_index_roi_inside(opti_roi,opti_transition_width,opti_roi_offset)
    transition_idx = cube.get_index_roi_transition(opti_roi,opti_transition_width,opti_roi_offset)
    stopband_idx = cube.get_index_roi_stopband(opti_roi,opti_transition_width,opti_roi_offset)

    # Weighting function for the loss
    lossweight = torch.ones_like(cube.df)
    if cube.mask == None: 
        mask = torch.ones_like(cube.df)
    else:
        mask = cube.mask
    num_obj = torch.sum(mask).item()
    num_roi_inner = len(roi_inside_idx)
    num_transi = len(transition_idx)
    roiweighting = (num_obj-num_roi_inner-num_transi)/num_roi_inner
    lossweight[roi_idx] = roiweighting
    lossweight[transition_idx] = 0.
    lossweight = lossweight*mask

    print('assigned weight for ROI: {}'.format(roiweighting))
    print('ROI B0 std: {}'.format(torch.std(cube.df[roi_inside_idx])))
    print('ROI B1 std: {}'.format(torch.std(cube.kappa[roi_inside_idx])))

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
        spio.savemat(os.path.join(output_dir,'design_setup.mat'),collector)


    # Optimization
    # ---------------------------------------------------
    if run_optimization:
        # folder for save the results
        print(output_dir)
        if not os.path.exists(output_dir):
            print('create: '+output_dir)
            os.mkdir(output_dir)

        print('optimization'.center(40,'*'))
        # Get transformed target for later optimization 
        spindomain_target,spindomain_para_fn = parameterlib.get_target_spindomain_parameters_list(
            cube, opti_roi, pulse_function='excitation',
            phase=opti_roi_phase,
            flip=opti_roi_flip, 
            roi_offset=opti_roi_offset
        )
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
        
        # Plot of some basic info
        # --------------------------------------------
        plot_opt_setup = True
        if plot_opt_setup:
            print('(plot subject info)')
            figname = os.path.join(output_dir,'fig-setup.png')
            fig = plt.figure(figsize=(10,10))
            pltm,pltn = 2,2
            subplot = lambda t: plt.subplot(pltm,pltn,t)

            ax = subplot(1)
            cube.plot(cube.df,title='B0 map',ax=ax)

            ax = subplot(2)
            cube.plot(cube.kappa,title='B1 map',ax=ax,cmap='hot')

            ax = subplot(3)
            cube.plot(target_Mxy.abs(),ax=ax)
            ax.set_title('target')

            ax = subplot(4)
            cube.plot(torch.log10(lossweight+1e-16),title='weighting',ax=ax)
            ax.set_title('log(weighting)')

            plt.suptitle('excitation pulse optimization')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close(fig)


        optimizer = PulseOptimization.Spindomain_opt_solver()
        loss_fn = losslib.loss_3term
        try:
            pulse_opt = optimizer.optimize(
                cube, pulse_init,
                spindomain_target, spindomain_para_fn, loss_fn, lossweight,
                rfmax, grmax, slewrate,
                niter=niter, rf_niter=rf_niter, gr_niter=gr_niter, lam=lam,
                eval_fn_list=[eval_fn_roimxy,eval_fn_outvolmxy,eval_fn_roi_mse,eval_fn_outvol_mse],
                eval_savename_list=['roi_Mxy','outvol_Mxy','roi_mse','outvol_mse'],
                # --------------------------------
                results_folder=output_dir,
                save=True,
                savetmppulse_name=output_pulsename,
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
            spio.savemat(os.path.join(output_dir,'optimization.mat'),optimizer.optinfos)
            optimizer.savelog(os.path.join(output_dir,'opti_logs.txt'))
            # save optimized pulse
            pulse_opt.save(os.path.join(output_dir,output_pulsename))
        except:
            pass


    # run_evaluation = False
    if run_evaluation:
        # ------------------------------------------
        # Get the optimized pulse
        # ------------------------------------------
        pulse_opt = mri.Pulse.load(os.path.join(output_dir,output_pulsename),device=device)
        pulse_opt.show_info()

        # simulation
        # ----------------------------------------
        print('(simulation)')
        with torch.no_grad():
            M = mri.blochsim(cube,pulse_init.Nt,pulse_init.dt,pulse_init.rf,pulse_init.gr,device=device)
            perf_ref_init = mri.SpinArray.calculate_Mxy(M)
        with torch.no_grad():
            M = mri.blochsim(
                cube,pulse_opt.Nt,pulse_opt.dt,pulse_opt.rf,pulse_opt.gr,device=device)
            perf_ref_opt = mri.SpinArray.calculate_Mxy(M)

        save_simulation_results = True
        if save_simulation_results:
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
                'mask': cube.mask.reshape(dim).cpu().numpy(),
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
                'kspace_init': pulse_init.get_kspace().cpu().numpy(),
                'rf_opti': pulse_opt.rf.cpu().numpy(),
                'gr_opti': pulse_opt.gr.cpu().numpy(),
                'slewrate_opti': pulse_opt.get_slewrate().cpu().numpy(),
                'kspace_opti': pulse_opt.get_kspace().cpu().numpy(),
            }
            spio.savemat(os.path.join(output_dir,output_simulation_name),collector)


        plot_simulation_results = True
        if plot_simulation_results:
            print('(plot simulation results)')
            figname = os.path.join(output_dir,'fig-simulation.png')
            fig = plt.figure(figsize=(15,10))
            pltm,pltn = 4,4
            subplot = lambda t: plt.subplot(pltm,pltn,t)
            # ----------------------------
            ax = subplot(1)
            pulse_init.plot_rf_magnitude(ax=ax)
            ax.set_title('initial rf amplitude')
            ax = subplot(2)
            pulse_init.plot_rf_phase(ax=ax,title='initial rf phase')
            ax = subplot(3)
            pulse_init.plot_gradients(ax=ax,title='initial gradient')
            ax = subplot(4)
            pulse_init.plot_slewrate(ax=ax,title='initial slew-rate')

            ax = subplot(5)
            pulse_opt.plot_rf_magnitude(ax=ax)
            ax.set_title('final rf amplitude')
            ax = subplot(6)
            pulse_opt.plot_rf_phase(ax=ax,title='final rf phase')
            ax = subplot(7)
            pulse_opt.plot_gradients(ax=ax,title='final gradient')
            ax = subplot(8)
            pulse_opt.plot_slewrate(ax=ax,title='final slew-rate')


            ax = plt.subplot(pltm,pltn,(9,13),projection='3d')
            pulse_init.plot_kspace3d(ax=ax)
            ax.set_title('initial k-space')

            ax = plt.subplot(pltm,pltn,(10,14),projection='3d')
            pulse_opt.plot_kspace3d(ax=ax)
            ax.set_title('final k-space')

            ax = plt.subplot(pltm,pltn,(11,15))
            cube.plot(perf_ref_init.abs(),ax=ax,title='initial Mxy')

            ax = plt.subplot(pltm,pltn,(12,16))
            cube.plot(perf_ref_opt.abs(),ax=ax,title='final Mxy')

            plt.suptitle('excitation pulse deisng simulation')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close(fig)

    
    return


def optimize_refocusing(pulse_init:mri.Pulse,cube:mri.SpinGrid,
                        # --- design info -----
                        opti_roi=[6,6,5], opti_roi_offset=[0.,0.,0.], 
                        opti_transition_width=[0.,0.,0.], opti_roi_phase=0,
                        rfmax=0.014, grmax=24, slewrate=120,
                        # --- parameters for optimization ----
                        niter=10,rf_niter=5,gr_niter=0,lam=1e4,
                        # --- output info ----
                        output_dir = '',
                        output_pulsename='opti_exc_pulse.mat',
                        output_simulation_name='simulation.mat',
                        run_optimization=True, run_evaluation=True,
                        device=torch.device('cpu')): # TODO
    '''Optimize a refocusing pulse using spin domatin representation.
    
    Args:
        pulse_init: initial rf pulse
        cube: optimization target 
        opti_roi: 
        opti_roi_offset:
        opti_transition_width:
        opti_roi_flip:
        opti_roi_phase:
        rf_max:
        gr_max:
        slewrate:
        niter:
    '''
    print('(optimizing refocusing pulse)')
    fov = cube.fov
    dim = cube.dim

    # get the index for different design region
    roi_idx = cube.get_index_roi(opti_roi,opti_roi_offset)
    roi_inside_idx = cube.get_index_roi_inside(opti_roi,opti_transition_width,opti_roi_offset)
    transition_idx = cube.get_index_roi_transition(opti_roi,opti_transition_width,opti_roi_offset)
    stopband_idx = cube.get_index_roi_stopband(opti_roi,opti_transition_width,opti_roi_offset)

    # Weighting function for the loss
    lossweight = torch.ones_like(cube.df)
    if cube.mask == None: 
        mask = torch.ones_like(cube.df)
    else:
        mask = cube.mask
    num_obj = torch.sum(mask).item()
    num_roi_inner = len(roi_inside_idx)
    num_transi = len(transition_idx)
    roiweighting = (num_obj-num_roi_inner-num_transi)/num_roi_inner
    lossweight[roi_idx] = roiweighting
    lossweight[transition_idx] = 0.
    lossweight = lossweight*mask

    print('assigned weight for ROI: {}'.format(roiweighting))
    print('ROI B0 std: {}'.format(torch.std(cube.df[roi_inside_idx])))
    print('ROI B1 std: {}'.format(torch.std(cube.kappa[roi_inside_idx])))


    # ------------------------------------------------
    # functions for measure the error
    # (refocusing case)
    # ---------------------------------------------------
    target_betasquare = cube.calculate_target_spindomain_refocusing(
        phase=opti_roi_phase, roi=opti_roi, roi_offset=opti_roi_offset
    )

    save_design_setups = False
    if save_design_setups:
        def get_mask(idx):
            mask = torch.zeros_like(cube.df)
            mask[idx] = 1
            return mask.reshape(cube.dim)
        collector = {
            # setups
            # 'mask': cube.mask.reshape(dim).cpu().numpy(),
            'b0map': cube.df.reshape(dim).cpu().numpy(),
            'b1map': cube.kappa.reshape(dim).cpu().numpy(),
            'target': target_betasquare.reshape(dim).cpu().numpy(),
            'lossweight': lossweight.reshape(dim).cpu().numpy(),
            'roi_mask': get_mask(roi_idx).cpu().numpy(),
            'roi_inside_mask': get_mask(roi_inside_idx).cpu().numpy(),
            'transition_mask': get_mask(transition_idx).cpu().numpy(),
            'stopband_mask': get_mask(stopband_idx).cpu().numpy(),
        }
        spio.savemat(os.path.join(output_dir,'design_setup.mat'),collector)

    print(''.center(40,'*'))
    # ******************************************
    # 
    # ******************************************
    # run_optimization = True
    if run_optimization:
        # folder for save the results
        print(output_dir)
        if not os.path.exists(output_dir):
            print('create: '+output_dir)
            os.mkdir(output_dir)

        # Get transformed target for later optimization 
        spindomain_target,spindomain_para_fn = parameterlib.get_target_spindomain_parameters_list(
            cube,opti_roi,pulse_function='refocusing',phase=opti_roi_phase, 
            flip=180,roi_offset=opti_roi_offset
        )

        def eval_fn_roi(alpha,beta):
            '''Return error for each spin.'''
            para = -beta**2
            return torch.mean(para[roi_inside_idx])
        def eval_fn_stopband(alpha,beta):
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
        
        # Plot of some basic info
        # --------------------------------------------
        plot_opt_setup = True
        if plot_opt_setup:
            print('(plot subject info)')
            figname = os.path.join(output_dir,'fig-setup.png')
            fig = plt.figure(figsize=(10,10))
            pltm,pltn = 2,2
            subplot = lambda t: plt.subplot(pltm,pltn,t)

            ax = subplot(1)
            cube.plot(cube.df,title='B0 map',ax=ax)

            ax = subplot(2)
            cube.plot(cube.kappa,title='B1 map',ax=ax,cmap='hot')

            ax = subplot(3)
            cube.plot(target_betasquare.abs(),ax=ax)
            ax.set_title('target')

            ax = subplot(4)
            cube.plot(torch.log10(lossweight+1e-16),title='weighting',ax=ax)
            ax.set_title('log(weighting)')

            # ax = subplot(5)
            # cube.plot(cube.mask,title='mask',ax=ax)
            # ax.set_title('mask')

            plt.suptitle('refocusing pulse optimization')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close(fig)
        
        # Optimization
        # ---------------------------------------------------
        optimizer = PulseOptimization.Spindomain_opt_solver()
        loss_fn = losslib.loss_2term
        try:
            pulse_opt = optimizer.optimize(
                cube,pulse_init,
                spindomain_target,
                spindomain_para_fn,
                loss_fn,
                lossweight,
                rfmax, grmax, slewrate,
                niter=niter, rf_niter=rf_niter, gr_niter=gr_niter,
                lam=lam,
                eval_fn_list=[eval_fn_roi,eval_fn_stopband,eval_fn_roi_mse,eval_fn_outvol_mse],
                eval_savename_list=['roi_effi','stopband_effi','roi_mse','outvol_mse'],
                results_folder=output_dir,
                save=True,
                savetmppulse_name=output_pulsename,
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
            print('logging #update = {}'.format(len(optimizer.optinfos['time_hist'])))
            # save optimization details
            # print(optimizer.optinfos.keys())
            spio.savemat(os.path.join(output_dir,'optimization.mat'),optimizer.optinfos)
            optimizer.savelog(os.path.join(output_dir,'opti_logs.txt'))
            # save optimized pulse
            pulse_opt.save(os.path.join(output_dir,output_pulsename))
        except:
            pass
    
    # run_evaluation = True
    if run_evaluation:
        
        # ------------------------------------------
        # Get the initial pulse
        # ------------------------------------------
        
        # ------------------------------------------
        # Get the optimized pulse
        # ------------------------------------------
        pulse_opt = mri.Pulse.load(os.path.join(output_dir,output_pulsename),device=device)
        # pulse_opt.show_info()

        # simulation
        # ----------------------------------------
        print('(simulation)')
        with torch.no_grad():
            a,b = mri.spinorsim_c_singlestep(cube,pulse_init.Nt,pulse_init.dt,pulse_init.rf,pulse_init.gr,device=device)
            perf_ref_init = b**2
        with torch.no_grad():
            a,b = mri.spinorsim_c_singlestep(
                cube,pulse_opt.Nt,pulse_opt.dt,pulse_opt.rf,pulse_opt.gr,device=device)
            perf_ref_opt = b**2

        save_simulation_results = True
        if save_simulation_results:
            def get_mask(idx):
                mask = torch.zeros_like(cube.df)
                mask[idx] = 1
                return mask.reshape(cube.dim)
            # save some results for plot
            # calculate slew-rate, k-space
            print('roi refocusing (initial): {}, {}'.format(
                torch.mean(perf_ref_init[roi_inside_idx]),torch.mean(perf_ref_init[roi_inside_idx]).abs()))
            print('roi refocusing (final): {}, {}'.format(
                torch.mean(perf_ref_opt[roi_inside_idx]), torch.mean(perf_ref_opt[roi_inside_idx]).abs()))
            collector = {
                # setups
                'mask': cube.mask.reshape(dim).cpu().numpy(),
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
            spio.savemat(os.path.join(output_dir,output_simulation_name),collector)

        plot_eval_pulses = True
        if plot_eval_pulses:
            print('(plot simulation results)')
            figname = os.path.join(output_dir,'fig-simulation.png')
            fig = plt.figure(figsize=(15,10))
            pltm,pltn = 4,4
            subplot = lambda t: plt.subplot(pltm,pltn,t)
            # ----------------------------
            ax = subplot(1)
            pulse_init.plot_rf_magnitude(ax=ax)
            ax.set_title('initial rf amplitude')
            ax = subplot(2)
            pulse_init.plot_rf_phase(ax=ax,title='initial rf phase')
            ax = subplot(3)
            pulse_init.plot_gradients(ax=ax,title='initial gradient')
            ax = subplot(4)
            pulse_init.plot_slewrate(ax=ax,title='initial slew-rate')

            ax = subplot(5)
            pulse_opt.plot_rf_magnitude(ax=ax)
            ax.set_title('final rf amplitude')
            ax = subplot(6)
            pulse_opt.plot_rf_phase(ax=ax,title='final rf phase')
            ax = subplot(7)
            pulse_opt.plot_gradients(ax=ax,title='final gradient')
            ax = subplot(8)
            pulse_opt.plot_slewrate(ax=ax,title='final slew-rate')


            ax = plt.subplot(pltm,pltn,(9,13),projection='3d')
            pulse_init.plot_kspace3d(ax=ax)
            ax.set_title('initial k-space')

            ax = plt.subplot(pltm,pltn,(10,14),projection='3d')
            pulse_opt.plot_kspace3d(ax=ax)
            ax.set_title('final k-space')

            ax = plt.subplot(pltm,pltn,(11,15))
            cube.plot(perf_ref_init.abs(),ax=ax,title='initial refocusing')

            ax = plt.subplot(pltm,pltn,(12,16))
            cube.plot(perf_ref_opt.abs(),ax=ax,title='final refocusing')

            plt.suptitle('refocusing pulse deisng simulation')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close(fig)
    
    return