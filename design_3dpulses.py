# example that design 3d refocusing pulse using spin-domain representation
# author: jiayao

import numpy as np
import torch
import os
import scipy.io as spio
import matplotlib.pyplot as plt
import sys

# ---------------------------------------------------------------------------
# Things that may need to configure: 
# ---------------------------------------------------------------------------
MultidRFSpinDomain_dir = ''        # where is the folder of 'Multid_RF_SpinDomain'

# if is not installed as a package, uncomment the following line
# sys.path.append(os.path.join(MultidRFSpinDomain_dir,'src')) 
# ---------------------------------------------------------------------------


# import of functions 
from sdrf import mri
from sdrf.mrpulse import spDmPulseOpt

# Some provided data examples 
def get_setups(b0num,b1num): 

    # B0 maps, Hz
    if b0num==0: 
        b0map = np.zeros((40,40,28))
    else:
        b0map = np.load(os.path.join(MultidRFSpinDomain_dir,'experimentsData/phantom_b0_{}.npy'.format(b0num)))
    # B1 maps
    if b1num==0: 
        b1map = np.ones((40,40,28))
    else:
        b1map = np.load(os.path.join(MultidRFSpinDomain_dir,'experimentsData/phantom_b1_{}.npy'.format(b0num)))
        b1map = np.abs(b1map)
    # Mask of the phantom
    mask = np.load(os.path.join(MultidRFSpinDomain_dir,'experimentsData/phantom_mask.npy'))

    # RF pulse initialization 
    rf_init = np.load(os.path.join(MultidRFSpinDomain_dir,'experimentsData/pulse3d_rf_init.npy')) # mT
    gr_init = np.load(os.path.join(MultidRFSpinDomain_dir,'experimentsData/pulse3d_gr_init.npy')) # mT/m
    dt = 5e-3 # ms
    p = {'rf_init': rf_init, 'gr_init': gr_init, 'dt': dt}

    return b0map, b1map, mask, p

if __name__=='__main__':
    
    # ------------------------------------------------------------------------------------
    # Setups : 
    #   pulse types, name of outputs, 
    # ------------------------------------------------------------------------------------
    # Use GPU if you have, it may take more than hours to run the experiments on CPU !!!
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if not torch.cuda.is_available():
        print('!!! Running without GPU can take hours !!!')

    # Design pulse type: 'excitation' or 'refocusing'
    # >>>>>>> 
    # 
    # design_pulse_type = 'refocusing'
    design_pulse_type = 'excitation'

    # Select B0, B1+ maps: 
    # some B0,B1+ data are already provided in the 'experimentData'
    # >>>>>>>
    # 
    exp_b0_id = 0
    exp_b1_id = 0





    # -----------------------------------------------------
    # Path to the output folder, for saving the results 
    # -------------------------------
    save_results_folder = os.path.join(
        MultidRFSpinDomain_dir, 
        'outputs',
        '{}_{}_{}'.format(design_pulse_type,exp_b0_id, exp_b1_id)
    )
    print('results save to: ')
    print(save_results_folder)
    _ = input('continue:')
    
    # Now load the B0, B1+ maps, and initialization for the RF pulse: 
    b0map, b1map, mask, p = get_setups(exp_b0_id, exp_b1_id)
    rf_init, gr_init, dt = p['rf_init'], p['gr_init'], p['dt']
    print('b0map: {}, b1map: {}, mask: {}'.format(b0map.shape, b1map.shape, mask.shape))
    print('rf: {}, gr: {}, dt: {}'.format(rf_init.shape, gr_init.shape, dt))







    # ------------------------------------------------------------------------------------
    # '''optimization spins target'''
    # ------------------------------------------------------------------------------------
    # Build the initialization pulse 
    pulse_init = mri.Pulse(dt=dt,                                   # unit: ms
                           rf=torch.from_numpy(rf_init).to(device), # unit: mT
                           gr=torch.from_numpy(gr_init).to(device), # unit: mT/m
                           device=device)
    # pulse_init.info()

    # Create the object of spins for simulation and optimization 
    fov = [30,30,10] # cm
    dim = [40,40,28] 
    cube = mri.SpinGrid(fov=fov,dim=dim,device=device)

    # Set B0,B1+ maps if have: 
    cube.set_mask(torch.from_numpy(mask).to(device))
    cube.set_B0map(torch.from_numpy(b0map).to(device))
    cube.set_B1map(torch.from_numpy(b1map).to(device))
    

    # Pulse optimization: 
    opti_roi = [8,8,6]                  # ROI size (unit: cm)
    opti_transition_width=[1.6,1.6,1.2] # transition region width: (unit: cm)

    # If design refocusing pulse
    if design_pulse_type=='refocusing':
        spDmPulseOpt.optimize_refocusing(
            pulse_init, cube, 
            opti_roi=opti_roi, opti_transition_width=opti_transition_width, 
            opti_roi_phase=90, 
            rfmax=0.014, grmax=24, slewrate=120, 
            niter=5, rf_niter=5, gr_niter=5, lam=1e4, 
            # 
            save_results_folder=save_results_folder,
            save_pulse_name='refocusing_pulse.mat', 
            save_simulation_results_name='simulation',
            run_optimization=True, 
            run_evaluation  =True,
            device=device
        )

            
        # Get the simulation results 
        pulse_opti = mri.Pulse.load(os.path.join(save_results_folder,'refocusing_pulse.mat'))
        results = spio.loadmat(os.path.join(save_results_folder,'simulation.mat'))
        # print(results.keys())
        
        t = pulse_opti.times.cpu().numpy()
        # 
        rf_init_c = rf_init[0] + 1j*rf_init[1]
        perf_init = results['performance_init']
        srate_init = pulse_init.get_slewrate().cpu().numpy()
        # 
        rf_opti = pulse_opti.rf.cpu().numpy()
        rf_opti_c = rf_opti[0] + 1j*rf_opti[1]
        gr_opti = pulse_opti.gr.cpu().numpy()
        perf_opti = results['performance_opti']
        srate_opti = pulse_opti.get_slewrate().cpu().numpy()
        

        # Plot the results 
        # ----------------------------------------------
        fig = plt.figure(figsize=(15,10))
        pm,pn = 4,4

        # Plot rf, gradient waveforms (initial and optimized)

        ax = plt.subplot(pm,pn,1)
        ax.plot(t,np.abs(rf_init_c))
        ax.set_title('initial rf magnitude')
        ax.set_ylabel('mT')

        ax = plt.subplot(pm,pn,2)
        ax.plot(t,np.angle(rf_init_c))
        ax.set_title('initial rf phase')

        ax = plt.subplot(pm,pn,3)
        ax.plot(t,np.abs(rf_opti_c))
        ax.set_title('optimized rf magnitude')
        ax.set_ylabel('mT')

        ax = plt.subplot(pm,pn,4)
        ax.plot(t,np.angle(rf_opti_c))
        ax.set_title('optimized rf phase')

        ax = plt.subplot(pm,pn,5)
        [ax.plot(t,gr_init[k]) for k in range(3)]
        ax.set_title('initial gradients')
        ax.set_ylabel('mT/m')
        ax.set_xlabel('ms')

        ax = plt.subplot(pm,pn,6)
        [ax.plot(t,srate_init[k]) for k in range(3)]
        ax.set_title('initial slewrate')
        ax.set_ylabel('mT/m/ms')

        ax = plt.subplot(pm,pn,7)
        [ax.plot(t,gr_opti[k]) for k in range(3)]
        ax.set_title('optimized gradients')
        ax.set_ylabel('mT/m')

        ax = plt.subplot(pm,pn,8)
        [ax.plot(t,srate_opti[k]) for k in range(3)]
        ax.set_title('final slewrate')
        ax.set_ylabel('mT/m/ms')
        
        # plot simulated profile (initial and optimized)
        
        ax = plt.subplot(pm,pn,(9,14))
        cube.plot(np.abs(perf_init),vrange=[0,1],ax=ax)
        ax.set_title(r'initial |$\beta^2$|')

        ax = plt.subplot(pm,pn,(11,16))
        cube.plot(np.abs(perf_opti),vrange=[0,1],ax=ax)
        ax.set_title(r'optimized |$\beta^2$|')

        # for axid,s in zip([9,10,13,14],[0,10,14,25]):    
        #     ax = plt.subplot(pm,pn,axid)
        #     im = ax.imshow(np.abs(perf_init[:,:,s]),vmin=0,vmax=1)
        #     plt.colorbar(im)
        #     ax.set_title(r'initial |$\beta^2$|, slice={}'.format(s+1))
        
        # for axid,s in zip([11,12,15,16],[0,10,14,25]):     
        #     ax = plt.subplot(pm,pn,axid)
        #     im = ax.imshow(np.abs(perf_opti[:,:,s]),vmin=0,vmax=1)
        #     plt.colorbar(im)
        #     ax.set_title(r'optimized |$\beta^2$|, slice={}'.format(s+1))
        

        plt.tight_layout()
        plt.savefig(os.path.join(save_results_folder,'plot.png'))


    # If design excitation pulse
    elif design_pulse_type=='excitation':
        spDmPulseOpt.optimize_excitation(
            pulse_init, cube, 
            opti_roi=opti_roi, opti_transition_width=opti_transition_width, 
            opti_roi_flip=90, opti_roi_phase=270, 
            rfmax=0.014, grmax=24, slewrate=120, 
            niter=5, rf_niter=5, gr_niter=5, lam=2e4, 
            # 
            save_results_folder=save_results_folder,
            save_pulse_name='excitation_pulse.mat', 
            save_simulation_results_name='simulation',
            run_optimization=True, 
            run_evaluation  =True,
            device=device
        )

        # Get the simulation results 
        pulse_opti = mri.Pulse.load(os.path.join(save_results_folder,'excitation_pulse.mat'))
        results = spio.loadmat(os.path.join(save_results_folder,'simulation.mat'))
        # print(results.keys())


        t = pulse_opti.times.cpu().numpy()
        # 
        rf_init_c = rf_init[0] + 1j*rf_init[1]
        perf_init = results['performance_init']
        srate_init = pulse_init.get_slewrate().cpu().numpy()
        # 
        rf_opti = pulse_opti.rf.cpu().numpy()
        rf_opti_c = rf_opti[0] + 1j*rf_opti[1]
        gr_opti = pulse_opti.gr.cpu().numpy()
        perf_opti = results['performance_opti']
        srate_opti = pulse_opti.get_slewrate().cpu().numpy()


        # Plot the results 
        # ----------------------------------------------
        fig = plt.figure(figsize=(15,10))
        pm,pn = 4,4

        # Plot rf, gradient waveforms (initial and optimized)

        ax = plt.subplot(pm,pn,1)
        ax.plot(t,np.abs(rf_init_c))
        ax.set_title('initial rf magnitude')
        ax.set_ylabel('mT')

        ax = plt.subplot(pm,pn,2)
        ax.plot(t,np.angle(rf_init_c))
        ax.set_title('initial rf phase')

        ax = plt.subplot(pm,pn,3)
        ax.plot(t,np.abs(rf_opti_c))
        ax.set_title('optimized rf magnitude')
        ax.set_ylabel('mT')

        ax = plt.subplot(pm,pn,4)
        ax.plot(t,np.angle(rf_opti_c))
        ax.set_title('optimized rf phase')

        ax = plt.subplot(pm,pn,5)
        [ax.plot(t,gr_init[k]) for k in range(3)]
        ax.set_title('initial gradients')
        ax.set_ylabel('mT/m')
        ax.set_xlabel('ms')
        
        ax = plt.subplot(pm,pn,6)
        [ax.plot(t,srate_init[k]) for k in range(3)]
        ax.set_title('initial slewrate')
        ax.set_ylabel('mT/m/ms')

        ax = plt.subplot(pm,pn,7)
        [ax.plot(t,gr_opti[k]) for k in range(3)]
        ax.set_title('optimized gradients')
        ax.set_ylabel('mT/m')

        ax = plt.subplot(pm,pn,8)
        [ax.plot(t,srate_opti[k]) for k in range(3)]
        ax.set_title('final slewrate')
        ax.set_ylabel('mT/m/ms')

        # plot simulated profile (initial and optimized)
        
        ax = plt.subplot(pm,pn,(9,14))
        cube.plot(np.abs(perf_init),vrange=[0,1],ax=ax)
        ax.set_title(r'initial |Mxy|')

        ax = plt.subplot(pm,pn,(11,16))
        cube.plot(np.abs(perf_opti),vrange=[0,1],ax=ax)
        ax.set_title(r'optimized |Mxy|')

        # for axid,s in zip([9,10,13,14],[0,10,14,25]):    
        #     ax = plt.subplot(pm,pn,axid)
        #     im = ax.imshow(np.abs(perf_init[:,:,s]),vmin=0,vmax=1)
        #     plt.colorbar(im)
        #     ax.set_title(r'initial |$Mxy$|, slice={}'.format(s+1))
        
        # for axid,s in zip([11,12,15,16],[0,10,14,25]):     
        #     ax = plt.subplot(pm,pn,axid)
        #     im = ax.imshow(np.abs(perf_opti[:,:,s]),vmin=0,vmax=1)
        #     plt.colorbar(im)
        #     ax.set_title(r'optimized |$Mxy$|, slice={}'.format(s+1))        

        plt.tight_layout()
        plt.savefig(os.path.join(save_results_folder,'plot.png'))

    else:
        pass
    
    