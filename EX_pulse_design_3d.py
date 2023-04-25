# optimize the pulse using mri module
# design of a 3D refocusing pulse
# author: jiayao


import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from time import time
from scipy import interpolate

import mri
import mriopt
import mripulse
import mriutils

# import my other python files
# import sys
# print(sys.path)
# sys.path.append('/scratch/MRI_programs/MRI_PY/')
# import MatlabFns


# make a difference for my PC and lab computer
if torch.cuda.is_available():
    LOG_FOLDER = 'mri_logs/'
    SAVED_GROUP_FOLDER = 'mri_logs/saved_group/'
    SAVE_FIG = True
else:
    LOG_FOLDER = 'mri_logs/'
    # LOG_FOLDER = 'mri_logs/group220722/'
    SAVED_GROUP_FOLDER = 'mri_logs/saved_group/'
    SAVE_FIG = False



if __name__ == '__main__':
    print()
    # mri.MR()
    
    '''
    Design of 3D refocusing pulse
    rf <= 0.25 Gauss, G <= 5 Gauss/cm, slew-rate <= 12 Gauss/cm/ms, d1,d2:passband and stopband ripples
    '''
    # ==================================================================================
    config = {
        # system
        'gpu':0,  # less than 0 means 'cpu'

        # hardware limits
        'rfmax': 0.25*0.1,  #(mT)
        'gmax': 5*10,  #(mT/ms)
        'smax': 12*10, # smax:(mT/m/ms)(slew-rate)

        # object
        'fov': [24,24,14], #[24,24,24], 
        'dim': [40,40,30], # matrix size [40,40,40] [45,45,35], [60,60,40] [58,58,40]

        # pulse requirements
        'rf_dt': 0.002,  #(ms) time resolution for RF
        'gr_dt': 0.01,  #(ms)
        'Gm':None,  #matrix for gr
        'pulse_type': 'refocusing',  # 'excitation', 'inversion'
        'roi_shape': 'MichiganM', # 'cube','sphere', 'MichiganM'
        'roi_xyz':[[-3,3],[-3,3],[-3,3]], # region or radius according to the shape
        'roi_r': 2, # if roi_shape == sphere
        'roi_offset':[0,0,0], # ROI offset
        'weighting_sigma':8,
        'weighting_amp':10,
        'transition_width': [0.5,0.5,0.5], #[0.4,0.4,0.4],
        'weighting_outside_custom':False, # wieghting shape
        # 'd1':0.01,
        # 'd2':0.05,
        'roi_index': None,  # computed later
        'transition_index': None,  # computed later
        'lossweight' : None,  # assign later
        'target_para_r': None,  # real part of target parameters
        'target_para_i': None,  # imag part of target parameters

        # what are the initial inputs
        'init_simple_readin': True,
        'init_pulse_folder': '/scratch/MRI_programs/RF_design_results_JY/',
        'init_pulse_file': 'pulse_opt_log_23mar25_1.mat',
        'set_B0map':False,
        'B0map':'/scratch/JY_ImageRecon/Recon_Images/init_fake_B0map.mat',
        'set_B1map':False,
        'B1map':'/scratch/JY_ImageRecon/Recon_Images/init_fake_B1map.mat',
        'init_maps_region_adjust': False,

        # optimization parameters
        'niter' : 6, #8
        'rf_niter' : 5,
        'gr_niter' : 5,
        'rf_niter_increase' : False,
        'rf_algo' : 'FW',
        'gr_algo' : 'GD',
        'rfloop_case' : 'all',
        'grloop_case' : 'skip_last',
        'rf_modification' : 'none',
        'rf_modify_back' : False,
        'loss_fn' : 'weighted_l1',
        'estimate_new_target' : True,

        # others
        'save_result': True,
        'outputfolder':'/scratch/MRI_programs/RF_design_results_JY/',
        'outputdatafile':'mri_opt_log_tmp.mat',
    }

    # what results to plot
    plot_config = {
        'save_fig': True,
        'initial_pulse': True,
        'initial_kspace': True,
        # 'target': True,
        'weighting': True,
        'weighting_1d': True,
        'optimized_pulse': True,
        'loss_hist': True,
    }

    # which block to run
    running_config = {
        'initial_3d_simu': True,
        'initial_spindomain_simu': True,
        'optimization': True,
        'final_test': True,
    }

    # --------------------------------------------------------------------
    # some pulses for initial:
    # initial_config['folder'] = LOG_FOLDER
    # initial_config['folder'] = SAVED_GROUP_FOLDER
    config['init_pulse_file'] = 'zoomedmri_demo-10x10x8-ex.mat' # (4.03ms)
    # config['init_pulse_file'] = 'mri_opt_log_tmp_1114_rf4.mat'
    # --------------------------------------------------------------------
    # initial_config['file'] = 'mri_opt_log_tmp.mat' 

    # initial_config['file'] = 'zoomedmri_demo-10x10x4-ex.mat'
    # initial_config['file'] = 'zoomedmri_demo-8x8x4-ex.mat'

    # initial_config['file'] = 'zoomedmri_demo-10x10x5-ex.mat'
    # initial_config['file'] = 'zoomedmri_demo-10x10x5-ex-off2x4.mat'
    
    # initial_config['file'] = 'zoomedmri_demo-6x6x6-ex.mat'
    # initial_config['file'] = 'zoomedmri_demo-7x7x6-ex.mat'
    # initial_config['file'] = 'zoomedmri_demo-7x7x6-ex4.mat'
    # initial_config['file'] = 'zoomedmri_demo-8x8x7-ex.mat'
    # initial_config['file'] = 'zoomedmri_demo-8x8x8-ex.mat'

    # initial_config['file'] = 'zoomedmri_demo_7x7x7_ex.mat'
    # initial_config['file'] = 'zoomedmri_demo_7x7x7_ex1.mat'
    # initial_config['file'] = 'zoomedmri_demo-8x8x6-ex.mat'
    # initial_config['file'] = 'zoomedmri_demo-sphere-d6.mat'
    # initial_config['file'] = 'zoomedmri_demo-sphere-d8.mat'





    # GPU, and others
    # =============================================================================
    print(' RF pulse design configuring '.center(50,'='))
    # 
    # cpu or gpu
    if config['gpu'] < 0:
        device = 'cpu'
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    config['device'] = device
    print('| device =',device)
    # 
    # whether save figures
    SAVE_FIG = plot_config['save_fig']


    # Get initialization pulse:
    # ======================================================================
    if config['init_simple_readin']:
        # simply read in some pulse:
        folder = config['init_pulse_folder']
        filename = config['init_pulse_file']
        data = mri.read_data(folder+filename)
        # Nt,dt,rf,gr = mri.data2pulse(data,device=device)
        # print(data.keys())
        # print(type(rf))

        rf,gr,dt,Nt = data['rf'],data['gr'],data['dt'].item(),data['Nt'].item()
        # Change to Torch tensor
        rf = torch.tensor(rf.tolist(),device=device)
        gr = torch.tensor(gr.tolist(),device=device)
        # print(rf.shape,gr.shape)

        # Construct the pulse object
        pulse = mri.Pulse(rf,gr,dt,device=device)

        # If needs to change time resolution
        if dt != 0.002:
            print('| need to change dt ...')
            pulse.change_dt(0.002)
        
        # additional step on gradient: (try to be consistant with scanner)
        # G = np.kron(np.identity(gr_new.shape[1]),np.ones(5).reshape(-1,1))

    else:
        # other operations to get the initial pulse:
        pass
    pulse.show_info()
    print('| --',pulse.rf.dtype, pulse.gr.dtype)

    # whether to plot something:
    # ----------------------------------------------
    # plot initial pulse
    if plot_config['initial_pulse']:
        mri.plot_pulse(rf,gr,dt,picname='pictures/initial_pulse.png',savefig=SAVE_FIG)
    # plot of initial excitation k-space
    if plot_config['initial_kspace']:
        mripulse.plot_kspace(gr,Nt,dt,case='excitation',picname='pictures/initial_kspace.png',save_fig=SAVE_FIG)
        kspace = mripulse.get_kspace(Nt,dt,gr,case='excitation') #(1/cm)
        # normalize by FOV
        dk = 1/torch.tensor(config['fov'],device=device) #(1/cm)
        print(dk)
        kspace = kspace/(dk.reshape(-1,1)) 
        print('kspace (max abs/fov^-1):',kspace.abs().amax(axis=1))
    # temporary, save initial
    # mri.save_infos(pulse,logname=pulse_requirements['outputfolder']+'zoomedmri_demo_10x10x8-ex_copy.mat',otherinfodic={})
    # exit(0)




    # Build and :
    # ===========================================================================
    print('\n'+' spin object '.center(50,'='))
    # cube info
    # -------------------------------------------------------
    fov = config['fov']
    dim = config['dim']
    print('| FOV:',fov)
    print('| dim:',dim)

    cube = mri.Build_SpinArray(fov=fov,dim=dim,device=device)
    # cube.show_info()

    # Get B0-map and B1-map
    # -------------------------------------------------------
    if config['set_B0map']:
        B0map,loc_x,loc_y,loc_z = mriutils.load_initial_b0map(config['B0map'])
        B0map = cube.map_interpolate_fn(B0map,loc_x,loc_y,loc_z)
        cube.set_B0map(B0map)
        print('| --', B0map.dtype)
    if config['set_B1map']:
        B1map,loc_x,loc_y,loc_z = mriutils.load_initial_b1map(config['B1map'])
        # print(np.max(B1map))
        # print('B1map', B1map.shape, np.count_nonzero(B1map==np.nan))
        B1map = cube.map_interpolate_fn(B1map,loc_x,loc_y,loc_z)
        cube.set_B1map(B1map)
        print('| --', B1map.dtype)
    
    # in addition, ignore the region outside the object in the map
    if config['init_maps_region_adjust']:
        select_idx = cube.get_index_ellipsoid(center=[0,-1.5,-1],abc=[9,9,3.5],inside=False)
        cube.kappa[select_idx] = 1.0
        cube.df[select_idx] = 0.0
    # -------------------------------------------------------
    # plot
    mri.plot_cube_slices(cube,cube.kappa,picname='pictures/initial_b1map.png',savefig=SAVE_FIG)
    mri.plot_cube_slices(cube,cube.df,picname='pictures/initial_b0map.png',savefig=SAVE_FIG)
    # show cube info
    cube.show_info()

    # exit(0)




    # Optimization parameters configureation
    # ======================================================================
    print('\n'+' opt parameters '.center(50,'='))
    # Set the weights for different regions:
    # -------------------------------------------------------
    if config['roi_shape'] == 'cube':
        roi = config['roi_xyz']
        offset = config['roi_offset']

        # get index of spins of ROI:
        x1,x2,y1,y2,z1,z2 = roi[0][0],roi[0][1],roi[1][0],roi[1][1],roi[2][0],roi[2][1]
        roi_idx = cube.get_index([x1,x2],[y1,y2],[z1,z2])

        # ----------- weighting ----------------
        weighting = torch.ones(cube.num,device=device)

        # assign weighting within ROI:
        distance_squ = (cube.loc[0,:]-offset[0])**2+(cube.loc[1,:]-offset[1])**2+(cube.loc[2,:]-offset[2])**2
        weighting = torch.exp(-0.5*distance_squ/(config['weighting_sigma']**2))

        # assign weighting outside ROI:
        outside_idx = mri.index_subtract(cube.get_index_all(),roi_idx)
        if config['weighting_outside_custom']:
            offset = config['roi_offset']
            distance_squ = (cube.loc[0,:]-offset[0])**2+(cube.loc[1,:]-offset[1])**2+(cube.loc[2,:]-offset[2])**2
            # weighting_out = 1*torch.log(4+0.5*torch.sqrt(distance_squ)) # works
            # weighting_out = 2*torch.log(0.5*torch.sqrt(distance_squ)) # ok
            # weighting_out = 3*torch.log(0.25*torch.sqrt(distance_squ))
            weighting_out = 0.2*torch.log(1+20*torch.sqrt(distance_squ))
            # weighting_out = 0.5*torch.sqrt(distance_squ)
            weighting[outside_idx] = weighting_out[outside_idx]
        else:
            weighting[outside_idx] = 2. #1.
        weighting[roi_idx] = weighting[roi_idx]*config['weighting_amp'] # 5.
        # 
        print('| #non-ROI / #ROI :',(cube.num-roi_idx.shape[0])/roi_idx.shape[0])

        # assign transition band weighting:
        d = [m/2 for m in config['transition_width']]
        inner_idx = cube.get_index([x1+d[0],x2-d[0]],[y1+d[1],y2-d[1]],[z1+d[2],z2-d[2]])
        larger_idx = cube.get_index([x1-d[0],x2+d[0]],[y1-d[1],y2+d[1]],[z1-d[2],z2+d[2]])
        transition_idx = mri.index_subtract(larger_idx,inner_idx)
        weighting[transition_idx] = 0.0

        # parameters for total variation within ROI:
        # TODO

    elif config['roi_shape'] == 'sphere':
        # TODO (check later)
        roi_radius = config['roi_r']
        roi_idx = cube.get_index_ball([0,0,0],radius=roi_radius)
        # ----------- weighting ----------------
        weighting = torch.ones(cube.num,device=device)

        # assign weighting within ROI:
        offset = config['roi_offset']
        distance_squ = (cube.loc[0,:]-offset[0])**2+(cube.loc[1,:]-offset[1])**2+(cube.loc[2,:]-offset[2])**2
        weighting = torch.exp(-0.5*distance_squ/(config['weighting_sigma']**2))
        
        # assign weighting outside ROI:
        outside_idx = mri.index_subtract(cube.get_index_all(),roi_idx)
        weighting[outside_idx] = 1. #1.
        weighting[roi_idx] = weighting[roi_idx]*config['weighting_amp'] # 5.
        # 
        print('| #non-ROI / #ROI :',(cube.num-roi_idx.shape[0])/roi_idx.shape[0])

        # assign weighting of transition band:
        d = [m/2 for m in config['transition_width']]
        inner_idx = cube.get_index_ball([0,0,0],roi_radius-d[0])
        larger_idx = cube.get_index_ball([0,0,0],roi_radius+d[0])
        transition_idx = mri.index_subtract(larger_idx,inner_idx)
        weighting[transition_idx] = 0.0
        
    elif config['roi_shape'] == 'MichiganM':
        trajx = [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
                -4, -3, -2, -1, 0, 1, 2, 3, 4,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        trajy = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                4, 3, 2, 1, 0, 1, 2, 3 ,4,
                5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]
        print('| build target region .. | ',len(trajx),len(trajy))
        # --------------------------------------
        def get_idx_along_traj(dx,dy,dz):
            cx,cy = trajx[0],trajy[0]
            roi_idx = cube.get_index(xlim=[cx-dx,cx+dx],ylim=[cy-dy,cy+dy],zlim=[-dz,dz])
            for t in range(len(trajx)):
                cx,cy = trajx[t],trajy[t]
                tmp_idx = cube.get_index(xlim=[cx-dx,cx+dx],ylim=[cy-dy,cy+dy],zlim=[-dz,dz])
                roi_idx = mri.index_union(roi_idx,tmp_idx)
            return roi_idx
        dx,dy,dz = 1.5,1.5,3
        roi_idx = get_idx_along_traj(dx,dy,dz)
        # 
        d = 0.5
        dx,dy,dz = 2+d,2+d,3+d
        larger_idx = get_idx_along_traj(dx,dy,dz)
        dx,dy,dz = 2-d,2-d,3-d
        inner_idx = get_idx_along_traj(dx,dy,dz)
        # ----------- weighting ----------------
        weighting = torch.ones(cube.num,device=device)
        
        # assign weighting outside ROI:
        outside_idx = mri.index_subtract(cube.get_index_all(),roi_idx)
        weighting[outside_idx] = 1. #1.
        weighting[roi_idx] = weighting[roi_idx]*config['weighting_amp'] # 5.
        # 
        print('| #non-ROI / #ROI :',(cube.num-roi_idx.shape[0])/roi_idx.shape[0])

        # assign weighting of transition band:
        print('| \tcurrent no transition band assigned')
        transition_idx = mri.index_subtract(larger_idx,inner_idx)
        weighting[transition_idx] = 0.0

    else:
        print('Error: no recognized shape!')
        exit(1)

    # add parameter
    # -------------------------------------------------
    config['lossweight'] = weighting
    config['target_foi_idx'] = roi_idx
    config['roi_index'] = roi_idx
    config['transition_index'] = transition_idx
    
    # plot of the weighting function:
    # -------------------------------------------------
    # plot of target
    if plot_config['weighting']:
        mri.plot_cube_slices(cube,weighting,picname='pictures/opt_weighting.png',savefig=SAVE_FIG)
    # plot of weighing in 1D
    if plot_config['weighting_1d']:
        tmppicname = 'pictures/opt_weighting_1d.png'
        tmp_idx = cube.get_index([-30,30],[-0.1,0.1],[-0.1,0.1])
        plt.figure()
        plt.scatter(np.array(cube.loc[0,tmp_idx].tolist()), np.array(weighting[tmp_idx].tolist()))
        plt.savefig(tmppicname)
        print('save ... | '+tmppicname)
    

    
    # Set target pulse spatial patterns:
    # -------------------------------------------------
    if config['pulse_type'] == 'refocusing':
        target_para_r = 0.0*torch.ones(cube.num,device=device)
        target_para_r[roi_idx] = -1.0 # 180 degree: beta^2 = 1, or -1
        target_para_i = 0.0*torch.ones(cube.num,device=device)
        target_para_i[roi_idx] = 0.0
        # may not be used, depends on method in optimization
    else:
        print('Error: no method for pulse_type',config['pulse_type'])
        exit(1)
    # -------------------------------------------------
    config['target_para_r'] = target_para_r
    config['target_para_i'] = target_para_i
    print('--- target',target_para_r.dtype)


    # exit(0)
    


    # Some other tests
    if False: # test of the target parameters:
        mri.plot_cube_slices(cube,target_para_r,picname='pictures/mri_tmp_pic_target.png',save_fig=SAVE_FIG)
    if False:
        # save the target
        data = {'para': target_para_r+1j*target_para_i}
        data['para'] = np.array(data['para'].reshape(dim).tolist())
        print('>> save data...'+'pattern_target.mat')
        print('\tsaved infos:',data.keys())
        spio.savemat('pattern_target.mat',data)
    if False:
        # save the target
        data = {'para': weighting.reshape(dim)}
        data['para'] = np.array(data['para'].tolist())
        print('>> save data...'+'pattern_weight.mat')
        print('\tsaved infos:',data.keys())
        spio.savemat('pattern_weight.mat',data)


    # ripples #TODO (not going to use this)
    # band_ripple = torch.zeros(cube.num,device=device) # ideal no var
    # pulse_requirements['band_ripple'] = band_ripple

    # Save variables for plotting for the paper
    # data4paper = {
    #     'weighting': np.array(weighting.reshape(dim).tolist()),
    #     'b1map': np.array(B1map.tolist()),
    #     'b0map': np.array(B0map.tolist()),
    #     'target': np.array((target_para_r+1j*target_para_i).reshape(dim).tolist()),
    # }
    # mriutils.save_variables(data4paper,'data4paper_init.mat')

    # exit(0)



    # --- Simulation with intial pulse: ---
    # ======================================================================
    if running_config['initial_3d_simu']:
        print('\n'+' simulation '.center(50,'='))
        M = mri.blochsim(cube,Nt,dt,rf,gr,device=device)
        # 3D plot of 2D slices:
        mri.plot_cube_slices(cube,M[0,:],picname='pictures/init_profile_Mx.png',savefig=SAVE_FIG)
        mri.plot_cube_slices(cube,M[1,:],picname='pictures/init_profile_My.png',savefig=SAVE_FIG)
        mri.plot_cube_slices(cube,M[2,:],picname='pictures/init_profile_Mz.png',savefig=SAVE_FIG)
    if running_config['initial_spindomain_simu']: 
        print('\n'+' simulation '.center(50,'='))
        # simulation of spin-domain parameters:
        a,b = mri.slrsim_c(cube,Nt,dt,rf,gr,device=device)
        para = b**2
        # 3D plot of spin-domain parameters:
        mri.plot_cube_slices(cube,para.abs(),valuerange=[-1,1],title=r'$|\beta^2|$',picname='pictures/init_profile_betasquare_mag.png',savefig=SAVE_FIG)
        mri.plot_cube_slices(cube,para.angle(),valuerange=[-1,1],title=r'$\angle\beta^2$',picname='pictures/init_profile_betasquare_phase.png',savefig=SAVE_FIG)
    


    # Optimization of the pulse:
    # ======================================================================
    if running_config['optimization']:
        print('\n'+' optimization '.center(50,'='))
        loss_para_fn = lambda ar,ai,br,bi: mriopt.abloss_para_fn(ar,ai,br,bi,case=config['pulse_type']) # 'inversion' or 'refocusing'
        loss_fn = lambda xr,xi,yr,yi,weight: mriopt.abloss_c_fn(xr,xi,yr,yi,weight,case=config['loss_fn'])
        
        # perform optimization:
        target_para = 0
        solver = mriopt.Spindomain_opt_solver()
        pulse,optinfos = solver.optimize(cube,pulse,target_para,loss_fn,loss_para_fn,config)
        # pulse,optinfos = solver.optimize_plus(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
        # exit(0)

        # additional operations after optimization

        # plot the optimization curve:
        if plot_config['loss_hist']>0:
            mriopt.plot_optinfo(optinfos,picname='pictures/optinfo.png',savefig=SAVE_FIG)
            # adding some comments in the log file:
            comments = '''design of 3d refocusing pulse'''
            optinfos['comments'] =  comments
            print(optinfos['comments'])

        # save logs:
        if config['save_result']>0:
            outputpath = config['outputfolder']+config['outputdatafile']
            mri.save_pulse(pulse,logname=outputpath,otherinfodic=optinfos)

    # just for change time resolution and save a pulse
    # if config['save_result']>0:
    #         outputpath = config['outputfolder']+config['outputdatafile']
    #         mri.save_pulse(pulse,logname=outputpath,otherinfodic={})
    

    # show optimized pulse info
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    pulse = mri.Pulse(rf,gr,dt,device=device)
    pulse.show_info()
    if plot_config['optimized_pulse']: 
        mri.plot_pulse(rf,gr,dt,picname='pictures/opti_pulse.png',savefig=SAVE_FIG)
    
    
    # Final test of designed pulse
    # ======================================================================
    # do simulation of optimized pulse
    if running_config['final_test']: 
        print()
        print(' final simu test '.center(50,'='))
        test_cube = mri.Build_SpinArray(fov=fov,dim=[40,40,40],device=device)
        if config['set_B0map']: # set B0, B1 maps
            B0map,loc_x,loc_y,loc_z = mriutils.load_initial_b0map(config['B0map'])
            B0map = test_cube.map_interpolate_fn(B0map,loc_x,loc_y,loc_z)
            test_cube.set_B0map(B0map)
        if config['set_B1map']:
            B1map,loc_x,loc_y,loc_z = mriutils.load_initial_b1map(config['B1map'])
            B1map = test_cube.map_interpolate_fn(B1map,loc_x,loc_y,loc_z)
            test_cube.set_B1map(B1map)
        if False: # let M start in transver plane
            test_cube.set_Mag(torch.tensor([0.,1.,0.],device=device))

        # --- bloch simulation ---
        with torch.no_grad():
            Mopt = mri.blochsim(test_cube,Nt,dt,rf,gr,device=device)
            if True:
                mri.plot_cube_slices(test_cube,Mopt[2,:],picname='pictures/opt_profile_end_Mz.png',savefig=SAVE_FIG)
            if False: 
                # plot the error
                mri.plot_cube_slices(cube,(target_cube.Mag[2,:]-Mopt[2,:]).abs(),picname='pictures/mri_pic_opt_error_map.png',save_fig=SAVE_FIG)
        
        # --- spinor simulation: ---
        if True: 
            # Try to empty space, wonder if it helps
            torch.cuda.empty_cache()
            # plot the refocusing coefficient
            with torch.no_grad():
                a,b = mri.slrsim_c(test_cube,Nt,dt,rf,gr,device=device)
                para = b**2
                mri.plot_cube_slices(test_cube,para.abs(),valuerange=[-1,1],picname='pictures/opt_profile_end_betasquare_mag.png',savefig=SAVE_FIG)
                mri.plot_cube_slices(test_cube,para.angle(),valuerange=[-1,1],picname='pictures/opt_profile_end_betasquare_phase.png',savefig=SAVE_FIG)
        

        if False: # another way show profile error
            mri.plot_1d_profiles([cube.loc[2,:],cube.loc[2,:]],[(Mopt[2,:]-target_cube.Mag[2,:]).abs(),weighting],picname='pictures/mri_pic_opt_1d_profile.png',save_fig=SAVE_FIG)
        # mri.plot_slices(cube,M,'z',valuerange=[-1,1])
        # a,b = mri.slrsim_c(cube,Nt,dt,rf,gr)
        # mri.plot_slr_profile(cube.loc[2,:],a,b,picname='pictures/mri_tmp_pic_slr_profile.png')
        # print(a.abs()**2 - b.abs()**2)
    
