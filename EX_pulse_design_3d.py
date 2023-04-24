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
import mripul
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
    # system configure:
    sys_config = {
        'gpu':0,
    }
    # adding the pulse requirements:
    pulse_requirements = {
        'rfmax':0.25*0.1, #mT
        'gmax':5*10,
        'smax':12*10, # smax:(mT/m/ms)(slew-rate)
        'd1':0.01,
        'd2':0.05,
        'rf_dt':0.002, #(ms)
        'gr_dt':0.01, #(ms)
        'Gm':None, #matrix for gr
        'pulse_type': 'refocusing',
        'fov': [24,24,14], #[24,24,24], 
        'dim': [40,40,30], # matrix size [40,40,40] [45,45,35], [60,60,40] [58,58,40]
        'roi_shape': 'cube', # 'cube','sphere'
        'roi_xyz':[[-3,3],[-3,3],[-3,3]], # region or radius according to the shape
        'roi_r': 2, # if roi_shape == sphere
        'roi_offset':[0,0,0], # ROI offset
        'weighting_sigma':8,
        'weighting_amp':70,
        'transition_width': [0.5,0.5,0.5], #[0.4,0.4,0.4],
        'weighting_outside_custom':False, # wieghting shape
        # optimization parameters
        'niter' : 5, #8
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
        'save': True,
        'outputfolder':'/scratch/MRI_programs/RF_design_results_JY/',
        'outputdatafile':'mri_opt_log_tmp.mat',
        }

    # what are the intial inputs
    initial_config = {
        'initial_pulse': None, # which initial pulse
        'folder': '/scratch/MRI_programs/RF_design_results_JY/',
        'file': 'pulse_opt_log_23mar25_1.mat',
        'format':1, # >0 means no other process needs
        'B0map':'/scratch/JY_ImageRecon/Recon_Images/init_B0map.mat',
        'B1map':'/scratch/JY_ImageRecon/Recon_Images/init_B1map.mat',
    }

    # what results to plot
    plot_config = {
        'save_fig': True,
        'initial_pulse': 1,
        'initial_kspace': 1,
        'target': 1,
        'weighting': 1,
        'optimized_pulse': 1,
        'loss_hist': 1,
    }

    # which block to run
    running_config = {
        'initial_3d_simu': False,
        'initial_spindomain_simu': False,
        'optimization': False,
        'final_test': False,
    }

    # --------------------------------------------------------------------
    # some pulses for initial:
    # initial_config['folder'] = LOG_FOLDER
    # initial_config['folder'] = SAVED_GROUP_FOLDER
    # initial_config['file'] = 'zoomedmri_demo-10x10x8-ex.mat' # (4.03ms)
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
    if sys_config['gpu'] < 0:
        device = 'cpu'
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    pulse_requirements['device'] = device
    print('>> device =',device)
    # 
    # whether save figures
    SAVE_FIG = plot_config['save_fig']


    # Get initialization pulse:
    # ======================================================================
    if initial_config['format'] > 0:
        # simply read in some pulse:
        folder = initial_config['folder']
        filename = initial_config['file']
        data = mri.read_data(folder+filename)
        # Nt,dt,rf,gr = mri.data2pulse(data,device=device)
        # print(data.keys())

        rf,gr,dt,Nt = data['rf'],data['gr'],data['dt'].item(),data['Nt'].item()
        # print(dt)
        # print(dt==0.002)
        if dt != 0.002:
            # Interpolate the pulse
            # Such that dt for rf is 2us, dt for gr is 10us
            print('>> Change of time resolution')
            told = dt*np.arange(Nt) #(ms)
            # Interpolate gradient
            dt_rf = pulse_requirements['rf_dt'] #(ms)
            duration = int((Nt-1)*dt/dt_rf)*dt_rf
            duration = int(duration/0.01)*0.01
            tnew = np.arange(0,duration,0.010) #(ms)(dt=10us)
            Nt_new = len(tnew)
            # print('duration:',Nt*dt, Nt_new*0.01)
            # print(told[-5:])
            # print(Nt_new)
            # print('gr',gr.shape)
            # print('told',told.shape)
            # print(gr[0,:].shape)
            # print(tnew[-5:])
            gr_new = np.zeros((3,Nt_new))
            for ch in range(3):
                gr_channl = gr[ch,:]
                # print(told.shape,gr_channl.shape)
                inter_gr_fn = interpolate.interp1d(told,gr_channl,kind='nearest')
                gr_new[ch,:] = inter_gr_fn(tnew)
            # Interpolate RF
            tnew = np.arange(0,duration,dt_rf) #(ms)(dt=1us)
            Nt_new = len(tnew)
            rf_new = np.zeros((2,Nt_new))
            # print(tnew[-5:])
            for ch in range(2):
                rf_channl = rf[ch,:]
                inter_rf_fn = interpolate.interp1d(told,rf_channl,kind='nearest')
                rf_new[ch,:] = inter_rf_fn(tnew)
            print(rf_new.shape)
            print(gr_new.shape)
            #
            G = np.kron(np.identity(gr_new.shape[1]),np.ones(5).reshape(-1,1))
            # print(G[:20,:3])
            #
            rf = rf_new
            gr = gr_new@G.T
            dt = dt_rf
            Nt = Nt_new
        else:
            print('dt =',dt)
        
        # Change to Torch tensor
        rf = torch.tensor(rf.tolist(),device=device)
        gr = torch.tensor(gr.tolist(),device=device)
        print(rf.shape,gr.shape)


        # Then construct pulse
        pulse = mri.Pulse(rf,gr,dt,device=device)
    else:
        # other operations to get the initial pulse:
        pass
    pulse.show_info()
    # whether to plot something:
    if plot_config['initial_pulse']>0:
        mri.plot_pulse(rf,gr,dt,picname='pictures/mri_pic_opt_pulse_init.png',savefig=SAVE_FIG)
    if plot_config['initial_kspace']>0:
        mripul.plot_kspace(gr,Nt,dt,case='excitation',save_fig=SAVE_FIG)
        kspace = mripul.get_kspace(Nt,dt,gr,case='excitation') #(1/cm)
        # normalize by FOV
        dk = 1/torch.tensor(pulse_requirements['fov'],device=device) #(1/cm)
        print(dk)
        kspace = kspace/(dk.reshape(-1,1)) 
        print('kspace (max abs/fov^-1):',kspace.abs().amax(axis=1))
    # temporary, save initial
    # mri.save_infos(pulse,logname=pulse_requirements['outputfolder']+'zoomedmri_demo_10x10x8-ex_copy.mat',otherinfodic={})
    # exit(0)




    # Build and translate configurations and setups:
    # ===========================================================================

    # cube info
    # -------------------------------------------------------
    fov = pulse_requirements['fov']
    dim = pulse_requirements['dim']
    cube = mri.Build_SpinArray(fov=fov,dim=dim,device=device)
    cube.show_info()

    # Get B0-map and B1-map
    # -------------------------------------------------------
    B0map,loc_x,loc_y,loc_z = mriutils.load_initial_b0map(initial_config['B0map'])
    B0map = cube.map_interpolate_fn(B0map,loc_x,loc_y,loc_z)

    B1map,loc_x,loc_y,loc_z = mriutils.load_initial_b1map(initial_config['B1map'])
    print(np.max(B1map))
    print('B1map', B1map.shape, np.count_nonzero(B1map==np.nan))
    B1map = cube.map_interpolate_fn(B1map,loc_x,loc_y,loc_z)

    cube.set_B0map(B0map)
    cube.set_B1map(B1map)

    cube.show_info()

    # 
    mri.plot_cube_slices(cube,cube.kappa,picname='pictures/initial_b1map.png',savefig=SAVE_FIG)
    mri.plot_cube_slices(cube,cube.df,picname='pictures/initial_b0map.png',savefig=SAVE_FIG)

    # exit(0)


    def load_initial_b0map():
        # Adding B0 map (off-resonance)
        try:
            data = spio.loadmat(initial_config['B0map'])
            B0map = data['b0map']
            x,y,z = data['loc_x'].flatten(),data['loc_y'].flatten(),data['loc_z'].flatten()
            # print(x.shape, max(x),max(y),max(z))
            print('>> read in measured B0map',B0map.shape)
            # print(B0map.shape)
            init_fail = False
        except:
            init_fail = True
            print('>> initial B0 maps does not exist')
        # ----------------------------------------------
        # if True:
        try:
            # Interpolation
            interp_fn = interpolate.RegularGridInterpolator((x,y,z),B0map)
            loc,loc_x,loc_y,loc_z = mri.Build_SpinArray_loc(fov=fov,dim=dim)
            loc_x,loc_y,loc_z = loc_x.numpy(),loc_y.numpy(),loc_z.numpy()
            # print(max(loc_x),max(loc_y),max(loc_z))
            loc = loc.numpy().reshape(3,-1).T
            B0map = interp_fn(loc)
            B0map = torch.tensor(B0map,device=device).reshape(dim)
            print('>> interpolate B0',B0map.shape)
            # print(B0map.shape)
        except:
            init_fail = True
            print('>> interpolation fails !!')
        # ----------------------------------------------
        if init_fail:
            B0map = torch.zeros(dim,device=device) # (Hz)
        print('>> adding B0 maps... max =',torch.max(B0map.abs()).item())
        return B0map
    def load_initial_b1map():
        # Get B1 map (rf transmit)
        try:
            data = spio.loadmat(initial_config['B1map'])
            B1map = data['b1map']
            x,y,z = data['loc_x'].flatten(),data['loc_y'].flatten(),data['loc_z'].flatten()
            print('>> read in measured B1map',B1map.shape)
            # print(B1map.shape)
            init_fail = False
        except:
            init_fail = True
            print('>> initial B1 maps does not exist')
        # ----------------------------------------------
        try:
            # Interpolation
            interp_fn = interpolate.RegularGridInterpolator((x,y,z),B1map)
            print(B1map.mean())
            loc,loc_x,loc_y,loc_z = mri.Build_SpinArray_loc(fov=fov,dim=dim)
            loc_x,loc_y,loc_z = loc_x.numpy(),loc_y.numpy(),loc_z.numpy()
            loc = loc.numpy().reshape(3,-1).T
            B1map = interp_fn(loc)
            print('nan:', np.count_nonzero(np.isnan(B1map)))
            B1map[np.nonzero(B1map==np.nan)] = 1.0
            B1map = torch.tensor(B1map,device=device).reshape(dim)
            print('>> interpolate B1',B1map.shape)
            # print(B0map.shape)
            B1map.read()
        except:
            init_fail = True
            print('>> interpolation fails !!')
        # ----------------------------------------------
        if init_fail:
            B1map = torch.ones(dim,device=device)
        print('>> adding B1 maps... max =',torch.max(B1map.abs()).item())
        return B1map
    # -------------------------------------------------------
    B0map = load_initial_b0map()
    B1map = load_initial_b1map()
    
    
    
    # Build the cube object of spins
    # -----------------------------------------------
    print('FOV:',fov,', dim:',dim)
    cube = mri.Build_SpinArray(fov=fov,dim=dim,B1map=B1map,B0map=B0map,device=device)
    cube.show_info()




    # Set the weights for different regions:
    # -------------------------------------------------------
    if pulse_requirements['roi_shape'] == 'cube':
        # get index of spins of ROI:
        roi = pulse_requirements['roi_xyz']
        x1,x2,y1,y2,z1,z2 = roi[0][0],roi[0][1],roi[1][0],roi[1][1],roi[2][0],roi[2][1]
        roi_idx = cube.get_index([x1,x2],[y1,y2],[z1,z2])
        pulse_requirements['target_foi_idx'] = roi_idx

        # ----------- weighting ----------------
        weighting = torch.ones(cube.num,device=device)

        # assign weighting within ROI:
        offset = pulse_requirements['roi_offset']
        distance_squ = (cube.loc[0,:]-offset[0])**2+(cube.loc[1,:]-offset[1])**2+(cube.loc[2,:]-offset[2])**2
        weighting = torch.exp(-0.5*distance_squ/(pulse_requirements['weighting_sigma']**2))

        # assign weighting outside ROI:
        outside_idx = mri.index_subtract(cube.get_index_all(),roi_idx)
        if pulse_requirements['weighting_outside_custom']:
            offset = pulse_requirements['roi_offset']
            distance_squ = (cube.loc[0,:]-offset[0])**2+(cube.loc[1,:]-offset[1])**2+(cube.loc[2,:]-offset[2])**2
            # weighting_out = 1*torch.log(4+0.5*torch.sqrt(distance_squ)) # works
            # weighting_out = 2*torch.log(0.5*torch.sqrt(distance_squ)) # ok
            # weighting_out = 3*torch.log(0.25*torch.sqrt(distance_squ))
            weighting_out = 0.2*torch.log(1+20*torch.sqrt(distance_squ))
            # weighting_out = 0.5*torch.sqrt(distance_squ)
            weighting[outside_idx] = weighting_out[outside_idx]
        else:
            weighting[outside_idx] = 2. #1.
        weighting[roi_idx] = weighting[roi_idx]*pulse_requirements['weighting_amp'] # 5.
        # 
        print('#non-foi / #foi :',(cube.num-roi_idx.shape[0])/roi_idx.shape[0])

        # assign transition band weighting:
        d = [m/2 for m in pulse_requirements['transition_width']]
        inner_idx = cube.get_index([x1+d[0],x2-d[0]],[y1+d[1],y2-d[1]],[z1+d[2],z2-d[2]])
        larger_idx = cube.get_index([x1-d[0],x2+d[0]],[y1-d[1],y2+d[1]],[z1-d[2],z2+d[2]])
        transition_idx = mri.index_subtract(larger_idx,inner_idx)
        weighting[transition_idx] = 0.0

        # parameters for total variation within ROI:


        # plot of the target:
        # if plot_config['target']:
        #     mri.plot_cube_slices(cube,weighting,picname='pictures/mri_tmp_pic_target.png',save_fig=SAVE_FIG)
    elif pulse_requirements['roi_shape'] == 'sphere':
        # TODO (check later)
        roi_radius = pulse_requirements['roi_r']
        roi_idx = cube.get_index_ball([0,0,0],radius=roi_radius)
        pulse_requirements['target_foi_idx'] = roi_idx
        # ----------- weighting ----------------
        weighting = torch.ones(cube.num,device=device)

        # assign weighting within ROI:
        offset = pulse_requirements['roi_offset']
        distance_squ = (cube.loc[0,:]-offset[0])**2+(cube.loc[1,:]-offset[1])**2+(cube.loc[2,:]-offset[2])**2
        weighting = torch.exp(-0.5*distance_squ/(pulse_requirements['weighting_sigma']**2))
        
        # assign weighting outside ROI:
        outside_idx = mri.index_subtract(cube.get_index_all(),roi_idx)
        weighting[outside_idx] = 1. #1.
        weighting[roi_idx] = weighting[roi_idx]*pulse_requirements['weighting_amp'] # 5.
        # 
        print('#non-foi / #foi :',(cube.num-roi_idx.shape[0])/roi_idx.shape[0])

        # assign weighting of transition band:
        d = [m/2 for m in pulse_requirements['transition_width']]
        inner_idx = cube.get_index_ball([0,0,0],roi_radius-d[0])
        larger_idx = cube.get_index_ball([0,0,0],roi_radius+d[0])
        transition_idx = mri.index_subtract(larger_idx,inner_idx)
        weighting[transition_idx] = 0.0
        
        # if plot_config['target']:
        #     mri.plot_cube_slices(cube,weighting,picname='pictures/mri_tmp_pic_target.png',save_fig=SAVE_FIG)
    else:
        print('Error: no recognized shape!')
        exit(1)
    # ------------------------------
    # plot of the weighting function:
    if plot_config['target']:
        mri.plot_cube_slices(cube,weighting,picname='pictures/mri_tmp_pic_target.png',save_fig=SAVE_FIG)
    # plot of weighing in 1D:
    if True:
        tmp_idx = cube.get_index([-30,30],[-0.1,0.1],[-0.1,0.1])
        plt.figure()
        plt.scatter(np.array(cube.loc[0,tmp_idx].tolist()), np.array(weighting[tmp_idx].tolist()))
        plt.savefig('pictures/weighting_1d.png')
        print('save weighting 1d plot...')
    pulse_requirements['lossweight'] = weighting



    # Set target pulse spatial patterns:
    # ----------------------------------------------
    if pulse_requirements['pulse_type'] == 'refocusing':
        target_para_r = 0.0*torch.ones(cube.num,device=device)
        target_para_r[roi_idx] = -1.0 # 180 degree: beta^2 = 1, or -1
        target_para_i = 0.0*torch.ones(cube.num,device=device)
        target_para_i[roi_idx] = 0.0
        # may not be used, depends on method in optimization
    else:
        print('Error: no method for pulse_type',pulse_requirements['pulse_type'])
        exit(1)
    pulse_requirements['target_para_r'] = target_para_r
    pulse_requirements['target_para_i'] = target_para_i
    print('>> target',target_para_r.dtype)


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
    data4paper = {
        'weighting': np.array(weighting.reshape(dim).tolist()),
        'b1map': np.array(B1map.tolist()),
        'b0map': np.array(B0map.tolist()),
        'target': np.array((target_para_r+1j*target_para_i).reshape(dim).tolist()),
    }
    mri.save_infos(None,'data4paper_init.mat',data4paper)



    # --- Simulation with intial pulse: ---
    # ======================================================================
    if running_config['initial_3d_simu']:
        print()
        print(' simulation '.center(50,'='))
        M = mri.blochsim(cube,Nt,dt,rf,gr,device=device)
        # 3D plot of 2D slices:
        mri.plot_cube_slices(cube,M[0,:],picname='pictures/mri_pic_opt_profile_init_Mx.png',save_fig=SAVE_FIG)
        mri.plot_cube_slices(cube,M[1,:],picname='pictures/mri_pic_opt_profile_init_My.png',save_fig=SAVE_FIG)
        mri.plot_cube_slices(cube,M[2,:],picname='pictures/mri_pic_opt_profile_init_Mz.png',save_fig=SAVE_FIG)
    if running_config['initial_spindomain_simu']: 
        print()
        print(' simulation '.center(50,'='))
        # simulation of spin-domain parameters:
        a,b = mri.slrsim_c(cube,Nt,dt,rf,gr,device=device)
        para = b**2
        # 3D plot of spin-domain parameters:
        mri.plot_cube_slices(cube,para.abs(),valuerange=[-1,1],picname='pictures/mri_pic_opt_profile_init_betasquare_mag.png',save_fig=SAVE_FIG)
        mri.plot_cube_slices(cube,para.angle(),valuerange=[-1,1],picname='pictures/mri_pic_opt_profile_init_betasquare_phase.png',save_fig=SAVE_FIG)
    


    # Optimization of the pulse:
    # ======================================================================
    if running_config['optimization']:
        print()
        print(' optimization '.center(50,'='))
        loss_para_fn = lambda ar,ai,br,bi: mriopt.abloss_para_fn(ar,ai,br,bi,case=pulse_requirements['pulse_type']) # 'inversion' or 'refocusing'
        loss_fn = lambda xr,xi,yr,yi,weight: mriopt.abloss_c_fn(xr,xi,yr,yi,weight,case=pulse_requirements['loss_fn'])
        
        # perform optimization:
        target_para = 0
        solver = mriopt.Spindomain_opt_solver()
        pulse,optinfos = solver.optimize(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
        # pulse,optinfos = solver.optimize_plus(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)

        # additional operations after optimization

        # plot the optimization curve:
        if plot_config['loss_hist']>0:
            mriopt.plot_optinfo(optinfos,picname='pictures/mri_tmp_pic_optinfo.png',save_fig=SAVE_FIG)
            # adding some comments in the log file:
            comments = '''design of 3d refocusing pulse'''
            optinfos['comments'] =  comments
            print(optinfos['comments'])

        # save logs:
        if pulse_requirements['save']>0:
            outputpath = pulse_requirements['outputfolder']+pulse_requirements['outputdatafile']
            mri.save_infos(pulse,logname=outputpath,otherinfodic=optinfos)




    # Final test of designed pulse
    # ======================================================================
    # show optimized pulse info
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    pulse = mri.Pulse(rf,gr,dt,device=device)
    pulse.show_info()
    if plot_config['optimized_pulse']: 
        mri.plot_pulse(rf,gr,dt,picname='pictures/mri_pic_opt_pulse_end.png',save_fig=SAVE_FIG)
    
    # do simulation of optimized pulse
    if running_config['final_test']: 
        print()
        print(' final simu test '.center(50,'='))
        test_cube = mri.Build_SpinArray(fov=fov,dim=[40,40,40],device=device)
        if False: # let M start in transver plane
            test_cube.set_Mag(torch.tensor([0.,1.,0.],device=device))

        # --- bloch simulation ---
        with torch.no_grad():
            Mopt = mri.blochsim(test_cube,Nt,dt,rf,gr,device=device)
            if True:
                mri.plot_cube_slices(test_cube,Mopt[2,:],picname='pictures/mri_pic_opt_profile_end_Mz.png',save_fig=SAVE_FIG)
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
                mri.plot_cube_slices(test_cube,para.abs(),valuerange=[-1,1],picname='pictures/mri_pic_opt_profile_end_betasquare_mag.png',save_fig=SAVE_FIG)
                mri.plot_cube_slices(test_cube,para.angle(),valuerange=[-1,1],picname='pictures/mri_pic_opt_profile_end_betasquare_phase.png',save_fig=SAVE_FIG)
        

        if False: # another way show profile error
            mri.plot_1d_profiles([cube.loc[2,:],cube.loc[2,:]],[(Mopt[2,:]-target_cube.Mag[2,:]).abs(),weighting],picname='pictures/mri_pic_opt_1d_profile.png',save_fig=SAVE_FIG)
        # mri.plot_slices(cube,M,'z',valuerange=[-1,1])
        # a,b = mri.slrsim_c(cube,Nt,dt,rf,gr)
        # mri.plot_slr_profile(cube.loc[2,:],a,b,picname='pictures/mri_tmp_pic_slr_profile.png')
        # print(a.abs()**2 - b.abs()**2)
    