# scripts for designing 3D refocusing pulse
# author: jiayao

# required modules
from argparse import ArgumentParser
import torch
import numpy as np
import matplotlib.pyplot as plt

# other required modules
import mri
import mriopt
import mriutils


if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument('--b1map', type=int, default=-1, help='>0 for using input B1 map')
    PARSER.add_argument('--b0map', type=int, default=-1, help='>0 for using input B0 map')
    PARSER.add_argument('--outpulse', type=str, default='pulse_opt_log.mat', help='name for output pulse file')
    PARSER.add_argument('--gpu', type=int, default=0, help='0:cuda; -1:cpu;')
    

    HPARAMS = PARSER.parse_args()
    # HPARAMS = PARSER.parse_args([])
    # print(HPARAMS)
    setB0map = True if HPARAMS.b0map>0 else False
    setB1map = True if HPARAMS.b1map>0 else False
    setgpu = HPARAMS.gpu
    setoutputfilename = HPARAMS.outpulse


    # configuration and do pulse optimization
    print()
    # mri.MR()

    '''
    Design of 3D refocusing pulse
    Current constraints:
        rf <= 0.25 Gauss, 
        G <= 5 Gauss/cm, 
        slew-rate <= 12 Gauss/cm/ms
    '''
    # ==================================================================================
    config = {
        # system
        'gpu':setgpu,  # less than 0 means 'cpu'

        # hardware limits
        'rfmax': 0.25*0.1,  #(mT)
        'gmax': 5*10,  #(mT/ms)
        'smax': 12*10, # smax:(mT/m/ms)(slew-rate)

        # object
        'fov': [24,24,12], #[24,24,24],  [24,24,14] [39,39,12]
        'dim': [40,40,32], # matrix size [40,40,40] [45,45,35], [60,60,40] [58,58,40]

        # pulse requirements
        'rf_dt': 0.002,  #(ms) time resolution for RF
        'gr_dt': 0.01,  #(ms)
        'Gm':None,  #matrix for gr
        'pulse_type': 'refocusing',  # 'excitation', 'inversion'
        'roi_shape': 'cube', # 'cube','sphere', 'blockM', 'triangle', 'cylinder'
        'roi_offset':[0,0,0], # ROI offset
        # 
        'roi_xyz':[[-3.5,3.5],[-3.5,3.5],[-3,3]], # region or radius according to the shape
        'roi_r': 3, # if roi_shape == sphere
        'roi_height': 6,
        'weighting_amp':50,
        'weighting_sigma':8,
        'transition_width': [0.7,0.7,0.7], #[0.4,0.4,0.4],
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
        'init_pulse_folder': './',
        'init_pulse_file':'init_zoomedmri_7x7x6_ex.mat', #(4.37ms)

        # initial B0,B1 maps
        'set_B0map':setB0map,
        'B0map':'./init_B0map_modified.mat', 
        'set_B1map':setB1map,
        'B1map':'./init_B1map_modified.mat', # init_fake_B1map

        # mask for the object
        'masked': True,
        'object_mask_B0': './mask_b0.mat',
        'object_mask_B1': './mask_b1.mat',
        

        # optimization parameters
        'niter' : 5, #10
        'rf_niter' : 5,
        'gr_niter' : 5,
        'rf_niter_increase' : False,
        'rf_algo' : 'FW',
        'gr_algo' : 'GD',
        'rfloop_case' : 'all',
        'grloop_case' : 'skip_last',
        'rf_modification' : 'none',
        'rf_modify_back' : False,
        'loss_fn' : 'weighted_complex_l1',  # 'weighted_l1'
        'estimate_new_target' : True,

        # others
        'save_result': True,
        'outputfolder':'./',
        'outputdatafile':setoutputfilename,
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



    ##################################################################################



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
            print('| change dt to configured dt ...')
            pulse.change_dt(0.002)
        
    else:
        # other operations to get the initial pulse:
        pass
    pulse.show_info()
    print('| --',pulse.rf.dtype, pulse.gr.dtype)


    # whether to plot something:
    # ----------------------------------------------
    if plot_config['initial_pulse']:
        pulse.plot(picname='pic_pulse_init.png',savefig=SAVE_FIG)




    # Build target spin array, and set B0,B1 maps, and object mask :
    # ===========================================================================
    print('\n'+' spin object '.center(50,'='))
    # cube info
    # -------------------------------------------------------
    fov = config['fov']
    dim = config['dim']
    print('| FOV:',fov)
    print('| dim:',dim)
    print('|')

    cube = mri.Build_SpinArray(fov=fov,dim=dim,device=device)
    # cube.show_info()
    # print(cube.loc[0,:].min())

    # Get B0-map and B1-map
    # -------------------------------------------------------
    if config['set_B0map']:
        B0map,loc_x,loc_y,loc_z = mriutils.load_initial_b0map(config['B0map'])
        B0map = cube.map_interpolate_fn(B0map,loc_x,loc_y,loc_z)
        cube.set_B0map(B0map)
        # print('| ---', B0map.dtype)
        # print('| B0 NaN:\t', torch.sum(torch.isnan(B0map)))
    if config['set_B1map']:
        B1map,loc_x,loc_y,loc_z = mriutils.load_initial_b1map(config['B1map'])
        # print(np.max(B1map))
        # print('B1map', B1map.shape, np.count_nonzero(B1map==np.nan))
        B1map = cube.map_interpolate_fn(B1map,loc_x,loc_y,loc_z)
        # B1map[torch.nonzero(B1map<0.1)] = 1
        # B1map[torch.nonzero(B1map==0)] = 1
        cube.set_B1map(B1map)
        # print('| ---', B1map.dtype)
        # print('| B1 is 0:\t', torch.sum(B1map==0))
        # print('| B1 NaN:\t', torch.sum(torch.isnan(B1map)))
        '''note: when B1-map have 0, the program fails'''

    # in addition, get object mask
    if config['masked']:
        # select_idx = cube.get_index_ellipsoid(center=[0,-1.5,-1],abc=[9,9,3.5],inside=False)
        # cube.kappa[select_idx] = 1.0
        # cube.df[select_idx] = 0.0
        # 
        mask_b0,loc_x,loc_y,loc_z = mriutils.load_object_mask(config['object_mask_B0'])
        mask_b0 = cube.map_interpolate_fn(mask_b0,loc_x,loc_y,loc_z)
        mask_b0[torch.nonzero(mask_b0>=0.5)] = 1
        mask_b0[torch.nonzero(mask_b0<0.5)] = 0
        # 
        mask_b1,loc_x,loc_y,loc_z = mriutils.load_object_mask(config['object_mask_B1'])
        mask_b1 = cube.map_interpolate_fn(mask_b1,loc_x,loc_y,loc_z)
        mask_b1[torch.nonzero(mask_b1>=0.5)] = 1
        mask_b1[torch.nonzero(mask_b1<0.5)] = 0
        #
        print('mask info:',torch.sum(mask_b0>0.5),torch.sum(mask_b1>0.5))
        # 
        mask = mask_b0*mask_b1
        mask[torch.nonzero(cube.get_kappagrid()==0)] = 0
        cube.set_maskmap(mask)
        # print(mask.unique())
        
        # extract masked target
        cube = cube.get_unmasked()
    else:
        mask = torch.ones(dim,device=device)    
    print('| mask NaN:\t', torch.sum(torch.isnan(mask)))
    # -------------------------------------------------------
    # testmap = (1 - mask)+B1map
    # print('test',torch.sum(testmap==0))
    # mri.plot_cube_slices(cube,testmap.view(-1),picname='pictures/test_cube_slices.png',savefig=SAVE_FIG)
    # --------------------------------------------
    # plot
    print('|')
    # mri.plot_cube_slices(cube,cube.df,masked=True,valuerange=None,picname='pictures/initial_b0map.png',savefig=SAVE_FIG)
    # mri.plot_cube_slices(cube,cube.kappa,masked=True,valuerange=None,picname='pictures/initial_b1map.png',savefig=SAVE_FIG)
    # mri.plot_cube_slices(cube,mask.view(-1),masked=False,picname='pictures/initial_mask.png',title='mask',savefig=SAVE_FIG)
    # show cube info
    cube.show_info()

    print('| spin array grid: {}x{}x{} = {}'.format(dim[0],dim[1],dim[2],dim[0]*dim[1]*dim[2]))
    masked = config['masked']

    # plot b0 and b1 maps
    mri.plot_cube_B0B1maps(cube,masked=masked,picname='phantom_b0b1maps.png',savefig=SAVE_FIG)


    # Optimization parameters configureation
    # ======================================================================
    print('\n'+' opt parameters '.center(50,'='))
    # Set the weights for different regions:
    # -------------------------------------------------------
    if config['roi_shape'] == 'cube':
        print('| ROI shape:',config['roi_shape'])
        print('| --- ROI offset:',config['roi_offset'])
        roi = config['roi_xyz']
        offset = config['roi_offset']

        # get index of spins of ROI:
        x1,x2,y1,y2,z1,z2 = roi[0][0]+offset[0],roi[0][1]+offset[0],roi[1][0]+offset[1],roi[1][1]+offset[1],roi[2][0]+offset[2],roi[2][1]+offset[2]
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
            weighting[outside_idx] = 1. #1.
        weighting[roi_idx] = weighting[roi_idx]*config['weighting_amp'] # 5.
        # 
        print('| --- #non-ROI / #ROI :',(cube.num-roi_idx.shape[0])/roi_idx.shape[0])

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

    elif config['roi_shape'] == 'cylinder':
        print('| ROI shape:',config['roi_shape'])

        roi_radius = config['roi_r']
        roi_height = config['roi_height']
        offset = config['roi_offset']
        def get_cylinder_idx(center,radius,height):
            roi_idx = cube.get_index_circle([center[0],center[1]],radius=radius)
            roi_idx_z = cube.get_index([-10,10],[-10,10],[-height/2,height/2])
            roi_idx = mri.index_intersect(roi_idx,roi_idx_z)
            return roi_idx
        roi_idx = get_cylinder_idx(offset,roi_radius,roi_height)
        # ----------- weighting ----------------
        weighting = torch.ones(cube.num,device=device)

        # assign weighting within ROI:
        # offset = [0,0,0]
        # distance_squ = (cube.loc[0,:]-offset[0])**2+(cube.loc[1,:]-offset[1])**2+(cube.loc[2,:]-offset[2])**2
        # weighting = torch.exp(-0.5*distance_squ/(config['weighting_sigma']**2))
        
        # assign weighting outside ROI:
        outside_idx = mri.index_subtract(cube.get_index_all(),roi_idx)
        weighting[outside_idx] = 1. #1.
        weighting[roi_idx] = weighting[roi_idx]*config['weighting_amp'] # 5.
        # 
        print('| #non-ROI / #ROI :',(cube.num-roi_idx.shape[0])/roi_idx.shape[0])

        # assign weighting of transition band:
        inner_idx = get_cylinder_idx(offset,roi_radius,roi_height)
        larger_idx = get_cylinder_idx(offset,roi_radius+0.5,roi_height+0.5)
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
    # if plot_config['weighting']:
    #     mri.plot_cube_slices(cube,weighting,masked=masked,picname='pictures/opt_weighting.png',savefig=SAVE_FIG)
    # # plot of weighing in 1D
    # if plot_config['weighting_1d']:
    #     tmppicname = 'pictures/opt_weighting_1d.png'
    #     tmp_idx = cube.get_index([-30,30],[-0.1,0.1],[-0.1,0.1])
    #     plt.figure()
    #     plt.scatter(np.array(cube.loc[0,tmp_idx].tolist()), np.array(weighting[tmp_idx].tolist()))
    #     plt.savefig(tmppicname)
    #     print('save ... | '+tmppicname)

    # exit(0)



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
    # print('--- target',target_para_r.dtype)


    # exit(0)


    # Save variables for plotting for the paper
    if False:
        weighting_grid = cube.match_value_grid(weighting)
        target_r_grid = cube.match_value_grid(target_para_r)
        target_i_grid = cube.match_value_grid(target_para_i)
        data4paper = {
            'info': 'rf:mT, gr:mT/m, dt:ms, ',
            'weighting': np.array(weighting_grid.reshape(dim).tolist()),
            'b1map': np.array(B1map.reshape(dim).tolist()),
            'b0map': np.array(B0map.reshape(dim).tolist()),
            'target': np.array((target_r_grid+1j*target_i_grid).reshape(dim).tolist()),
            'mask': np.array(mask.reshape(dim).tolist()),
        }
        mriutils.save_variables(data4paper,'data4paper_phantom.mat')

        exit(0)
    # exit(0)



    # --- Simulation with intial pulse: ---
    # ======================================================================
    if running_config['initial_3d_simu']:
        print('\n'+' simulation '.center(50,'='))
        M = mri.blochsim(cube,Nt,dt,rf,gr,device=device)
        # 3D plot of 2D slices:
        mri.plot_cube_magnetization(cube,M,masked=masked,picname='pic_init_profile_M.png',savefig=SAVE_FIG)
        # mri.plot_cube_slices(cube,M[0,:],masked=masked,picname='pictures/init_profile_Mx.png',savefig=SAVE_FIG)
        # mri.plot_cube_slices(cube,M[1,:],masked=masked,picname='pictures/init_profile_My.png',savefig=SAVE_FIG)
        # mri.plot_cube_slices(cube,M[2,:],masked=masked,picname='pictures/init_profile_Mz.png',savefig=SAVE_FIG)
    if running_config['initial_spindomain_simu']: 
        print('\n'+' simulation '.center(50,'='))
        # simulation of spin-domain parameters:
        a,b = mri.slrsim_c(cube,Nt,dt,rf,gr,device=device)
        para = b**2
        # 3D plot of spin-domain parameters:
        mri.plot_cube_betasquare(cube,para,masked=masked,picname='pic_init_profile_betasquare.png',savefig=SAVE_FIG)
        # mri.plot_cube_slices(cube,para.abs(),masked=masked,valuerange=[0,1],picname='pictures/init_profile_betasquare_mag.png',savefig=SAVE_FIG)
        # mri.plot_cube_slices(cube,para.angle(),masked=masked,valuerange=[-3.15,3.15],picname='pictures/init_profile_betasquare_phase.png',savefig=SAVE_FIG)



    # Optimization of the pulse:
    # ======================================================================
    if running_config['optimization']:
        print('\n'+' optimization '.center(50,'='))
        loss_para_fn = lambda ar,ai,br,bi: mriopt.abloss_para_fn(ar,ai,br,bi,case=config['pulse_type']) # 'inversion' or 'refocusing'
        loss_fn = lambda xr,xi,yr,yi,weight: mriopt.abloss_c_fn(xr,xi,yr,yi,weight,case=config['loss_fn'])
        
        # perform optimization:
        '''
        - the target para real and imag is in config
        - 
        '''
        target_para = 0
        solver = mriopt.Spindomain_opt_solver()
        pulse,optinfos = solver.optimize(cube,pulse,loss_fn,loss_para_fn,config)
        # pulse,optinfos = solver.optimize_plus(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
        # exit(0)

        # additional operations after optimization

        # plot the optimization curve:
        if plot_config['loss_hist']>0:
            mriopt.plot_optinfo(optinfos,picname='optinfo.png',savefig=SAVE_FIG)
            # adding some comments in the log file:
            comments = '''design of 3d refocusing pulse'''
            optinfos['comments'] =  comments
            print(optinfos['comments'])

        # save logs:
        if config['save_result']>0:
            outputpath = config['outputfolder']+config['outputdatafile']
            mri.save_pulse(pulse,logname=outputpath,otherinfodic=optinfos)


    # show final/optimized pulse info
    # ======================================================================
    print()
    print(' final pulse '.center(50,'='))
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    pulse = mri.Pulse(rf,gr,dt,device=device)
    pulse.show_info()
    if plot_config['optimized_pulse']: 
        # mri.plot_pulse(rf,gr,dt,picname='pictures/opti_pulse.png',savefig=SAVE_FIG)
        pulse.plot(picname='pic_pulse_opti.png',savefig=SAVE_FIG)


    # Final test of designed pulse
    # ======================================================================
    # do simulation of optimized pulse
    if running_config['final_test']: 
        print()
        print(' final simu test '.center(50,'='))
        test_cube = mri.Build_SpinArray(fov=fov,dim=[40,40,dim[2]],device=device)
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
        if False:
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
                mri.plot_cube_betasquare(test_cube,para,picname='pic_opti_profile_betasqaure.png',savefig=SAVE_FIG)
                # mri.plot_cube_slices(test_cube,para.abs(),valuerange=[-1,1],picname='pictures/opt_profile_end_betasquare_mag.png',savefig=SAVE_FIG)
                # mri.plot_cube_slices(test_cube,para.angle(),valuerange=[-3.2,3.2],picname='pictures/opt_profile_end_betasquare_phase.png',savefig=SAVE_FIG)
        
