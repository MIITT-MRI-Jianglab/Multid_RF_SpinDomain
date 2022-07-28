# optimize the pulse using mri module
# author: jiayao


import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from time import time

import mri
import mriopt
import mritrajectory

print()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print('>> using device:',device)
datatype = torch.float


# make a difference for my PC and lab computer
if torch.cuda.is_available():
    LOG_FOLDER = 'logs/mri_logs/'
    SAVE_FIG = True
else:
    # LOG_FOLDER = 'mri_logs/'
    LOG_FOLDER = 'mri_logs/group220722/'
    SAVE_FIG = False



if __name__ == '__main__':
    print()
    mri.MR()

    # define some functions:
    def plot_target_illustrate(value,loc=None,valuerange=None,picname='pictures/mri_tmp_pic_target.png'):
        '''value:tensor, '''
        plt.figure(figsize=(12,8))
        # plot:
        if loc==None:
            plt.plot(value.tolist())
        else:
            loc = loc.tolist()
            plt.plot(loc,value.tolist())
        # ylim:
        if valuerange!=None:
            plt.ylim(valuerange[0],valuerange[1])
        else:
            plt.ylim([-1.05,1.05])
        # show or save:
        if SAVE_FIG:
            print('save fig...'+picname)
            plt.savefig(picname)
        else:
            plt.show()

    # ==========================================================
    # ------------------------------------
    # ==========================================================
    if True: # design 2D pulse
        # choice of initial:
        if False:
            # data = mri.read_data(LOG_FOLDER+'slr_excitation.mat')
            data = mri.read_data(LOG_FOLDER+'slr_inversion.mat')
            # data = mri.read_data(LOG_FOLDER+'slr_refocusing.mat')
            Nt,dt,rf,gr = mri.data2pulse(data)
            print(data['info'],data.keys())
            # rf = torch.rand_like(rf)*0.01
            if False:
                rf = torch.ones_like(rf)*0.01
            if False:
                rf = torch.zeros_like(rf)
                rf[0,:] = torch.ones(Nt,device=device)*0.005
        if True:
            data = mri.read_data(LOG_FOLDER+'mri_tmp_log_spiral_pulse.mat')
            Nt,dt,rf,gr = mri.data2pulse(data)
            rf = rf*0.8
            if False:
                rf = torch.zeros_like(rf)
                rf[0,:] = torch.ones(Nt,device=device)*0.01
            if False:
                gr = torch.zeros_like(gr)
                gr[0,:] = 0.01
                gr[1,:] = 0.01
        if False:
            data = mri.read_data(LOG_FOLDER+'mri_opt_log_tmp_2d_3.mat')
            try:
                Nt,dt = data['init_Nt'].item(),data['init_dt'].item()
                rf = torch.tensor(data['init_rf'].tolist(),device=device)
                gr = torch.tensor(data['init_gr'].tolist(),device=device)
                pulse_init = mri.Pulse(rf,gr,dt)
                # mri.plot_pulse(rf,gr,dt,picname='pictures/mri_tmp_pic_pulse_init.png')
            except:
                print('no init_rf and init_gr!')
        if False: # self build a random initial
            dt = 0.004 # ms
            Nt = int(10/dt)
            rf = torch.ones((2,Nt),device=device)
            gr = torch.zeros((3,Nt),device=device)
            gr[2,:] = torch.ones(Nt,device=device)*0.01
        
        # adding the pulse requirements:
        # rf <= 0.25 Gauss, G <= 5 Gauss/cm, slew-rate <= 12 Gauss/cm/ms, d1,d2:passband and stopband ripples
        pulse_requirements = {'rfmax':0.25*0.1,'gmax':5*10,'smax':12*10,'d1':0.01,'d2':0.05} # smax:(mT/m/ms)
        pulse = mri.Pulse(rf,gr,dt)
        pulse.show_info()
        if True:
            mri.plot_pulse(rf,gr,dt,picname='pictures/mri_pic_opt_pulse_init.png',save_fig=SAVE_FIG)



        # build a cube for simulation:
        cube = mri.Build_SpinArray(fov=[20,20,1],dim=[50,50,1])
        if False:
            cube.show_info()

        # build desired target:
        if True: # build a same cube, and get desired target:
            target_cube = mri.Build_SpinArray(fov=[20,20,1],dim=[50,50,1])
            # target_idx = target_cube.get_index([-4,4],[-4,4],[-1.,1.])
            target_idx = target_cube.get_index_circle(center=[0,0],radius=5)
            target_cube.set_selected_Mag(target_idx,torch.tensor([0.,0.,-1.],device=device)) # inversion
            # target_cube.set_selected_Mag(target_idx,torch.tensor([0.,-1.,0.],device=device)) # refocusing
            if True: # test of the target parameters:
                mri.plot_cube_slices(target_cube,target_cube.Mag[2,:],picname='pictures/mri_tmp_pic_target.png',save_fig=SAVE_FIG)
        if True: # target SLR parameters:
            # target_idx = cube.get_index([-4,4],[-4,4],[-1.,1.])
            target_idx = target_cube.get_index_circle(center=[0,0],radius=5)
            if False: # for inversion
                target_para = 0.0*torch.ones(cube.num,device=device) # 0 degree: |beta|^2 = 1
                target_para[target_idx] = 1.0 # 180 degree: |beta|^2 = 1
            if True: # refocusing
                target_para = 0.0*torch.ones(cube.num,device=device)
                target_para[target_idx] = -1.0 # 180 degree: beta^2 = 1
            if False: # excitation
                pass
            if False: # test of the target parameters:
                mri.plot_cube_slices(cube,target_para,picname='pictures/mri_tmp_pic_target.png',save_fig=SAVE_FIG)
                # plot_target_illustrate(target_para,loc=cube.loc[2,:])
        # wighting function at diff locations:
        weighting = torch.ones(cube.num,device=device)
        if False: # weighting for a square
            transi_wd = 0.1*8.0 # the half width of transition band
            transition_idx = cube.get_index([-4-transi_wd,-4+transi_wd],[-4-transi_wd,4+transi_wd],[-1,1])
            weighting[transition_idx] = 0.0
            transition_idx = cube.get_index([-4-transi_wd,4+transi_wd],[-4-transi_wd,-4+transi_wd],[-1,1])
            weighting[transition_idx] = 0.0
            transition_idx = cube.get_index([4-transi_wd,4+transi_wd],[-4-transi_wd,4+transi_wd],[-1,1])
            weighting[transition_idx] = 0.0
            transition_idx = cube.get_index([-4-transi_wd,4+transi_wd],[4-transi_wd,4+transi_wd],[-1,1])
            weighting[transition_idx] = 0.0
            if True:
                mri.plot_cube_slices(cube,weighting,picname='pictures/mri_tmp_pic_target.png',save_fig=SAVE_FIG)
                # plot_target_illustrate(weighting,loc=cube.loc[2,:],valuerange=[-0.1,3])
        if False: # weighting for a circle
            transi_wd = 0.1*10 # cm

        pulse_requirements['lossweight'] = weighting
        # ripples in the bands:
        band_ripple = torch.zeros(cube.num,device=device) # ideal no var
        if False: # todo, how to change for 2d region
            band_ripple = pulse_requirements['d2']*torch.ones_like(band_ripple)
            passband_idx = cube.get_index([-1,1],[-1,1],[-0.25,0.25])
            band_ripple[passband_idx] = pulse_requirements['d1']
            if False:
                plot_target_illustrate(band_ripple,loc=cube.loc[2,:],valuerange=[-0.01,1.])
        pulse_requirements['band_ripple'] = band_ripple
        # for task learning the gradient:
        if True: #TODO
            print('todo')
            # # gradient task target: alpha=+/-1 the magnetization remains the same
            # gr_task_para = 0.0*torch.ones(cube.num,device=device)
            # # weighting for not-care region
            # notcare_weighting = torch.ones(cube.num,device=device)
            # target_idx = cube.get_index([-1,1],[-1,1],[-0.25,0.25])
            # notcare_weighting[target_idx] = 0.
            # # if need opti based on a cube:
            # # gr_task_cube = mri.Build_SpinArray(fov=[1,1,1],dim=[1,1,1000])
            # if True:
            #     plot_target_illustrate(notcare_weighting,loc=cube.loc[2,:],valuerange=[-0.01,1.2])

        
        


        # simulation with intial pulse:
        if True:
            M = mri.blochsim(cube,Nt,dt,rf,gr)
            mri.plot_cube_slices(cube,M[2,:],picname='pictures/mri_pic_opt_profile_init.png',save_fig=SAVE_FIG)
            # mri.plot_magnetization_profile(cube.loc[2,:],M,picname='pictures/mri_pic_opt_profile_init.png',save_fig=SAVE_FIG)
        # simulation of SLR parameters:
        if True: # plot the refocusing measure
            a,b = mri.slrsim_c(cube,Nt,dt,rf,gr)
            para = b**2
            mri.plot_cube_slices(cube,para.real,valuerange=[-1,1],picname='pictures/mri_pic_opt_profile_init_beta.png',save_fig=SAVE_FIG)
        




        # --- optimize the pulse: ---
        if False: # >> based on bloch simulation
            def loss_para_fn(Mag):
                # Mag = Mag[2,:]
                return Mag
            def loss_fn_l2(M,Mt,weight):
                err = M[2,:]-Mt[2,:]
                return torch.sum(err**2)
            def loss_fn_weighted_l2(M,Mt,weight):
                err = M[2,:]-Mt[2,:]
                l = torch.sum((err*weight)**2)
                return l
            def loss_fn_l1(M,Mt,weight):
                err = (M-Mt).abs()
                return torch.sum(err)
            def loss_fn_weighted_l1(M,Mt,weight):
                l = (M-Mt).abs()*weight
                return l
            def loss_fn_custom1(M,Mt,weight):
                return 0
            def loss_fn(M,Mt,weight=None):
                if False:
                    threshold_layer = torch.nn.Threshold(0.0,0.0)
                    err = (M[2,:] - Mt[2,:]).abs() - band_ripple
                    err = threshold_layer(err)
                if False:
                    err = (M[2,:] - Mt[2,:]).abs()
                    # err = err/(band_ripple+1e-6)
                # err = torch.exp(10*err)-1.0 # change the error
                # l = (err**2)*torch.exp(err) # add exp weighting
                if True:
                    l = (M[2,:] - Mt[2,:])**2
                # l = err # l1-norm
                # l = err*torch.exp(err)
                if False: # my exponential transform loss function
                    nu = torch.log(band_ripple+1)/band_ripple
                    l = torch.exp(nu*err)-1.0
                    # l = l**2
                # l = err
                if weight==None:
                    weight = torch.ones_like(l)
                l = torch.sum(l*weight)
                return l
            if True: # a compute of the initial loss
                M = mri.blochsim(cube,pulse.Nt,pulse.dt,pulse.rf,pulse.gr)
                loss = loss_fn(M,target_cube.Mag,weighting)
                print('>> pulse loss measure:',loss.item())
            # ------ choose opti method ------
            if True:
                pulse,optinfos = mriopt.LBFGS(cube,pulse,target_cube.Mag,loss_fn,weighting,pulse_requirements)
                # pulse,optinfos = mriopt.FW(cube,pulse,target_cube.Mag,loss_fn,weighting,pulse_requirements)
        if True: # >> based on slr
            # choose loss parameter function:
            def loss_fn(x,y,weight=None):
                # loss function try decrease the ripples, but not effective
                if weight==None:
                    weight = torch.ones_like(x)
                if True: # l2-norm square error
                    loss_err = torch.sum(((x-y)**2)*weight)
                    l = loss_err
                if False:
                    loss_err = (x - y).abs() - band_ripple
                    l = (loss_err*weight)**2
                if False: # l1-norm error
                    loss_err = torch.sum((x-y).abs()*weight)
                    l = loss_err
                if False:
                    loss_err = ((x-y).abs()-band_ripple).abs()*weight
                    l = loss_err
                if False: # add penalty
                    err_var = torch.diff(x-y)
                    idx = target_idx[10:]
                    idx = target_idx[:-10]
                    err_var = err_var[idx]
                    loss_pen = torch.sum(err_var**2)
                    l = loss_err + 10*loss_pen
                return l
            def loss_fn_custom1(x,y,weight):
                return 0
            # choose loss function:
            loss_para_fn = lambda ar,ai,br,bi: mriopt.abloss_para_fn(ar,ai,br,bi,case='refocusing')
            loss_fn = lambda x,y,weight: mriopt.abloss_fn(x,y,weight,case='l1')
            if True: # a compute of the initial loss
                a_real,a_imag,b_real,b_imag = mri.slrsim_(cube,pulse.Nt,pulse.dt,pulse.rf,pulse.gr)
                para = loss_para_fn(a_real,a_imag,b_real,b_imag)
                loss = loss_fn(para,target_para,weighting)
                print('>> pulse loss measure:',loss.item())
            # ------ choose opti method ------
            pulse_requirements['niter'] = 5
            pulse_requirements['rf_niter'] = 2
            pulse_requirements['gr_niter'] = 2
            if True:
                # rf,gr = mriopt.slr_GD(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
                # pulse,optinfos = mriopt.slr_GD_transform(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
                # pulse,optinfos = mriopt.slr_LBFGS(cube,pulse,target_para,loss_fn_l2,loss_para_fn,pulse_requirements)
                pulse,optinfos = mriopt.slr_FW(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
                # pulse,optinfos = mriopt.slr_FW_gcon(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
        

        # plot the optimization curve:
        if True:
            mriopt.plot_optinfo(optinfos,picname='pictures/mri_tmp_pic_optinfo.png',save_fig=SAVE_FIG)
            # adding some comments in the log file:
            comments = '''
design of 2d pulse, use FW, loss function is l1
'''
            optinfos['comments'] =  comments
            print(optinfos['comments'])

        # save logs:
        if True:
            mri.save_infos(pulse,logname=LOG_FOLDER+'mri_opt_log_tmp.mat',otherinfodic=optinfos)

        # -------------------------------
        # final simple test of optimized results:
        Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
        if True: # show pulse info
            pulse = mri.Pulse(rf,gr,dt)
            pulse.show_info()
            mri.plot_pulse(rf,gr,dt,picname='pictures/mri_pic_opt_pulse_end.png',save_fig=SAVE_FIG)
        if True: # do simulation
            test_cube = mri.Build_SpinArray(fov=[20,20,1],dim=[50,50,1])
            if True: # let M start in transver plane
                test_cube.set_Mag(torch.tensor([0.,1.,0.],device=device))
            # --- bloch simulation ---
            Mopt = mri.blochsim(test_cube,Nt,dt,rf,gr)
            if True:
                mri.plot_cube_slices(test_cube,Mopt[1,:],picname='pictures/mri_pic_opt_profile_end.png',save_fig=SAVE_FIG)
            # --- spin domain simulation: ---
            if True: # plot the refocusing coefficient
                a,b = mri.slrsim_c(cube,Nt,dt,rf,gr)
                para = b**2
                mri.plot_cube_slices(cube,para.real,valuerange=[-1,1],picname='pictures/mri_pic_opt_profile_end_beta.png',save_fig=SAVE_FIG)
            # mri.plot_magnetization_profile(test_cube.loc[2,:],Mopt,picname='pictures/mri_pic_opt_profile_end.png',save_fig=SAVE_FIG)
            # mri.plot_magnetization_profile_two(cube.loc[2,:],M,test_cube.loc[2,:],Mopt,method='z',picname='pictures/mri_pic_opt_profile_compare.png',save_fig=SAVE_FIG)
            if True: # plot the error
                mri.plot_cube_slices(cube,(target_cube.Mag[2,:]-Mopt[2,:]).abs(),picname='pictures/mri_pic_opt_error_map.png',save_fig=SAVE_FIG)
                # mri.plot_1d_profile(cube.loc[2,:],(Mopt[2,:]-target_cube.Mag[2,:]).abs()*weighting,picname='pictures/mri_pic_opt_1d_profile.png',save_fig=SAVE_FIG)
            
            if False: # another way show profile error
                mri.plot_1d_profiles([cube.loc[2,:],cube.loc[2,:]],[(Mopt[2,:]-target_cube.Mag[2,:]).abs(),weighting],picname='pictures/mri_pic_opt_1d_profile.png',save_fig=SAVE_FIG)
            # mri.plot_slices(cube,M,'z',valuerange=[-1,1])
            # a,b = mri.slrsim_c(cube,Nt,dt,rf,gr)
            # mri.plot_slr_profile(cube.loc[2,:],a,b,picname='pictures/mri_tmp_pic_slr_profile.png')
            # print(a.abs()**2 - b.abs()**2)





































    # =====================================================================
    # read logs and test results
    # =====================================================================
    if False: # read in pulse and test results
        '''
        the only test of pulse simulations
        '''
        # read in the initial pulse:
        if False: # read in the tse pulse data:
            data = mri.read_data(LOG_FOLDER+'tse_eval.mat')
            Nt,dt = data['rf'].shape[0], torch.tensor(data['dt'],device=device).item()*1e3
            rf = torch.zeros((2,Nt),device=device)
            gr = torch.zeros((3,Nt),device=device)
            rf[0,:] = torch.tensor(data['rf'].tolist(),device=device).view(-1)*1e-3 # uT -> mT
            gr[2,:] = torch.tensor(data['g'].tolist(),device=device).view(-1) # mT/m/ms
            # print(Nt,dt)
            # print(data.keys())
            # print(data['dt'])
            # print(type(data['g']))
        if True:
            # data = mri.read_data(LOG_FOLDER+'mri_log_slr_exc.mat')
            # data = mri.read_data(LOG_FOLDER+'slr_excitation.mat')
            # data = mri.read_data(LOG_FOLDER+'slr_inversion.mat')
            # data = mri.read_data(LOG_FOLDER+'slr_refocusing.mat')
            data = mri.read_data(LOG_FOLDER+'mri_opt_test_log.mat')
            Nt,dt,rf,gr = mri.data2pulse(data)
            if False: # try to get the initial pulse (if it contains an initial pulse info)
                try:
                    Nt,dt = data['init_Nt'].item(),data['init_dt'].item()
                    rf = torch.tensor(data['init_rf'].tolist(),device=device)
                    gr = torch.tensor(data['init_gr'].tolist(),device=device)
                    mri.plot_pulse(init_rf,init_gr,dt,picname='pictures/mri_tmp_pic_pulse_init.png',save_fig=SAVE_FIG)
                except:
                    print('no init_rf and init_gr!')
        pulse = mri.Pulse(rf,gr,dt)
        pulse.show_info()
        mri.plot_pulse(rf,gr,dt,picname='pictures/mri_tmp_pic_pulse.png',save_fig=SAVE_FIG)

        # build a cube for simulation:
        cube = mri.Build_SpinArray(fov=[1,1,2],dim=[1,1,2000])
        if True: # M start in transverse plane
            cube.set_Mag(torch.tensor([0.,1.,0.],device=device))

        # simulation with Blcoh equation:
        M = mri.blochsim(cube,Nt,dt,rf,gr)
        mri.plot_magnetization_profile(cube.loc[2,:],M,picname='pictures/mri_tmp_pic_M_profile.png',save_fig=SAVE_FIG)
        # mri.plot_slices(cube,M,'z',valuerange=[-1,1])

        # simulation with SLR:
        a,b = mri.slrsim_c(cube,Nt,dt,rf,gr)
        mri.plot_slr_profile(cube.loc[2,:],a,b,picname='pictures/mri_tmp_pic_slr_profile.png',save_fig=SAVE_FIG)
        # mri.plot_slr_profile(cube.loc[2,:],a,b,case='z',picname='pictures/mri_tmp_pic_slr_profile.png',save_fig=SAVE_FIG)

        if True: # test B1+ sensitivity

            if True:
                # B1+ change: scale B1 at all locations
                Beff_hist = mri.spinarray_Beffhist(cube,Nt,dt,rf,gr)
                Beff_hist[:2,:,:] = Beff_hist[:2,:,:]*0.5
            if False:
                # adding B1+ changes: scale the B1 at selected locations
                B1var_idx = cube.get_index([-1,1],[-1,1],[0,10])
                Beff_hist = mri.spinarray_Beffhist(cube,Nt,dt,rf,gr)
                Beff_hist[:2,B1var_idx,:] = Beff_hist[:2,B1var_idx,:]*1.2

            # compute normalized B and phi, for all time points:
            Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
            Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
            phi_hist = -(dt*2*torch.pi*cube.gamma)*Beff_norm_hist.T #(Nt*num)
            phi_hist = phi_hist.T #(num*Nt)

            # compute the simulation
            M = mri.blochsim_array(cube,Nt,dt,Beff_unit_hist,phi_hist)
            mri.plot_magnetization_profile(cube.loc[2,:],M,picname='pictures/mri_tmp_pic_B1_variation.png',save_fig=SAVE_FIG)

            
        