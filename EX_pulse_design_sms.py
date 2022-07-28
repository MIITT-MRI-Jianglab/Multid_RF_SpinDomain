# optimize the pulse using mri module
# author: jiayao

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import mri
import mriopt
from time import time

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
    LOG_FOLDER = 'mri_logs/'
    SAVE_FIG = False



if __name__ == '__main__':
    print()

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


    # ==================================================================
    if False: # based on bloch simulation
        '''
        do optimization based on the bloch simulation
        '''
        # read in the initial pulse:
        if True:
            # data = mri.read_data(LOG_FOLDER+'mri_log_slr_exc.mat')
            # data = mri.read_data(LOG_FOLDER+'slr_excitation.mat')
            # data = mri.read_data(LOG_FOLDER+'slr_inversion.mat')
            data = mri.read_data(LOG_FOLDER+'slr_refocusing.mat')
            Nt,dt,rf,gr = mri.data2pulse(data)
            # if start with random number:
            # rf = 0.01*torch.rand((2,Nt),device=device)
            rf = 0.01*torch.ones((2,Nt),device=device)
        pulse_requirements = {'rfmax':0.25*0.1,'gmax':5*10,'smax':12*10} # smax:(mT/m/ms)
        pulse = mri.Pulse(rf,gr,dt)
        pulse.show_info()
        mri.plot_pulse(rf,gr,dt,picname='pictures/mri_pic_opt_pulse_init.png',save_fig=SAVE_FIG)

        # build a cube for simulation:
        cube = mri.Build_SpinArray(fov=[1,1,2],dim=[1,1,1000])
        if False: # set magnetization starts in transverse plane
            cube.set_Mag(torch.tensor([0.,1.,0.],device=device))

        # build a same cube for design target:
        target_cube = mri.Build_SpinArray(fov=[1,1,2],dim=[1,1,1000])
        if False: # modification for the refocusing case:
            target_cube.set_Mag(torch.tensor([0.,1.,0.],device=device))
            # M = mri.blochsim(target_cube,Nt,dt,torch.zeros((2,Nt),device=device),gr)
            M = mri.blochsim(target_cube,Nt,dt,rf,gr)
            target_cube.Mag = M
        if True:
            target_idx = target_cube.get_index([-1,1],[-1,1],[-0.25,0.25])
            target_cube.set_selected_Mag(target_idx,torch.tensor([0.,0.,-1.],device=device)) # inversion
            # target_cube.set_selected_Mag(target_idx,torch.tensor([0.,-1.,0.],device=device)) # refocusing
        if False: # build a same cube but for multislice:
            target_slice_list = []
            for target_slice_pos in [-11,-9,-7,-5,-3,-1,1,3,5,7,9,11]:
                target_slice_list.append([target_slice_pos-0.05,target_slice_pos+0.05])        
            for target_slice in target_slice_list:
                target_idx = cube.get_index([-1,1],[-1,1],target_slice)
                target_cube.set_selected_Mag(target_idx,torch.tensor([0.,0.,-1.],device=device))
        if True: # test of the target parameters:
            plt.figure(figsize=(12,8))
            plt.plot(target_cube.Mag[2,:].tolist())
            plt.ylim([-1.05,1.05])
            if SAVE_FIG:
                plt.savefig('pictures/mri_tmp_pic_target.png')
            else:
                plt.show()

        
        # simulation with intial pulse:
        M = mri.blochsim(cube,Nt,dt,rf,gr)
        mri.plot_magnetization_profile(cube.loc[2,:],M,picname='pictures/mri_pic_opt_profile_init.png',save_fig=SAVE_FIG)
        # mri.plot_slices(cube,M,'z',valuerange=[-1,1])

        # choose loss function:
        def loss_para_fn(Mag):
            # Mag = Mag[2,:]
            return Mag
        def loss_fn(M,Mt):
            l = (M[2,:] - Mt[2,:])**2
            l = torch.sum(l)
            return l
        # loss_fn = mriopt.loss_l2
        
        # optimize the pulse, choose methods for optimizations:
        # pulse,optinfos = mriopt.GD(cube,pulse,target_cube.Mag,loss_fn,loss_para_fn,pulse_requirements)
        pulse,optinfos = mriopt.LBFGS(cube,pulse,target_cube.Mag,loss_fn,loss_para_fn,pulse_requirements)
        # pulse,optinfos = mriopt.FW(cube,pulse,target_cube.Mag,loss_fn,loss_para_fn,pulse_requirements)
        # rf,gr = mriopt.window_GD(cube,pulse,target_cube.Mag,loss_fn,loss_para_fn,pulse_requirements)
        
        if True:
            mri.save_infos(pulse,logname=LOG_FOLDER+'mri_opt_test_log.mat',otherinfodic=optinfos)

        # -------------------------------
        # final test of optimized results:
        Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
        if True:
            pulse = mri.Pulse(rf,gr,dt)
            pulse.show_info()
        if True:
            test_cube = mri.Build_SpinArray(fov=[1,1,2],dim=[1,1,2000])
            if False: # let M start in transver plane
                test_cube.set_Mag(torch.tensor([0.,1.,0.],device=device))
            M = mri.blochsim(test_cube,Nt,dt,rf,gr)
            mri.plot_pulse(rf,gr,dt,picname='pictures/mri_pic_opt_pulse_end.png',save_fig=SAVE_FIG)
            mri.plot_magnetization_profile(test_cube.loc[2,:],M,picname='pictures/mri_pic_opt_profile_end.png',save_fig=SAVE_FIG)
            # mri.plot_slices(cube,M,'z',valuerange=[-1,1])
        # test of B1 sensitivity:
        if False:
            notes = '''simulate the B1 sensitivity,'''
            print(notes)
            test_cube = mri.Build_SpinArray(fov=[1,1,1],dim=[1,1,100])

    # ===================================================================
    if False: # based on slr
        '''
        optimization, based on SLR parameters
        '''
        # read in the initial pulse:
        if False: # read in the tse pulse data: for SMS
            data = mri.read_data(LOG_FOLDER+'tse_eval.mat')
            Nt,dt = data['rf'].shape[0], torch.tensor(data['dt'],device=device).item()*1e3
            rf = torch.zeros((2,Nt),device=device)
            gr = torch.zeros((3,Nt),device=device)
            rf[0,:] = torch.tensor(data['rf'].tolist(),device=device).view(-1)*1e-3
            gr[2,:] = torch.tensor(data['g'].tolist(),device=device).view(-1)*1e-2
            # print(Nt,dt)
            # print(data.keys())
        if True:
            # data = mri.read_data(LOG_FOLDER+'mri_log_slr_exc.mat')
            # data = mri.read_data(LOG_FOLDER+'slr_excitation.mat')
            data = mri.read_data(LOG_FOLDER+'slr_inversion.mat')
            # data = mri.read_data(LOG_FOLDER+'slr_refocusing.mat')
            Nt,dt,rf,gr = mri.data2pulse(data)
            # rf = 0.01*torch.rand((2,Nt),device=device)
            rf = 0.01*torch.ones((2,Nt),device=device)
        pulse_requirements = {'rfmax':0.25*0.1,'gmax':5*10,'smax':12*10} # smax:(mT/m/ms)
        # rf <= 0.25 Gauss, G <= 5 Gauss/cm, slew-rate <= 12 Gauss/cm/ms
        pulse = mri.Pulse(rf,gr,dt)
        pulse.show_info()
        mri.plot_pulse(rf,gr,dt,picname='pictures/mri_pic_opt_pulse_init.png',save_fig=SAVE_FIG)

        # build a cube for simulation:
        cube = mri.Build_SpinArray(fov=[1,1,2],dim=[1,1,1000])

        # build a same cube for design target:
        target_idx = cube.get_index([-1,1],[-1,1],[-0.25,0.25])
        target_para = 0.0*torch.ones(cube.num,device=device)
        target_para[target_idx] = 1.0
        if True: # test of the target parameters:
            plt.figure(figsize=(12,8))
            plt.plot(target_para.tolist())
            plt.ylim([-1.05,1.05])
            if SAVE_FIG:
                plt.savefig('pictures/mri_tmp_pic_target.png')
            else:
                plt.show()
        
        # simulation with intial pulse:
        M = mri.blochsim(cube,Nt,dt,rf,gr)
        # mri.plot_slices(cube,M,'z',valuerange=[-1,1])
        mri.plot_magnetization_profile(cube.loc[2,:],M,picname='pictures/mri_pic_opt_profile_init.png',save_fig=SAVE_FIG)
        
        # simulation of SLR parameters:
        a,b = mri.slrsim_c(cube,Nt,dt,rf,gr)
        # mri.plot_slr_profile(cube.loc[2,:],a,b)
        # print(a.abs()**2 - b.abs()**2)
        # print(b.abs())

        # choose loss function:
        def loss_para_fn(ar,ai,br,bi):
            # turn the slr para. into sth. for measuring
            # for excitation:
            # para = 4*(ar*br + ai*bi)**2 + 4*(ar*bi - ai*br)**2 # Mxy
            # para = 2*(ar*bi - ai*br)
            if False:
                para = br**2 + bi**2 # use this for excitation and inversion
            # for inversion: also use this for excitation seems to be more effective
            # para = ar**2+ai**2-(br**2+bi**2) # Mz
            if True:
                para = br**2 + bi**2 # use this for inversion: 0 and 1
            # for crushed spin echoes:
            # para = (br**2 - bi**2)**2 + 4*(br*bi)**2 # Mxy
            if False:
                para = br**2 - bi**2 # this for refocusing, target -1, otherwise 0/1
            return para
        def loss_fn(x,y,weight=None):
            # loss function try decrease the ripples, but not effective
            loss_err = torch.sum((x-y)**2)
            err_var = torch.diff(x-y)
            idx = target_idx[10:]
            idx = target_idx[:-10]
            err_var = err_var[idx]
            loss_pen = torch.sum(err_var**2)
            return loss_err + 100*loss_pen
        loss_fn = lambda x,y: torch.sum((x-y)**2)
        # loss_fn = mriopt.loss_l2

        # optimize the pulse:
        # rf,gr = mriopt.slr_GD(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
        pulse,optinfos = mriopt.slr_LBFGS(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
        # pulse,optinfos = mriopt.slr_GD_transform(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
        # pulse,optinfos = mriopt.slr_FW(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
        # pulse,optinfos = mriopt.slr_FW_2(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
        # pulse,optinfos = mriopt.slr_FW_3(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
        
        if True:
            mri.save_infos(pulse,logname=LOG_FOLDER+'mri_opt_test_log.mat',otherinfodic=optinfos)

        # -------------------------------
        # final test of optimized results:
        Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
        if True:
            pulse = mri.Pulse(rf,gr,dt)
            pulse.show_info()
        if True:
            test_cube = mri.Build_SpinArray(fov=[1,1,2],dim=[1,1,1000])
            if False: # let M start in transver plane
                test_cube.set_Mag(torch.tensor([0.,1.,0.],device=device))
            M = mri.blochsim(test_cube,Nt,dt,rf,gr)
            mri.plot_pulse(rf,gr,dt,picname='pictures/mri_pic_opt_pulse_end.png',save_fig=SAVE_FIG)
            mri.plot_magnetization_profile(test_cube.loc[2,:],M,picname='pictures/mri_pic_opt_profile_end.png',save_fig=SAVE_FIG)
            # mri.plot_slices(cube,M,'z',valuerange=[-1,1])
            # a,b = mri.slrsim_c(cube,Nt,dt,rf,gr)
            # mri.plot_slr_profile(cube.loc[2,:],a,b,picname='pictures/mri_tmp_pic_slr_profile.png')
            # print(a.abs()**2 - b.abs()**2)
        if False:
            # test of B1 sensitivity    
            notes = '''simulate the B1 sensitivity,'''
            print(notes)
            test_cube = mri.Build_SpinArray(fov=[1,1,1],dim=[1,1,100])



















    # ===================================================================
    # 
    # 
    # ===================================================================
    if True: # the multislice pulse
        '''
        optimization, based on SLR parameters
        '''
        # read in the initial pulse:
        if False: # read in the tse pulse data: for SMS
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
        if False:
            # data = mri.read_data(LOG_FOLDER+'mri_log_slr_exc.mat')
            # data = mri.read_data(LOG_FOLDER+'slr_excitation.mat')
            data = mri.read_data(LOG_FOLDER+'slr_inversion.mat')
            # data = mri.read_data(LOG_FOLDER+'slr_refocusing.mat')
            Nt,dt,rf,gr = mri.data2pulse(data)
            # rf = torch.rand_like(rf)*0.01
            if False:
                rf = torch.ones_like(rf)*0.01 # get intersting periodic result pulse
            if True:
                rf = torch.zeros_like(rf)
                rf[0,:] = torch.ones(Nt,device=device)*0.01
            if False:
                gr = torch.zeros_like(gr)
                gr[2,:] = torch.ones(Nt,device=device)*5 # mT/m
        if True:
            dt = 0.005 # ms
            Nt = int(6/dt)
            rf = torch.zeros((2,Nt),device=device)
            rf[0,:] = torch.ones(Nt,device=device)*0.01
            gr = torch.zeros((3,Nt),device=device)
            gr[2,:] = torch.ones(Nt,device=device)*5


        # adding the requirements:
        # rf <= 0.25 Gauss, G <= 5 Gauss/cm, slew-rate <= 12 Gauss/cm/ms
        pulse_requirements = {'rfmax':0.25*0.1,'gmax':5*10,'smax':12*10,'d1':0.01,'d2':0.05} # smax:(mT/m/ms)
        pulse = mri.Pulse(rf,gr,dt)
        pulse.show_info()
        if True:
            mri.plot_pulse(rf,gr,dt,picname='pictures/mri_pic_opt_pulse_init.png',save_fig=SAVE_FIG)



        # build a cube for simulation:
        cube = mri.Build_SpinArray(fov=[1,1,24],dim=[1,1,2000])
        if False: # start magnetization in the transverse plane
            cube.set_Mag(torch.tensor([0.,1.,0.],device=device))



        # requirements for the multislices:
        target_slice_pos_list = [-8,-4,0,4,8] # cm
        # target_slice_pos_list = [-10,-6,-2,2,6,10] # cm
        slice_thickness = 0.2 # cm
        # build a same cube for design target:
        if True:
            target_cube = mri.Build_SpinArray(fov=[1,1,24],dim=[1,1,2000])
            target_slice_list = []
            for target_slice_pos in target_slice_pos_list:
                target_slice_list.append([target_slice_pos-0.5*slice_thickness,target_slice_pos+0.5*slice_thickness])        
            for target_slice in target_slice_list:
                target_idx = cube.get_index([-1,1],[-1,1],target_slice)
                target_cube.set_selected_Mag(target_idx,torch.tensor([0.,0.,-1.],device=device)) # inversion
            if True:
                plot_target_illustrate(target_cube.Mag[2,:],loc=target_cube.loc[2,:])
        # specify the slices: for spin-domain:
        if True:
            target_para = 0.0*torch.ones(cube.num,device=device)
            target_slice_list = []
            for target_slice_pos in target_slice_pos_list:
                target_slice_list.append([target_slice_pos-0.5*slice_thickness,target_slice_pos+0.5*slice_thickness])        
            for target_slice in target_slice_list:
                target_idx = cube.get_index([-1,1],[-1,1],target_slice)
                target_para[target_idx] = -1.0
            if False: # test of the target parameters:
                plot_target_illustrate(target_para,loc=cube.loc[2,:])
        # design weighting function:
        weighting = torch.ones(cube.num,device=device)
        if True:
            transi_wd = 0.15 # 15% of the passband
            for target_slice_pos in target_slice_pos_list:
                left_transi_pos = target_slice_pos-0.5*slice_thickness
                left_transi_band = [left_transi_pos-0.5*transi_wd*slice_thickness, left_transi_pos+0.5*transi_wd*slice_thickness]
                target_idx = cube.get_index([-1,1],[-1,1],left_transi_band)
                weighting[target_idx] = 0.0
                right_transi_pos = target_slice_pos+0.5*slice_thickness
                right_transi_band = [right_transi_pos-0.5*transi_wd*slice_thickness, right_transi_pos+0.5*transi_wd*slice_thickness]
                target_idx = cube.get_index([-1,1],[-1,1],right_transi_band)
                weighting[target_idx] = 0.0
            if False:
                plot_target_illustrate(weighting,loc=cube.loc[2,:],valuerange=[-0.1,3])
        pulse_requirements['lossweight'] = weighting
        # the ripples in the different bands:
        band_ripple = torch.zeros(cube.num,device=device) # ideal no var
        if False: # todo, how to change for multislices
            band_ripple = pulse_requirements['d2']*torch.ones_like(band_ripple)
            # passband_idx = cube.get_index([-1,1],[-1,1],[-0.25,0.25])
            # band_ripple[passband_idx] = pulse_requirements['d1']
            if False:
                plot_target_illustrate(band_ripple,loc=cube.loc[2,:],valuerange=[-0.01,1.])
        pulse_requirements['band_ripple'] = band_ripple
        # for task learning the gradient:
        if True:
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
            # mri.plot_slices(cube,M,'z',valuerange=[-1,1])
            mri.plot_magnetization_profile(cube.loc[2,:],M,picname='pictures/mri_pic_opt_profile_init.png',save_fig=SAVE_FIG)
        # simulation of SLR parameters:
        if False:
            a,b = mri.slrsim_c(cube,Nt,dt,rf,gr)
            # mri.plot_slr_profile(cube.loc[2,:],a,b)
            # print(a.abs()**2 - b.abs()**2)
            # print(b.abs())



        # === optimization based on bloch simulation: ===
        if False:
            def loss_fn():
                return
            # TODO
        # === optimization based on spin domain: ===
        if True:
            # choose loss function:
            def loss_para_fn(ar,ai,br,bi):
                # turn the slr para. into sth. for measuring
                # for excitation:
                # para = 4*(ar*br + ai*bi)**2 + 4*(ar*bi - ai*br)**2 # Mxy
                # para = 2*(ar*bi - ai*br)
                if False:
                    '''excitation: '''
                    para = br**2 + bi**2 # use this for excitation and inversion
                # for inversion: also use this for excitation seems to be more effective
                # para = ar**2+ai**2-(br**2+bi**2) # Mz
                if False: 
                    '''inversion |beta|^2 = 1 in-slice, = 0 out-slice'''
                    para = br**2 + bi**2
                if False:
                    '''for crushed spin echoes:'''
                    # para = (br**2 - bi**2)**2 + 4*(br*bi)**2 # Mxy
                    pass
                if True:
                    '''refocusing: in-slice: beta^2 = -1, out-slice: beta^2 = 0.
                    or Imag(beta) = +/- 1'''
                    para = br**2 - bi**2 # this for refocusing, the real part of beta^2
                    # para = bi**2
                return para
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
            # loss_fn = lambda x,y: torch.sum((x-y)**2)
            def loss_fn_l2(x,y,weight):
                l = torch.sum((x-y)**2)
                return l
            def loss_fn_weighted_l2(x,y,weight):
                l = torch.sum(((x-y)**2)*weight)
                return l
            def loss_fn_l1(x,y,weight):
                l = torch.sum((x-y).abs())
                return l
            def loss_fn_weighted_l1(x,y,weight):
                l = torch.sum((x-y).abs()*weight)
                return l
            def loss_fn_custom1(x,y,weight):
                nu = torch.log(band_ripple+1)/band_ripple
                err = (x-y).abs()
                l = torch.sum(torch.exp(nu*err)-1.0)
                return l
            def loss_fn_custom2(x,y,weight):
                al = torch.log(band_ripple+1)/band_ripple
                err = (x-y).abs()
                l = torch.sum(torch.exp(al*err)*err)
                return l
            if True: # a compute of the initial loss
                a_real,a_imag,b_real,b_imag = mri.slrsim_(cube,pulse.Nt,pulse.dt,pulse.rf,pulse.gr)
                para = loss_para_fn(a_real,a_imag,b_real,b_imag)
                loss = loss_fn_l2(para,target_para,weighting)
                print('>> pulse loss measure:',loss.item())
                loss = loss_fn_weighted_l2(para,target_para,weighting)
                print('>> pulse loss measure:',loss.item())
                loss = loss_fn_l1(para,target_para,weighting)
                print('>> pulse loss measure:',loss.item())
                loss = loss_fn_weighted_l1(para,target_para,weighting)
                print('>> pulse loss measure:',loss.item())
                loss = loss_fn_custom1(para,target_para,weighting)
                print('>> pulse loss measure:',loss.item())
            # ---- choose optimization method ---:
            pulse_requirements['niter'] = 4
            pulse_requirements['rf_niter'] = 1
            pulse_requirements['gr_niter'] = 0
            if True:
                # rf,gr = mriopt.slr_GD(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
                # pulse,optinfos = mriopt.slr_GD_transform(cube,pulse,target_para,loss_fn,loss_para_fn,pulse_requirements)
                pulse,optinfos = mriopt.slr_LBFGS(cube,pulse,target_para,loss_fn_weighted_l1,loss_para_fn,pulse_requirements)
                # pulse,optinfos = mriopt.slr_FW(cube,pulse,target_para,loss_fn_weighted_l1,loss_para_fn,pulse_requirements)
        

        
        # plot the optimization curve:
        if True:
            mriopt.plot_optinfo(optinfos,picname='pictures/mri_tmp_pic_optinfo.png',save_fig=SAVE_FIG)
            # adding some comments in the log file:
            comments = '''
design of multislice pulse, use lbfgs, loss function is weighted l1
'''
            optinfos['comments'] =  comments
            print(optinfos['comments'])

        # save logs:
        if True:
            mri.save_infos(pulse,logname=LOG_FOLDER+'mri_opt_log_tmp.mat',otherinfodic=optinfos)




        # -------------------------------
        # final test of optimized results:
        Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
        if True:
            pulse = mri.Pulse(rf,gr,dt)
            pulse.show_info()
        if True:
            test_cube = mri.Build_SpinArray(fov=[1,1,24],dim=[1,1,2000])
            if False: # start in transverse plane
                test_cube.set_Mag(torch.tensor([0.,1.,0.],device=device))
            Mopt = mri.blochsim(test_cube,Nt,dt,rf,gr)
            mri.plot_pulse(rf,gr,dt,picname='pictures/mri_pic_opt_pulse_end.png',save_fig=SAVE_FIG)
            mri.plot_magnetization_profile(test_cube.loc[2,:],Mopt,picname='pictures/mri_pic_opt_profile_end.png',save_fig=SAVE_FIG)
            mri.plot_magnetization_profile_two(cube.loc[2,:],M,test_cube.loc[2,:],Mopt,method='z',picname='pictures/mri_pic_opt_profile_compare.png',save_fig=SAVE_FIG)
            if False: # plot the error distribution
                mri.plot_1d_profiles([cube.loc[2,:],cube.loc[2,:]],[(Mopt[2,:]-target_cube.Mag[2,:]).abs(),weighting],picname='pictures/mri_pic_opt_1d_profile.png',save_fig=SAVE_FIG)
            if True:
                a,b = mri.slrsim_c(cube,Nt,dt,rf,gr)
                mri.plot_slr_profile(cube.loc[2,:],a,b,picname='pictures/mri_tmp_pic_slr_profile.png',save_fig=SAVE_FIG)
                # print(a.abs()**2 - b.abs()**2)
            # mri.plot_slices(cube,M,'z',valuerange=[-1,1])













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

            
        