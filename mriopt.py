# optimization using mri module
# author: jiayao

import torch
import numpy as np
import mri
from time import time
import math
import matplotlib.pyplot as plt

import mripulse


# print()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
# print('>> mri: using device:',device)


# ---------------------------------------------


# --------------------------------------------------
# -                 loss functions                 -
# --------------------------------------------------
def loss_l2(M,Md):
    '''M:(3*num), Md:(3*num),desired'''
    M_err = M - Md
    err = M_err.norm()**2
    return err
def loss_angle(M,Md):
    '''dot product of M and Md, to represent the angle'''
    M_unit = torch.nn.functional.normalize(M,dim=0)
    # Md_unit = torch.nn.functional.normalize(Md,dim=0)
    # loss = -torch.sum(M_unit*Md,dim=0)
    err_ang = 1 - M_unit*Md
    loss = torch.sum(err_ang**2)
    return loss
def loss_wl2(x,y,weight=None):
    '''x:1d tensor, y:1d tensor'''
    if weight == None:
        loss = torch.sum((x - y)**2)
    else:
        loss = torch.sum(weight*(x - y)**2)
    return loss
def loss_fn(M,Md,case='l2'):
    if case == 'l2':
        l = torch.sum((M-Md)**2)
    return l
# -------------------------------------------
# for spin domain optimization:
def abloss_para_fn(ar,ai,br,bi,case='inversion'):
    '''
    turn the slr para. into sth. for measuring
    some case:
        excitation
        inversion
        refocusing
    '''
    # para = 4*(ar*br + ai*bi)**2 + 4*(ar*bi - ai*br)**2 # Mxy
    # para = 2*(ar*bi - ai*br)
    if case=='excitation':
        # para_r = br**2 + bi**2 # use this for excitation and inversion
        # para_i = 0.0
        para_r = ar*br + ai*bi
        para_i = ar*bi - ai*br
    # for inversion: also use this for excitation seems to be more effective
    # para = ar**2+ai**2-(br**2+bi**2) # Mz
    if case=='inversion': 
        '''inversion |beta|^2 = 1 in-slice, = 0 out-slice'''
        para_r = br**2 + bi**2
        para_i = 0.0
    if case=='refocusing':
        '''refocusing: in-slice: beta^2 = -1, out-slice: beta^2 = 0.
        or Imag(beta) = +/- 1'''
        para_r = br**2 - bi**2 # this for refocusing, the real part of beta^2
        para_i = br*bi*2 # this for refocusing, the imag part of beta^2
        # para = bi**2
    # if case=='crusedSE':
    #     '''for crushed spin echoes:'''
    #     para = (br**2 - bi**2)**2 + 4*(br*bi)**2 # Mxy
    return para_r,para_i
def abloss_fn(x,y,weight=None,case='l2'):
    '''return different types of loss functions'''
    if case == 'l2':
        l = torch.sum((x - y)**2)
    if case == 'weighted_l2':
        l = torch.sum(((x-y)**2)*weight)
    if case == 'l1':
        l = torch.sum((x-y).abs())
    if case == 'weighted_l1':
        l = torch.sum((x-y).abs()*weight)
    return l
def abloss_c_fn(xr,xi,yr,yi,weight=None,case='l2'):
    '''return different types of loss functions'''
    if case == 'l2':
        l = torch.sum((xr - yr)**2+(xi-yi)**2)
    if case == 'weighted_l2':
        l = torch.sum(((xr-yr)**2 + (xi-yi)**2)*weight)
    if case == 'l1':
        l = torch.sum((xr-yr).abs()+(xi-yi).abs())
    if case == 'weighted_l1':
        l = torch.sum(((xr-yr).abs() + (xi-yi).abs())*weight)
    if case == 'complex_l1':
        l = torch.sum(torch.sqrt((xr-yr)**2 + (xi-yi)**2))
    if case == 'weighted_complex_l1':
        l = torch.sum(torch.sqrt((xr-yr)**2 + (xi-yi)**2)*weight)
    if case == 'angle':
        l = -torch.sum(xr*yr + xi*yi)
    if case == 'weighted_angle':
        l = -torch.sum((xr*yr + xi*yi)*weight)
    return l
def show_diff_abloss(x,y,weight):
    loss = abloss_fn(x,y,weight,case='l2')
    print('>> pulse loss measure:',loss.item())
    loss = abloss_fn(x,y,weight,case='weighted_l2')
    print('>> pulse loss measure:',loss.item())
    loss = abloss_fn(x,y,weight,case='l1')
    print('>> pulse loss measure:',loss.item())
    loss = abloss_fn(x,y,weight,case='weighted_l1')
    print('>> pulse loss measure:',loss.item())
# --------------------------------------------------
# -              transform functions               -
# --------------------------------------------------
def transform_rf(rf,rfmax):
    '''rf:(2*Nt)(mT), rfmax:(mT)'''
    rfmag = rf.norm(dim=0)
    trfmag = (rfmag/rfmax*torch.pi/2).tan()
    rfang = torch.atan2(rf[1,:],rf[0,:])
    return trfmag,rfang
def transform_rf_back(trfmag,rfang,rfmax):
    rfmag = trfmag.atan()/torch.pi*2*rfmax
    rfang = rfang.reshape(1,-1)
    rf = torch.cat((rfmag*rfang.cos(),rfmag*rfang.sin()),dim=0)
    return rf
def transform_gr(gr,dt,smax):
    '''g(n+1) - g(n), gr:(3*Nt)(mT/m), smax:(mT/m/ms), dt:(ms)'''
    # assume smax > gmax in value!
    diff = torch.diff(gr,dim=1)/dt # mT/m/ms
    # make all within the constraints:
    idx = torch.nonzero(diff>smax)
    diff[idx] = smax-1e-6 #smax*0.99999
    idx = torch.nonzero(diff<-smax)
    diff[idx] = -smax+1e-6 #-smax*0.99999
    # cat the first one
    s = torch.cat((gr[:,0].reshape(3,1),diff),dim=1)
    # print(s[:,:5])
    # print('s',s)
    '''tan'''
    s_normal = s/smax*torch.pi/2 # in range (-1,1)
    ts = s_normal.tan()
    return ts
def transform_gr_back(ts,dt,smax):
    s = ts.atan()/torch.pi*2*smax
    # print('s',s)
    # dg = torch.cat((s[:,0].reshape(3,1),s*dt),dim=1) # mT/m
    # print(s[:,:5])
    g = torch.cumsum(s,dim=1)
    return g
# test those transforms:
def test_transforms():
    Nt,dt = 6,0.5
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev)
    if True: # test on gr:    
        gr = torch.zeros((3,Nt),device=dev)
        gr[2,:] = 12.0
        smax = 10
        # smax = 14
        # print(gr[:,:5])
        print(gr)
        ts = transform_gr(gr,dt,smax)
        # print(ts[:,:5])
        print(ts)
        gr_ = transform_gr_back(ts,dt,smax)
        # print(gr_[:,:5])
        print(gr_)
        print()
    # test on rf:
    if False:
        rf = torch.rand((2,Nt),device=dev)
        rfmax = 10
        trfmag,rfang = transform_rf(rf,rfmax)
        rf_ = transform_rf_back(trfmag,rfang,rfmax)
        print(rf[:,:5])
        print(trfmag[:5])
        print(rfang[:5])
        # print(rfang.shape)
        # print(rf_.shape)
        print(rf_[:,:5])
    return


def smoother(x):
    '''x shape: (n*Nt), out:(n*Nt), smoothed along Nt'''
    def local_filter(input):
        n = len(input)
        h = torch.tensor([1.,2.,1.],device=device)/4
        # print('input',input)
        output = torch.zeros(n,device=device)
        output = output + input*h[1]
        output[:-1] = output[:-1] + input[1:]*h[2]
        output[1:] = output[1:] + input[:-1]*h[0]
        return output
    if len(x.shape)==1:
        out = local_filter(x)
    if len(x.shape)==2:
        out = torch.zeros_like(x)
        for k in range(x.shape[0]):
            out[k,:] = local_filter(x[k,:])
    return out
def test_smoother():
    x = torch.rand((2,100),device=device)
    t = torch.arange(100,device=device)
    # x[1,:] = torch.cos(0.01*t)
    # x = torch.rand(10,device=device)
    y = smoother(x)
    x = np.array(x.tolist())
    y = np.array(y.tolist())
    plt.figure()
    plt.plot(x[1,:])
    plt.plot(y[1,:],ls='--')
    plt.show()
    return

def gradient_smooth(G,smax,dt): # need to be more carefully designed, to mantain the similar k-trajectory
    '''
    G:(3*Nt)(mT/m), smax:(mT/m/ms), gamma:(MHz/T)
    '''
    # compute the slew rate:
    slewrate = torch.diff(G,dim=1)/dt #(mT/m/ms)
    for ch in range(3):
        s = slewrate[ch,:]
        # find where the slew-rate is large:
        idx = torch.nonzero(s.abs()>smax)
        if len(idx) == 0:
            return G
        # adjust slew-rate:
        def srate_check(si,simax):
            if si > smax:
                dssum = si - simax
                # si = simax
                flag = 1
            elif si < -simax:
                dssum = s[i] - (-simax)
                # si = -simax
                flag = -1
            else:
                flag = 0
                dssum = 0
            return flag,dssum
        ssum = 0
        for i in range(len(s)):
            if torch.abs(s[i]) <= smax:
                # adjustable
                if ssum != 0:
                    s[i] = s[i]
            flag,dssum = srate_check(s[i],smax)
            if flag == 0:
                inner_flag,inner_dssum = srate_check(s[i]+ssum,smax)
                if inner_flag == 0: # everything is good
                    s[i] = s[i] + ssum
                    ssum = 0
                elif inner_flag > 0: # adding too much
                    s[i] = smax
                    ssum = inner_dssum
                else: # inner_flag < 0: # decrease too much
                    s[i] = -smax
                    ssum = inner_dssum
            elif flag > 0:
                s[i] = smax
                ssum = ssum + dssum
            else: # flag < 0
                s[i] = -smax
                ssum = ssum + dssum
        slewrate[ch,:] = s
    # change back to gradients
    slewrate = torch.cat((G[:,0].reshape(3,1),slewrate),dim=1)
    gr = torch.cumsum(slewrate,dim=1)
    return gr
def test_gradient_smooth():
    gr = torch.randn((3,30),device=device)*10
    gr_new = gradient_smooth(gr,10,1)
    k = mripulse.get_kspace(30,1,gr)
    k_new = mripulse.get_kspace(30,1,gr_new)
    print(gr.sum(dim=1))
    print(gr_new.sum(dim=1))
    if True:
        plt.figure()
        plt.plot(gr[1,:].diff().tolist())
        plt.plot(gr_new[1,:].diff().tolist(),ls='--')
        plt.savefig('pictures/tmp.png')
    if True:
        plt.figure()
        plt.plot(k[1,:].tolist())
        plt.plot(k_new[1,:].tolist(),ls='--')
        plt.savefig('pictures/tmp2.png')

    return





# --------------------------------------------------
# -                   optimizers                   -
# --------------------------------------------------
def GD(spinarray,pulse,Mtarget,loss_fn,loss_para_fn,requirements):
    # Gradient descent, and with exchange of parameters to avoid the constraints
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    print(Nt,dt,rf.shape,gr.shape)
    print(rfmax,gmax,smax)

    # initial loss:
    M = mri.blochsim(spinarray,Nt,dt,rf,gr)
    loss = loss_fn(M,Mtarget)
    optinfos = {}
    optinfos['init_rf'] = np.array(rf.tolist())
    optinfos['init_gr'] = np.array(gr.tolist())
    optinfos['init_Nt'] = Nt
    optinfos['init_dt'] = dt
    optinfos['time_hist'] = [0.0]
    optinfos['loss_hist'] = [loss.item()]
    print('initial loss =',loss.item())

    # optimization parameters setting:
    niter = 2
    rf_niter = 4
    gr_niter = 4
    lr_rf_init = 1.0
    lr_gr_init = 1.0

    # line search function:
    def backtrack(lr1,lr2,rf,gr,d1,d2):
        c = 1e-6
        while False:
            tmprf = rf + lr*d1
            tmpgr = gr + lr*d2
            loss = 0
            # if ...:
            #     break
            # else:
            #     lr1 = lr1*0.5
            #     lr2 = lr2*0.5
        return lr1,lr2

    starttime = time()
    # function record opt infos:
    def addinfo_fn(lossvalue):
        optinfos['time_hist'].append(time()-starttime)
        optinfos['loss_hist'].append(lossvalue.item())
        return

    # optimization:
    trfmag,rfang = transform_rf(rf,rfmax)
    ts = transform_gr(gr,dt,smax)
    trfmag.requires_grad = rfang.requires_grad = ts.requires_grad = True
    for k in range(niter):
        # update rf:
        if True:
            for rf_iter in range(rf_niter):
                rf = transform_rf_back(trfmag,rfang,rfmax)
                gr = transform_gr_back(ts,dt,smax)
                M = mri.blochsim(spinarray,Nt,dt,rf,gr)
                loss = loss_fn(M,Mtarget)
                loss.backward()
                trfmag_grad, rfang_grad = trfmag.grad, rfang.grad
                with torch.no_grad():
                    lr_rf = 1e-3
                    # lr_rf,_ = backtrack(lr_rf,0,rf,gr,-rf_grad,-gr_grad)
                    # rf = rf - lr_rf*rf_grad
                    trfmag = trfmag - lr_rf*trfmag_grad
                    rfang = rfang - lr_rf*rfang_grad
                    # add log infos:
                    M = mri.blochsim(spinarray,Nt,dt,rf,gr)
                    loss = loss_fn(M,Mtarget)
                    addinfo_fn(loss)
                    # print details:
                    print('\t--> rf update, loss =',loss.item())
                trfmag.grad = rfang.grad = None
                trfmag.requires_grad = rfang.requires_grad = True
        # update gr:
        if True:
            for _ in range(gr_niter):
                rf = transform_rf_back(trfmag,rfang,rfmax)
                gr = transform_gr_back(ts,dt,smax)
                M = mri.blochsim(spinarray,Nt,dt,rf,gr)
                loss = loss_fn(M,Mtarget)
                loss.backward()
                ts_grad = ts.grad
                with torch.no_grad():
                    lr_gr = 1e-3
                    # _,lr_gr = backtrack(0,lr_gr,rf,gr,-rf_grad,-gr_grad)
                    # gr = gr - lr_gr*gr_grad
                    # print(gr_grad)
                    # add log inofs:
                    M = mri.blochsim(spinarray,Nt,dt,rf,gr)
                    loss = loss_fn(M,Mtarget)
                    addinfo_fn(loss)
                    # print details:
                    print('\t--> gr update, loss =',loss.item())
                ts.grad = None
                ts.requires_grad = True
                        
        # print details:
        if (k % 2) == 0:
            print('>> iteration:',k)
            # print('\tgrad:',rf.grad.shape,gr.grad.shape)
            print('\tloss =',loss.item())
    print('>> end optimization, time:',time()-starttime,'s')
    pulse = mri.Pulse(rf,gr,dt)
    return pulse,optinfos
def FW(spinarray,pulse,Mtarget,loss_fn,lossweight,requirements):
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    print(Nt,dt,rf.shape,gr.shape,rfmax,gmax,smax)
    # print(requirements['band_ripple'])

    # initial loss:
    M = mri.blochsim(spinarray,Nt,dt,rf,gr)
    loss = loss_fn(M,Mtarget,lossweight)
    optinfos = {}
    optinfos['init_rf'] = np.array(rf.tolist())
    optinfos['init_gr'] = np.array(gr.tolist())
    optinfos['init_Nt'] = Nt
    optinfos['init_dt'] = dt
    optinfos['time_hist'] = [0.0]
    optinfos['loss_hist'] = [loss.item()]
    print('initial loss =',loss.item())

    # function record opt infos:
    def addinfo_fn(time0,lossvalue):
        optinfos['time_hist'].append(time()-time0)
        optinfos['loss_hist'].append(lossvalue.item())
        return

    # line search functions:
    def linesearch_rf(lr,rf,gr,currentloss,grad,d):
        c = 1e-6
        for _ in range(20):
            tmprf = rf + lr*d
            M = mri.blochsim(spinarray,Nt,dt,tmprf,gr)
            newloss = loss_fn(M,Mtarget,lossweight)
            expectedloss = currentloss + c*lr*torch.dot(grad.view(-1),d.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = lr*0.5
        return lr,newloss
    
    # optimization parameters setting:
    niter = 3
    rf_niter = 10
    gr_niter = 5
    show_detail_step = 1

    starttime = time()

    # optimize:
    rf.requires_grad = True
    gr.requires_grad = True
    for k in range(niter):
        # update rf:
        if True:
            for rf_itr in range(rf_niter):
                M = mri.blochsim(spinarray,Nt,dt,rf,gr)
                loss = loss_fn(M,Mtarget,lossweight)
                loss.backward()
                rf_grad = rf.grad
                with torch.no_grad():
                    # update:
                    v = -rfmax*torch.nn.functional.normalize(rf_grad,dim=0)
                    d = v-rf
                    # lr_rf = 2/(rf_itr+2)
                    lr_rf,loss = linesearch_rf(1.0,rf,gr,loss,rf_grad,d)
                    rf = rf + lr_rf*d
                    # record logs:
                    M = mri.blochsim(spinarray,Nt,dt,rf,gr)
                    loss = loss_fn(M,Mtarget,lossweight)
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (rf_itr%1) == 0:
                        print('\trf update: loss =',loss.item())
                    # termination:
                    if abs((optinfos['loss_hist'][-2] - optinfos['loss_hist'][-1])/optinfos['loss_hist'][-2]) < 1e-6:
                        break
                rf.grad = None
                rf.requires_grad = True
                gr.grad = None
                gr.requires_grad = True
        # update gr:
        if False:
            for gr_itr in range(gr_niter):
                M = mri.blochsim(spinarray,Nt,dt,rf,gr)
                loss = loss_fn(M,Mtarget,lossweight)
                loss.backward()
                gr_grad = gr.grad
                with torch.no_grad():
                    # update:
                    # record logs:
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (gr_itr%1) == 0:
                        print('\tgr update: loss =',loss.item())
                    # termination:
                    if abs((optinfos['loss_hist'][-2] - optinfos['loss_hist'][-1])/optinfos['loss_hist'][-2]) < 1e-4:
                        break
                rf.grad = None
                rf.requires_grad = True
                gr.grad = None
                gr.requires_grad = True
        # show opt details:
        if (k%show_detail_step) == 0:
            print('>> end iteration:',k+1,', loss={}'.format(optinfos['loss_hist'][-1]))
    print('>> end optimization, time cost:',time()-starttime)
    pulse = mri.Pulse(rf,gr,dt)
    return pulse,optinfos
def CG(spinarray,pulse,Mtarget,loss_fn,loss_para_fn,requirements):
    # Gradient descent, and with exchange of parameters to avoid the constraints
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    print(Nt,dt,rf.shape,gr.shape)
    print(rfmax,gmax,smax)


    return rf,gr
def LBFGS(spinarray,pulse,Mtarget,loss_fn,lossweight,requirements):
    # LBFGS based on pytorch solver, and with exchange of parameters to avoid the constraints
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    print(Nt,dt,rf.shape,gr.shape)
    print(rfmax,gmax,smax)

    # initial loss:
    M = mri.blochsim(spinarray,Nt,dt,rf,gr)
    loss = loss_fn(M,Mtarget,lossweight)
    optinfos = {}
    optinfos['init_rf'] = np.array(rf.tolist())
    optinfos['init_gr'] = np.array(gr.tolist())
    optinfos['init_Nt'] = Nt
    optinfos['init_dt'] = dt
    optinfos['time_hist'] = [0.0]
    optinfos['loss_hist'] = [loss.item()]
    print('initial loss =',loss.item())

    # function record opt infos:
    def addinfo_fn(time0,lossvalue):
        optinfos['time_hist'].append(time()-time0)
        optinfos['loss_hist'].append(lossvalue.item())
        return
    def total_loss(trfmag,rfang,ts):
        tmprf = transform_rf_back(trfmag,rfang,rfmax)
        tmpgr = transform_gr_back(ts,dt,smax)
        M = mri.blochsim(spinarray,Nt,dt,tmprf,tmpgr)
        loss = loss_fn(M,Mtarget,lossweight)
        return loss

    # optimization parameters setting:
    niter = 4
    rf_niter = 1
    gr_niter = 0
    print('>> optimization settings: niter={}, rf niter={}, gr niter={}'.format(niter,rf_niter,gr_niter))

    trfmag,rfang = transform_rf(rf,rfmax)
    ts = transform_gr(gr,dt,smax)
    trfmag.requires_grad = rfang.requires_grad = ts.requires_grad = True
    opt_rf = torch.optim.LBFGS([trfmag,rfang], lr=3., max_iter=30, history_size=30,
                                tolerance_change=1e-6,
                                line_search_fn='strong_wolfe')
    opt_gr = torch.optim.LBFGS([ts], lr=3., max_iter=5, history_size=20,
                                tolerance_change=1e-6,
                                line_search_fn='strong_wolfe')
    # optimize:
    starttime = time()
    for k in range(niter):
        print('iteration number {}'.format(k+1))

        def closure():
            opt_rf.zero_grad()
            opt_gr.zero_grad()
            rf = transform_rf_back(trfmag,rfang,rfmax)
            gr = transform_gr_back(ts,dt,smax)
            M = mri.blochsim(spinarray,Nt,dt,rf,gr)
            loss = loss_fn(M,Mtarget,lossweight)
            loss.backward()
            return loss

        # optimize rf:
        for _ in range(rf_niter):
            opt_rf.step(closure)
            # record infos:
            loss = total_loss(trfmag,rfang,ts)
            addinfo_fn(starttime,loss)
            # details:
            print('\trf updata: loss =',loss.item())

        # optimize gr:
        for _ in range(gr_niter):
            opt_gr.step(closure)
            # record infos:
            loss = total_loss(trfmag,rfang,ts)
            addinfo_fn(starttime,loss)
            # details:
            print('\tgr updata: loss =',loss.item())

    time_cost = time() - starttime
    print('>> end of optimization, cost time:',time_cost)

    rf = transform_rf_back(trfmag,rfang,rfmax)
    gr = transform_gr_back(ts,dt,smax)
    pulse = mri.Pulse(rf,gr,dt)
    return pulse,optinfos
def LBFGS_(spinarray,pulse,Mtarget,loss_fn,loss_para_fn,requirements):
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    print(Nt,dt,rf.shape,gr.shape)
    return rf,gr
def TR_SR1(spinarray,pulse,Mtarget,loss_fn,loss_para_fn,requirements):
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    print(Nt,dt,rf.shape,gr.shape)
    return rf,gr



# --------------------------------------------------
# -                SLR optimizers                  -
# --------------------------------------------------
def slr_GD(spinarray,pulse,para_target,loss_fn,loss_para_fn,requirements):
    # only a test on gradient descent scheme
    print('>> Optimization method: GD')
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    lossweight = requirements['lossweight']
    print(Nt,dt,rf.shape,gr.shape,rfmax,gmax,smax)

    # initial loss:
    # def loss_para_fn(ar,ai,br,bi):
    #     para = ar**2+ai**2-br**2-bi**2
    #     return para
    a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
    para = loss_para_fn(a_real,a_imag,b_real,b_imag)
    loss = loss_fn(para,para_target,lossweight)
    optinfos = {}
    optinfos['init_rf'] = np.array(rf.tolist())
    optinfos['init_gr'] = np.array(gr.tolist())
    optinfos['init_Nt'] = Nt
    optinfos['init_dt'] = dt
    optinfos['time_hist'] = [0.0]
    optinfos['loss_hist'] = [loss.item()]
    print('initial loss =',loss.item())

    # function record opt infos:
    def addinfo_fn(time0,lossvalue):
        optinfos['time_hist'].append(time()-time0)
        optinfos['loss_hist'].append(lossvalue.item())
        return
    def total_loss(rf,gr):
        return

    # optimization parameters setting:
    niter = requirements['niter']
    rf_niter = requirements['rf_niter']
    gr_niter = requirements['gr_niter']
    lr_rf_init = 1.0
    lr_gr_init = 1.0
    show_detail_step = 1

    # line search function:
    def backtrack(lr,rf,gr,d,currentloss,grad1):
        c = 1e-6
        for _ in range(20):
            tmprf = rf + lr*d
            a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,tmprf,gr)
            para = loss_para_fn(a_real,a_imag,b_real,b_imag)
            newloss = loss_fn(para,para_target,lossweight)
            if newloss < currentloss+c*lr*torch.dot(grad1.view(-1),d.view(-1)):
                break
            lr = lr*0.5
        return lr

    starttime = time()

    # optimize:
    rf.requires_grad = True
    gr.requires_grad = True
    for k in range(niter):
        # update rf:
        if True:
            for rf_itr in range(rf_niter):
                a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
                para = loss_para_fn(a_real,a_imag,b_real,b_imag)
                loss = loss_fn(para,para_target,lossweight)
                loss.backward()
                rf_grad = rf.grad
                with torch.no_grad():
                    # update:
                    lr = 1.0
                    lr = backtrack(lr,rf,gr,-rf_grad,loss,rf_grad)
                    rf = rf - lr*rf_grad
                    a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
                    para = loss_para_fn(a_real,a_imag,b_real,b_imag)
                    loss = loss_fn(para,para_target,lossweight)
                    # record logs:
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (rf_itr%1) == 0:
                        print('\trf update:')
                        print('\tloss =',loss.item())
                rf.grad = None
                rf.requires_grad = True
        # update gr:
        if True:
            for gr_itr in range(gr_niter):
                a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
                para = loss_para_fn(a_real,a_imag,b_real,b_imag)
                loss = loss_fn(para,para_target,lossweight)
                loss.backward()
                gr_grad = gr.grad
                with torch.no_grad():
                    # update:
                    # record logs:
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (gr_itr%1) == 0:
                        print('\tgr update:')
                        print('\tloss =',loss.item())
                gr.grad = None
                gr.requires_grad = True
        # show opt details:
        if ((k+1)%show_detail_step) == 0:
            print('>> end iteration:',k+1,', loss={}'.format(optinfos['loss_hist'][-1]))
    # save information and logs:
    # print(optinfos)
    # mri.save_infos(Nt,dt,rf,gr,logname=,otherinfodic=optinfos)
    pulse = mri.Pulse(rf,gr,dt)
    return pulse,optinfos
def slr_GD_transform(spinarray,pulse,para_target,loss_fn,loss_para_fn,requirements):
    # gradient descent, with exchange of variables
    print('>> Optimization method: GD + exchange of variables')
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    lossweight = requirements['lossweight']
    print(Nt,dt,rf.shape,gr.shape,rfmax,gmax,smax)

    a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
    para = loss_para_fn(a_real,a_imag,b_real,b_imag)
    loss = loss_fn(para,para_target,lossweight)
    optinfos = {}
    optinfos['init_rf'] = np.array(rf.tolist())
    optinfos['init_gr'] = np.array(gr.tolist())
    optinfos['init_Nt'] = Nt
    optinfos['init_dt'] = dt
    optinfos['time_hist'] = [0.0]
    optinfos['loss_hist'] = [loss.item()]
    print('initial loss =',loss.item())

    # function record opt infos:
    def addinfo_fn(time0,lossvalue):
        optinfos['time_hist'].append(time()-time0)
        optinfos['loss_hist'].append(lossvalue.item())
        return
    def total_loss(trfmag,rfang,ts):
        rf = transform_rf_back(trfmag,rfang,rfmax)
        gr = transform_gr_back(ts,dt,smax)
        a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
        para = loss_para_fn(a_real,a_imag,b_real,b_imag)
        loss = loss_fn(para,para_target,lossweight)
        return loss

    # optimization parameters setting:
    niter = 2
    rf_niter = 2
    gr_niter = 2
    lr_rf_init = 1.0
    lr_gr_init = 1.0
    show_detail_step = 1

    # line search function:
    def backtrack(lr1,lr2,trfmag,rfang,ts,currentloss,grad11,grad12,grad2,d11,d12,d2):
        c = 1e-6
        for _ in range(20):
            tmptrfmag = trfmag + lr1*d11
            tmprfang = rfang + lr1*d12
            tmpts = ts + lr2*d2
            newloss = total_loss(tmptrfmag,tmprfang,tmpts)
            expectedloss_p1 = c*lr1*(torch.dot(grad11.view(-1),d11)+torch.dot(grad12.view(-1),d12.view(-1)))
            expectedloss_p2 = c*lr2*(torch.dot(grad2.view(-1),d2.view(-1)))
            expectedloss = currentloss + expectedloss_p1 + expectedloss_p2
            if newloss < expectedloss:
                break
            lr1, lr2 = lr1*0.5, lr2*0.5
        return lr1,lr2,newloss

    starttime = time()

    # optimize:
    trfmag,rfang = transform_rf(rf,rfmax)
    ts = transform_gr(gr,dt,smax)
    trfmag.requires_grad = rfang.requires_grad = ts.requires_grad = True
    for k in range(niter):
        # update rf:
        if True:
            for rf_itr in range(rf_niter):
                # rf = transform_rf_back(trfmag,rfang,rfmax)
                # gr = transform_gr_back(ts,dt,smax)
                # a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
                # para = loss_para_fn(a_real,a_imag,b_real,b_imag)
                # loss = loss_fn(para,para_target)
                loss = total_loss(trfmag,rfang,ts)
                loss.backward()
                trfmag_grad,rfang_grad = trfmag.grad,rfang.grad
                ts_grad = ts.grad
                with torch.no_grad():
                    # update:
                    # lr = 1e-3
                    lr_rf,_,loss = backtrack(lr_rf_init,0.,trfmag,rfang,ts,loss,
                        trfmag_grad,rfang_grad,ts_grad,
                        -trfmag_grad,-rfang_grad,-ts_grad)
                    trfmag = trfmag - lr_rf*trfmag_grad
                    rfang = rfang - lr_rf*rfang_grad
                    # record logs:
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (rf_itr%1) == 0:
                        print('\t--> rf update:')
                        print('\tloss =',loss.item())
                trfmag.grad = rfang.grad = None
                ts.grad = None
                trfmag.requires_grad = rfang.requires_grad = True
                ts.requires_grad = True
        # update gr:
        if True:
            for gr_itr in range(gr_niter):
                loss = total_loss(trfmag,rfang,ts)
                loss.backward()
                trfmag_grad,rfang_grad = trfmag.grad,rfang.grad
                ts_grad = ts.grad
                with torch.no_grad():
                    _,lr_gr,loss = backtrack(lr_rf_init,0.,trfmag,rfang,ts,loss,
                        trfmag_grad,rfang_grad,ts_grad,
                        -trfmag_grad,-rfang_grad,-ts_grad)
                    ts = ts - lr_gr*ts_grad
                    # record logs:
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (gr_itr%1) == 0:
                        print('\t--> gr update:')
                        print('\tloss =',loss.item())
                trfmag.grad = rfang.grad = None
                ts.grad = None
                trfmag.requires_grad = rfang.requires_grad = True
                ts.requires_grad = True
        # show opt details:
        if ((k+1)%show_detail_step) == 0:
            print('>> end iteration:',k+1,', loss={}'.format(optinfos['loss_hist'][-1]))
    # change the variable back
    rf = transform_rf_back(trfmag,rfang,rfmax)
    gr = transform_gr_back(ts,dt,smax)
    pulse = mri.Pulse(rf,gr,dt)
    return pulse,optinfos
def slr_LBFGS(spinarray,pulse,para_target,loss_fn,loss_para_fn,requirements):
    print('>> Optimization method: LBFGS by pytorch solver')
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    lossweight = requirements['lossweight']
    print(Nt,dt,rf.shape,gr.shape,rfmax,gmax,smax)

    # initial initial loss:
    a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
    para = loss_para_fn(a_real,a_imag,b_real,b_imag)
    loss = loss_fn(para,para_target,lossweight)

    # record initial info:
    optinfos = {}
    optinfos['init_rf'] = np.array(rf.tolist())
    optinfos['init_gr'] = np.array(gr.tolist())
    optinfos['init_Nt'] = Nt
    optinfos['init_dt'] = dt
    optinfos['time_hist'] = [0.0]
    optinfos['loss_hist'] = [loss.item()]
    print('initial loss =',loss.item())

    # function to record opt infos:
    def addinfo_fn(time0,lossvalue):
        optinfos['time_hist'].append(time()-time0)
        optinfos['loss_hist'].append(lossvalue.item())
        return

    # optimization parameters setting:
    niter = requirements['niter']
    rf_niter = requirements['rf_niter']
    gr_niter = requirements['gr_niter']
    show_detail_step = 1

    # optimization:
    trfmag,rfang = transform_rf(rf,rfmax)
    ts = transform_gr(gr,dt,smax)
    trfmag.requires_grad = rfang.requires_grad = ts.requires_grad = True
    
    opt_rf = torch.optim.LBFGS([trfmag,rfang], lr=3., max_iter=20, history_size=30,
                                tolerance_change=1e-6,
                                line_search_fn='strong_wolfe')
    opt_gr = torch.optim.LBFGS([ts], lr=3., max_iter=10, history_size=20,
                                tolerance_change=1e-6,
                                line_search_fn='strong_wolfe')

    starttime = time()
    for k in range(niter):
        print('iteration number {}'.format(k+1))

        def closure():
            opt_rf.zero_grad()
            opt_gr.zero_grad()
            rf = transform_rf_back(trfmag,rfang,rfmax)
            gr = transform_gr_back(ts,dt,smax)
            a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
            para = loss_para_fn(a_real,a_imag,b_real,b_imag)
            loss = loss_fn(para,para_target,lossweight)
            loss.backward()
            return loss

        # optimize rf:
        for _ in range(rf_niter):
            opt_rf.step(closure)
            # record infos:
            rf = transform_rf_back(trfmag,rfang,rfmax)
            gr = transform_gr_back(ts,dt,smax)
            a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
            para = loss_para_fn(a_real,a_imag,b_real,b_imag)
            loss = loss_fn(para,para_target,lossweight)
            addinfo_fn(starttime,loss)
            # details:
            print('\trf updata: loss =',loss.item())

        # optimize gr:
        for _ in range(gr_niter):
            opt_gr.step(closure)
            # record infos:
            rf = transform_rf_back(trfmag,rfang,rfmax)
            gr = transform_gr_back(ts,dt,smax)
            a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
            para = loss_para_fn(a_real,a_imag,b_real,b_imag)
            loss = loss_fn(para,para_target,lossweight)
            addinfo_fn(starttime,loss)
            # details:
            print('\tgr updata: loss =',loss.item())

    time_cost = time() - starttime
    print('>> end of optimization, cost time:',time_cost)

    rf = transform_rf_back(trfmag,rfang,rfmax)
    gr = transform_gr_back(ts,dt,smax)
    pulse = mri.Pulse(rf,gr,dt)
    return pulse,optinfos

def slr_LBFGS_(spinarray,pulse,para_target,loss_fn,loss_para_fn,requirements):
    print('>> Optimization method: LBFGS from scratch')
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    lossweight = requirements['lossweight']
    print(Nt,dt,rf.shape,gr.shape,rfmax,gmax,smax)

    # initial initial loss:
    a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
    para = loss_para_fn(a_real,a_imag,b_real,b_imag)
    loss = loss_fn(para,para_target,lossweight)

    # record initial info:
    optinfos = {}
    optinfos['init_rf'] = np.array(rf.tolist())
    optinfos['init_gr'] = np.array(gr.tolist())
    optinfos['init_Nt'] = Nt
    optinfos['init_dt'] = dt
    optinfos['time_hist'] = [0.0]
    optinfos['loss_hist'] = [loss.item()]
    print('initial loss =',loss.item())

    # function to record opt infos:
    def addinfo_fn(time0,lossvalue):
        optinfos['time_hist'].append(time()-time0)
        optinfos['loss_hist'].append(lossvalue.item())
        return
    def total_loss(trfmag,rfang,ts):
        rf = transform_rf_back(trfmag,rfang,rfmax)
        gr = transform_gr_back(ts,dt,smax)
        a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
        para = loss_para_fn(a_real,a_imag,b_real,b_imag)
        loss = loss_fn(para,para_target,lossweight)
        return loss

    # optimization parameters setting:
    niter = 2
    rf_niter = 2
    gr_niter = 2
    show_detail_step = 1

    # line search function:
    def linesearch_rf():
        return

    # optimization:
    trfmag,rfang = transform_rf(rf,rfmax)
    ts = transform_gr(gr,dt,smax)

    n_rf = trfmag.numel() + rfang.numel()
    H_rf = torch.eye(n_rf)
    rf_grad_prev = torch.zeros(n_rf)
    rf_sk = torch.zeros(n_rf)

    trfmag.requires_grad = rfang.requires_grad = ts.requires_grad = True
    starttime = time()
    for k in range(niter):
        if True: # rf update
            for rf_itr in range(rf_niter):
                loss = total_loss(trfmag,rfang,ts)
                loss.backward()
                trfmag_grad,rfang_grad = trfmag.grad,rfang.grad
                ts_grad = ts.grad
                with torch.no_grad():
                    # update:

                    # add optinfo:
                    loss = total_loss(trfmag,rfang,ts)
                    addinfo_fn(starttime,loss)
                    # details:
                    print('\trf updata: loss =',loss.item())
                trfmag.grad = rfang.grad = None
                ts.grad = None
                trfmag.requires_grad = rfang.requires_grad = True
                ts.requires_grad = True
        if False:
            for gr_itr in range(gr_niter):
                loss = total_loss(trfmag,rfang,ts)
                loss.backward()
                # !! == TODO ==
                with torch.no_grad():
                    pass

    time_cost = time() - starttime
    print('>> end of optimization, cost time:',time_cost)

    rf = transform_rf_back(trfmag,rfang,rfmax)
    gr = transform_gr_back(ts,dt,smax)
    pulse = mri.Pulse(rf,gr,dt)
    return pulse,optinfos

def slr_FW_pre(spinarray,pulse,para_target,loss_fn,loss_para_fn,requirements):
    # Frank-Wolfe, with arctan change of slew-rate
    print('>> Optimization method: FW')
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    print('\t',Nt,dt,rf.shape,gr.shape,rfmax,gmax,smax)

    a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
    para = loss_para_fn(a_real,a_imag,b_real,b_imag)
    loss = loss_fn(para,para_target)
    optinfos = {}
    optinfos['init_rf'] = np.array(rf.tolist())
    optinfos['init_gr'] = np.array(gr.tolist())
    optinfos['init_Nt'] = Nt
    optinfos['init_dt'] = dt
    optinfos['time_hist'] = [0.0]
    optinfos['loss_hist'] = [loss.item()]
    print('initial loss =',loss.item())

    # function record opt infos:
    def addinfo_fn(time0,lossvalue):
        optinfos['time_hist'].append(time()-time0)
        optinfos['loss_hist'].append(lossvalue.item())
        return
    def total_loss(rf,ts):
        gr = transform_gr_back(ts,dt,smax)
        a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
        para = loss_para_fn(a_real,a_imag,b_real,b_imag)
        loss = loss_fn(para,para_target)
        return loss

    # optimization parameters setting:
    niter = 2
    rf_niter = 8
    gr_niter = 2
    lr_rf_init = 1.0
    lr_gr_init = 1.0
    show_detail_step = 1

    # line search function:
    def linesearch_rf(lr,rf,ts,currentloss,grad,d):
        c = 1e-6
        for _ in range(20):
            tmprf = rf + lr*d
            newloss = total_loss(tmprf,ts)
            expectedloss = currentloss + c*lr*torch.dot(grad.view(-1),d.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss
    def linesearch_gr(lr,rf,ts,currentloss,grad,d):
        c = 1e-6
        for _ in range(20):
            tmpts = ts + lr*d
            newloss = total_loss(rf,tmpts)
            expectedloss = currentloss + c*lr*torch.dot(grad.view(-1),d.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss

    starttime = time()

    # optimize:
    ts = transform_gr(gr,dt,smax)
    rf.requires_grad = ts.requires_grad = True
    for k in range(niter):
        # update rf:
        if True:
            for rf_itr in range(rf_niter):
                loss = total_loss(rf,ts)
                loss.backward()
                rf_grad = rf.grad
                ts_grad = ts.grad
                with torch.no_grad():
                    # update:
                    v = -rfmax*torch.nn.functional.normalize(rf_grad,dim=0)
                    d = v-rf
                    # lr_rf = 2/(rf_itr+2)
                    lr_rf,loss = linesearch_rf(1.0,rf,ts,loss,rf_grad,d)
                    rf = rf + lr_rf*d
                    # record logs:
                    # loss = total_loss(rf,ts)
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (rf_itr%1) == 0:
                        print('\t--> rf update, loss =',loss.item())
                rf.grad = None
                rf.requires_grad = True
                ts.grad = None
                ts.requires_grad = True
        # update gr:
        if True:
            for gr_itr in range(gr_niter):
                loss = total_loss(rf,ts)
                loss.backward()
                rf_grad = rf.grad
                ts_grad = ts.grad
                with torch.no_grad():
                    # lr_gr = 1e-3
                    lr_gr,loss = linesearch_gr(1.0,rf,ts,loss,ts_grad,-ts_grad)
                    ts = ts - lr_gr*ts_grad
                    # record logs:
                    # loss = total_loss(rf,ts)
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (gr_itr%1) == 0:
                        print('\t--> gr update, loss =',loss.item())
                rf.grad = None
                rf.requires_grad = True
                ts.grad = None
                ts.requires_grad = True
        # show opt details:
        if ((k+1)%show_detail_step) == 0:
            print('>> end iteration:',k+1,', loss={}'.format(optinfos['loss_hist'][-1]))
    # change the variable back
    gr = transform_gr_back(ts,dt,smax)
    # information and logs:
    # print(optinfos)
    # mri.save_infos(Nt,dt,rf,gr,logname=,otherinfodic=optinfos)
    pulse = mri.Pulse(rf,gr,dt)
    return pulse,optinfos

def slr_FW(spinarray,pulse,para_target,loss_fn,loss_para_fn,requirements):
    # Frank-Wolfe, FW on rf, GD on gr
    print('>> Optimization method: constrained optimizations')
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    lossweight = requirements['lossweight']
    print('\t',Nt,dt,rf.shape,gr.shape,rfmax,gmax,smax)

    a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
    para = loss_para_fn(a_real,a_imag,b_real,b_imag)
    loss = loss_fn(para,para_target,lossweight)
    optinfos = {}
    optinfos['init_rf'] = np.array(rf.tolist())
    optinfos['init_gr'] = np.array(gr.tolist())
    optinfos['init_Nt'] = Nt
    optinfos['init_dt'] = dt
    optinfos['time_hist'] = [0.0]
    optinfos['loss_hist'] = [loss.item()]
    print('initial loss =',loss.item())

    # function record opt infos:
    def addinfo_fn(time0,lossvalue):
        optinfos['time_hist'].append(time()-time0)
        optinfos['loss_hist'].append(lossvalue.item())
        return
    def total_loss(rf,gr):
        a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
        para = loss_para_fn(a_real,a_imag,b_real,b_imag)
        loss = loss_fn(para,para_target,lossweight)
        return loss
    def max_change(seq):
        m = torch.max(torch.diff(seq,dim=1).abs())/dt
        return m
    print('max rf change:',max_change(rf).item())

    # optimization parameters setting:
    niter = requirements['niter']
    rf_niter = requirements['rf_niter'] #100
    gr_niter = requirements['gr_niter']
    lr_rf_init = 1.0
    lr_gr_init = 1.0
    show_detail_step = 1
    print('optimization settings: niter={}, rf niter={}, gr niter={}'.format(niter,rf_niter,gr_niter))

    # line search function:
    def  linesearch_rf(lr,rf,gr,currentloss,grad,d):
        c = 1e-6
        for _ in range(20):
            tmprf = rf + lr*d
            newloss = total_loss(tmprf,gr)
            expectedloss = currentloss + c*lr*torch.dot(grad.view(-1),d.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss
    def linesearch_gr(lr,rf,gr,currentloss,grad,d):
        c = 1e-6
        for _ in range(20):
            tmpgr = gr + lr*d
            newloss = total_loss(rf,tmpgr)
            expectedloss = currentloss + c*lr*torch.dot(grad.view(-1),d.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss

    starttime = time()

    # optimize:
    rf.requires_grad = gr.requires_grad = True
    for k in range(niter):
        # update rf:
        if True:
            for rf_itr in range(rf_niter):
                loss = total_loss(rf,gr)
                loss.backward()
                rf_grad = rf.grad
                gr_grad = gr.grad
                with torch.no_grad():
                    # update:
                    v = -rfmax*torch.nn.functional.normalize(rf_grad,dim=0)
                    # plt.figure()
                    # plt.plot(rf_grad[0,:].tolist())
                    # plt.plot(rf_grad_s[0,:].tolist(),ls='--')
                    # plt.show()
                    d = v-rf
                    if False:
                        print('\t\tmax rf change:',max_change(rf).item(),'max d change:',max_change(d).item())
                        print('\t\t',(0.1-max_change(rf))/max_change(d))

                    # different constraints for step size:
                    # lr_rf_max = (0.1-max_change(rf))/max_change(d)
                    # lr_rf = 2/(rf_itr+2)
                    lr_rf_max = 1.0

                    # line search:
                    lr_rf,loss = linesearch_rf(lr_rf_max,rf,gr,loss,rf_grad,d)
                    rf = rf + lr_rf*d
                    # rf = smoother(rf)

                    # record logs:
                    # loss = total_loss(rf,gr)
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (rf_itr%1) == 0:
                        print('\t--> rf update, loss =',loss.item())
                    # termination:
                    if abs((optinfos['loss_hist'][-1] - optinfos['loss_hist'][-2])/optinfos['loss_hist'][-2]) < 1e-6:
                        break
                rf.grad = None
                rf.requires_grad = True
                gr.grad = None
                gr.requires_grad = True
        # update gr:
        if True:
            for gr_itr in range(gr_niter):
                loss = total_loss(rf,gr)
                loss.backward()
                rf_grad = rf.grad
                gr_grad = gr.grad
                with torch.no_grad():
                    d = -gr_grad
                    print('\td change',max_change(d).item(),'gr change',max_change(gr).item())
                    lr_gr_max = 1.0
                    lr_gr,loss = linesearch_gr(lr_gr_max,rf,gr,loss,gr_grad,d)
                    gr = gr + lr_gr*d
                    # record logs:
                    # loss = total_loss(rf,gr)
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (gr_itr%1) == 0:
                        print('\t--> gr update, loss =',loss.item())
                    # termination:
                    if abs((optinfos['loss_hist'][-1] - optinfos['loss_hist'][-2])/optinfos['loss_hist'][-2]) < 1e-6:
                        break
                rf.grad = None
                rf.requires_grad = True
                gr.grad = None
                gr.requires_grad = True
        # show opt details:
        if ((k+1)%show_detail_step) == 0:
            print('>> end iteration:',k+1,', loss={}'.format(optinfos['loss_hist'][-1]))
    # change the variable back
    # information and logs:
    # print(optinfos)
    # mri.save_infos(Nt,dt,rf,gr,logname=,otherinfodic=optinfos)
    pulse = mri.Pulse(rf,gr,dt)
    return pulse,optinfos


def slr_AFW(spinarray,pulse,para_target,loss_fn,loss_para_fn,requirements):
    # accelereated Frank-Wolfe, FW on rf
    print('>> Optimization method: constrained optimizations')
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    lossweight = requirements['lossweight']
    print('\t',Nt,dt,rf.shape,gr.shape,rfmax,gmax,smax)

    a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
    para = loss_para_fn(a_real,a_imag,b_real,b_imag)
    loss = loss_fn(para,para_target,lossweight)
    optinfos = {}
    optinfos['init_rf'] = np.array(rf.tolist())
    optinfos['init_gr'] = np.array(gr.tolist())
    optinfos['init_Nt'] = Nt
    optinfos['init_dt'] = dt
    optinfos['time_hist'] = [0.0]
    optinfos['loss_hist'] = [loss.item()]
    print('initial loss =',loss.item())

    # function record opt infos:
    def addinfo_fn(time0,lossvalue):
        optinfos['time_hist'].append(time()-time0)
        optinfos['loss_hist'].append(lossvalue.item())
        return
    def total_loss(rf,gr):
        a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
        para = loss_para_fn(a_real,a_imag,b_real,b_imag)
        loss = loss_fn(para,para_target,lossweight)
        return loss
    def max_change(seq):
        m = torch.max(torch.diff(seq,dim=1).abs())/dt
        return m
    print('max rf change:',max_change(rf).item())

    # optimization parameters setting:
    niter = 2
    rf_niter = 4
    gr_niter = 0
    lr_rf_init = 1.0
    lr_gr_init = 1.0
    show_detail_step = 1
    print('optimization settings: niter={}, rf niter={}, gr niter={}'.format(niter,rf_niter,gr_niter))

    # line search function:
    def  linesearch_rf(lr,rf,gr,currentloss,grad,d):
        c = 1e-6
        for _ in range(20):
            tmprf = rf + lr*d
            newloss = total_loss(tmprf,gr)
            expectedloss = currentloss + c*lr*torch.dot(grad.view(-1),d.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss
    def linesearch_gr(lr,rf,gr,currentloss,grad,d):
        c = 1e-6
        for _ in range(20):
            tmpgr = gr + lr*d
            newloss = total_loss(rf,tmpgr)
            expectedloss = currentloss + c*lr*torch.dot(grad.view(-1),d.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss

    starttime = time()

    # optimize:
    rf.requires_grad = gr.requires_grad = True
    for k in range(niter):
        # update rf:
        if True:
            for rf_itr in range(rf_niter):
                loss = total_loss(rf,gr)
                loss.backward()
                rf_grad = rf.grad
                gr_grad = gr.grad
                with torch.no_grad():
                    # update:
                    v = -rfmax*torch.nn.functional.normalize(rf_grad,dim=0)
                    # plt.figure()
                    # plt.plot(rf_grad[0,:].tolist())
                    # plt.plot(rf_grad_s[0,:].tolist(),ls='--')
                    # plt.show()
                    d = v-rf
                    print('\t\tmax rf change:',max_change(rf).item(),'max d change:',max_change(d).item())
                    print('\t\t',(0.1-max_change(rf))/max_change(d))

                    # different constraints for step size:
                    # lr_rf_max = (0.1-max_change(rf))/max_change(d)
                    # lr_rf = 2/(rf_itr+2)
                    lr_rf_max = 1.0

                    # line search:
                    lr_rf,loss = linesearch_rf(lr_rf_max,rf,gr,loss,rf_grad,d)
                    rf = rf + lr_rf*d
                    # rf = smoother(rf)

                    # record logs:
                    # loss = total_loss(rf,gr)
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (rf_itr%1) == 0:
                        print('\t--> rf update, loss =',loss.item())
                    # termination:
                    if abs((optinfos['loss_hist'][-1] - optinfos['loss_hist'][-2])/optinfos['loss_hist'][-2]) < 1e-6:
                        break
                rf.grad = None
                rf.requires_grad = True
                gr.grad = None
                gr.requires_grad = True
        # update gr:
        if False:
            for gr_itr in range(gr_niter):
                loss = total_loss(rf,gr)
                loss.backward()
                rf_grad = rf.grad
                gr_grad = gr.grad
                with torch.no_grad():
                    d = -gr_grad
                    print('\td change',max_change(d).item(),'gr change',max_change(gr).item())
                    lr_gr_max = 1.0
                    lr_gr,loss = linesearch_gr(lr_gr_max,rf,gr,loss,gr_grad,d)
                    gr = gr + lr_gr*d
                    # record logs:
                    # loss = total_loss(rf,gr)
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (gr_itr%1) == 0:
                        print('\t--> gr update, loss =',loss.item())
                    # termination:
                    if abs((optinfos['loss_hist'][-1] - optinfos['loss_hist'][-2])/optinfos['loss_hist'][-2]) < 1e-6:
                        break
                rf.grad = None
                rf.requires_grad = True
                gr.grad = None
                gr.requires_grad = True
        # show opt details:
        if ((k+1)%show_detail_step) == 0:
            print('>> end iteration:',k+1,', loss={}'.format(optinfos['loss_hist'][-1]))
    # change the variable back
    # information and logs:
    # print(optinfos)
    # mri.save_infos(Nt,dt,rf,gr,logname=,otherinfodic=optinfos)
    pulse = mri.Pulse(rf,gr,dt)
    return pulse,optinfos

def slr_FW_gcon(spinarray,pulse,para_target,loss_fn,loss_para_fn,requirements):
    # Frank-Wolfe, FW on rf, and FW on gr, and constrain the leanning rate
    print('>> Optimization method: constrained optimizations')
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    print('\t',Nt,dt,rf.shape,gr.shape,rfmax,gmax,smax)

    a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
    para = loss_para_fn(a_real,a_imag,b_real,b_imag)
    loss = loss_fn(para,para_target)
    optinfos = {}
    optinfos['init_rf'] = np.array(rf.tolist())
    optinfos['init_gr'] = np.array(gr.tolist())
    optinfos['init_Nt'] = Nt
    optinfos['init_dt'] = dt
    optinfos['time_hist'] = [0.0]
    optinfos['loss_hist'] = [loss.item()]
    print('initial loss =',loss.item())

    # function record opt infos:
    def addinfo_fn(time0,lossvalue):
        optinfos['time_hist'].append(time()-time0)
        optinfos['loss_hist'].append(lossvalue.item())
        return
    def total_loss(rf,gr):
        a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
        para = loss_para_fn(a_real,a_imag,b_real,b_imag)
        loss = loss_fn(para,para_target)
        return loss
    def max_change(seq):
        m = torch.max(torch.diff(seq,dim=1).abs())/dt
        return m
    print('max rf change:',max_change(rf).item())

    # optimization parameters setting:
    niter = 4
    rf_niter = 20
    gr_niter = 0
    lr_rf_init = 1.0
    lr_gr_init = 1.0
    show_detail_step = 1

    # line search function:
    def linesearch_rf(lr,rf,gr,currentloss,grad,d):
        c = 1e-6
        for _ in range(20):
            tmprf = rf + lr*d
            newloss = total_loss(tmprf,gr)
            expectedloss = currentloss + c*lr*torch.dot(grad.view(-1),d.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss
    def linesearch_gr(lr,rf,gr,currentloss,grad,d):
        c = 1e-6
        for _ in range(20):
            tmpgr = gr + lr*d
            newloss = total_loss(rf,tmpgr)
            expectedloss = currentloss + c*lr*torch.dot(grad.view(-1),d.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss

    starttime = time()

    # optimize:
    rf.requires_grad = gr.requires_grad = True
    for k in range(niter):
        # update rf:
        if True:
            for rf_itr in range(rf_niter):
                loss = total_loss(rf,gr)
                loss.backward()
                rf_grad = rf.grad
                gr_grad = gr.grad
                with torch.no_grad():
                    # update:
                    v = -rfmax*torch.nn.functional.normalize(rf_grad,dim=0)
                    # plt.figure()
                    # plt.plot(rf_grad[0,:].tolist())
                    # plt.plot(rf_grad_s[0,:].tolist(),ls='--')
                    # plt.show()
                    d = v-rf
                    # print('\t\tmax rf change:',max_change(rf).item(),'max d change:',max_change(d).item())
                    # print('\t\t',(0.1-max_change(rf))/max_change(d), '->', max_change(smoother(d)))
                    lr_rf_max = (0.1-max_change(rf))/max_change(d)
                    # lr_rf = 2/(rf_itr+2)
                    lr_rf,loss = linesearch_rf(1.0,rf,gr,loss,rf_grad,d)
                    rf = rf + lr_rf*d
                    # rf = smoother(rf)
                    # record logs:
                    # loss = total_loss(rf,gr)
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (rf_itr%1) == 0:
                        print('\t--> rf update, loss =',loss.item())
                    # termination:
                    if abs((optinfos['loss_hist'][-1] - optinfos['loss_hist'][-2])/optinfos['loss_hist'][-2]) < 1e-6:
                        break
                rf.grad = None
                rf.requires_grad = True
                gr.grad = None
                gr.requires_grad = True
        # update gr:
        if True:
            for gr_itr in range(gr_niter):
                loss = total_loss(rf,gr)
                loss.backward()
                rf_grad = rf.grad
                gr_grad = gr.grad
                with torch.no_grad():
                    # v = -gmax*torch.nn.functional.normalize(smoother(gr_grad),dim=0)
                    v = -gmax*torch.nn.functional.normalize(gr_grad,dim=0)
                    d = v - gr
                    # lr_gr_max = (smax - max_change(gr))/max_change(d)
                    # lr_gr_max = (smax - max_change(gr))/(max_change(v)-max_change(gr))
                    lr_gr_max = 1e-1
                    lr_gr,loss = linesearch_gr(lr_gr_max,rf,gr,loss,gr_grad,d)
                    # print('\td change',max_change(d).item(),'gr change',max_change(gr).item())
                    # print('\tmax lr:',lr_gr_max)
                    # print('\tgr lr:',lr_gr)
                    # plt.figure()
                    # plt.plot(gr_grad[2,:].tolist())
                    # plt.show()
                    gr = gr + lr_gr*d
                    # record logs:
                    # loss = total_loss(rf,gr)
                    addinfo_fn(starttime,loss)
                    # print details:
                    if (gr_itr%1) == 0:
                        print('\t--> gr update, loss =',loss.item())
                rf.grad = None
                rf.requires_grad = True
                gr.grad = None
                gr.requires_grad = True
        # show opt details:
        if ((k+1)%show_detail_step) == 0:
            print('>> end iteration:',k+1,', loss={}'.format(optinfos['loss_hist'][-1]))
    print('>> end optimization, time:',time()-starttime,'s')
    # change the variable back
    # information and logs:
    # print(optinfos)
    # mri.save_infos(Nt,dt,rf,gr,logname=,otherinfodic=optinfos)
    pulse = mri.Pulse(rf,gr,dt)
    return pulse,optinfos





# -------------------------------------
# My ultimate problem solver !!!
# -------------------------------------
class Spindomain_opt_solver:
    def __init__(self,memory=10):
        # self.device = device
        self.optinfos = {'time_hist':[], 'loss_hist':[]}

        # reserve for LBFGS:
        self.sk_list = []
        self.yk_list = []
        self.pk_list = []
        self.mem_num = memory
        #NOTE: memory is saved for LBFGS method, while not used in the final design methods

    # functions for evaluate some properyies
    def max_change(self,seq,dt):
        m = torch.max(torch.diff(seq,dim=1).abs())/dt
        return m
    # functions for recording history
    def addinfo_fn(self,timedur,lossvalue,roi_err=np.nan):
        self.optinfos['time_hist'].append(timedur)
        self.optinfos['loss_hist'].append(lossvalue.item())
        self.optinfos['roi_err_hist'].append(roi_err.item())
        return
    def optinfo_add(self,name_list,var_list):
        for n,v in zip(name_list,var_list):
            if n not in self.optinfos.keys():
                self.optinfos[n] = v
            else:
                print('[error] add optinfos error... already existing field... {}'.format(n))
    def optinfo_additem(self,name,var):
        if name not in self.optinfos.keys():
            self.optinfos[name] = var
        else:
            print('[error] add optinfos error... already existing field... {}'.format(name))
    # def optinfo_update_requirements(self,requirements):
    #     self.optinfos['rfmax'] = requirements['rfmax']

    # line search functions
    def linesearch_rf(self,lr,rf,gr,currentloss,loss_fn,rf_grad,rf_d):
        '''backtracking linesearch of rf'''
        c = 1e-6
        for _ in range(20):
            tmprf = rf + lr*rf_d
            newloss = loss_fn(tmprf,gr,self.target_para_r,self.target_para_i)
            expectedloss = currentloss + c*lr*torch.dot(rf_grad.view(-1),rf_d.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss
    def linesearch_gr(self,lr,rf,gr,currentloss,loss_fn,gr_grad,gr_d):
        '''linesearch of gr'''
        c = 5*1e-7
        for _ in range(20):
            tmpgr = gr + lr*gr_d
            # newloss = loss_fn(rf,tmpgr)
            newloss = loss_fn(rf,tmpgr,self.target_para_r,self.target_para_i)
            # expectedloss = currentloss + c*lr*torch.dot(gr_grad.view(-1),gr_d.view(-1))
            expectedloss = currentloss
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss
    def linesearch_together(self,lr,rf,gr,currentloss,loss_fn,rf_grad,gr_grad,rf_d,gr_d):
        '''backtracking line search of rf and gr together. --- done'''
        c = 1e-6
        for _ in range(20):
            tmprf = rf + lr*rf_d
            tmpgr = gr + lr*gr_d
            newloss = loss_fn(tmprf,tmpgr)
            g = torch.cat((rf_grad.view(-1),gr_grad.view(-1)))
            d = torch.cat((rf_d.view(-1),gr_d.view(-1)))
            expectedloss = currentloss + c*lr*torch.dot(g,d)
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss
    def linesearch_arctan_rf(self,lr,trfmag,rfang,ts,currentloss,loss_fn,rf_grad,rf_d):
        '''linesearch when doing arctan transformation
        #NOTE not used in the end'''
        c = 1e-6
        d = rf_d.reshape(2,-1)
        for _ in range(20):
            tmptrfmag = trfmag + lr*d[0,:]
            tmprfang = rfang + lr*d[1,:]
            newloss = loss_fn(tmptrfmag,tmprfang,ts)
            expectedloss = currentloss + c*lr*torch.dot(rf_grad.view(-1),rf_d.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = 0.5*lr
        # print('\t lr =',lr)
        return lr,newloss
    def golden_linesearch(self,a,b,fun,delta):
        '''golden ratio line search'''
        lam = a + 0.382*(b-a)
        mu = a + 0.618*(b-a)
        for _ in range(30):
            if fun(lam) > fun(mu):
                if b - lam < delta:
                    return mu
                    # break
                else:
                    a = lam
                    lam = mu
                    mu = a + 0.618*(b - a)
            else:
                if mu - a < delta:
                    return lam
                    break
                else:
                    b = mu
                    lam = a + 0.382*(b - a)
                    mu = lam
            print(a,b)
        return lam
    def easylinesearch(self,a,b,fun,N=100):
        '''search smallest point within [a,b] of fun(x)'''
        plsit = torch.linspace(a,b,N)
        fstar = fun(a)
        pstar = a
        k = 0
        for p in plsit:
            k = k+1
            if (k%10) == 0:
                print('\t\tsearch time:',k)
            fnew = fun(p)
            if fnew < fstar:
                pstar = p
                fstar = fnew
        return pstar
    def constraint_lr(self,x,d,dt,maxdiff):
        '''max lr due to slew-rate, x:(n*N), d:(n*N)'''
        m1 = torch.max(torch.diff(x,dim=1).abs())/dt
        m2 = torch.max(torch.diff(x,dim=1).abs())/dt
        lr_max = ((maxdiff - m1).abs())/m2
        return lr_max
    # functions for optimization
    def lbfgs_dir(self,g):
        '''g: the new gradient'''
        q = g
        m = len(self.sk_list)
        if m == 0:
            d = -q
        else:
            a = []
            for i in range(m):
                # from k-1, k-2, ..., k-m
                '''a: [a(k-1), a(k-2), a(k-3), ...,a(k-m)]'''
                k = m-i-1
                a.append(self.pk_list[k]*torch.dot(self.sk_list[k],q))
                q = q - a[i]*self.yk_list[k]
            # q = q - a*yk_list
            # r = q # suppose H0 is indentity
            for i in range(m):
                # from k-m, ..., k-1
                b = 1/torch.dot(self.sk_list[i],self.yk_list[i]) * torch.dot(self.yk_list[i],q)
                q = q + self.sk_list[i]*(a[m-i-1] - b)
            d = -q
        return d
    def skykpk_memory_update(self,sk,yk):
        c = 1e-4
        pk_inv = torch.dot(yk,sk)
        if pk_inv <= c*torch.norm(sk)*torch.norm(yk):
            print('\t\tmemory update skip')
            pass
        else:
            if len(self.sk_list)>=self.mem_num:
                self.sk_list = self.sk_list[1:]
                self.yk_list = self.yk_list[1:]
                self.pk_list = self.pk_list[1:]
            self.yk_list.append(yk)
            self.sk_list.append(sk)
            self.pk_list.append(1/pk_inv)
            # print(len(self.sk_list),len(self.yk_list))
    def skykpk_memory_clear(self):
        self.sk_list = []
        self.yk_list = []
        self.pk_list = []
    # def rf_update(self):
    #     return
    # def gr_update(self):
    #     return
    # def rfgr_update(self):
    #     pass
    # def gd_step(self,rf,gr,rf_grad,gr_grad):
    def cond_check(self,k,niter,case='skip_first'):
        if case == 'skip_first':
            if k == 0:
                return False
            else:
                return True
        if case == 'skip_last':
            if k == niter-1:
                return False
            else:
                return True
        if case == 'all':
            return True
    def termination_cond(self):
        if abs((self.optinfos['loss_hist'][-1] - self.optinfos['loss_hist'][-2])/self.optinfos['loss_hist'][-2]) < 1e-4:
            return True
        else:
            return False
    def estimate_new_target_para(self,tr,ti,para_r,para_i,roi_idx):
        '''estimate new target para for ROI'''
        r = tr[roi_idx].sum()
        i = ti[roi_idx].sum()
        m = torch.sqrt(r**2 + i**2)
        # print(r.dtype,i.dtype,m.dtype)
        # print(foi_idx.dtype)
        # print(para_r.dtype)
        if m == 0:
            para_r[roi_idx] = -1.
            para_i[roi_idx] = 0.
        else:
            para_r[roi_idx] = (r/m).item()
            para_i[roi_idx] = (i/m).item()
        # print('\t>> new target: beta^2 =({},{})'.format(para_r[roi_idx[0]],para_i[roi_idx[0]]))
        return para_r,para_i
    def gr_dir_mask(self,d,len=20):
        len = int(len/2)
        mask = torch.ones_like(d)*1e-5
        d_norm = d.norm(dim=0)
        if False:
            max_idx = torch.argmax(d_norm)
            if max_idx - len < 0:
                mask[:,:2*len] = 1.0
            elif max_idx + len > d_norm.shape[0]:
                mask[:,-2*len:] = 1.0
            else:
                mask[:,max_idx-len:max_idx+len] = 1.0
        if True:
            th = d_norm.max()*0.95
            idx = (d_norm > th)
            unmask_idx = torch.nonzero(idx).view(-1)
            mask[:,unmask_idx] = 1.0
        return mask
    # main optimization process:
    def optimize(self,spinarray,pulse,para_target,loss_fn,loss_para_fn,requirements):
        '''
        optimization process, 
        - spinarray
        - pulse
        - loss_fn(real,imag,tarreal,tarimag,weight): compute loss
        - loss_para_fn(areal,aimag,breal,bimag): translate the simulation result to parameters
        - requirements: dictionary
        '''
        # get info from requirements:
        # -----------------------------------------------
        print('| requirements:')
        # print(requirements.keys())
        # for k in requirements.keys():
        #     print('|\t',k)  
        
        def check_req_var(requ,namespace,default_val):
            '''function read values in requirements'''
            val = requ.get(namespace)
            if val==None:
                val = default_val
            return val
        
        Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
        # print('max rf change:',self.max_change(rf,dt).item())

        device = requirements['device']
        rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
        
        # optimization parameters setting:
        niter = requirements['niter']
        rf_niter = requirements['rf_niter']
        gr_niter = requirements['gr_niter']
        rf_niter_increase = check_req_var(requirements,'rf_niter_increase',False)
        rfloop_case = check_req_var(requirements,'rfloop_case','all')  #'all'
        grloop_case = check_req_var(requirements,'grloop_case','all')  #'skip_last'
        # 
        lr_rf_init = 1.0
        lr_gr_init = 1.0
        lr_init = 1.0
        # 
        rf_algo = requirements['rf_algo']
        gr_algo = requirements['gr_algo']
        rf_modification = check_req_var(requirements,'rf_modification','none')  # ['noise','shrink',]
        rf_modify_back = requirements['rf_modify_back']
        # 
        roi_idx = requirements['roi_index']
        target_foi_idx = requirements['target_foi_idx']
        lossweight = requirements['lossweight'] # loss weighting
        # 
        show_details = True
        show_details_rfstep = 1
        show_details_grstep = 1
        
        
        # saved to the solver:
        self.target_para_r = requirements['target_para_r']
        self.target_para_i = requirements['target_para_i']
        self.device = requirements['device']

        
        estimate_new_target_cutoff = 3
        estimate_new_target = check_req_var(requirements,'estimate_new_target',False)

        # some infos add to optinfos
        self.optinfo_add(['rfmax','gmax','smax','rf_algo','gr_algo','loss_fn',
                        'niter','rf_niter','gr_niter',
                        'rfloop_case','grloop_case','rf_modification'],
                    [rfmax,gmax,smax,rf_algo,gr_algo,requirements['loss_fn'],
                    niter,rf_niter,gr_niter,
                    rfloop_case,grloop_case,rf_modification])
        self.optinfo_additem('gpu',requirements['gpu'])
        self.optinfo_additem('spin_num',spinarray.num)
        self.optinfos['roi_target'] = [self.target_para_r[target_foi_idx[0]].item(),self.target_para_i[target_foi_idx[0]].item()]
        self.optinfos['pulse_type'] = requirements['pulse_type']
        self.optinfos['fov'] = requirements['fov']
        self.optinfos['dim'] = requirements['dim']
        self.optinfos['roi_shape'] = requirements['roi_shape']
        self.optinfos['roi_xyz'] = requirements['roi_xyz']
        self.optinfos['roi_r'] = requirements['roi_r']
        self.optinfos['roi_offset'] = requirements['roi_offset']
        self.optinfos['weighting_sigma'] = requirements['weighting_sigma']
        self.optinfos['weighting_amp'] = requirements['weighting_amp']
        self.optinfos['transition_width'] = requirements['transition_width']
        self.optinfos['estimate_new_target'] = estimate_new_target
        self.optinfos['estimate_new_target_cutoff'] = estimate_new_target_cutoff
        self.optinfos['initial_roi_target'] = [self.target_para_r[roi_idx[0]].item(), self.target_para_i[roi_idx[0]].item()]
        self.optinfos['final_roi_target'] = [self.target_para_r[roi_idx[0]].item(), self.target_para_i[roi_idx[0]].item()]


        # display requirements:
        if True:
            print(''.center(40,'-'))
            print('| rf:',rf.shape, ', gr:',gr.shape)
            print('| dt:',dt,',rfmax:',rfmax,', gmax:',gmax,', smax:',smax)
            print(''.center(40,'-'))
            print('| spin number: {}'.format(self.optinfos['spin_num']))
            print('| rf_algo: {},\tgr_algo: {}'.format(rf_algo,gr_algo))
            print('| niter: {}, rf_niter: {}, gr_niter: {}'.format(niter,rf_niter,gr_niter))
            print('|\tin addition, rf loop: {}, gr loop: {}'.format(rfloop_case,grloop_case))
            print('|\tbetween rf and gr: rf modification: [{}]'.format(rf_modification))
            print('|\tin opt, estimate new target:',estimate_new_target)
            print('| loss fun: {}'.format(self.optinfos['loss_fn']))
            print(''.center(40,'-'))
        
        

        # Construct functions for loss and error
        # ------------------------------------------------------------------------------
        def total_loss(rf,gr,target_para_r,target_para_i):
            '''total loss fucntion'''
            a_real,a_imag,b_real,b_imag = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
            para_r,para_i = loss_para_fn(a_real,a_imag,b_real,b_imag)
            loss = loss_fn(para_r,para_i,target_para_r,target_para_i,lossweight)
            return loss
        # def arctan_total_loss(trfmag,rfang,ts):
        #     rf = transform_rf_back(trfmag,rfang,rfmax)
        #     gr = transform_gr_back(ts,dt,smax)
        #     a_real,a_imag,b_real,b_imag = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
        #     para = loss_para_fn(a_real,a_imag,b_real,b_imag)
        #     loss = loss_fn(para,para_target,lossweight)
        #     return loss
        def roi_error_innerfn(para_r,para_i,target_para_r,target_para_i,roi_idx):
            '''compute ROI error if already has simulation results'''
            err = torch.sqrt((para_r-target_para_r)**2 + (para_i-target_para_i)**2)
            # err = torch.abs(para_r-target_para_r) + torch.abs(para_i-target_para_i)
            err = torch.sum(err[roi_idx])/roi_idx.size(0)
            return err
        def roi_error_fn(rf,gr,target_para_r,target_para_i,roi_idx):
            '''error within ROI per spin'''
            a_real,a_imag,b_real,b_imag = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
            para_r,para_i = loss_para_fn(a_real,a_imag,b_real,b_imag)
            err = roi_error_innerfn(para_r,para_i,target_para_r,target_para_i,roi_idx)
            return err
        # def total_loss(rf,k,target_para_r,target_para_i):
        #     # gr = kspace2gr(k)
        #     a_real,a_imag,b_real,b_imag = mri.spinorsim(spinarray,Nt,dt,rf,gr)
        #     para_r,para_i = loss_para_fn(a_real,a_imag,b_real,b_imag)
        #     loss = loss_fn(para_r,para_i,target_para_r,target_para_i,lossweight)
        #     return loss
        # if rf_algo in ['arctan_LBFGS','arctan_LBFGS_']:
        #     total_loss = arctan_total_loss

        # def print_details(header)
        def print_details(name='  ',nitr=None,subitr=None,loss=None,roi_err=None,para_r=None,para_i=None,timedur=None,header=False):
            if header:
                print('------------------------------------------------')
                print('|  | iteration | loss | roi err | beta^2 | time ')
                print('------------------------------------------------')
            else:
                s = '|'+name+'|'
                s = s+' -:' if nitr==None else s+' {} :'.format(nitr)
                s = s+'- |' if subitr==None else s+' {} |'.format(subitr)
                s = s+' - |' if loss==None else s+' {:.10f} \t|'.format(loss)
                s = s+' - |' if roi_err==None else s+' {:.5f} |'.format(roi_err)
                if (para_r!=None) & (para_i!=None):
                    s = s+' ({:.5f},{:.5f}) |'.format(para_r,para_i)
                else:
                    s = s+' (-,-) |'
                s = s+'' if timedur==None else s+' {:.2f} s'.format(timedur)
                print(s)
            return


        # Initial objective value and error
        # -------------------------------------------------------------------------------
        with torch.no_grad():
            loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
            roi_err = roi_error_fn(rf,gr,self.target_para_r,self.target_para_i,roi_idx)
            # 
            self.optinfos['init_rf'] = np.array(rf.tolist())
            self.optinfos['init_gr'] = np.array(gr.tolist())
            self.optinfos['init_Nt'] = Nt
            self.optinfos['init_dt'] = dt
            self.optinfos['time_hist'] = [0.0]
            self.optinfos['loss_hist'] = [loss.item()]
            self.optinfos['roi_err_hist'] = [roi_err.item()]
            
            
            # print the initial loss:
            print('| initial loss = {}, ROI error = {}, beta^2 = ({},{})'.format(loss.item(),
                                                                                roi_err.item(),
                                                                                self.target_para_r[roi_idx[0]],
                                                                                self.target_para_i[roi_idx[0]]))

        # new target: for refocusing
        if estimate_new_target:
            with torch.no_grad():
                ar,ai,br,bi = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
                para_r,para_i = loss_para_fn(ar,ai,br,bi)
                self.target_para_r,self.target_para_i = self.estimate_new_target_para(para_r,para_i,self.target_para_r,self.target_para_i,target_foi_idx)
                loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
                roi_err = roi_error_fn(rf,gr,self.target_para_r,self.target_para_i,roi_idx)
                # print('\tloss after new beta^2:',loss.item())
                # print('| ({},{})'.format(self.target_para_r[roi_idx[0]],self.target_para_i[roi_idx[0]]))
                print_details(header=True)
                print_details(loss=loss,roi_err=roi_err,para_r=self.target_para_r[roi_idx[0]],
                    para_i=self.target_para_i[roi_idx[0]])
        
        torch.cuda.empty_cache() # helps with memory

        # print(self.optinfos.keys())

        # return pulse, {}

        # ===================== optimization =============================
        starttime = time()        

        # variable preparation:
        if rf_algo in ['arctan_LBFGS','arctan_LBFGS_','arctan_CG']: # arctan transform methods
            trfmag,rfang = transform_rf(rf,rfmax)
            ts = transform_gr(gr,dt,smax)
            trfmag.requires_grad = rfang.requires_grad = ts.requires_grad = True
            
            if rf_algo == 'arctan_LBFGS':
                opt_rf = torch.optim.LBFGS([trfmag,rfang], lr=3., max_iter=20, history_size=30,
                                            tolerance_change=1e-6,
                                            line_search_fn='strong_wolfe')
                
                opt_gr = torch.optim.LBFGS([ts], lr=3., max_iter=10, history_size=20,
                                            tolerance_change=1e-6,
                                            line_search_fn='strong_wolfe')
            # NOTE: since I didn't use LBFGS in the end, the above block is not used in final design.
        else:
            rf.requires_grad = gr.requires_grad = True
        # rf.requires_grad = gr.requires_grad = True


        # optimization:
        for k in range(niter):
            print('k =',k)

            # when the case using LBFGS with exchange of variables in optimization:
            if rf_algo == 'arctan_LBFGS':
                def closure():
                    opt_rf.zero_grad()
                    opt_gr.zero_grad()
                    rf = transform_rf_back(trfmag,rfang,rfmax)
                    gr = transform_gr_back(ts,dt,smax)
                    a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
                    para = loss_para_fn(a_real,a_imag,b_real,b_imag)
                    loss = loss_fn(para,para_target,lossweight)
                    loss.backward()
                    return loss

            # optimizing rf:
            # ------------------------
            if True:
                if self.cond_check(k,niter,case=rfloop_case):
                # if self.cond_check(k,niter,case='skip_first'):
                # if k>0: # k>0,skip the first update of rf
                    print_details(header=True)
                    for rf_iter in range(rf_niter):
                        if rf_algo == 'AmplitudeSearch':
                            with torch.no_grad():
                                amp_loss_fn = lambda x: total_loss(x*rf,gr)
                                lr_rf_max = rfmax/(rf.max())
                                # p = self.golden_linesearch(0.,lr_rf_max,amp_loss_fn,1e-4)
                                p = self.easylinesearch(0,lr_rf_max,amp_loss_fn,100)
                                rf = p*rf
                                loss = total_loss(rf,gr)
                        if rf_algo == 'GD': # GD
                            loss = total_loss(rf,gr)
                            loss.backward()
                            rf_grad = rf.grad
                            gr_grad = gr.grad
                            with torch.no_grad():
                                lr_rf,loss = self.linesearch_rf(lr_rf_init,rf,gr,loss,total_loss,rf_grad,-rf_grad)
                                rf = rf - lr_rf*rf_grad
                        if rf_algo == 'FGD':
                            loss = total_loss(rf,gr)
                            loss.backward()
                            rf_grad = rf.grad
                            gr_grad = gr.grad
                            with torch.no_grad():
                                pass
                        if rf_algo == 'FW':
                            loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
                            loss.backward()
                            rf_grad = rf.grad
                            gr_grad = gr.grad
                            with torch.no_grad():
                                v = -rfmax*torch.nn.functional.normalize(rf_grad,dim=0)
                                d = v-rf
                                # print('\t\tmax rf change:',max_change(rf).item(),'max d change:',max_change(d).item())
                                # print('\t\t',(0.1-max_change(rf))/max_change(d))

                                # different constraints for step size:
                                # lr_rf_max = (0.1-max_change(rf))/max_change(d)
                                # lr_rf = 2/(rf_itr+2)
                                lr_rf_max = 1.0

                                # line search:
                                lr_rf,loss = self.linesearch_rf(lr_rf_init,rf,gr,loss,total_loss,rf_grad,d)
                                rf = rf + lr_rf*d
                            # NOTE: FW method is the method used in the final design
                        if rf_algo == 'AFW':
                            # TODO: accelerated Frank-Wolfe method
                            if rf_iter == 0:
                                pass
                            else:
                                pass
                            loss = total_loss(rf,gr)
                            loss.backward()
                            rf_grad = rf.grad
                            gr_grad = gr.grad
                            with torch.no_grad():
                                pass
                        if rf_algo == 'LBFGS':
                            # self-implemented LBFGS approach
                            loss = total_loss(rf,gr)
                            loss.backward()
                            rf_grad = rf.grad
                            gr_grad = gr.grad
                            with torch.no_grad():
                                if rf_iter == 0:
                                    self.rf_grad_prev = rf_grad.view(-1)
                                    d = self.lbfgs_dir(rf_grad.view(-1)).view_as(rf)
                                    # 
                                    lr_rf,loss = self.linesearch_rf(lr_rf_init,rf,gr,loss,total_loss,rf_grad,d)
                                    sk = lr_rf*d.view(-1)
                                    rf.add_(lr_rf*d)
                                else:
                                    yk = rf_grad.view(-1) - self.rf_grad_prev
                                    self.rf_grad_prev = rf_grad.view(-1)
                                    self.skykpk_memory_update(sk,yk)
                                    d = self.lbfgs_dir(rf_grad.view(-1)).view_as(rf)
                                    # 
                                    lr_rf,loss = self.linesearch_rf(lr_rf_init,rf,gr,loss,total_loss,rf_grad,d)
                                    sk = lr_rf*d.view(-1)
                                    rf.add_(lr_rf*d)
                            # NOTE: this is not the approach used for the final design
                        if rf_algo == 'arctan_LBFGS':
                            opt_rf.step(closure)
                            # compute the new loss:
                            rf = transform_rf_back(trfmag,rfang,rfmax)
                            gr = transform_gr_back(ts,dt,smax)
                            loss = total_loss(rf,gr)
                            # note: this is using defualt solver in Pytorch
                        if rf_algo == 'arctan_LBFGS_':
                            # loss = arctan_total_loss(trfmag,rfang,ts)
                            # loss.backward()
                            # rf_grad = torch.cat((trfmag.grad,rfang.grad))
                            # ts_grad = ts.grad
                            # # print(rf_grad.shape)
                            # with torch.no_grad():
                            #     if rf_iter == 0:
                            #         self.rf_grad_prev = rf_grad
                            #         d = self.lbfgs_dir(rf_grad)
                            #         # lr_rf = 1e-3
                            #         lr_rf,loss = self.linesearch_arctan_rf(lr_rf_init,trfmag,rfang,ts,loss,arctan_total_loss,rf_grad,d)
                            #         sk = lr_rf*d
                            #         skm = sk.view_as(rf)
                            #         trfmag.add_(skm[0,:])
                            #         rfang.add_(skm[1,:])
                            #     else:
                            #         yk = rf_grad - self.rf_grad_prev
                            #         self.rf_grad_prev = rf_grad
                            #         self.skykpk_memory_update(sk,yk)
                            #         d = self.lbfgs_dir(rf_grad)
                            #         lr_rf,loss = self.linesearch_arctan_rf(lr_rf_init,trfmag,rfang,ts,loss,arctan_total_loss,rf_grad,d)
                            #         # lr = 1e-2
                            #         # update the variables:
                            #         sk = lr_rf*d
                            #         skm = sk.view_as(rf)
                            #         trfmag.add_(skm[0,:])
                            #         rfang.add_(skm[1,:])
                            # # NOTE: this is self-implemented
                            pass
                        if rf_algo == 'arctan_CG':
                            pass
                        
                        timedur = time() - starttime
                        rf = rf.detach()
                        gr = gr.detach()
                        
                        with torch.no_grad():
                            # estimate of error within ROI
                            roi_err = roi_error_fn(rf,gr,self.target_para_r,self.target_para_i,roi_idx)

                            # save optimization info to optinfos:    
                            self.addinfo_fn(timedur=timedur,lossvalue=loss,roi_err=roi_err)
                            
                            if show_details & (((rf_iter+1)%show_details_rfstep) == 0):
                                # print('\t\tmemory: {}'.format(len(self.sk_list)))
                                # print('\t(rf)--> {}, loss:'.format(rf_iter),loss.item())
                                print_details(name='rf',nitr=k,subitr=rf_iter,loss=loss,roi_err=roi_err,
                                    para_r=self.target_para_r[roi_idx[0]],para_i=self.target_para_i[roi_idx[0]],timedur=timedur)
                        
                        # new target: for refocusing
                        if estimate_new_target & (k<estimate_new_target_cutoff):
                            with torch.no_grad():
                                ar,ai,br,bi = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
                                para_r,para_i = loss_para_fn(ar,ai,br,bi)
                                self.target_para_r,self.target_para_i = self.estimate_new_target_para(para_r,para_i,self.target_para_r,self.target_para_i,target_foi_idx)
                                self.optinfos['final_roi_target'] = [self.target_para_r[target_foi_idx[0]].item(),self.target_para_i[target_foi_idx[0]].item()]
                                
                                # compute new loss:
                                loss = loss_fn(para_r,para_i,self.target_para_r,self.target_para_i,lossweight)
                                # loss_ = total_loss(rf,gr,self.target_para_r,self.target_para_i)
                                # print('\tloss after new beta^2:',loss_.item())
                                
                                # compute new ROI error
                                roi_err = roi_error_innerfn(para_r,para_i,self.target_para_r,self.target_para_i,roi_idx)

                                print_details(nitr=k,subitr=rf_iter,loss=loss,roi_err=roi_err,
                                    para_r=self.target_para_r[roi_idx[0]],para_i=self.target_para_i[roi_idx[0]])

                        # reprepare the variables' gradient:
                        if rf_algo in ['arctan_LBFGS_']:
                            trfmag.grad = rfang.grad = None
                            ts.grad = None
                            trfmag.requires_grad = rfang.requires_grad = ts.requires_grad = True
                        elif rf_algo in ['FW','GD','LBFGS']:
                            rf.grad = gr.grad = None
                            rf.requires_grad = gr.requires_grad = True
                        else:
                            pass
                        
                        # termination condition:
                        if self.termination_cond():
                            break
                    
                    # After rf loops:
                    if rf_algo in ['arctan_LBFGS_']:
                        rf = transform_rf_back(trfmag,rfang,rfmax)
                        gr = transform_gr_back(ts,dt,smax)
                    self.skykpk_memory_clear()
                    if rf_niter_increase: # if gradually increase rf iteration number
                        rf_niter = rf_niter + 5

            # between rf and gr update:
            # -----------------------------
            if False:
                if self.cond_check(k,niter,case='skip_last'): # some modifications through the optimization procedure
                    if rf_modification == 'none':
                        print('\t\t----------')
                    else:
                        with torch.no_grad():
                            if rf_modification == 'shrink': # shrink rf
                                self.saved_rf = rf
                                p = 0.5
                                for _ in range(20):
                                    tmprf = (1-p)*rf
                                    newloss = total_loss(tmprf,gr,self.target_para_r,self.target_para_i)
                                    if newloss < loss*1.05:
                                        break
                                    else:
                                        p=p*0.7
                                print('\t\t\tp =',p,', 1-p =',1-p)
                                rf = (1-p)*rf
                            if False:
                                if k < 2: 
                                    # magnitude matching TODO: make it more accurate
                                    p = 1.0
                                    for _ in range(20):
                                        tmprf = p*rf
                                        newloss = total_loss(tmprf,gr)
                                        if newloss > loss:
                                            break
                                        else:
                                            p = 1.1*p
                                    rf = p*rf
                                    print('\t\tp =',p)
                            if rf_modification == 'noise': # add noise to rf:
                                self.saved_rf = rf
                                p = 1.0
                                # nmax0 = rf[0,:].mean()
                                nmax0 = rf[0,:].max()
                                # nmax1 = rf[1,:].mean()
                                nmax1 = rf[1,:].max()
                                for _ in range(20):
                                    tmprf = torch.zeros_like(rf)
                                    tmprf[0,:] = rf[0,:] + (torch.randn_like(rf[0,:])-0.5)*nmax0*p
                                    tmprf[1,:] = rf[1,:] + (torch.randn_like(rf[0,:])-0.5)*nmax1*p
                                    if total_loss(tmprf,gr) < 1.5*loss:
                                        break
                                    else:
                                        p = p*0.7
                                rf = tmprf
                                print('\t\t',nmax0,nmax1, 'p =',p)
                            if rf_modification == 'noisy_shrink':
                                self.saved_rf = rf
                                nmax = rf[0,:].max()
                                rf[0,:] = rf[0,:] - torch.rand_like(rf[0,:])*nmax*0.05
                                nmax = rf[1,:].max()
                                rf[1,:] = rf[1,:] - torch.rand_like(rf[1,:])*nmax*0.05
                            print('\t--> modify rf')
                            print('\t\t----------')
                    if rf_algo in ['arctan_LBFGS','arctan_LBFGS_']:
                        trfmag.grad = rfang.grad = None
                        trfmag.requires_grad = rfang.requires_grad = True
                        ts.grad = None
                        ts.requires_grad = True
                    else:
                        rf.grad = None
                        rf.requires_grad = True
                        gr.grad = None
                        gr.requires_grad = True
                # NOTE: if done some change between rf and gr update, in the final design, 
                # no change was made to the pulse during this
            
            # optimizing gradient:
            # ------------------------------
            if True:
                if self.cond_check(k,niter,case=grloop_case): # k<niter-1,skip in the last loop
                # if self.cond_check(k,niter,case='skip_last'): # k<niter-1,skip in the last loop
                    print_details(header=True)
                    for gr_iter in range(gr_niter):
                        if gr_algo == 'GD':
                            loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
                            loss.backward()
                            gr_grad = gr.grad
                            with torch.no_grad():
                                d = -gr_grad
                                # d = d*self.gr_dir_mask(d,len=10)
                                if False:
                                    plt.figure()
                                    plt.plot(d[0,:].tolist(),label='x')
                                    plt.plot(d[1,:].tolist(),label='y')
                                    plt.legend()
                                    plt.savefig('pictures/tmp_pic.png')
                                    print('save fig...pictures/tmp_pic.png')
                                # lr_gr_max = (gmax-gr.abs().max())/d.abs().max()
                                lr_gr_max = min(((gmax-gr)/d).abs().min(),((-gmax-gr)/d).abs().min())
                                # lr_gr_max = 1.0
                                # print('\tlr_gr_max:',lr_gr_max)
                                lr_gr,loss = self.linesearch_gr(lr_gr_max,rf,gr,loss,total_loss,gr_grad,d)
                                gr = gr + lr_gr*d
                            #NOTE: this the final algorithm for gradient update
                        if gr_algo == 'FW':
                            # print('rf.requires_gra =',rf.requires_grad)
                            loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
                            loss.backward()
                            gr_grad = gr.grad
                            with torch.no_grad():
                                # gr_grad_masked = gr_grad*self.gr_dir_mask(gr_grad,len=20)
                                # print('\t\t\tmax |gr_grad|:',gr_grad.abs().max())
                                v = -gmax*torch.nn.functional.normalize(gr_grad,dim=0)
                                d = v-gr
                                d = d*self.gr_dir_mask(d,len=10)
                                # print('\t\tmax rf change:',max_change(rf).item(),'max d change:',max_change(d).item())
                                # print('\t\t',(0.1-max_change(rf))/max_change(d))

                                # different constraints for step size:
                                # lr_rf_max = (0.1-max_change(rf))/max_change(d)
                                # lr_rf = 2/(rf_itr+2)

                                # lr_gr_max = (gmax-gr.abs().max())/d.abs().max()
                                lr_gr_max = 1.0
                                # print('\tlr_gr_max:',lr_gr_max)

                                # line search:
                                lr_gr,loss = self.linesearch_gr(lr_gr_max,rf,gr,loss,total_loss,gr_grad,d)
                                gr = gr + lr_gr*d
                        if gr_algo == 'arctan_LBFGS':
                            opt_gr.step(closure)
                            # compute the new loss:
                            rf = transform_rf_back(trfmag,rfang,rfmax)
                            gr = transform_gr_back(ts,dt,smax)
                            loss = total_loss(rf,gr)
                        if gr_algo == 'LBFGS':
                            loss = total_loss(rf,gr)
                            loss.backward()
                            rf_grad = rf.grad
                            gr_grad = gr.grad
                            with torch.no_grad():
                                if gr_iter == 0:
                                    self.gr_grad_prev = gr_grad.view(-1)
                                    d = self.lbfgs_dir(gr_grad.view(-1)).view_as(gr)
                                    lr_gr,loss = self.linesearch_gr(lr_gr_init,rf,gr,loss,total_loss,gr_grad,d)
                                    sk = lr_gr*d.view(-1)
                                    gr.add_(lr_gr*d)
                                else:
                                    yk = gr_grad.view(-1) - self.gr_grad_prev
                                    self.gr_grad_prev = gr_grad.view(-1)
                                    self.skykpk_memory_update(sk,yk)
                                    d = self.lbfgs_dir(gr_grad.view(-1)).view_as(gr)
                                    lr_gr,loss = self.linesearch_gr(lr_rf_init,rf,gr,loss,total_loss,gr_grad,d)
                                    sk = lr_gr*d.view(-1)
                                    gr.add_(lr_gr*d)
                        
                        timedur = time()-starttime
                        gr = gr.detach()
                        rf = rf.detach()
                        
                        # additiaonl steps after updating:
                        
                        with torch.no_grad():
                            # estimate ROI error
                            roi_err = roi_error_fn(rf,gr,self.target_para_r,self.target_para_i,roi_idx)

                            # save opt infos:    
                            self.addinfo_fn(timedur=timedur,lossvalue=loss,roi_err=roi_err)
                            if show_details & (((gr_iter+1)%show_details_grstep) == 0):
                                # print('\t\tmemory: {}'.format(len(self.sk_list)))
                                # print('\t(gr)--> {}, loss:'.format(gr_iter),loss.item())
                                print_details('gr',nitr=k,subitr=gr_iter,loss=loss,roi_err=roi_err,
                                    para_r=self.target_para_r[roi_idx[0]],para_i=self.target_para_i[roi_idx[0]],timedur=timedur)

                        with torch.no_grad():
                            # if False:
                            #     # smopthing:
                            #     gr = gradient_smooth(gr,smax,dt)

                            # new target: for refocusing
                            if estimate_new_target & (k<estimate_new_target_cutoff):
                                ar,ai,br,bi = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
                                para_r,para_i = loss_para_fn(ar,ai,br,bi)
                                self.target_para_r,self.target_para_i = self.estimate_new_target_para(para_r,para_i,self.target_para_r,self.target_para_i,target_foi_idx)
                                self.optinfos['final_roi_target'] = [self.target_para_r[target_foi_idx[0]].item(),self.target_para_i[target_foi_idx[0]].item()]
                                
                                # compute new loss
                                loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
                                # print('\tloss after new beta^2:',loss.item())

                                # compute new ROI err
                                roi_err = roi_error_innerfn(para_r,para_i,self.target_para_r,self.target_para_i,roi_idx)

                                print_details(nitr=k,subitr=gr_iter,loss=loss,roi_err=roi_err,
                                    para_r=self.target_para_r[roi_idx[0]],para_i=self.target_para_i[roi_idx[0]],timedur=None)

                            # reprepare the variables' gradients:
                            if gr_algo in ['FW','GD','LBFGS']:
                                rf.grad = None
                                rf.requires_grad = True
                                gr.grad = None
                                gr.requires_grad = True

                        # termination:
                        # if self.termination_cond():
                        #     break

                    # After the gradient update loops:
                    if rf_modify_back:
                        if rf_modification in ['noise','noisy_shrink']:
                            with torch.no_grad():
                                rf = self.saved_rf
                                print('\t--> modify rf back')
                            if rf_algo in ['FW','GD']:
                                rf.grad = None
                                rf.requires_grad = True
                        #NOTE: not used in final design
                    self.skykpk_memory_clear()

            # ----------------------------
            # Another choice: update rf and gr together:
            # -------------------------
            if False:
                loss = total_loss(rf,gr)
                loss.backward()
                rf_grad = rf.grad
                gr_grad = gr.grad
                with torch.no_grad():
                    self.rfgr_update()
                    if True: # GD
                        # lr = 1e-4
                        lr,loss = self.linesearch_together(lr_init,rf,gr,loss,total_loss,rf_grad,gr_grad,-rf_grad,-gr_grad)
                        rf = rf - lr*rf_grad
                        gr = gr - lr*gr_grad
                        # loss = total_loss(rf,gr)
                    if True: # FW
                        pass
                    if True: # LBFGS
                        pass

                    # save opt infos:
                    self.addinfo_fn(time0=starttime,lossvalue=loss)
                    print('\t--> rf and gr update, loss =',loss.item())
                rf.grad = None
                rf.requires_grad = True
                gr.grad = None
                gr.requires_grad = True
        
        # print(self.optinfos.keys())

        # Output the final results:
        pulse = mri.Pulse(rf,gr,dt,device=device)
        return pulse,self.optinfos
    def optimize_plus(self,spinarray,pulse,para_target,loss_fn,loss_para_fn,requirements):
        '''
        the same solver, but trying to simulate and optimize over more huge # of spins
        especially for backward computation
        '''
        # get info from requirements:
        device = requirements['device']
        self.device = requirements['device']

        Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
        rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
        
        print('\t',Nt,dt,rf.shape,gr.shape,rfmax,gmax,smax)
        # print('max rf change:',self.max_change(rf,dt).item())

        
        lossweight = requirements['lossweight'] # loss weighting
        # optimization parameters setting:
        niter = requirements['niter']
        rf_niter = requirements['rf_niter']
        gr_niter = requirements['gr_niter']
        rf_niter_increase = requirements['rf_niter_increase']
        lr_rf_init = 1.0
        lr_gr_init = 1.0
        lr_init = 1.0
        rf_algo = requirements['rf_algo']
        gr_algo = requirements['gr_algo']
        rf_modification = requirements['rf_modification'] # ['noise','shrink',]
        rf_modify_back = requirements['rf_modify_back']
        rfloop_case = requirements['rfloop_case'] #'all'
        grloop_case = requirements['grloop_case'] #'skip_last'
        show_details = True
        show_details_rfstep = 1
        show_details_grstep = 1
        target_foi_idx = requirements['target_foi_idx']
        simu_batch_num = 60000 # the batch size for computing the gradient separatively
        
        # saved to the solver:
        self.target_para_r = requirements['target_para_r']
        self.target_para_i = requirements['target_para_i']
        
        estimate_new_target_cutoff = 3
        try:
            if requirements['estimate_new_target'] == True:
                estimate_new_target = True
            else:
                estimate_new_target = False
        except:
            estimate_new_target = False

        # some infos add to optinfos
        self.optinfo_add(['rfmax','gmax','smax','niter','rf_niter','gr_niter','rf_algo','gr_algo','rfloop_case','grloop_case','rf_modification'],
                    [rfmax,gmax,smax,niter,rf_niter,gr_niter,rf_algo,gr_algo,rfloop_case,grloop_case,rf_modification])
        self.optinfos['loss_fn'] = requirements['loss_fn']
        self.optinfos['pulse_type'] = requirements['pulse_type']
        self.optinfos['spin_num'] = spinarray.num
        self.optinfos['roi_target'] = [self.target_para_r[target_foi_idx[0]].item(),self.target_para_i[target_foi_idx[0]].item()]
        self.optinfos['fov'] = requirements['fov']
        self.optinfos['dim'] = requirements['dim']
        self.optinfos['roi_shape'] = requirements['roi_shape']
        self.optinfos['roi_xyz'] = requirements['roi_xyz']
        self.optinfos['roi_r'] = requirements['roi_r']
        self.optinfos['roi_offset'] = requirements['roi_offset']
        self.optinfos['weighting_sigma'] = requirements['weighting_sigma']
        self.optinfos['weighting_amp'] = requirements['weighting_amp']
        self.optinfos['transition_width'] = requirements['transition_width']
        self.optinfos['estimate_new_target'] = estimate_new_target
        self.optinfos['estimate_new_target_cutoff'] = estimate_new_target_cutoff
        self.optinfos['estimate_new_target'] = requirements['estimate_new_target']


        # display requirements:
        if True:
            print(''.center(20,'-'))
            print('spin number: {}'.format(self.optinfos['spin_num']))
            print('rf_algo: {},\tgr_algo: {}'.format(rf_algo,gr_algo))
            print('niter: {}, rf_niter: {}, gr_niter: {}'.format(niter,rf_niter,gr_niter))
            print('in addition, rf loop: {}, gr loop: {}'.format(rfloop_case,grloop_case))
            print('between rf and gr: rf modification: [{}]'.format(rf_modification))
            print('loss fun: {}'.format(self.optinfos['loss_fn']))
            print(''.center(20,'-'))
        
        
        def total_loss(rf,gr,target_para_r,target_para_i):
            a_real,a_imag,b_real,b_imag = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
            para_r,para_i = loss_para_fn(a_real,a_imag,b_real,b_imag)
            loss = loss_fn(para_r,para_i,target_para_r,target_para_i,lossweight)
            return loss
        def partial_loss(rf,gr,target_para_r,target_para_i,spin_idx):
            part_spinarray = spinarray.get_spins(spin_idx)
            a_real,a_imag,b_real,b_imag = mri.spinorsim(part_spinarray,Nt,dt,rf,gr,device=device)
            para_r,para_i = loss_para_fn(a_real,a_imag,b_real,b_imag)
            loss = loss_fn(para_r,para_i,target_para_r[spin_idx],target_para_i[spin_idx],lossweight[spin_idx])
            return loss
        def arctan_total_loss(trfmag,rfang,ts):
            rf = transform_rf_back(trfmag,rfang,rfmax)
            gr = transform_gr_back(ts,dt,smax)
            a_real,a_imag,b_real,b_imag = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
            para = loss_para_fn(a_real,a_imag,b_real,b_imag)
            loss = loss_fn(para,para_target,lossweight)
            return loss
        def roi_error_fn(rf,gr,target_para_r,target_para_i,target_foi_idx):
            # l1 error within ROI
            a_real,a_imag,b_real,b_imag = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
            para_r,para_i = loss_para_fn(a_real,a_imag,b_real,b_imag)
            err = torch.abs(para_r-target_para_r) + torch.abs(para_i-target_para_i)
            err = torch.sum(err[target_foi_idx])/target_foi_idx.size(0)
            return err
        if rf_algo in ['arctan_LBFGS','arctan_LBFGS_']:
            total_loss = arctan_total_loss
        
        

        # Initial simulation and objective
        # ------------------------------------------------
        # a_real,a_imag,b_real,b_imag = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
        # para_r,para_i = loss_para_fn(a_real,a_imag,b_real,b_imag)
        # loss = loss_fn(para_r,para_i,self.target_para_r,self.target_para_r,lossweight)
        with torch.no_grad():
            loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
            # 
            self.optinfos['init_rf'] = np.array(rf.tolist())
            self.optinfos['init_gr'] = np.array(gr.tolist())
            self.optinfos['init_Nt'] = Nt
            self.optinfos['init_dt'] = dt
            self.optinfos['time_hist'] = [0.0]
            self.optinfos['loss_hist'] = [loss.item()]
            
            # print the initial loss:
            print('initial loss =',loss.item())
        
        total_idx = spinarray.get_index_all()
        Nbatch = int(len(total_idx)/simu_batch_num)+1
        batch_idx_list = total_idx.chunk(Nbatch)
        print(Nbatch)

        torch.cuda.empty_cache()
        # print('1')
        # loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
        # print('2')
        # loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
        


        # ===================== optimization =============================
        starttime = time()

        # new target: for refocusing
        if estimate_new_target:
            with torch.no_grad():
                ar,ai,br,bi = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
                para_r,para_i = loss_para_fn(ar,ai,br,bi)
                self.target_para_r,self.target_para_i = self.estimate_new_target_para(para_r,para_i,self.target_para_r,self.target_para_i,target_foi_idx)
                loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
                print('\tloss after new beta^2:',loss.item())

        # variable preparation:
        if rf_algo in ['arctan_LBFGS','arctan_LBFGS_','arctan_CG']: # arctan transform methods
            trfmag,rfang = transform_rf(rf,rfmax)
            ts = transform_gr(gr,dt,smax)
            trfmag.requires_grad = rfang.requires_grad = ts.requires_grad = True
            
            if rf_algo == 'arctan_LBFGS':
                opt_rf = torch.optim.LBFGS([trfmag,rfang], lr=3., max_iter=20, history_size=30,
                                            tolerance_change=1e-6,
                                            line_search_fn='strong_wolfe')
                
                opt_gr = torch.optim.LBFGS([ts], lr=3., max_iter=10, history_size=20,
                                            tolerance_change=1e-6,
                                            line_search_fn='strong_wolfe')
            # NOTE: since I didn't use LBFGS in the end, the above block is not used in final design.
        else:
            rf.requires_grad = gr.requires_grad = True
        


        # optimization:
        for k in range(niter):
            print('k =',k)

            # when the case using LBFGS with exchange of variables in optimization:
            if rf_algo == 'arctan_LBFGS':
                def closure():
                    opt_rf.zero_grad()
                    opt_gr.zero_grad()
                    rf = transform_rf_back(trfmag,rfang,rfmax)
                    gr = transform_gr_back(ts,dt,smax)
                    a_real,a_imag,b_real,b_imag = mri.slrsim_(spinarray,Nt,dt,rf,gr)
                    para = loss_para_fn(a_real,a_imag,b_real,b_imag)
                    loss = loss_fn(para,para_target,lossweight)
                    loss.backward()
                    return loss

            # optimizing rf:
            # ------------------------
            if True:
                if self.cond_check(k,niter,case=rfloop_case):
                # if self.cond_check(k,niter,case='skip_first'):
                # if k>0: # k>0,skip the first update of rf
                    for rf_iter in range(rf_niter):
                        if rf_algo == 'AmplitudeSearch':
                            with torch.no_grad():
                                amp_loss_fn = lambda x: total_loss(x*rf,gr)
                                lr_rf_max = rfmax/(rf.max())
                                # p = self.golden_linesearch(0.,lr_rf_max,amp_loss_fn,1e-4)
                                p = self.easylinesearch(0,lr_rf_max,amp_loss_fn,100)
                                rf = p*rf
                                loss = total_loss(rf,gr)
                        if rf_algo == 'GD': # GD
                            loss = total_loss(rf,gr)
                            loss.backward()
                            rf_grad = rf.grad
                            gr_grad = gr.grad
                            with torch.no_grad():
                                lr_rf,loss = self.linesearch_rf(lr_rf_init,rf,gr,loss,total_loss,rf_grad,-rf_grad)
                                rf = rf - lr_rf*rf_grad
                        if rf_algo == 'FGD':
                            loss = total_loss(rf,gr)
                            loss.backward()
                            rf_grad = rf.grad
                            gr_grad = gr.grad
                            with torch.no_grad():
                                pass
                        if rf_algo == 'FW':
                            # use multiple steps compute the loss and combine the gradient together
                            loss = 0
                            rf_grad = 0
                            gr_grad = 0
                            batch_nitr = 0
                            for batch_idx in batch_idx_list:
                                batch_nitr = batch_nitr + 1
                                print('batch_nitr =',batch_nitr)

                                lossp = partial_loss(rf,gr,self.target_para_r,self.target_para_i,batch_idx)
                                lossp.backward()
                                rf_grad_b = rf.grad
                                gr_grad_b = gr.grad
                                with torch.no_grad():
                                    loss = loss+lossp.detach()
                                    rf_grad = rf_grad + rf_grad_b
                                    gr_grad = gr_grad + gr_grad_b
                                # rf = rf.detach()
                                # gr = gr.detach()
                                rf.grad = gr.grad = None
                                rf.requires_grad = gr.requires_grad = True
                            # update
                            with torch.no_grad():
                                v = -rfmax*torch.nn.functional.normalize(rf_grad,dim=0)
                                d = v-rf
                                # print('\t\tmax rf change:',max_change(rf).item(),'max d change:',max_change(d).item())
                                # print('\t\t',(0.1-max_change(rf))/max_change(d))

                                # different constraints for step size:
                                # lr_rf_max = (0.1-max_change(rf))/max_change(d)
                                # lr_rf = 2/(rf_itr+2)
                                lr_rf_max = 1.0

                                # line search:
                                lr_rf,loss = self.linesearch_rf(lr_rf_init,rf,gr,loss,total_loss,rf_grad,d)
                                rf = rf + lr_rf*d
                            # NOTE: FW method is the method used in the final design
                        if rf_algo == 'AFW':
                            # TODO: accelerated Frank-Wolfe method
                            if rf_iter == 0:
                                pass
                            else:
                                pass
                            loss = total_loss(rf,gr)
                            loss.backward()
                            rf_grad = rf.grad
                            gr_grad = gr.grad
                            with torch.no_grad():
                                pass
                        if rf_algo == 'LBFGS':
                            # self-implemented LBFGS approach
                            loss = total_loss(rf,gr)
                            loss.backward()
                            rf_grad = rf.grad
                            gr_grad = gr.grad
                            with torch.no_grad():
                                if rf_iter == 0:
                                    self.rf_grad_prev = rf_grad.view(-1)
                                    d = self.lbfgs_dir(rf_grad.view(-1)).view_as(rf)
                                    # 
                                    lr_rf,loss = self.linesearch_rf(lr_rf_init,rf,gr,loss,total_loss,rf_grad,d)
                                    sk = lr_rf*d.view(-1)
                                    rf.add_(lr_rf*d)
                                else:
                                    yk = rf_grad.view(-1) - self.rf_grad_prev
                                    self.rf_grad_prev = rf_grad.view(-1)
                                    self.skykpk_memory_update(sk,yk)
                                    d = self.lbfgs_dir(rf_grad.view(-1)).view_as(rf)
                                    # 
                                    lr_rf,loss = self.linesearch_rf(lr_rf_init,rf,gr,loss,total_loss,rf_grad,d)
                                    sk = lr_rf*d.view(-1)
                                    rf.add_(lr_rf*d)
                            # NOTE: this is not the approach used for the final design
                        if rf_algo == 'arctan_LBFGS':
                            opt_rf.step(closure)
                            # compute the new loss:
                            rf = transform_rf_back(trfmag,rfang,rfmax)
                            gr = transform_gr_back(ts,dt,smax)
                            loss = total_loss(rf,gr)
                            # note: this is using defualt solver in Pytorch
                        if rf_algo == 'arctan_LBFGS_':
                            loss = arctan_total_loss(trfmag,rfang,ts)
                            loss.backward()
                            rf_grad = torch.cat((trfmag.grad,rfang.grad))
                            ts_grad = ts.grad
                            # print(rf_grad.shape)
                            with torch.no_grad():
                                if rf_iter == 0:
                                    self.rf_grad_prev = rf_grad
                                    d = self.lbfgs_dir(rf_grad)
                                    # lr_rf = 1e-3
                                    lr_rf,loss = self.linesearch_arctan_rf(lr_rf_init,trfmag,rfang,ts,loss,arctan_total_loss,rf_grad,d)
                                    sk = lr_rf*d
                                    skm = sk.view_as(rf)
                                    trfmag.add_(skm[0,:])
                                    rfang.add_(skm[1,:])
                                else:
                                    yk = rf_grad - self.rf_grad_prev
                                    self.rf_grad_prev = rf_grad
                                    self.skykpk_memory_update(sk,yk)
                                    d = self.lbfgs_dir(rf_grad)
                                    lr_rf,loss = self.linesearch_arctan_rf(lr_rf_init,trfmag,rfang,ts,loss,arctan_total_loss,rf_grad,d)
                                    # lr = 1e-2
                                    # update the variables:
                                    sk = lr_rf*d
                                    skm = sk.view_as(rf)
                                    trfmag.add_(skm[0,:])
                                    rfang.add_(skm[1,:])
                            # note: this is self-implemented
                        if rf_algo == 'arctan_CG':
                            pass
                        with torch.no_grad():
                            # save to optinfos:
                            self.addinfo_fn(time0=starttime,lossvalue=loss)
                            if show_details & (((rf_iter+1)%show_details_rfstep) == 0):
                                # print('\t\tmemory: {}'.format(len(self.sk_list)))
                                print('\t(rf)--> {}, loss:'.format(rf_iter),loss.item())
                            #NOTE: save optimization info
                        rf = rf.detach()
                        if True:
                            with torch.no_grad():
                                roi_err = roi_error_fn(rf,gr,self.target_para_r,self.target_para_i,target_foi_idx)
                                print('\t\tROI error:',roi_err.item())
                            # NOTE: this is a estimate of error within ROI
                        
                        # new target: for refocusing
                        if estimate_new_target & (k<estimate_new_target_cutoff):
                            with torch.no_grad():
                                ar,ai,br,bi = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
                                para_r,para_i = loss_para_fn(ar,ai,br,bi)
                                self.target_para_r,self.target_para_i = self.estimate_new_target_para(para_r,para_i,self.target_para_r,self.target_para_i,target_foi_idx)
                                loss_ = total_loss(rf,gr,self.target_para_r,self.target_para_i)
                                print('\tloss after new beta^2:',loss_.item())
                                self.optinfos['roi_target'] = [self.target_para_r[target_foi_idx[0]].item(),self.target_para_i[target_foi_idx[0]].item()]
                        
                        # reprepare the variables' gradient:
                        if rf_algo in ['arctan_LBFGS_']:
                            trfmag.grad = rfang.grad = None
                            ts.grad = None
                            trfmag.requires_grad = rfang.requires_grad = ts.requires_grad = True
                        elif rf_algo in ['FW','GD','LBFGS']:
                            rf.grad = gr.grad = None
                            rf.requires_grad = gr.requires_grad = True
                        else:
                            pass
                        
                        # termination condition:
                        if self.termination_cond():
                            break
                    
                    # After rf loops:
                    if rf_algo in ['arctan_LBFGS_']:
                        rf = transform_rf_back(trfmag,rfang,rfmax)
                        gr = transform_gr_back(ts,dt,smax)
                    self.skykpk_memory_clear()
                    if rf_niter_increase: # if gradually increase rf iteration number
                        rf_niter = rf_niter + 5

            # between rf and gr update:
            # -----------------------------
            if False:
                if self.cond_check(k,niter,case='skip_last'): # some modifications through the optimization procedure
                    if rf_modification == 'none':
                        print('\t\t----------')
                    else:
                        with torch.no_grad():
                            if rf_modification == 'shrink': # shrink rf
                                self.saved_rf = rf
                                p = 0.5
                                for _ in range(20):
                                    tmprf = (1-p)*rf
                                    newloss = total_loss(tmprf,gr,self.target_para_r,self.target_para_i)
                                    if newloss < loss*1.05:
                                        break
                                    else:
                                        p=p*0.7
                                print('\t\t\tp =',p,', 1-p =',1-p)
                                rf = (1-p)*rf
                            if False:
                                if k < 2: 
                                    # magnitude matching TODO: make it more accurate
                                    p = 1.0
                                    for _ in range(20):
                                        tmprf = p*rf
                                        newloss = total_loss(tmprf,gr)
                                        if newloss > loss:
                                            break
                                        else:
                                            p = 1.1*p
                                    rf = p*rf
                                    print('\t\tp =',p)
                            if rf_modification == 'noise': # add noise to rf:
                                self.saved_rf = rf
                                p = 1.0
                                # nmax0 = rf[0,:].mean()
                                nmax0 = rf[0,:].max()
                                # nmax1 = rf[1,:].mean()
                                nmax1 = rf[1,:].max()
                                for _ in range(20):
                                    tmprf = torch.zeros_like(rf)
                                    tmprf[0,:] = rf[0,:] + (torch.randn_like(rf[0,:])-0.5)*nmax0*p
                                    tmprf[1,:] = rf[1,:] + (torch.randn_like(rf[0,:])-0.5)*nmax1*p
                                    if total_loss(tmprf,gr) < 1.5*loss:
                                        break
                                    else:
                                        p = p*0.7
                                rf = tmprf
                                print('\t\t',nmax0,nmax1, 'p =',p)
                            if rf_modification == 'noisy_shrink':
                                self.saved_rf = rf
                                nmax = rf[0,:].max()
                                rf[0,:] = rf[0,:] - torch.rand_like(rf[0,:])*nmax*0.05
                                nmax = rf[1,:].max()
                                rf[1,:] = rf[1,:] - torch.rand_like(rf[1,:])*nmax*0.05
                            print('\t--> modify rf')
                            print('\t\t----------')
                    if rf_algo in ['arctan_LBFGS','arctan_LBFGS_']:
                        trfmag.grad = rfang.grad = None
                        trfmag.requires_grad = rfang.requires_grad = True
                        ts.grad = None
                        ts.requires_grad = True
                    else:
                        rf.grad = None
                        rf.requires_grad = True
                        gr.grad = None
                        gr.requires_grad = True
                # NOTE: if done some change between rf and gr update, in the final design, 
                # no change was made to the pulse during this
            
            # optimizing gradient:
            # ------------------------------
            if True:
                if self.cond_check(k,niter,case=grloop_case): # k<niter-1,skip in the last loop
                # if self.cond_check(k,niter,case='skip_last'): # k<niter-1,skip in the last loop
                    for gr_iter in range(gr_niter):
                        if gr_algo == 'GD':
                            # use multiple steps compute the loss and combine the gradient together
                            loss = 0
                            rf_grad = 0
                            gr_grad = 0
                            batch_nitr = 0
                            for batch_idx in batch_idx_list:
                                batch_nitr = batch_nitr + 1
                                print('batch_nitr =',batch_nitr)

                                lossp = partial_loss(rf,gr,self.target_para_r,self.target_para_i,batch_idx)
                                lossp.backward()
                                rf_grad_b = rf.grad
                                gr_grad_b = gr.grad
                                with torch.no_grad():
                                    loss = loss+lossp.detach()
                                    rf_grad = rf_grad + rf_grad_b
                                    gr_grad = gr_grad + gr_grad_b
                                # rf = rf.detach()
                                # gr = gr.detach()
                                rf.grad = gr.grad = None
                                rf.requires_grad = gr.requires_grad = True
                            # update:
                            with torch.no_grad():
                                d = -gr_grad
                                # d = d*self.gr_dir_mask(d,len=10)
                                if False:
                                    plt.figure()
                                    plt.plot(d[0,:].tolist(),label='x')
                                    plt.plot(d[1,:].tolist(),label='y')
                                    plt.legend()
                                    plt.savefig('pictures/tmp_pic.png')
                                    print('save fig...pictures/tmp_pic.png')
                                # lr_gr_max = (gmax-gr.abs().max())/d.abs().max()
                                lr_gr_max = min(((gmax-gr)/d).abs().min(),((-gmax-gr)/d).abs().min())
                                # lr_gr_max = 1.0
                                # print('\tlr_gr_max:',lr_gr_max)
                                lr_gr,loss = self.linesearch_gr(lr_gr_max,rf,gr,loss,total_loss,gr_grad,d)
                                gr = gr + lr_gr*d
                            #NOTE: this the final algorithm for gradient update
                        if gr_algo == 'FW':
                            # print('rf.requires_gra =',rf.requires_grad)
                            loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
                            loss.backward()
                            gr_grad = gr.grad
                            with torch.no_grad():
                                # gr_grad_masked = gr_grad*self.gr_dir_mask(gr_grad,len=20)
                                # print('\t\t\tmax |gr_grad|:',gr_grad.abs().max())
                                v = -gmax*torch.nn.functional.normalize(gr_grad,dim=0)
                                d = v-gr
                                d = d*self.gr_dir_mask(d,len=10)
                                # print('\t\tmax rf change:',max_change(rf).item(),'max d change:',max_change(d).item())
                                # print('\t\t',(0.1-max_change(rf))/max_change(d))

                                # different constraints for step size:
                                # lr_rf_max = (0.1-max_change(rf))/max_change(d)
                                # lr_rf = 2/(rf_itr+2)

                                # lr_gr_max = (gmax-gr.abs().max())/d.abs().max()
                                lr_gr_max = 1.0
                                # print('\tlr_gr_max:',lr_gr_max)

                                # line search:
                                lr_gr,loss = self.linesearch_gr(lr_gr_max,rf,gr,loss,total_loss,gr_grad,d)
                                gr = gr + lr_gr*d
                        if gr_algo == 'arctan_LBFGS':
                            opt_gr.step(closure)
                            # compute the new loss:
                            rf = transform_rf_back(trfmag,rfang,rfmax)
                            gr = transform_gr_back(ts,dt,smax)
                            loss = total_loss(rf,gr)
                        if gr_algo == 'LBFGS':
                            loss = total_loss(rf,gr)
                            loss.backward()
                            rf_grad = rf.grad
                            gr_grad = gr.grad
                            with torch.no_grad():
                                if gr_iter == 0:
                                    self.gr_grad_prev = gr_grad.view(-1)
                                    d = self.lbfgs_dir(gr_grad.view(-1)).view_as(gr)
                                    lr_gr,loss = self.linesearch_gr(lr_gr_init,rf,gr,loss,total_loss,gr_grad,d)
                                    sk = lr_gr*d.view(-1)
                                    gr.add_(lr_gr*d)
                                else:
                                    yk = gr_grad.view(-1) - self.gr_grad_prev
                                    self.gr_grad_prev = gr_grad.view(-1)
                                    self.skykpk_memory_update(sk,yk)
                                    d = self.lbfgs_dir(gr_grad.view(-1)).view_as(gr)
                                    lr_gr,loss = self.linesearch_gr(lr_rf_init,rf,gr,loss,total_loss,gr_grad,d)
                                    sk = lr_gr*d.view(-1)
                                    gr.add_(lr_gr*d)
                        
                        
                        # additiaonl steps after updating:
                        with torch.no_grad():
                            # save opt infos:
                            self.addinfo_fn(time0=starttime,lossvalue=loss)
                            if show_details & (((gr_iter+1)%show_details_grstep) == 0):
                                # print('\t\tmemory: {}'.format(len(self.sk_list)))
                                print('\t(gr)--> {}, loss:'.format(gr_iter),loss.item())
                        gr = gr.detach()
                        if True:
                            with torch.no_grad():
                                roi_err = roi_error_fn(rf,gr,self.target_para_r,self.target_para_i,target_foi_idx)
                                print('\t\tROI error:',roi_err.item())
                            # NOTE: this is a estimate of error within ROI
                        with torch.no_grad():
                            if False:
                                # smopthing:
                                gr = gradient_smooth(gr,smax,dt)
                            # new target: for refocusing
                            if estimate_new_target & (k<estimate_new_target_cutoff):
                                ar,ai,br,bi = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
                                para_r,para_i = loss_para_fn(ar,ai,br,bi)
                                self.target_para_r,self.target_para_i = self.estimate_new_target_para(para_r,para_i,self.target_para_r,self.target_para_i,target_foi_idx)
                                loss = total_loss(rf,gr,self.target_para_r,self.target_para_i)
                                print('\tloss after new beta^2:',loss.item())
                                self.optinfos['roi_target'] = [self.target_para_r[target_foi_idx[0]].item(),self.target_para_i[target_foi_idx[0]].item()]
                            # reprepare the variables' gradients:
                            if gr_algo in ['FW','GD','LBFGS']:
                                rf.grad = None
                                rf.requires_grad = True
                                gr.grad = None
                                gr.requires_grad = True
                        # termination:
                        # if self.termination_cond():
                        #     break
                    # After the gradient update loops:
                    if rf_modify_back:
                        if rf_modification in ['noise','noisy_shrink']:
                            with torch.no_grad():
                                rf = self.saved_rf
                                print('\t--> modify rf back')
                            if rf_algo in ['FW','GD']:
                                rf.grad = None
                                rf.requires_grad = True
                        #NOTE: not used in final design
                    self.skykpk_memory_clear()

        # Output the final results:
        pulse = mri.Pulse(rf,gr,dt,device=device)

        # Try to empty space, wonder if it helps
        torch.cuda.empty_cache()

        return pulse,self.optinfos
    # another optimization with gr parameterized on k-space:
    def optimize_kspacebased(self,):
        print('not implemented')
        return
    def optimize_excitation(self,spinarray,pulse,para_target,loss_fn,loss_para_fn,requirements):
        return




# -----------------------------------------
# optimization as a window
# ----------------------------------------
def window_GD(spinarray,pulse,Mtarget,loss_fn,requirements):
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    rfmax,gmax,smax = requirements['rfmax'],requirements['gmax'],requirements['smax']
    print(Nt,dt,rf.shape,gr.shape,rfmax,gmax,smax)

    niter = 2
    rf_niter = 2
    gr_niter = 2
    window_len = 300
    window_niter = math.ceil(Nt/window_len)
    show_detail_step = 1
    # print(Nt/window_len,(Nt+1)//window_len, window_niter)


    for k in range(niter):
        # 
        if True:
            for _ in range(rf_niter):
                for w_itr in range(window_niter):
                    w_idx = torch.arange(w_itr*window_len,min((w_itr+1)*window_len,Nt))
                    print(w_idx)
        # 
        if False:
            for _ in range(gr_niter):
                for w_itr in range(window_niter):
                    w_idx = torch.arange(w_itr*window_len,min((w_itr+1)*window_len,Nt))
                    print(w_idx)
        if ((k+1)%show_detail_step) == 0:
            print('>> end iteration:',k+1)#,', loss={}'.format(optinfos['loss_hist'][-1]))

    return rf,gr











# --------------------------------------------
# plot functions
# --------------------------------------------
def plot_optinfo(optinfos,picname='tmppic_optinfo.png',title='optimization',savefig=False):
    fig, axs = plt.subplots(2, 1)
    LogPlot = True
    
    # loss curve
    if LogPlot:
        axs[0].semilogy(optinfos['time_hist'],optinfos['loss_hist'],
            marker='.',markersize=10,ls='--',label='loss')
    else:
        axs[0].plot(optinfos['time_hist'],optinfos['loss_hist'],
            marker='.',markersize=10,ls='--',label='loss')
    axs[0].text(optinfos['time_hist'][0],optinfos['loss_hist'][0],str(optinfos['loss_hist'][0]),fontsize=8)
    axs[0].text(0.8*optinfos['time_hist'][-1],1.5*optinfos['loss_hist'][-1],str(optinfos['loss_hist'][-1]),fontsize=8)
    axs[0].legend()
    # axs[0].set_xlabel('time(s)')
    axs[0].set_ylabel('total loss')
    axs[0].set_title(title)
    axs[0].grid(True)
    
    # error curve
    if optinfos.get('roi_err_hist') != None:
        axs[1].plot(optinfos['time_hist'],optinfos['roi_err_hist'],
            marker='.',markersize=10,ls='--',label='ROI err',color='green')
        axs[1].text(optinfos['time_hist'][0],optinfos['roi_err_hist'][0],str(optinfos['roi_err_hist'][0]),fontsize=8)
        axs[1].text(0.8*optinfos['time_hist'][-1],1.5*optinfos['roi_err_hist'][-1],str(optinfos['roi_err_hist'][-1]),fontsize=8)
        axs[1].set_xlabel('time(s)')
        axs[1].set_ylabel('ROI error per spin')
        axs[1].legend()
        axs[1].grid(True)

    if savefig:
        print('save fig ... | '+picname)
        plt.savefig(picname)
    else:
        plt.show()
    return










# --------------------------------

def optimization_example():
    # load initial pulse:
    # data = mri.read_data('logs/mri_log_slr_exc.mat')
    # rf = torch.tensor(data['rf'],device=device)
    # gr = torch.tensor(data['gr'],device=device)
    # Nt,dt = data['Nt'].item(),data['dt'].item()
    Nt,dt = 400,1e-2
    rf = torch.ones((2,Nt),device=device)
    # rf[0,:] = torch.sin(torch.linspace(0,torch.pi,Nt))
    gr = torch.zeros((3,Nt),device=device)
    gr[2,:] = 0.1
    #
    # print(Nt,dt)
    # print(type(Nt))
    # mri.plot_pulse(rf,gr,dt)
    # print(rf)

    # define text spin cube:
    cube = mri.Build_SpinArray(fov=[4,4,1],dim=[1,1,50])
    cube.show_info()
    print(cube.T1.dtype)
    Md = torch.zeros_like(cube.Mag, device=device)
    Md[2,:] = 1.0

    # define loss function:
    def loss_fun(spinarray,Nt,dt,rf,gr,Md):
        M = mri.blochsim(spinarray,Nt,dt,rf,gr) # todo, complete the simulation function
        loss = loss_angle(M,Md)
        return loss
    # print initial loss:
    loss = loss_fun(cube,Nt,dt,rf,gr,Md)
    M = mri.blochsim(cube,Nt,dt,rf,gr)
    print(M)
    print('initial loss:',loss)

    # optimization the objective function:
    # rf_opt, gr_opt = .....
    rf.requires_grad = gr.requires_grad = True
    for k in range(10):
        loss = loss_fun(cube,Nt,dt,rf,gr,Md)
        loss.backward()
        with torch.no_grad():
            lr = 1e-4
            rf = rf - lr*rf.grad
            gr = gr - lr*gr.grad
            loss = loss_fun(cube,Nt,dt,rf,gr,Md)
            print(k,loss)
        rf.grad = None
        gr.grad = None
        rf.requires_grad = True
        gr.requires_grad = True
    # M = mri.blochsim(cube,Nt,dt,rf,gr)
    # print(M)
    mri.plot_pulse(rf,gr,dt)
# ------------------------------






if __name__ == '__main__':
    print()
    # pulse = mri.example_pulse()
    # print(pulse.Nt,pulse.dt)
    # data = mri.read_data('logs/mri_log.mat')
    # optimization_example()

    # test_transforms() # pass
    # test_smoother() # pass
    # test_gradient_smooth()