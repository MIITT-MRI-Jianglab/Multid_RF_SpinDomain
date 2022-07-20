# optimization using mri module
# author: jiayao

import torch
import numpy as np
import mri
from time import time
import math
import matplotlib.pyplot as plt


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
    diff[idx] = smax*0.99999
    idx = torch.nonzero(diff<-smax)
    diff[idx] = -smax*0.99999
    # cat the first one
    s = torch.cat((gr[:,0].reshape(3,1),diff),dim=1)
    # print(s[:,:5])
    '''tan'''
    s_normal = s/smax*torch.pi/2 # in range (-1,1)
    ts = s_normal.tan()
    return ts
def transform_gr_back(ts,dt,smax):
    s = ts.atan()/torch.pi*2*smax
    dg = torch.cat((s[:,0].reshape(3,1),s*dt),dim=1) # mT/m
    # print(s[:,:5])
    g = torch.cumsum(s,dim=1)
    return g
# test those transforms:
def test_transforms():
    Nt,dt = 100,0.5
    if True: # test on gr:
        gr = torch.zeros((3,Nt),device=device)
        gr[2,:] = 12.0
        smax = 10
        print(gr[:,:5])
        ts = transform_gr(gr,dt,smax)
        print(ts[:,:5])
        gr_ = transform_gr_back(ts,dt,smax)
        print(gr_[:,:5])
        print()
    # test on rf:
    if False:
        rf = torch.rand((2,Nt),device=device)
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
    niter = 2
    rf_niter = 2
    gr_niter = 2
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
        if False:
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
    rf_niter = 1
    gr_niter = 0
    show_detail_step = 1

    # optimization:
    trfmag,rfang = transform_rf(rf,rfmax)
    ts = transform_gr(gr,dt,smax)
    trfmag.requires_grad = rfang.requires_grad = ts.requires_grad = True
    
    opt_rf = torch.optim.LBFGS([trfmag,rfang], lr=3., max_iter=20, history_size=30,
                                tolerance_change=1e-6,
                                line_search_fn='strong_wolfe')
    opt_gr = torch.optim.LBFGS([ts], lr=3., max_iter=5, history_size=20,
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
    niter = 1
    rf_niter = 100
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
def plot_optinfo(optinfos,picname='pictures/mri_tmp_pic_optinfo.png',save_fig=False,title='optimization'):
    plt.figure()
    LogPlot = False
    if LogPlot:
        plt.semilogy(optinfos['time_hist'],optinfos['loss_hist'],
            marker='.',markersize=10,ls='--',label='loss')
    else:
        plt.plot(optinfos['time_hist'],optinfos['loss_hist'],
            marker='.',markersize=10,ls='--',label='loss')
    plt.legend()
    plt.xlabel('time(s)')
    plt.title(title)
    if save_fig:
        print('save fig...'+picname)
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