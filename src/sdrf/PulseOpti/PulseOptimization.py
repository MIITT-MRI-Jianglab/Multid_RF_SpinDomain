# optimization functions for the project
# author: jiayao
import os
import torch
import typing
from time import time
import scipy.io as spio
# from tqdm import tqdm
from .. import mri
from ..mrsim import SpinDomain
# from ..mrsim import Bloch

class Solver:
    """Base for different pulse optimizers."""
    def __init__(self):
        self.optinfos = {}
        self.log = []
        pass
    def _check_termination(self):
        '''Check termination condition'''
        return False
    def savelog(self,logfile):
        '''Write log strings to a file.'''
        with open(logfile,'w') as f:
            for s in self.log:
                f.write(s)
                f.write('\n')
        return
    def _optinfos_append(self,key_list,val_list):
        '''Append values to optinfos with matched keys.'''
        num = len(key_list)
        for n in range(num):
            key = key_list[n]
            if self.optinfos.get(key):
                self.optinfos[key].append(val_list[n])
            else:
                self.optinfos[key] = [val_list[n]]
        return
    def linesearch_rf(self,lr,rf,gr,currentloss,loss_fn,rf_grad,rf_dir):
        '''Line search function of rf.

        Use backtracking line search, stopped by Armijo condition. 
        
        input:
            lr: learning rate
            rf: ...
            gr: ...
            currentloss: current loss value
            loss_fn: 
            rf_grad:
            rf_dir: rf search direction
        output:
            lr: searched learning rate
            newloss: new loss value
        '''
        c = 1e-6
        cntr = 0.5  #contraction
        for _ in range(20):
            tmprf = rf + lr*rf_dir
            newloss = loss_fn(tmprf,gr)
            expectedloss = currentloss + c*lr*torch.dot(rf_grad.view(-1),rf_dir.view(-1))
            if newloss < expectedloss:
                break
            else:
                lr = cntr*lr
        # print('learning rate:',lr)
        return lr,newloss
    def linesearch_gr(self,lr,rf,gr,currentloss,loss_fn,gr_grad,gr_dir,smax=120,dt=5e-3):
        '''Line search function of gr.
        
        Use backtracking line search.

        input:
            lr: learning rate
            rf: ...
            gr: ...
            currentloss: current loss value
            loss_fn: 
            rf_grad:
            rf_dir: rf search direction
            smax: slew-rate (mT/m/ms)
            dt: ms
        output:
            lr:
            newloss:
        '''
        c = 1e-6
        cntr = 0.5
        for _ in range(20):
            tmpgr = gr + lr*gr_dir
            newloss = loss_fn(rf,tmpgr)
            expectedloss = currentloss + c*lr*torch.dot(gr_grad.view(-1),gr_dir.view(-1))
            if newloss < expectedloss:
                srate = torch.diff(gr,dim=1)/dt
                # print(srate.abs().max())
                if srate.abs().max() < smax:
                    break
                else:
                    lr = cntr*lr
            else:
                lr = cntr*lr
        # print('learning rate:',lr)
        return lr,newloss
    def update_log(self,name='  ',
                   nitr=None,subitr=None,
                   objective=0.,
                   erreval_list=[],erreval_name_list=[],
                   otherval_list=[],otherval_name=[],
                   timedur=None,header=False,display=True):
        '''Display iteration infos during the optimization.'''
        if header:
            s = '--------------------------------------------------------------------\n'
            s += '|  | iteration | objective | '
            for n in erreval_name_list:
                s += ' '+n+' |'
            for n in otherval_name:
                s += ' '+n+' |'
            s = s + ' time \n'
            s += '--------------------------------------------------------------------'
            if display: print(s)
        else:
            s = '|'+name+'|'
            
            # iteration & objective
            s = s+' -:' if nitr==None else s+' {} :'.format(nitr)
            s = s+'- |' if subitr==None else s+' {} |'.format(subitr)
            s = s+' - |' if objective==None else s+' {:.10f} \t|'.format(objective)
            
            # other info
            for v in erreval_list:
                s += ' {:.5f} |'.format(v)
            for v in otherval_list:
                s += ' {:.5f} |'.format(v)

            # Last, update run time info
            s = s+'' if timedur==None else s+' {:.2f} min'.format(timedur/60)
            if display: print(s)
        self.log.append(s)
        return
    def optimize(self): # TODO
        '''optimization (todo)'''


def example_eval_fn(*a):
    '''for testing'''
    return 0


# Spin-domain optimization solvers : 
# ---------------------------------------

class Spindomain_opt_solver(Solver):
    '''Spin-domain optimization solver for RF pulse (e.g., refocusing) and gradient optimization.

    solve for target (refocusing) pulse, for target spin-domain parameters
    - constrained optimization
    - update of rf, and gr using Frank-Wolfe method
    '''
    def __init__(self) -> None:
        super().__init__()
    def optimize(self,spinarray:mri.SpinArray,pulse:mri.Pulse,
                #  --- for build objective functions
                 losspara_target_list, losspara_fn, loss_fn, lossweight=1.,
                #  --- for system constraints 
                 rfmax=0.014,gmax=24,smax=170,
                #  --- for optimization parameters
                 niter=2,rf_niter=2,gr_niter=2, lam=0,
                #  --- for monitoring the performance
                eval_fn_list=[], 
                eval_savename_list=[],
                eval_terminate_fn=None,
                #  eval_selectedspins_list=[],
                #  eval_fn=example_eval_fn,
                #  --- for save results
                 results_folder='', savetmppulse_name = 'pulse_opti.mat', save=False, 
                 pulse_function='refocusing',
                 details=False):
        '''Optimization.

        A framework for either refocusing or excitation pulse design.

        ::notes
        about optimization iterations
        1. update variables
        2. calculate new simulation
        3. calculate performance depends on input specifications


        Args:
            spinarray: 
            pulse: 
            losspara_target_list:   [(1,#spins), (1,#spins), ...]
            losspara_fn:            f(a_real,a_imag,b_real,b_imag), translate to variables for objective function
            loss_fn:                objective loss function
            lossweight:             (1,#spins)
            rfmax:   (mT)
            gmax:    (mT/m)
            smax:    (mT/m/ms)
            niter:                  total big iteration number
            rf_niter:               inner rf iteration number
            gr_niter:               inner gr iteration number
            lam:                    regularization parameter

            eval_fn_list:           list of functions for monitoring performance, e.g., (alpha,beta) -> metrics
            eval_savename_list:     saved name for corresponding error evaluation
            
            results_folder: 
            savetmppulse_name:
            save:                    whether to save temp results
            details: ...

        Returns:
            pulse_opt:              optimized pulse object
        '''
        # get configuration and requirements
        device = pulse.device
        Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
        rf = rf.contiguous() # not clear why this step needed in some cases
        gr = gr.contiguous()

        # currently the number of terms is either 2 or 3
        num_loss_terms = len(losspara_target_list)
        self.target_para_1 = losspara_target_list[0]
        self.target_para_2 = losspara_target_list[1]
        self.target_para_3 = losspara_target_list[2] if num_loss_terms==3 else 0.
        
        # function that compute the total loss in different cases
        if num_loss_terms==2:
            def totalloss_fn(rf,gr):
                '''total loss function for two terms'''
                a_real,a_imag,b_real,b_imag = SpinDomain.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
                para_r,para_i = losspara_fn(a_real,a_imag,b_real,b_imag)
                loss = loss_fn(self.target_para_1,self.target_para_2,para_r,para_i,lossweight)
                loss = loss + lam*torch.sum(rf**2)
                return loss
        elif num_loss_terms==3:
            def totalloss_fn(rf,gr):
                '''total loss function for three terms'''
                a_real,a_imag,b_real,b_imag = SpinDomain.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
                para_1,para_2,para_3 = losspara_fn(a_real,a_imag,b_real,b_imag)
                loss = loss_fn(self.target_para_1,self.target_para_2,self.target_para_3,
                               para_1,para_2,para_3,lossweight)
                loss = loss + lam*torch.sum(rf**2)
                return loss
        else:
            print('error in build total loss function')
            raise BaseException
        

        if eval_terminate_fn == None:
            eval_terminate_fn = lambda a,b: False

        def estimate_new_target_para(tr,ti,para_r,para_i,roi_idx):
            '''Estimate new target parameters for ROI.'''
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

        # function that evaluate some performance
        def erreval(rf,gr):
            '''Evaluation of mean errors for selected groups of spins, + termination condition.'''
            alpha,beta = SpinDomain.spinorsim_c(spinarray,Nt,dt,rf,gr,device)
            # print(alpha[:10])
            # _ = input('continue:')
            errlist = []
            for fn in eval_fn_list:
                err = fn(alpha,beta)
                errlist.append(err.item())
            # get termination instruction
            termination_flag = eval_terminate_fn(alpha,beta)
            return errlist, termination_flag
        
        # compute initial objective value and error -----------
        with torch.no_grad():
            loss = totalloss_fn(rf,gr)
            errlist, gonna_stop = erreval(rf,gr)
        self._optinfos_append(eval_savename_list,errlist)
        # some other infos to save
        self.optinfos['init_rf'] = rf.cpu().numpy()
        self.optinfos['init_gr'] = gr.cpu().numpy()
        self.optinfos['init_dt'] = dt
        self.optinfos['init_Nt'] = Nt
        self.optinfos['time_hist'] = [0]
        self.optinfos['loss_hist'] = [loss.item()]
        # 
        self.update_log(header=True,erreval_name_list=eval_savename_list)
        self.update_log(
            name='initial',objective=loss,
            erreval_list=errlist,erreval_name_list=eval_savename_list,
        )
        

        # optimization ----------------------------------------
        # things to save: 1) time; 2) loss function; 3) eval_list;
        starttime = time()
        for itr in range(niter):
            self.update_log(header=True,erreval_name_list=eval_savename_list)
            # optimizing rf
            # ---------------------------
            for rf_iter in range(rf_niter):
                # using FW method
                rf.requires_grad = gr.requires_grad = True
                loss = totalloss_fn(rf,gr)
                # a_real,a_imag,b_real,b_imag = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
                # para_r,para_i = losspara_fn(a_real,a_imag,b_real,b_imag)
                # loss = loss_fn(para_r,para_i,self.targetpara_r,self.targetpara_i,lossweight)
                loss.backward()
                rf_grad = rf.grad
                gr_grad = gr.grad
                with torch.no_grad():
                    v = -rfmax*torch.nn.functional.normalize(rf_grad,dim=0)
                    d = v-rf
                    # line search
                    lr_rf = 1.0
                    lr_rf,loss = self.linesearch_rf(lr_rf,rf,gr,loss,totalloss_fn,rf_grad,d)
                    rf = rf + lr_rf*d
                    
                    # some evaluations
                    errlist, gonna_stop = erreval(rf,gr)

                timedur = time() - starttime
                self.optinfos['time_hist'].append(timedur)
                self.optinfos['loss_hist'].append(loss.item())
                self._optinfos_append(eval_savename_list,errlist)
                display = True if details else (rf_iter+1)==rf_niter
                self.update_log(
                    name='rf',nitr=itr,subitr=rf_iter,objective=loss,
                    erreval_list=errlist,erreval_name_list=eval_savename_list,
                    timedur=timedur, display=display
                )

                # re-prepare the gradient
                rf.grad = gr.grad = None
                rf.requires_grad = gr.requires_grad = False

            if gonna_stop: break

            # optimizing gr
            # ---------------------------
            for gr_iter in range(gr_niter):
                # using FW method
                rf.requires_grad = gr.requires_grad = True
                loss = totalloss_fn(rf,gr)
                loss.backward()
                rf_grad = rf.grad
                gr_grad = gr.grad
                with torch.no_grad():
                    v = -gmax*torch.nn.functional.normalize(gr_grad,dim=0)
                    d = v-gr
                    # line search
                    lr_gr = 1.0
                    lr_gr,loss = self.linesearch_gr(lr_gr,rf,gr,loss,totalloss_fn,gr_grad,d,smax=smax,dt=dt)
                    gr = gr + lr_gr*d

                    # Some evaluations
                    errlist, gonna_stop = erreval(rf,gr)

                # display optimization info
                timedur = time() - starttime
                self.optinfos['time_hist'].append(timedur)
                self.optinfos['loss_hist'].append(loss.item())
                self._optinfos_append(eval_savename_list,errlist)
                display = True if details else (gr_iter+1)==gr_niter
                self.update_log(
                    name='gr',nitr=itr,subitr=gr_iter,objective=loss,
                    erreval_list=errlist,erreval_name_list=eval_savename_list,
                    timedur=timedur, display=display
                )

                # re-prepare the gradient
                rf.grad = gr.grad = None
                rf.requires_grad = gr.requires_grad = False
                
            if gonna_stop: break

            # save the pulse results for these steps
            # in case want to stop the algorithm earlier
            if save:
                pulsetmp = mri.Pulse(dt=dt,rf=rf,gr=gr,device=device)
                pulsetmp.save(os.path.join(results_folder,savetmppulse_name))
                
                # also save the result for each iteration 
                # pulsetmp.save(os.path.join(results_folder,'iter{}_'.format(itr)+savetmppulse_name))
                # pulsetmp.plot(results_folder+'tmp_pulse_opt.png',savefig=True)
        
        # rf = rf.detach()
        # gr = gr.detach()
        # return the results ----------------------------------
        pulse_opt = mri.Pulse(dt=dt,rf=rf,gr=gr,device=device)
        return pulse_opt



# Bloch simualtion based optimization solvers : 
# ---------------------------------------
class Bloch_opt_solver(Solver):
    ''''''

