'''Spin-domain simulation functions.

for one spin:
- spinorsim_spin: return complex numbers (not for jacobian operation)

for spins array:
- spinorsim_c: return compelx numbers (not for jacobian operation)
- spinorsim: return real numbers (explicit jacobian implemented for faster backward!)

Reference: 
    Pauly, John, et al. 
    Parameter relations for the Shinnar-Le Roux selective excitation pulse design algorithm (NMR imaging).
    IEEE transactions on medical imaging 10.1 (1991): 53-65.
'''
import numpy as np
import torch
from .. import mri


# ################################################################################
# spin-domain simulation for a single spin
# 
# ################################################################################

# Simulation for single spin
def _spinorsim_spin_singlestep(spin:mri.Spin,Nt,dt,rf,gr,device=torch.device('cpu'),history=False): #TODO
    '''One-step spin-domain simulation function (use effective B field) for a single spin.
    Consider rf,gr at the same time to obtain the effective field.
    
    input:
        spin:
        Nt: 
        dt: 		(ms)
        rf:(2*Nt)	(mT)
        gr:(3*Nt)	(mT/m)
        device:		"cpu"/"gpu"
        history:	True/False
    output:
        alpha: (complex)
        beta: (complex)
    '''
    # Calculate the effective magnetic field [mT]
    # Beff = B1*kappa + B_offres + B_gradient
    # <r,G> = cm * mT/m * 1e-2 = mT
    # df[Hz] / gamma[MHz/T] * 1e-3 = mT
    # --------------------------------------
    # Beff_hist = torch.zeros((3,Nt),device=device)*1.0
    # Beff_hist[:2,:] = rf*spin.kappa # mT
    # Beff_hist[2,:] = spin.loc@gr*1e-2 + spin.df/spin.gamma*1e-3
    Beff_hist = spin.get_Beffective(rf=rf,gr=gr)


    # for recording:
    ahist = (1.0+0.0j)*torch.zeros(Nt+1,device=device)
    bhist = (1.0+0.0j)*torch.zeros(Nt+1,device=device)
    ahist[0] = 1.0+0.0j
    bhist[0] = 0.0+0.0j
    for t in range(Nt):
        Beff = Beff_hist[:,t]
        Beff_norm = torch.linalg.norm(Beff,2)
        if Beff_norm == 0:
            Beff_unit = torch.zeros(3,device=device)
        else:
            Beff_unit = Beff/torch.linalg.norm(Beff,2)

        # compute parameters for the rotation
        # phi[rad] = - 2*pi*gamma[MHz/T]*Beff[mT]*dt[ms]  
        # Caution: the sign/direction of rotation!
        # ----------------------------------------
        phi = -(2*torch.pi*spin.gamma)*Beff_norm*dt

        # Compute the spin-domain update parameters
        # -----------------------------------------
        aj = torch.cos(phi/2) - 1j*Beff_unit[2]*torch.sin(phi/2)
        bj = (Beff_unit[1]-1j*Beff_unit[0])*torch.sin(phi/2)
        ahist[t+1] = aj*ahist[t] - bj.conj()*bhist[t]
        bhist[t+1] = bj*ahist[t] + aj.conj()*bhist[t]

    # Get the final state
    a,b = ahist[-1],bhist[-1]	
    if history:
        return ahist,bhist
    else:
        return a,b
def spinorsim_spin(spin:mri.Spin,Nt,dt,rf,gr,device=torch.device('cpu'),history=False):
    '''simulation in spin-domain. -> (alpha,beta)
    
    Args:
        spin:     mri.Spin
        Nt: 
        dt: ms
        rf: (2,Nt), mT
        gr: (3,Nt), mT/m
        device: 
        history: 
    
    Returns:
        alpha: 
        beta: 
    '''
    alpha,beta = _spinorsim_spin_singlestep(spin,Nt,dt,rf,gr,device=device,history=history)
    # alpha,beta = _spinorsim_spin_seperatestep(spin,Nt,dt,rf,gr,device=device,history=history)
    return alpha,beta


# ################################################################################
# spin-domain simulation for a group of spins
# 
# ################################################################################

# simulation functions using complex numbers for calculations 
def _spinorsim_c_singlestep(spinarray:mri.SpinArray,Nt,dt,rf,gr,device=torch.device('cpu'),details=False,history=False):
    '''Spin-domain simulation for spinarray, compute using complex number.
    Consider rf and gr and the same time to get effective field.
    
    input:
        spinarray:
        Nt:
        dt:(ms)
        rf:(2*Nt)(mT)
        gr:(3*Nt)(mT/m)
        device:
    output:
        a: (1*num)(complex)
        b: (1*num)(complex)
    '''

    # if want to simulate smaller Nt
    Nt_p = min(rf.shape[1],gr.shape[1])
    if Nt > Nt_p:
        Nt = Nt_p
        if details:
            print('modified Nt to {}'.format(Nt))
    rf = rf[:,:Nt]
    gr = gr[:,:Nt]

    num = spinarray.num

    # Compute effective magnetic field
    Beff_hist = spinarray.get_Beffective(Nt=Nt, rf=rf, gr=gr, device=device)

    # compute normalized B and phi, for all time points:
    # --------------------------------------------------
    Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
    Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
    phi_hist = -Beff_norm_hist*(spinarray.gamma.reshape(-1,1))*dt*2*torch.pi #(num*Nt)

    # for recording:
    ahist = (1.0+0.0j)*torch.zeros((num,Nt+1),device=device)
    bhist = (1.0+0.0j)*torch.zeros((num,Nt+1),device=device)
    ahist[:,0] = 1.0+0.0j
    bhist[:,0] = 0.0+0.0j
    for t in range(Nt):
        phi = phi_hist[:,t]
        nr = Beff_unit_hist[:,:,t] # rotation axis (3*num)

        aj = torch.cos(phi/2) - 1j*nr[2,:]*torch.sin(phi/2)
        bj = (nr[1,:] - 1j*nr[0,:])*torch.sin(phi/2)

        # Update
        ahist[:,t+1] = aj*ahist[:,t] - bj.conj()*bhist[:,t]
        bhist[:,t+1] = bj*ahist[:,t] + aj.conj()*bhist[:,t]

    # return the final value
    a = ahist[:,-1]
    b = bhist[:,-1]
    return a,b
def spinorsim_c(spinarray:mri.SpinArray,Nt,dt,rf,gr,device=torch.device('cpu'),details=False,history=False)->tuple[torch.Tensor,torch.Tensor]:
    '''Spin-domain simulation as complex numbers. -> (alpha,beta)
    
    Args:
        spinarray:  mri.SpinArray
        Nt:         number of time points
        dt:         (ms) 
        rf:         (2,Nt), mT
        gr:         (3,Nt), mT/m
        device:     torch.device

    Returns:
        alpha:   (1,Nt) (complex number)
        beta:    (1,Nt) (complex number)
    '''
    alpha,beta = _spinorsim_c_singlestep(spinarray,Nt,dt,rf,gr,device=device,details=False,history=history)
    # alpha,beta = _spinorsim_c_seperatestep(spinarray,Nt,dt,rf,gr,device=device,details=False,history=history)
    return alpha,beta 

# spin-domain simulation avoid complex numbers, by using real numbers
# for testing/developing purpose 
def _spinorsim_r_singlestep(spinarray:mri.SpinArray,Nt,dt,rf,gr,device=torch.device('cpu'),details=False):
    '''Spin-domain simulation of spins. Treat all values as real numbers. 
    Also used for testing computing gradients. 
    Use the effect field of rf,gr from dt, single-step simulation.

    intermediate values:
        Beff_hist:(3*num*Nt), phi_hist:(num*Nt)
    output: 
        a_real:(1*num)
        a_imag:(1*num)
        b_real:(1*num)
        b_imag:(1*num)
    '''
    num = spinarray.num

    # Compute effective magnetic field
    # Beff_hist = spinarray_Beffhist(spinarray,Nt,dt,rf,gr,device=device)
    Beff_hist = spinarray.get_Beffective(Nt=Nt,rf=rf,gr=gr,device=device)

    # compute normalized B and phi, for all time points:
    # --------------------------------------------------
    Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
    Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
    phi_hist = -Beff_norm_hist*(spinarray.gamma.reshape(-1,1))*dt*2*torch.pi #(num*Nt)

    # compute the updates spin-domain para for each time step 
    aj_real_hist = torch.cos(phi_hist/2) # (num*Nt)
    aj_imag_hist = -Beff_unit_hist[2]*torch.sin(phi_hist/2)
    bj_real_hist = Beff_unit_hist[1]*torch.sin(phi_hist/2)
    bj_imag_hist = -Beff_unit_hist[0]*torch.sin(phi_hist/2)

    # initialize states for all spins
    a_real = torch.ones(num,device=device)
    a_imag = torch.zeros(num,device=device)
    b_real = torch.zeros(num,device=device)
    b_imag = torch.zeros(num,device=device)

    for t in range(Nt):
        aj_real = aj_real_hist[:,t]
        aj_imag = aj_imag_hist[:,t]
        bj_real = bj_real_hist[:,t]
        bj_imag = bj_imag_hist[:,t]

        # updating to tempory variables to avoid mistakes
        atmp_real = aj_real*a_real - aj_imag*a_imag - bj_real*b_real - bj_imag*b_imag
        atmp_imag = aj_real*a_imag + aj_imag*a_real - bj_real*b_imag + bj_imag*b_real
        btmp_real = bj_real*a_real - bj_imag*a_imag + aj_real*b_real + aj_imag*b_imag
        btmp_imag = bj_real*a_imag + bj_imag*a_real + aj_real*b_imag - aj_imag*b_real

        # update the spin-domain states
        a_real = atmp_real
        a_imag = atmp_imag
        b_real = btmp_real
        b_imag = btmp_imag
        
    # return the final value
    return a_real, a_imag, b_real, b_imag


# ---------------------------------------------------
# Simulation use effective field from rf,gr. With explicit Jacobian.
class _Spinorsim_SpinArray_SingleStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.any, spinarray,Nt,dt,aj_real_hist,aj_imag_hist,bj_real_hist,bj_imag_hist,device):
        num = spinarray.num

        # initialize states for all spins
        a_real = torch.ones(num,device=device)
        a_imag = torch.zeros(num,device=device)
        b_real = torch.zeros(num,device=device)
        b_imag = torch.zeros(num,device=device)

        a_real_hist = torch.zeros((num,Nt+1),device=device)
        a_imag_hist = torch.zeros((num,Nt+1),device=device)
        b_real_hist = torch.zeros((num,Nt+1),device=device)
        b_imag_hist = torch.zeros((num,Nt+1),device=device)
        a_real_hist[:,0] = a_real

        for t in range(Nt):
            aj_real = aj_real_hist[:,t]
            aj_imag = aj_imag_hist[:,t]
            bj_real = bj_real_hist[:,t]
            bj_imag = bj_imag_hist[:,t]

            # updating to tempory variables to avoid mistakes
            atmp_real = aj_real*a_real - aj_imag*a_imag - bj_real*b_real - bj_imag*b_imag
            atmp_imag = aj_imag*a_real + aj_real*a_imag + bj_imag*b_real - bj_real*b_imag
            btmp_real = bj_real*a_real - bj_imag*a_imag + aj_real*b_real + aj_imag*b_imag
            btmp_imag = bj_imag*a_real + bj_real*a_imag - aj_imag*b_real + aj_real*b_imag

            # update the spin-domain states
            a_real = atmp_real
            a_imag = atmp_imag
            b_real = btmp_real
            b_imag = btmp_imag
            a_real_hist[:,t+1] = a_real
            a_imag_hist[:,t+1] = a_imag
            b_real_hist[:,t+1] = b_real
            b_imag_hist[:,t+1] = b_imag
        
        # test
        # print(a_real_hist)
        # print(aj_real_hist)
        
        # saved variables for backward
        ctx.save_for_backward(a_real_hist,a_imag_hist,b_real_hist,b_imag_hist,
                        aj_real_hist,aj_imag_hist,bj_real_hist,bj_imag_hist)

        # return the final value
        return a_real, a_imag, b_real, b_imag
    @staticmethod
    def backward(ctx: torch.any, outputgrads1, outputgrads2, outputgrads3, outputgrads4):
        grad_spinarray = grad_Nt = grad_dt = None
        grad_aj_real_hist = grad_aj_imag_hist = grad_bj_real_hist = grad_bj_imag_hist = None
        grad_device = None
        needs_grad = ctx.needs_input_grad

        # get saved tensors
        a_real_hist,a_imag_hist,b_real_hist,b_imag_hist,aj_real_hist,aj_imag_hist,bj_real_hist,bj_imag_hist = ctx.saved_tensors
        Nt = aj_real_hist.shape[1]

        # test
        # print(a_real_hist)
        # print(aj_real_hist)

        grad_aj_real_hist = torch.zeros_like(aj_real_hist) # (num*Nt)
        grad_aj_imag_hist = torch.zeros_like(aj_imag_hist)
        grad_bj_real_hist = torch.zeros_like(bj_real_hist)
        grad_bj_imag_hist = torch.zeros_like(bj_imag_hist)

        pl_pareal = outputgrads1
        pl_paimag = outputgrads2
        pl_pbreal = outputgrads3
        pl_pbimag = outputgrads4

        for k in range(Nt):
            t = Nt - k - 1
            # print('backward at t={}'.format(t))

            # get previous state alpha and beta:
            ar = a_real_hist[:,t]
            ai = a_imag_hist[:,t]
            br = b_real_hist[:,t]
            bi = b_imag_hist[:,t]

            pl_pajreal = pl_pareal*ar + pl_paimag*ai + pl_pbreal*br + pl_pbimag*bi
            pl_pajimag = -pl_pareal*ai + pl_paimag*ar + pl_pbreal*bi - pl_pbimag*br
            pl_pbjreal = -pl_pareal*br - pl_paimag*bi + pl_pbreal*ar + pl_pbimag*ai
            pl_pbjimag = -pl_pareal*bi + pl_paimag*br - pl_pbreal*ai + pl_pbimag*ar

            grad_aj_real_hist[:,t] = pl_pajreal
            grad_aj_imag_hist[:,t] = pl_pajimag
            grad_bj_real_hist[:,t] = pl_pbjreal
            grad_bj_imag_hist[:,t] = pl_pbjimag

            # update for back to next iteration
            ajr = aj_real_hist[:,t]
            aji = aj_imag_hist[:,t]
            bjr = bj_real_hist[:,t]
            bji = bj_imag_hist[:,t]

            pl_pareal_prev = pl_pareal*ajr + pl_paimag*aji + pl_pbreal*bjr + pl_pbimag*bji
            pl_paimag_prev = -pl_pareal*aji + pl_paimag*ajr - pl_pbreal*bji + pl_pbimag*bjr
            pl_pbreal_prev = -pl_pareal*bjr + pl_paimag*bji + pl_pbreal*ajr - pl_pbimag*aji
            pl_pbimag_prev = -pl_pareal*bji - pl_paimag*bjr + pl_pbreal*aji + pl_pbimag*ajr

            pl_pareal = pl_pareal_prev
            pl_paimag = pl_paimag_prev
            pl_pbreal = pl_pbreal_prev
            pl_pbimag = pl_pbimag_prev

        # output the grads:
        return grad_spinarray,grad_Nt,grad_dt,grad_aj_real_hist,grad_aj_imag_hist,grad_bj_real_hist,grad_bj_imag_hist,grad_device
_spinorsim_spinarray_singlestep = _Spinorsim_SpinArray_SingleStep.apply
def _spinorsim_singlestep(spinarray:mri.SpinArray,Nt,dt,rf:torch.Tensor,gr:torch.Tensor,device=torch.device('cpu')):
    """Spin-domain simulation. With explicit Jacobian."""
    # num = spinarray.num

    # Compute effective magnetic field
    # -----------------------------------------------------
    # Beff_hist = spinarray_Beffhist(spinarray,Nt,dt,rf,gr,device=device)
    # 
    # offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num) Hz/(MHz/T) = mT
    # offBeff = offBeff.reshape(num,1)
    # # Effective magnetic field given by: 1)rf, 2)gradient,and 3)B1 transmission
    # Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
    # Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt) # if no B1 map
    # Beff_hist[:2,:,:] = Beff_hist[:2,:,:]*spinarray.kappa.reshape(1,num,1) # consider with the B1 transmit map
    # Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr*1e-2 + offBeff
    # 
    Beff_hist = spinarray.get_Beffective(Nt=Nt,rf=rf,gr=gr,device=device)


    # compute normalized B and phi, for all time points:
    # --------------------------------------------------
    Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
    Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
    phi_hist = -Beff_norm_hist*(spinarray.gamma.reshape(-1,1))*dt*2*torch.pi #(num*Nt)

    # compute the updates spin-domain para for each time step 
    aj_real_hist = torch.cos(phi_hist/2) # (num*Nt)
    aj_imag_hist = -Beff_unit_hist[2]*torch.sin(phi_hist/2)
    bj_real_hist = Beff_unit_hist[1]*torch.sin(phi_hist/2)
    bj_imag_hist = -Beff_unit_hist[0]*torch.sin(phi_hist/2)

    # apply the simulation function
    a_real,a_imag,b_real,b_imag = _spinorsim_spinarray_singlestep(spinarray,Nt,dt,
        aj_real_hist,aj_imag_hist,bj_real_hist,bj_imag_hist,device)
    return a_real, a_imag, b_real, b_imag
def spinorsim(spinarray:mri.SpinArray,Nt,dt,rf:torch.Tensor,gr:torch.Tensor,device=torch.device('cpu'))->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    '''Spin-domain simulation using real number operations. -> (alpha_re,alpha_im,beta_re,beta_im)

    with explicit derived Jacobian operation...
    
    Args:
        spinarray:  mri.SpinArray
        Nt:         number of time points
        dt:         (ms) 
        rf:         (2,Nt), mT
        gr:         (3,Nt), mT/m
        device:     torch.device

    Returns:
        alpha_re:
        alpha_im: 
        beta_re:
        beta_im: 
    '''
    return _spinorsim_singlestep(spinarray,Nt,dt,rf,gr,device=device)

