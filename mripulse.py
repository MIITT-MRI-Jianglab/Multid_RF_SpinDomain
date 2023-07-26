# some functions for building pulses
# author: jiayao

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal

import mri



device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
# print('>> mri: using device:',device)

# make a difference for my PC and lab computer
if torch.cuda.is_available():
    SAVE_FIG = True
else:
    SAVE_FIG = False


# some functions
def reverse(g):
    N = g.shape[1]
    y = torch.zeros_like(g)
    for n in range(N):
        y[:,N-n-1] = g[:,n]
    return y



# ----------------------------------------------------------------
# different functions design the shape of the gradient
# ----------------------------------------------------------------
def triangle(Nt):
    '''with maximum as 1'''
    x = torch.zeros(Nt,device=device)
    if (Nt%2) == 0:
        N = int(Nt/2)
        x[0:N] = torch.arange(N,device=device)
        x[N:Nt] = N - 1 - torch.arange(N)
    else:
        N = int((Nt-1)/2)
        x[0:N] = torch.arange(N,device=device)
        x[N:Nt-1] = N - 1 - torch.arange(N,device=device)
    x = x/(x.max())
    return x
def trapezoid(Nt,flat=0.5):
    '''with maximum value 1
    flat: mean the length where is flat
    '''
    x = torch.zeros(Nt,device=device)
    N1 = int(Nt*(1-flat)*0.5)
    N2 = int(Nt*flat)
    N3 = Nt - N1 - N2
    x[0:N1] = torch.arange(N1,device=device)/(N1-1)
    x[N1:N1+N2] = torch.ones(N2)
    x[N1+N2:Nt] = (N3 - 1 - torch.arange(N3,device=device))/(N3-1)
    return x
def sin_shape(Nt,periods=2,phase=0):
    '''1d tensor, sin shape
    phase: [0,1], when phase=1, the shape is cos
    '''
    T = Nt/periods
    omega = 2*torch.pi/T
    t = torch.arange(Nt,device=device)
    x = torch.sin(omega*t+phase*2*torch.pi)
    return x
# ----------------------------------------------------
def constant_g(Nt,dir='z'):
    '''return constant g in one direction with magnitude 1'''
    # Nt = int(T/dt)
    gr = torch.zeros((3,Nt),device=device)
    if dir == 'x':
        gr[0,:] = 1.0
    if dir == 'y':
        gr[1,:] = 1.0
    if dir == 'z':
        gr[2,:] = 1.0
    return gr
def pins(Nt,dt):
    gr = torch.zeros((3,Nt),device=device)
    return gr
def EPI(Nt,dt):
    periods = 3
    gr = torch.zeros((3,Nt),device=device)
    smallT = int(Nt/periods/4)
    T = smallT*4
    blipT = int(smallT*0.1)
    for i in range(periods):
        gr[0,i*T+smallT-blipT:i*T+smallT] = torch.ones(blipT,device=device)*10
        gr[1,i*T:i*T+smallT] = torch.arange(smallT,device=device)/(smallT-1)
        gr[1,i*T+smallT:i*T+2*smallT] = (smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
        gr[1,i*T+2*smallT:i*T+3*smallT] = -torch.arange(smallT,device=device)/(smallT-1)
        gr[1,i*T+3*smallT:i*T+4*smallT] = -(smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
    return gr
def squares_shape(Nt,periods=2): # TODO no rising 
    '''with maximum gz = 1.0'''
    smallT = int(Nt/periods/6)
    T = smallT*6
    if smallT*6*periods < Nt:
        gr = torch.zeros((3,Nt),device=device)
    else:
        gr = torch.zeros((3,smallT*6*periods),device=device)
    for i in range(periods):
        gr[2,i*T:i*T+smallT] = torch.arange(smallT,device=device)/(smallT-1)
        gr[2,i*T+smallT:i*T+2*smallT] = torch.ones(smallT,device=device)
        gr[2,i*T+2*smallT:i*T+3*smallT] = (smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
        gr[2,i*T+3*smallT:i*T+4*smallT] = -torch.arange(smallT,device=device)/(smallT-1)
        gr[2,i*T+4*smallT:i*T+5*smallT] = -torch.ones(smallT,device=device)
        gr[2,i*T+5*smallT:i*T+6*smallT] = -(smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
    if gr.shape[1] > Nt:
        gr = gr[:,:Nt]
    return gr
def squares(Nt,periods=2):
    '''with maximum gz = 1.0'''
    smallT = int(Nt/periods/6)
    T = smallT*6
    if smallT*6*periods < Nt:
        gr = torch.zeros((3,Nt),device=device)
    else:
        gr = torch.zeros((3,smallT*6*periods),device=device)
    for i in range(periods):
        gr[2,i*T:i*T+smallT] = torch.arange(smallT,device=device)/(smallT-1)
        gr[2,i*T+smallT:i*T+2*smallT] = torch.ones(smallT,device=device)
        gr[2,i*T+2*smallT:i*T+3*smallT] = (smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
        gr[2,i*T+3*smallT:i*T+4*smallT] = -torch.arange(smallT,device=device)/(smallT-1)
        gr[2,i*T+4*smallT:i*T+5*smallT] = -torch.ones(smallT,device=device)
        gr[2,i*T+5*smallT:i*T+6*smallT] = -(smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
    if gr.shape[1] > Nt:
        gr = gr[:,:Nt]
    return gr
def sin_shape_1d(Nt,dt=1.0,periods=2):
    '''Nt, dt:(ms)
    maximum value is 1.0 in z direction'''
    gr = torch.zeros((3,Nt),device=device)
    T = Nt/periods
    omega = 2*torch.pi/T
    t = torch.arange(Nt,device=device)
    gr[2,:] = torch.sin(omega*t)
    return gr
def sin_shape_2d(Nt,dt=1.0,periods=2):
    '''Nt, dt:(ms)
    maximum value is 1.0 in z direction'''
    gr = torch.zeros((3,Nt),device=device)
    T = Nt/periods
    omega = 2*torch.pi/T
    t = torch.arange(Nt,device=device)
    gr[0,:] = torch.cos(omega*t)
    gr[1,:] = torch.sin(omega*t)
    return gr
def spiral(Nt,cir_num=8,dt=1.0,dirction='in'):
    '''Nt:, dt: (ms),
    only the spiral shape, with maximum magnitude 1
    '''
    gr = torch.zeros((3,Nt),device=device)
    # n = 8
    # T = 1000 #ms
    # 
    n = cir_num
    T = Nt # ms
    t = torch.arange(Nt,device=device)
    A = 1.0
    phi = 2*torch.pi*n*t/T
    gamma = 1
    # gr[0,:] = -A/gamma/T*(2*torch.pi*n*(1-t/T)*torch.sin(phi) + torch.cos(phi))
    # gr[1,:] = A/gamma/T*(2*torch.pi*n*(1-t/T)*torch.cos(phi) - torch.sin(phi))
    if dirction == 'in':
        # case 1:
        # gr[0,:] = (2*torch.pi*n*(1-t/T)*torch.cos(phi) - torch.sin(phi))
        # gr[1,:] = -(2*torch.pi*n*(1-t/T)*torch.sin(phi) + torch.cos(phi))
        # case 2:
        gr[0,:] = (2*torch.pi*n*(1-t/T)*torch.sin(phi) + torch.cos(phi))
        gr[1,:] = (2*torch.pi*n*(1-t/T)*torch.cos(phi) - torch.sin(phi))
    elif dirction == 'out':
        gr[0,:] = -(2*torch.pi*n*(t/T)*torch.sin(phi) + torch.cos(phi))
        gr[1,:] = (2*torch.pi*n*(t/T)*torch.cos(phi) - torch.sin(phi))
    else:
        pass
    gr = gr/n/2/torch.pi
    # print(gr[:,0])
    return gr
def spiral_3d(Nt,dt=1.0):
    '''Nt:, dt: (ms)'''
    # TODO
    gr = torch.zeros((3,Nt),device=device)
    n = 8
    T = 1000 #ms
    t = torch.arange(Nt,device=device)
    A = 1.0
    phi = 2*torch.pi*n*t/T
    gamma = 1
    gr[0,:] = -A/gamma/T*(2*torch.pi*n*(1-t/T)*torch.sin(phi) + torch.cos(phi))
    gr[1,:] = A/gamma/T*(2*torch.pi*n*(1-t/T)*torch.cos(phi) - torch.sin(phi))
    return gr
def EchoVolumar(Nt,dt):
    kxloc = [0,-1,-1,0,1,1,1,0,-1,-2,-2,-2,-1,0,1,2,2,2,1,0,-1]
    kyloc = [0,0,-1,-1,-1,0,1,1,1,1,0,-1,-2,-2,-2,-1,0,1,2,2,2]
    # nloc = len(kxloc)
    nloc = 12
    gr = torch.zeros((3,Nt),device=device)
    
    if True:
        smallT = int(Nt/nloc/4)
        T = smallT*4
        blipT = int(smallT*0.1)

        prevkx = 0
        prevky = 0
        for i in range(nloc):
            # gr[0,i*T+2*smallT-blipT:i*T+2*smallT] = torch.ones(blipT,device=device)*5
            # gr[1,i*T+2*smallT-blipT:i*T+2*smallT] = torch.ones(blipT,device=device)*5
            gr[0,i*T+2*smallT-blipT:i*T+2*smallT] = triangle(blipT)*(kxloc[i] - prevkx)
            gr[1,i*T+2*smallT-blipT:i*T+2*smallT] = triangle(blipT)*(kyloc[i] - prevky)
            prevkx = kxloc[i]
            prevky = kyloc[i]
            # fast z-trajectory:
            gr[2,i*T:i*T+smallT] = torch.arange(smallT,device=device)/(smallT-1)
            gr[2,i*T+smallT:i*T+2*smallT] = (smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
            gr[2,i*T+2*smallT:i*T+3*smallT] = -torch.arange(smallT,device=device)/(smallT-1)
            gr[2,i*T+3*smallT:i*T+4*smallT] = -(smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
    if False:
        smallT = int(Nt/nloc/4)
        T = smallT*4
        blipT = int(smallT*0.1)
        
        prevkx = 0
        prevky = 0
        for i in range(nloc):
            # gr[0,i*T+2*smallT-blipT:i*T+2*smallT] = torch.ones(blipT,device=device)*5
            # gr[1,i*T+2*smallT-blipT:i*T+2*smallT] = torch.ones(blipT,device=device)*5
            gr[0,i*T+2*smallT-blipT:i*T+1*smallT] = triangle(blipT)*(kxloc[i] - prevkx)
            gr[1,i*T+2*smallT-blipT:i*T+1*smallT] = triangle(blipT)*(kyloc[i] - prevky)
            prevkx = kxloc[i]
            prevky = kyloc[i]
        for i in range(int(Nt/T)):
            # fast z-trajectory:
            gr[2,i*T:i*T+smallT] = torch.arange(smallT,device=device)/(smallT-1)
            gr[2,i*T+smallT:i*T+2*smallT] = (smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
            gr[2,i*T+2*smallT:i*T+3*smallT] = -torch.arange(smallT,device=device)/(smallT-1)
            gr[2,i*T+3*smallT:i*T+4*smallT] = -(smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
    # gr[0:2] = gr[0:2]*10
    return gr
def EchoVolumar2(Nt,dt=1.0):
    kxmove = [-1,-1,-1,-1,0,0,0,1,1,1,0,0,-1,-1,0,1]
    kymove = [0,0,0,0,1,1,1,0,0,0,-1,-1,0,0,1,0]
    nloc = len(kxmove)
    # nloc = 12
    gr = torch.zeros((3,Nt),device=device)
    
    if True:
        periods = nloc+1
        smallT = int(Nt/periods/2)
        T = smallT*2
        blipT = int(smallT*0.1)

        prevkx = 0
        prevky = 0
        sign = 1
        for i in range(nloc):
            # gr[0,i*T+2*smallT-blipT:i*T+2*smallT] = torch.ones(blipT,device=device)*5
            # gr[1,i*T+2*smallT-blipT:i*T+2*smallT] = torch.ones(blipT,device=device)*5
            gr[0,i*T+2*smallT-blipT:i*T+2*smallT] = triangle(blipT)*kxmove[i]
            gr[1,i*T+2*smallT-blipT:i*T+2*smallT] = triangle(blipT)*kymove[i]
            # fast z-trajectory:
            gr[2,i*T:i*T+2*smallT] = triangle(2*smallT)*sign
            sign = - sign
            # gr[2,i*T:i*T+smallT] = torch.arange(smallT,device=device)/(smallT-1)
            # gr[2,i*T+smallT:i*T+2*smallT] = (smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
            # gr[2,i*T+2*smallT:i*T+3*smallT] = -torch.arange(smallT,device=device)/(smallT-1)
            # gr[2,i*T+3*smallT:i*T+4*smallT] = -(smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
        gr[2,(periods-1)*T:(periods-1)*T+2*smallT] = triangle(2*smallT)*sign
    return gr
def EchoVolumar3(Nt,dt):
    kxmove = [-1,-1,-1,-1,0,0,0,1,1,1,0,0,-1,-1,0,1]
    kymove = [0,0,0,0,1,1,1,0,0,0,-1,-1,0,0,1,0]
    nloc = len(kxmove)
    # nloc = 12
    gr = torch.zeros((3,Nt),device=device)
    
    if True:
        periods = nloc+1
        smallT = int(Nt/periods/2)
        T = smallT*2
        blipT = int(smallT*0.1)

        flat_factor = 0.8
        sign = 1
        for i in range(nloc):
            # gr[0,i*T+2*smallT-blipT:i*T+2*smallT] = torch.ones(blipT,device=device)*5
            # gr[1,i*T+2*smallT-blipT:i*T+2*smallT] = torch.ones(blipT,device=device)*5
            gr[0,i*T+2*smallT-blipT:i*T+2*smallT] = triangle(blipT)*kxmove[i]
            gr[1,i*T+2*smallT-blipT:i*T+2*smallT] = triangle(blipT)*kymove[i]
            # fast z-trajectory:
            gr[2,i*T:i*T+2*smallT] = trapezoid(2*smallT,flat=flat_factor)*sign
            sign = - sign
            # gr[2,i*T:i*T+smallT] = torch.arange(smallT,device=device)/(smallT-1)
            # gr[2,i*T+smallT:i*T+2*smallT] = (smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
            # gr[2,i*T+2*smallT:i*T+3*smallT] = -torch.arange(smallT,device=device)/(smallT-1)
            # gr[2,i*T+3*smallT:i*T+4*smallT] = -(smallT - 1 - torch.arange(smallT,device=device))/(smallT-1)
        gr[2,(periods-1)*T:(periods-1)*T+2*smallT] = trapezoid(2*smallT,flat=flat_factor)*sign
    return gr


# --------------------------------------------------------
# functions for the rf pulse, small-flip-angle
# ---------------------------------------------------------
def get_kspace(Nt,dt,gr,gamma=42.48,case='excitation'):
    '''
    gamma:(MHz/T), gr:(3*Nt)(mT/m), dt:(ms)

    output: kspace (1/cm)
    '''
    if case == 'excitation':
        kt = torch.cumsum(gr,dim=1) #(3*Nt)
        # kt = torch.cat((torch.zeros((3,1),device=device),kt),dim=1) #(3*(Nt+1))
        kt = kt - kt[:,-1].reshape(3,1) #make final be 0
        # MHz/T * mT/m * ms = 1/m = 1/(100cm)
        kt = -(gamma*dt*kt)*100 # 1/cm
        # kt = kt/(torch.pi*2)
    elif case == 'imaging':
        kt = torch.cumsum(gr,dim=1)
        kt = torch.cat((torch.zeros((3,1),device=device),kt),dim=1) #(3*(Nt+1))
        kt = gamma*kt*dt/100 # 1/cm
        # kt = kt/(2*torch.pi)
    # print(kt.max())
    return kt

def kspaceweighting_to_rf(kspaceweighting,Nt,dt,gr,gamma=42.48):
    '''
    kspaceweighting:(Nt), gamma:(MHz/T)
    '''
    rf = torch.zeros((2,Nt),device=device)
    gr_norm = gr.norm(dim=0)
    rf[0,:] = kspaceweighting*gamma*gr_norm
    return rf

def get_slewrate(Nt,dt,gr):
    '''
    compute the slew rate of a given gradient waveform
    '''
    # TODO
    return



# -------------------------------------------------------
# Some pulse design methods
# -------------------------------------------------------



# SLR transform
# --------------------------------------
def slr_transform(Nt,dt,rf,gamma=42.48):
	"""assume g is constant
    gammma:(MHz/)
    A:[a0,a1,...,an-1], a0~z^0, a1~z^{-1}
    B:[b0,b1,...,bn-1]
    """
	#
	rf = rf[0,:] + (0.0+1.0j)*rf[1,:] #mT
	rf_norm_hist = rf.abs() #mT
	phi_hist = 2*torch.pi*gamma*rf_norm_hist*dt #2pi* MHz/T * mT * ms  #rotation given by rf
	A = (1.0+0.0j)*torch.zeros(Nt,device=device)
	B = (1.0+0.0j)*torch.zeros(Nt,device=device)
	A[0] = 1.0 # at time 0, only has 0-order coefficient
	B[0] = 0.0 # at time 0, only has 0-order coefficient
	Cj = torch.cos(phi_hist/2)
	Sj = -(0.0+1.0j)*(torch.exp((0.0+1.0j)*rf.angle()))*torch.sin(phi_hist/2)
	for t in range(Nt):
        # A+ = C*A - S.conj()*z^{-1}*B
		A_tmp1 = Cj[t]*A
		A_tmp2 = -Sj[t].conj()*B # term z^{-1}
        # B+ = S*A + C*z^{-1}*B
		B_tmp1 = Sj[t]*A
		B_tmp2 = Cj*B # term z^{-1}
        # 
		B = B_tmp1
		B[1:] = B[1:] + B_tmp2[:-1]
		A = A_tmp1
		A[1:] = A[1:] + A_tmp2[:-1]
	return A,B

def mag2mp(x): # copied from sigpy, but not using!
    n = np.size(x)
    xl = np.log(np.abs(x))  # Log of mag spectrum
    # TODO: a fft transform using other functions
    # xlf = sp.fft(xl, center=False, norm=None)
    xlf = 0.

    xlfp = xlf
    xlfp[0] = xlf[0]  # Keep DC the same
    xlfp[1:(n // 2):1] = 2 * xlf[1:(n // 2):1]  # Double positive frequencies
    xlfp[n // 2] = xlf[n // 2]  # keep half Nyquist the same
    xlfp[n // 2 + 1:n:1] = 0  # zero negative frequencies

    # TODO: a ifft
    # xlaf = sp.ifft(xlfp, center=False, norm=None)
    xlaf = 0.

    a = np.exp(xlaf)  # complex exponentiation

    return a


# SLR inverse transform
# ----------------------------------------
def slr_transform_inverse(A,B,dt,gamma=42.48):
    '''
    A,B:(Nt)(complex numbers)
    gamma: (MHz/T)
    dt: (ms)
    # TODO: almost done, but the inverse transform result is reversed the sign
    '''
    Nt = len(B)
    theta_hist = torch.zeros(Nt,device=device)
    rf_mag_hist = torch.zeros(Nt,device=device)
    rf_c = (1.+0.j)*torch.zeros(Nt,device=device)
    for t in range(Nt):
        # rotation given by rf:
        phi = 2*torch.arctan((B[0]/A[0]).abs()) 
        
        # compute rf phase:
        theta = (-(0.+1.0j)*(B[0]/A[0])).angle() # phase of rf
        # theta_hist[Nt-t-1] = theta

        # compute rf:
        rf_mag = 1/(2*torch.pi*gamma)/dt*phi*torch.exp((0.+1.j)*theta)
        rf_c[Nt-t-1] = rf_mag

        # compute new A,B parameters:
        Cj = 1/torch.sqrt((B[0]/A[0])**2+1)
        Sj = Cj*B[0]/A[0]
        # A- = C*A + S.conj()*B
        A_tmp = Cj*A + Sj.conj()*B
        # B- = -S*z*A + C*z*B
        B_tmp = -Sj*A + Cj*B

        A = A_tmp
        B = torch.zeros_like(B)
        B[0:-1] = B_tmp[1:]
    # compute the rf
    rf = torch.zeros((2,Nt),device=device)
    rf[0,:] = rf_c.real
    rf[1,:] = rf_c.imag
    return rf

def SLR_compute_A_from_B(B):
    '''compute A-parameter from the B-parameter
    '''
    # to make sure magnitude not larger than 1:
    Bmax = B.abs().max()
    if Bmax >= 1:
        B = Bmax/(1e-7 + Bmax)
    Amag = torch.sqrt(1-B.abs()**2)
    logAmag = torch.log(Amag)
    # do a Hilbert transform:
    logAmagfft = torch.fft(logAmag)

    Aphase = logAmagfft # here is actually wrong, to correct it!

    A = Amag*torch.exp((1j)*Aphase)
    return A

# SLR pulse design
# ----------------------------------------
def SLR_pulsedesign(g,W,m,dt,Nt,d1,d2):
    '''
    the design method of slr pulse, but move than 90 or 180-pulse design
    g: the strength of gradient field (mT/m)
    W: the target region width (cm) (center located at 0cm)
    m: the magnitude of the desired B, [0,1]
    dt: (ms)
    Nt: number of time points of the pulse
    d1,d2: the passband,stopband ripple

    output: rf(2*Nt)
    '''
    if (Nt%2) == 0:
        N = N+1
    else:
        N = Nt

    # filter design for parameter B
    bands = [0,2,3,5]
    desired = [1,1,0,0]
    weight = [1,d1/d2]
    fir_firls = signal.firls(N, bands=bands, desired=desired, weight=weight)
    
    freq, response = signal.freqz(fir_firls)
    print(freq.shape, response.shape, response.real.dtype)

    fig, axs = plt.subplots()
    plt.plot(freq,np.abs(response))
    if SAVE_FIG:
        picname = 'pictures/tmp_pic.png'
        print('save fig...'+picname)
        plt.savefig(picname)
    else:
        plt.show()
    
    B = 0

    # compute A:
    # A = SLR_compute_A_from_B(B)

    # compute rf pulse:
    # rf = slr_transform_inverse(A,B,dt)

    # A question to consider: what is the minimum pulse length?

    # return rf,Nt,dt
    return 0


def test_slr_transform():
    spin = mri.Spin()
    spin.show_info()
    # 
    pulse = mri.example_pulse()
    pulse.show_info()
    Nt,dt,rf,gr = pulse.Nt,pulse.dt,pulse.rf,pulse.gr
    mri.plot_pulse(rf,gr,dt,save_fig=SAVE_FIG)

    # slr transform:
    A,B = slr_transform(Nt,dt,rf,gamma=spin.gamma)
    # print(A)
    # print(B)
    # Fourier transform of B
    N = len(B)
    freq = torch.arange(0,N)
    print(len(freq),N)

    if False: # plot
        plt.figure()
        # plt.plot(A.abs().tolist())
        plt.plot(B.abs().tolist(),ls='--')
        if SAVE_FIG:
            picname = 'pictures/mri_tmp_pic.png'
            print('save fig...'+picname)
            plt.savefig(picname)
        else:
            plt.show()

    rf = slr_transform_inverse(A,B,dt)

    if True:
        plt.figure()
        plt.plot(rf[0,:].tolist(),label='real')
        plt.plot(rf[1,:].tolist(),label='imag')
        if SAVE_FIG:
            picname = 'pictures/mri_tmp_pic.png'
            print('save fig...'+picname)
            plt.savefig(picname)
        else:
            plt.show()

    pass


# --------------------------------------------------------
# plot functions
# ---------------------------------------------------------
def plot_phase(x,picname='pictures/mri_tmp_pic_phase.png',save_fig=False):
    '''x:(2*N)'''
    x = x[0,:] + (0.+1j)*x[1,:]
    phase = x.angle()
    phase = np.array(phase.tolist())
    phase = 180*phase/np.pi
    plt.figure()
    plt.plot(phase,label='angle')
    plt.legend()
    plt.ylabel('angle')
    if save_fig:
        print('save fig...'+picname)
        plt.savefig(picname)
    else:
        plt.show()
    return
def plot_G(gr,dt,picname='pictures/mri_tmp_pic_G.png',save_fig=False):
    '''gr:(3*Nt)(mT/cm), dt:(ms)'''
    gr = np.array(gr.tolist())
    Nt = gr.shape[1]
    time = np.arange(Nt)*dt
    fig = plt.figure()
    plt.plot(time,gr[0,:],label='gr,x')
    plt.plot(time,gr[1,:],label='gr,y')
    plt.plot(time,gr[2,:],label='gr,z')
    plt.ylabel('mT/cm')
    plt.xlabel('time(ms)')
    plt.legend()
    if save_fig:
        print('save fig...'+picname)
        plt.savefig(picname)
    else:
        plt.show()
    return
def plot_kspace(gr,Nt,dt=1.0,gamma=42.48,case='excitation',picname='pictures/mri_tmp_pic_kspace.png',save_fig=False):
    '''gr:(3*Nt)(mT/cm), dt:(1)(ms)'''
    # if case == 'excitation':
    #     kt = torch.cumsum(gr,dim=1)
    #     kt = np.array(kt.tolist())
    #     kt = np.concatenate((np.array([[0],[0],[0]]),kt),axis=1)
    #     kt = kt - kt[:,-1].reshape(3,1)
    # elif case == 'imaging':
    #     kt = torch.cumsum(gr,dim=1)
    #     kt = np.array(kt.tolist())
    #     kt = np.concatenate((np.array([[0],[0],[0]]),kt),axis=1)

    kt = get_kspace(Nt,dt,gr,gamma=gamma,case=case)
    kt = np.array(kt.tolist())
    
    # get maximum range of the plot:
    maxk = np.max(np.absolute(kt),axis=1)
    maxkx,maxky,maxkz = max(maxk),max(maxk),max(maxk)
    # print(maxkx)
    # print(maxky)
    # print(max(np.abs(kt.reshape(-1))))

    # plot:
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(kt[0,:],kt[1,:],kt[2,:],label=r'k-trajectory $(cm^{-1})$')
    # ax.plot((0,M[0,0]),(0,M[1,0]),(0,M[2,0]),linewidth=1,linestyle='--')
    # ax.plot((0,M[0,-1]),(0,M[1,-1]),(0,M[2,-1]),linewidth=1,linestyle='--')
    # ax.text(M[0,0],M[1,0],M[2,0],r'$k_0$',fontsize=8)
    # ax.text(M[0,-1],M[1,-1],M[2,-1],r'end',fontsize=8)
    ax.plot(kt[0,0],kt[1,0],kt[2,0],marker='o',color='green')
    ax.plot(kt[0,-1],kt[1,-1],kt[2,-1],marker='o',color='red')
    ax.text(kt[0,0],kt[1,0],kt[2,0],r'start',fontsize=8)
    ax.text(kt[0,-1],kt[1,-1],kt[2,-1],r'end',fontsize=8)
    ax.legend()
    ax.set_xlim(-1.1*maxkx,1.1*maxkx)
    ax.set_ylim(-1.1*maxky,1.1*maxky)
    ax.set_zlim(-1.1*maxkz,1.1*maxkz)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('kz')
    if save_fig:
        print('save fig... | '+picname)
        plt.savefig(picname)
    else:
        plt.show()
    return

def test():
    x = torch.rand(5)
    print(x)
    print(x[1:])
    print(x[:-1])

if __name__ == '__main__':
    print('mripul.py\n')
    if True:
        # gr = spiral(1000)
        # gr = sin_shape_1d(1000,periods=2)
        # gr = sin_shape_2d(1000,periods=2)
        # gr = EPI(1000,1.0)
        # gr = EchoVolumar(10000,1.0)
        gr = EchoVolumar3(10000,1.0)
        gr[2,:] = gr[2,:]*0.1
        # gr[1,:300] = sin_shape(300,phase=0.25)
        # gr = torch.randn_like(gr)
        plot_G(gr,1,save_fig=SAVE_FIG)
        plot_kspace(gr,1000,case='excitation',save_fig=SAVE_FIG)
    if False:
        test_slr_transform()
    if False:
        print(int(0.1))
        print(int(0.7))
    if False: # test slr pulse design
        rf = SLR_pulsedesign(g=5,W=1,m=0.5,dt=0.1,Nt=101,d1=0.01,d2=0.01)
