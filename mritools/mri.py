'''Basic functions and definitions, including core simulation functions.
author: jiayao

## Acknowledgements:
this work has inspired and take reference of: 
- https://github.com/tianrluo/AutoDiffPulses
- and others works

## Notes
                  H^1: gamma = 4.24kHz/Gauss = 42.48 MHz/T
          1 Gauss = 0.1 mT,     1 T = 10,000 Gauss,     1 mT = 10 Gauss

change of units
        1 Gauss/cm = 10 mT/m
        1 mT/m/ms = 100 Gauss/cm/s

what is in this file:
1. basic objects in mri such as spins, spin arrays,
2. simulation functions
3. operations of magnetization
'''
import os
import math
import numpy as np
import torch
# from time import time
import warnings

# packages for plot 
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.io as spio
from scipy.interpolate import interp1d
from scipy import interpolate

# some testing modules
# import mritools.Siemens.hardwareinfo as SiemensInfo
# import mritools.Siemens.SAFEpnsPred as siemensSAFEpnsPred
# import mritools.Siemens.SAFEpnsPredVec as siemensSAFEpnsPredVec
# from mritools.Siemens import pypulseqSAFEpnsPred
# import mritools.callMatlab.safePNSpredict as matlabSAFEpnsPred

# my other files
# import CONFIG
# Gamma = CONFIG.GAMMA
Gamma = 42.48 # MHz/T (gyromagnetic ratio) (normalized by 2*pi)



################################################################################
# Functions for index operation:
# index are all torch.int64
def index_intersect(idx1,idx2):
    t1 = set(idx1.unique().cpu().numpy())
    t2 = set(idx2.unique().cpu().numpy())
    idx = t1.intersection(t2)
    idx = list(idx)
    idx = torch.tensor(idx,device=idx1.device,dtype=idx1.dtype)
    return idx
def index_union(idx1,idx2):
    '''idx1,idx2: are 1d tensors, int type'''
    idx = torch.unique(torch.cat((idx1,idx2)))
    return idx
def index_subtract(idx1,idx2):
    '''idx1 - idx2, (tensor)'''
    t1 = set(idx1.unique().cpu().numpy())
    t2 = set(idx2.unique().cpu().numpy())
    idx = t1-t2
    idx = list(idx)
    idx = torch.tensor(idx,device=idx1.device,dtype=idx1.dtype)
    return idx



###############################################################################

class Pulse:
    def __init__(self,dt=1.0,rf=None,gr=None,device=torch.device("cpu"),name='pulse'):
        '''Initialize of an instance of pulse.

        Args:
            dt:        (ms)
            rf: (2,Nt) (mT)     the real and imaginary part (torch.tensor)
            gr: (3,Nt) (mT/m)   three x-y-z channels (torch.tensor)
            device:             cpu or cuda
            name:               name for identification

        .. notes::

        other attributes (if needed):
            sys_rfmax:        (mT)
            sys_gmax:         (mT/m)
            sys_slewratemax:  (mT/m/ms)
        '''
        self.device = device
        self.dtype = torch.float32      # may be useful
        self.name = name
        # (some checks)
        if not isinstance(self.name,str): self.name=''

        # if (rf==None) & (gr==None):
        #     print('Error: rf = gr = None!')
        # else:

        if rf==None:
            Nt = gr.shape[1]
            rf = torch.zeros(2,Nt)
        elif gr==None:
            Nt = rf.shape[1]
            gr = torch.zeros(3,Nt)
        else:
            assert rf.shape[1] == gr.shape[1]
            Nt = rf.shape[1]
        self.rf = rf.to(self.dtype).to(self.device)
        self.gr = gr.to(self.dtype).to(self.device)
        self.dt = dt
        self.Nt = Nt

        self._rf_unit = 'mT'
        self._gr_unit = 'mT/m'
        self._dt_unit = 'ms'
        
        self.sys_rfmax = float('Inf')         # system peak rf
        self.sys_gmax = float('Inf')          # system max gradient
        self.sys_slewratemax = float('Inf')   # system max slew-rate
        self.pnsTH_fn = lambda t: None

    # -------------------------------------------------------------
    def set_hardwarelimits(self,rfmax,gmax,slewrate):
        '''Set system hardware limits'''
        self.sys_rfmax = rfmax
        self.sys_gmax = gmax
        self.sys_slewratemax = slewrate
        return
    def __set_PNS_threshold(self,f): # TODO
        '''Set the pns thershold function (specificly, limit of slew-rate)
        
        Args:
            f: a function of time
        '''
        self.pnsTH_fn = f
        return
    # (staticmethods that are most related to a pulse)
    @staticmethod
    def calculate_slewrate(gr:torch.Tensor,dt):
        '''Return the slew-rate (mT/m/ms)(3,Nt).
        
        Args:
            gr: (3,Nt)(Tensor)(mT/m)
            dt: (ms)

        Returns:
            srate: (3,Nt)(mT/m/ms)
        '''
        srate = torch.diff(gr,dim=1)/dt # mT/m/ms
        # add 0 for the beginning (to have the same length as gr)
        srate = torch.cat((torch.zeros((3,1),device=srate.device),srate),dim=1)
        return srate
    @staticmethod
    def gradient_to_kspace(gr:torch.Tensor,dt,use='excitation',gamma=Gamma,zero_included=False):
        '''Return the k-space (1/m).

        get k-space trajectory of the gradient, 
        excitation: $$k(t) = - \gamma \int_t^T G(s) ds$$
        imaging: $$k(t) = \gamma \int_0^t G(s) ds$$

        if it is excitation k-space trajectory, then assume there is one additional
        point (0,0,0) (but not in the returned result)
        
        Args:
            gr: (3,Nt)(mT/m) gradient (tensor)
            dt: (ms) time resolution 
            use: 'excitation' or 'imaging'
            zero_included: True: include last/1st point, an additional zero in results
        
        Returns:
            ksp: (3,Nt)         (1/m)=(cycle/m)

        ------
        example:
            g1 = [[1,1,1,1],[...],[...]]
            ksp = [[-4,-3,-2,-1],[...],[...]]
        '''
        device = gr.device
        if use == 'excitation':
            # for the excitation kspace, the last k-point is zero
            # last point can be included or ignored
            # ----------------------------------------
            ksp = torch.cumsum(gr,dim=1)  #(3*Nt) (mT/m*ms)
            ksp = ksp - ksp[:,-1].reshape(3,1)
            ksp = ksp - gr
            ksp = ksp*gamma*dt  #unit: (MHz/T * mT/m * ms)=(1/m) or (cycle/m)
            # ksp = 2*torch.pi*ksp  #unit: (rad/m)
            if zero_included:
                ksp = torch.cat((ksp,torch.zeros((3,1),device=device)),dim=1)
        elif use == 'imaging':
            # for the imaging kspace, the first k-point is zero
            # the first point can be included or ignored
            # ----------------------------------------------
            ksp = torch.cumsum(gr,dim=1)  #(3*Nt) (mT/m*ms)
            ksp = ksp*gamma*dt # (1/m) or (cycle/m)
            if zero_included:
                ksp = torch.cat((torch.zeros((3,1),device=device),ksp),dim=1)
        else:
            return None
        return ksp
    @staticmethod
    def kspace_to_gradient(ktraj:torch.Tensor,dt,use:str='excitation',gamma=Gamma,zero_included=False):
        '''Return gradient (mT/m) from given k-space trajectory.

        if is excitation k-space trajectory, then the assume there 
        is an additional point which is (0,0,0)

        Args:
            ktraj: (3,Nt)   (1/m)=(cycle/m)     k-space (tensor)
            dt:             (ms)                time resolution
            use:                               'excitation' or 'imaging'
            zero_included:                      include 1st/last point
        
        Returns:
            gr: (3,Nt)      (mT/m)              gradients
        '''
        if use == 'excitation':
            # add 0 to the end, ex-kspace defaultly assume returns to zero
            if not zero_included:
                ktraj = torch.cat((ktraj,torch.zeros((3,1),device=ktraj.device)),dim=1)
            ktraj = ktraj/(gamma*dt) #unit: mT/m
            gr = torch.diff(ktraj,dim=1)
        elif use == 'imaging':
            if not zero_included:
                ktraj = torch.cat((torch.zeros((3,1),device=ktraj.device),ktraj),dim=1)
            ktraj = ktraj/(gamma*dt)  # (mT/m)
            gr = torch.diff(ktraj,dim=1)
        else:
            return None
        return gr
    # ---------------------------------------------------------------
    # (Some properties)
    # (they are calculated based on the current attributes)
    # ------------------------------------------------------------
    @property
    def duration(self):
        '''total duration (ms)'''
        return self.dt*self.Nt
    @property
    def times(self)->torch.Tensor:
        '''get time array (1,Nt)(ms)'''
        t = torch.arange(self.Nt)*self.dt
        t = t.to(self.device)
        return t
    
    # (relate to RF waveform)
    @property
    def rf_complex(self):
        '''complex valued rf waveform (mT)'''
        return self.rf[0,:] + self.rf[1,:]*1j
    @property
    def rf_amplitude(self):
        '''array (mT)'''
        return torch.linalg.norm(self.rf,dim=0)
    @property
    def rf_phase(self):
        '''phase of rf waveform (-pi to pi)'''
        return torch.angle(self.rf_complex)
    @property
    def rf_peak(self):
        '''peak rf amplitude (mT)'''
        return torch.max(torch.linalg.norm(self.rf,dim=0)).item()
    @property
    def rf_energy(self):
        '''total energy of rf (mT*mT*ms)'''
        return torch.sum(self.rf**2)*self.dt

    # (relate to Gradient waveform)
    @property
    def gr_peak(self):
        '''peak gradient magnitude of three channel''' 
        return torch.max(torch.abs(self.gr)).item()
    @property
    def gr_peak_rotinvar(self):
        '''rotation invariant peak value (mT/m)'''
        return torch.max(torch.linalg.norm(self.gr,dim=0)).item()
    
    # ---------------------------------------------------------------
    # Methods for computing some properties
    # (methods)
    # ------------------------------------------------------------
    def get_slewrate(self)->torch.Tensor:
        '''Return the slew-rate (mT/m/ms), zero is added at the beginning'''
        # srate = torch.diff(self.gr,dim=1)/self.dt # mT/m/ms
        # # add 0 for the beginning
        # srate = torch.cat((torch.zeros((3,1),device=srate.device),srate),dim=1)
        return self.calculate_slewrate(self.gr,self.dt)
    def get_kspace(self,case='imaging',gamma=Gamma,zero_included=True):
        '''get excitation kspace (1/m) or (cycle/m)

        for imaging:
                $$k(t) \propto \gamma \int G(t) dt$$

        input:
            gamma:(MHz/T)
        var used:
            self.gr:(3*Nt)(mT/m), 
            self.dt:(ms)
        output: 
            kspace: (1/m) or (cycle/m)
        '''
        if case == 'excitation':
            # for the excitation kspace, the last k-point is zero
            # last point can be included or ignored
            # ----------------------------------------
            ksp = torch.cumsum(self.gr,dim=1)  #(3*Nt) (mT/m*ms)
            ksp = ksp - ksp[:,-1].reshape(3,1)
            ksp = ksp - self.gr
            ksp = ksp*gamma*self.dt  #unit: (MHz/T * mT/m * ms)=(1/m) or (cycle/m)
            # ksp = 2*torch.pi*ksp  #unit: (rad/m)
            if zero_included:
                ksp = torch.cat((ksp,torch.zeros((3,1),device=self.device)),dim=1)
        elif case == 'imaging':
            # for the imaging kspace, the first k-point is zero
            # the first point can be included or ignored
            # ----------------------------------------------
            ksp = torch.cumsum(self.gr,dim=1)  #(3*Nt) (mT/m*ms)
            ksp = ksp*gamma*self.dt # (1/m) or (cycle/m)
            if zero_included:
                ksp = torch.cat((torch.zeros((3,1),device=self.device),ksp),dim=1)
        else:
            return None
        return ksp
    def get_gradient_acoustic_resonance(self,nf=500) -> torch.Tensor:
        '''return spectrum of gradient resonance frequency'''
        freq, spect = self.calculate_gradient_spectrum(self.gr,self.dt,nf=nf)
        return freq, spect
    # ---------------------------------------------------------------
    # Methods make changes/operations to the pulse 
    # ---------------------------------------------------------------
    def cut_rf_amplitude_to_system(self):
        '''Cut the rf amplitude to be within system limits.'''
        rf_amp = self.rf_amplitude
        rf_amp = torch.clamp(rf_amp,min=0,max=self.sys_rfmax)
        rf_real = rf_amp*torch.cos(self.rf_phase)
        rf_imag = rf_amp*torch.sin(self.rf_phase)
        self.rf[0] = rf_real
        self.rf[1] = rf_imag
        return
    def cut_gr_amplitude_to_system(self):
        '''Cut the gradient values to be within system limits.'''
        self.gr = torch.clamp(self.gr,min=-self.sys_gmax,max=self.sys_gmax)
        return
    def add_to_rf_phase(self,phase):
        '''add to rf phase (deg)(0-180)'''
        phi = torch.tensor(phase).to(self.device)
        rf = self.rf_complex*torch.exp(1j*torch.pi*phi/180)
        self.rf[0] = rf.real
        self.rf[1] = rf.imag
        return
    def add_to_xy_gradient_phase(self,phase):
        '''rotate the xy gradients (as a vector), phase (deg)(0-180)'''
        phi = torch.tensor(phase).to(self.device)
        gxy = self.gr[0]+1j*self.gr[1]
        gxy = gxy*torch.exp(1j*torch.pi*phi/180)
        self.gr[0] = gxy.real
        self.gr[1] = gxy.imag
        return
    def __add_to_yz_gradient_phase(self,phase):
        '''rotate the xy gradients (as a vector), phase (deg)(0-180)'''
        phi = torch.tensor(phase).to(self.device)
        gyz = self.gr[1]+1j*self.gr[2]
        gyz = gyz*torch.exp(1j*torch.pi*phi/180)
        self.gr[1] = gyz.real
        self.gr[2] = gyz.imag
        return
    def rotate_gradient_x_to_y(self,phase):
        '''rotate the gradients from x to y (as a vector), phase (deg)(0-360)'''
        phi = torch.tensor(phase).to(self.device)
        gxy = self.gr[0]+1j*self.gr[1]
        gxy = gxy*torch.exp(1j*torch.pi*phi/180)
        self.gr[0] = gxy.real
        self.gr[1] = gxy.imag
        return
    def rotate_gradient_y_to_z(self,angle):
        '''rotate the gradient from y to z direction (deg)(0-360)'''
        phi = torch.tensor(angle).to(self.device)
        gyz = self.gr[1]+1j*self.gr[2]
        gyz = gyz*torch.exp(1j*torch.pi*phi/180)
        self.gr[1] = gyz.real
        self.gr[2] = gyz.imag
        return
    def rotate_gradient_z_to_x(self,angle):
        '''rotate the gradient from z to x direction (deg)(0-360)'''
        phi = torch.tensor(angle).to(self.device)
        gzx = self.gr[2]+1j*self.gr[0]
        gzx = gzx*torch.exp(1j*torch.pi*phi/180)
        self.gr[2] = gzx.real
        self.gr[0] = gzx.imag
        return
    def scale_rf_to_flipangle(self,flip): # TODO
        '''re-scale the rf pulse to required flip angle. May cause error for large flip angles.'''
    def modulate_rf_by(self,freq_shift):
        '''frequency shift of rf waveform. (Hz)'''
        delta_omega = 2*torch.pi*freq_shift # omega = 2*pi*f, 2*pi*Hz = rad/s
        rf_c = self.rf_complex*torch.exp(1j*delta_omega*self.times*1e-3)
        self.rf[0] = rf_c.real
        self.rf[1] = rf_c.imag
        return
    def change_device(self,device:torch.device):
        '''change device to cuda or cpu'''
        self.device = device
        self.rf = self.rf.to(device)
        self.gr = self.gr.to(device)
        return
    def change_dt(self,newdt):
        '''change of time resolution, (ms)
        
        Args:
            newdt: (ms) new time-step

        use method = 'linear', 'nearest'
        '''
        # print('>> Change of time resolution')
        if self.dt != newdt:
            Nt_new = math.floor(self.dt*(self.Nt-1)/newdt)+1
            # print(Nt_new)
            
            gr_old = np.array(self.gr.tolist())
            rf_old = np.array(self.rf.tolist())
            told = self.dt*np.arange(self.Nt) #(Nt*)(ms)
            tnew = np.arange(Nt_new)*newdt
            # print(tnew.shape)
            # print(told[-4:])
            # print(tnew[-4:])
            
            # Interpolate gradient
            gr_new = np.zeros((3,Nt_new))
            for ch in range(3):
                gr_channl = gr_old[ch,:]
                # print(told.shape,gr_channl.shape)
                inter_gr_fn = interpolate.interp1d(told,gr_channl,kind='linear')
                gr_new[ch,:] = inter_gr_fn(tnew)
            
            # Interpolate RF
            rf_new = np.zeros((2,Nt_new))
            for ch in range(2):
                rf_channl = rf_old[ch,:]
                inter_rf_fn = interpolate.interp1d(told,rf_channl,kind='linear')
                rf_new[ch,:] = inter_rf_fn(tnew)


            # duration = int(self.Nt*self.dt/newdt)*newdt
            # duration = int(duration/0.01)*0.01
            # tnew = np.arange(0,duration,0.010) #(ms)(dt=10us)
            # Nt_new = len(tnew)
            # # print('duration:',Nt*dt, Nt_new*0.01)
            # # print(told[-5:])
            # # print(Nt_new)
            # # print('gr',gr.shape)
            # # print('told',told.shape)
            # # print(gr[0,:].shape)
            # # print(tnew[-5:])

            # gr_new = np.zeros((3,Nt_new))
            # for ch in range(3):
            # 	gr_channl = gr[ch,:]
            # 	# print(told.shape,gr_channl.shape)
            # 	inter_gr_fn = interpolate.interp1d(told,gr_channl,kind='nearest')
            # 	gr_new[ch,:] = inter_gr_fn(tnew)
            # # Interpolate RF
            # tnew = np.arange(0,duration,dt_rf) #(ms)(dt=1us)
            # Nt_new = len(tnew)
            # rf_new = np.zeros((2,Nt_new))
            # # print(tnew[-5:])
            # for ch in range(2):
            # 	rf_channl = rf[ch,:]
            # 	inter_rf_fn = interpolate.interp1d(told,rf_channl,kind='nearest')
            # 	rf_new[ch,:] = inter_rf_fn(tnew)

            # print(rf_new.shape)
            # print(gr_new.shape)
            #
            # G = np.kron(np.identity(gr_new.shape[1]),np.ones(5).reshape(-1,1))
            # print(G[:20,:3])

            # update
            self.dt = newdt
            self.Nt = Nt_new
            self.rf = torch.tensor(rf_new,device=self.device, dtype=self.dtype)
            self.gr = torch.tensor(gr_new,device=self.device, dtype=self.dtype)
        return
    def balancing_gr(self):
        '''make the summation of gradient to be zero'''
        gr_diff = torch.sum(self.gr,dim=1)
        self.gr[:,1:self.Nt-1] = self.gr[:,1:self.Nt-1] - gr_diff.reshape(3,1)/(self.Nt-2)
        return
    # ---------------------------------------------------------------
    # Methods of plot function:
    # ---------------------------------------------
    def plot_rf_magnitude(self,ax:matplotlib.axes.Axes,title=''):
        times = self.times.cpu().numpy()
        xlimit = [0,times[-1]]
        rf = self.rf.cpu().numpy()
        rf_complex = rf[0,:] + 1j*rf[1,:]
        # 
        ax.plot(times,np.abs(rf_complex),lw=1)
        ax.set_ylabel('RF magnitude\nmT')
        ax.set_xlabel('time (ms)')
        ax.set_xlim(xlimit)
        ax.set_title(title)
        return
    def plot_rf_phase(self,ax:matplotlib.axes.Axes,title=''):
        times = self.times.cpu().numpy()
        xlimit = [0,times[-1]]
        rf = self.rf.cpu().numpy()
        rf_complex = rf[0,:] + 1j*rf[1,:]
        # 
        ax.plot(times,np.angle(rf_complex),lw=1)
        ax.set_ylabel('RF angle\nrad')
        ax.set_xlabel('time (ms)')
        ax.set_xlim(xlimit)
        ax.set_title(title)
        return
    def plot_gradients(self,ax:matplotlib.axes.Axes,title='',legend=True):
        # N = self.rf.shape[1]
        # time = np.arange(N)*self.dt
        times = self.times.cpu().numpy()
        xlimit = [0,times[-1]]
        gr = self.gr.cpu().numpy()
        gr_mag = self.gr.norm(dim=0).cpu().numpy()
        # 
        ax.plot(times,gr[0,:],label='Gx')
        ax.plot(times,gr[1,:],label='Gy')
        ax.plot(times,gr[2,:],label='Gz')
        ax.plot(times,gr_mag,ls='--',alpha=0.5)
        ax.set_xlim(xlimit)
        if legend: ax.legend()
        ax.set_ylabel('gradients\nmT/m')
        ax.set_xlabel('time (ms)')
        ax.set_title(title)
        return
    def plot_slewrate(self,ax:matplotlib.axes.Axes,title='',legend=True):
        # N = self.rf.shape[1]
        times = self.times.cpu().numpy() #np.arange(N)*self.dt
        xlimit = [0,times[-1]]
        srate = self.get_slewrate().cpu().numpy()
        srate_mag = self.get_slewrate().norm(dim=0).cpu().numpy()
        # 
        ax.plot(times,srate[0,:],label='Sx')
        ax.plot(times,srate[1,:],label='Sy')
        ax.plot(times,srate[2,:],label='Sz')
        ax.plot(times,srate_mag,ls='--',alpha=0.5) # vector magnitude
        ax.set_xlim(xlimit)
        ax.set_ylabel('slewrate\nmT/m/ms')
        ax.set_xlabel('time (ms)')
        if legend: ax.legend()
        ax.set_title(title)
        return
    def plot_kspace(self,ax:matplotlib.axes.Axes,title='',legend=True):
        kt = self.get_kspace().cpu().numpy()
        N = kt.shape[1]
        time = np.arange(N)*self.dt
        xlimit = [0,time[-1]]
        # 
        ax.plot(time,kt[0,:],label='kx')
        ax.plot(time,kt[1,:],label='ky')
        ax.plot(time,kt[2,:],label='kz')
        ax.set_xlim(xlimit)
        ax.set_ylabel('kspace\n1/m')
        ax.set_xlabel('time (ms)')
        if legend: ax.legend()
        ax.set_title(title)
        return
    def plot_pns_prediction(self,ax:matplotlib.axes.Axes,title='',model='SAFE',plot_threshold=False,threshold=80):
        '''Plot the PNS threshold which is translated into a constraint of slew-rate.
        '''
    def plot_kspace3d(self,method='excitation',title='',legend=False,figname='tmp.png',savefig=False,ax=None):
        '''plot of the k-space'''
        # N = self.rf.shape[1]
        # time = np.arange(N)*self.dt
        
        # Use rf to demonstrate the deposit of power
        # rf = self.rf.cpu().numpy()
        # rf_complex = rf[0,:] + 1j*rf[1,:]
        rf_complex = self.rf_complex.cpu().numpy()

        # Get k-space trajectory
        kt = self.get_kspace().cpu().numpy()

        if ax==None:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(projection='3d')
            newfig = True
        else:
            newfig = False
        
        # Plot
        show_rf_weighting = False
        if show_rf_weighting:
            ax.scatter(kt[0,:],kt[1,:],kt[2,:],s=np.abs(rf_complex)*1e4,marker='d',
                color='red',alpha=0.02
            )
        ax.plot(kt[0,:],kt[1,:],kt[2,:],label=r'k-trajectory $(cycle{\cdot}m^{-1})$')
        # ax.plot((0,M[0,0]),(0,M[1,0]),(0,M[2,0]),linewidth=1,linestyle='--')
        # ax.plot((0,M[0,-1]),(0,M[1,-1]),(0,M[2,-1]),linewidth=1,linestyle='--')
        # ax.text(M[0,0],M[1,0],M[2,0],r'$k_0$',fontsize=8)
        # ax.text(M[0,-1],M[1,-1],M[2,-1],r'end',fontsize=8)
        ax.scatter(kt[0,0],kt[1,0],kt[2,0],marker='o',color='green',label='start')
        ax.scatter(kt[0,-1],kt[1,-1],kt[2,-1],marker='o',color='red',label='end')
        # ax.text(kt[0,0],kt[1,0],kt[2,0],r'start',fontsize=8)
        # ax.text(kt[0,-1],kt[1,-1],kt[2,-1],r'end',fontsize=8)
        if legend:
            ax.legend()
        # ax5.set_xlim(-1.1*maxkx,1.1*maxkx)
        # ax5.set_ylim(-1.1*maxky,1.1*maxky)
        # ax5.set_zlim(-1.1*maxkz,1.1*maxkz)
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_zlabel('kz')
        ax.set_title(title)
        plt.axis('equal')

        if newfig:
            print('save fig: '+figname)
            plt.savefig(figname)
            plt.close(fig)
        return
    def plot_gradient_spectrum(self,ax:matplotlib.axes.Axes,title=''):
        '''Plot acoustic sprecturm of the gradients'''
    # --------------------------------------------------------------
    def plot(self,title='pulse',figname='tmp.png',savefig=False,use='excitation'):
        '''plot characteristics of a pulse object'''
        pltr,pltc = 6,4
        thesubplot = lambda x,y: plt.subplot(pltr,pltc,x*pltc+y+1)
        thesubplot_long = lambda x,y,xlen: plt.subplot(pltr,pltc,(x*pltc+y+1,x*pltc+y+xlen))
        fig = plt.figure(figsize=(12,12))
        
        # ax = plt.subplot(pltr,pltc,(1,2))
        ax = thesubplot_long(0,0,2)
        self.plot_rf_magnitude(ax=ax)
        
        ax = thesubplot_long(1,0,2)
        self.plot_rf_phase(ax=ax)

        ax = thesubplot_long(2,0,2)
        self.plot_gradients(ax=ax)

        ax = thesubplot_long(3,0,2)
        self.plot_slewrate(ax=ax)

        ax = thesubplot_long(4,0,2)
        self.plot_kspace(ax=ax)

        ax = thesubplot_long(5,0,2)
        self.plot_pns_prediction(ax=ax,plot_threshold=True,threshold=80)

        ax = plt.subplot(pltr,pltc,(3,8),projection='3d')
        self.plot_kspace3d(ax=ax,method=use)
        ax.set_title('3d k-space (1/m)')

        ax = thesubplot_long(2,2,2)
        self.plot_gradient_spectrum(ax=ax)
        ax.set_title('acoustic spectrum')

        ax = thesubplot_long(3,2,2)
        ax.set_title('gradient PSF')

        ax = thesubplot_long(5,2,2)
        info = 'duration={}ms, dt={}ms'.format(self.duration,self.dt)
        info += '\nrfmax={:.4f}mT, Gmax={:.2f}mT/m'.format(self.rf_peak,self.gr_peak)
        ax.set_title(info)
        ax.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        if savefig:
            print('save fig: '+figname)
            plt.savefig(figname)
            plt.close(fig)
        return
    def _plot_pulse_kspace(self,figname='tmp.png',savefig=False):
        # fig = plt.figure(figsize=(13,6),layout='constrained')
        fig = plt.figure(figsize=(13,6))
        ax1 = plt.subplot(4,2,1)
        self.plot_rf_magnitude(ax=ax1)

        ax2 = plt.subplot(4,2,3)
        self.plot_rf_phase(ax=ax2)
        
        ax3 = plt.subplot(4,2,5)
        self.plot_gradients(ax=ax3)

        ax4 = plt.subplot(4,2,7)
        self.plot_slewrate(ax=ax4)

        ax5 = plt.subplot(4,2,8)
        self.plot_kspace(ax=ax5)

        ax5 = plt.subplot(4,2,(2,6),projection='3d')
        self.plot_kspace3d(ax=ax5)
        # ax5.text(kt[0,0],kt[1,0],kt[2,0],r'start',fontsize=8)
        # ax5.text(kt[0,-1],kt[1,-1],kt[2,-1],r'end',fontsize=8)

        plt.tight_layout()
        if savefig:
            print('save fig: '+figname)
            plt.savefig(figname)
            plt.close(fig)
        return
    # -----------------------------------
    # Save pulse
    # ----------------------------------
    def save(self,outpath,details=False,exc_kspace=False,img_kspace=False,slewrate=False,pns=False,spectrum=False,saveall=False):
        '''save pulse to a .mat file'''
        outdic = {
            'rf': self.rf.cpu().numpy(),
            'gr': self.gr.cpu().numpy(),
            'dt': self.dt,
            'Nt': self.Nt,
            'name': self.name,
        }
        if saveall:
            exc_kspace = img_kspace = slewrate = pns = spectrum = True
        if exc_kspace: outdic['k_exc'] = self.get_kspace(case='excitation').cpu().numpy()
        if img_kspace: outdic['k_img'] = self.get_kspace(case='imaging').cpu().numpy()
        if slewrate: outdic['srate'] = self.get_slewrate().cpu().numpy()
        if pns: outdic['pns'] = self.get_PNS_prediction_SAFE().cpu().numpy()
        if spectrum: 
            freq,spec = self.get_gradient_acoustic_resonance()
            freq = freq.cpu().numpy()
            spec = spec.cpu().numpy()
            outdic['freq'] = freq
            outdic['spectrum'] = spec
        spio.savemat(outpath,outdic)
        logstr = 'save pulse: '+outpath
        if details: print(logstr)
        return
    @staticmethod
    def load(filepath,device=torch.device('cpu'),details=False) -> 'Pulse':
        '''load pulse from save data, e.g., from Pulse.save()'''
        data = spio.loadmat(filepath)
        # print(data)
        rf = torch.from_numpy(data['rf']).to(device)
        gr = torch.from_numpy(data['gr']).to(device)
        dt = data['dt'].item()
        # name = data.get('name')[0]
        return Pulse(dt=dt,rf=rf,gr=gr,device=device)
    
    # Display of some info
    # --------------------------------------------------
    def show_info(self):
        # print('>> Pulse:')
        print('Pulse: {}, '.format(self.name), self.device)
        print('\t    | shape | max | rate | energy')
        print('\t--------------------------------------------------')
        s = '\trf: | ({},{}) | {:.5f}   mT |'.format(self.rf.shape[0],self.rf.shape[1],self.rf_peak)
        s += ' {:.6f}   mT/ms |'.format(torch.diff(self.rf,dim=1).abs().max().item()/(self.dt))
        s += ' {:.6f} mT*mT*ms |'.format(self.rf_energy)
        print(s)
        s = '\tgr: | ({},{}) | {:.5f} mT/m |'.format(self.gr.shape[0],self.gr.shape[1],self.gr_peak)
        s += ' {:.6f} mT/m/ms |'.format(self.get_slewrate().abs().max().item())
        print(s)
        print('\tduration={} ms, #time points={}, dt={} ms'.format(
            self.duration, self.Nt, self.dt))
        print('\t'+''.center(40,'-'))
        return
    # ------------------------------
    # provided examples
    # ------------------------------
    @staticmethod
    def example_pulse(device=torch.device('cpu')): # TODO
        Nt = 400 #100
        dt = 0.01 # ms
        t0 = 100 # time parameter for a sinc pulse
        rf = 0.0*torch.zeros((2,Nt))
        rf[0,:] = 6*1e-3*torch.special.sinc((torch.arange(Nt)-200)/t0)
        gr = 0.0*torch.zeros((3,Nt)) # mT/m
        gr[2,:] = 10*torch.ones(Nt) # mT/m
        # --------
        ntrise = 10
        tref = torch.arange(ntrise)
        gr_rise = gr[:,1].reshape(3,1) * tref/ntrise
        gr_drop = gr[:,-1].reshape(3,1) * (tref[-1] - tref)/ntrise
        gr = torch.hstack((gr_rise,gr,gr_drop))
        rf = torch.hstack((torch.zeros(2,ntrise),rf,torch.zeros(2,ntrise)))
        # ---------------------
        pulse = Pulse(rf=rf,gr=gr,dt=dt,device=device,name='example')
        return pulse
    @staticmethod
    def example_gradient(device=torch.device('cpu')):
        Nt = 500
        dt = 1e-2
        rf = torch.zeros((2,Nt))
        gmax = 5
        gr = torch.ones((3,Nt)) * gmax
        n_rise = 50
        tref = torch.arange(n_rise)
        gr[:,:n_rise] = tref/n_rise * gmax
        gr[:,-n_rise:] = (tref[-1] - tref)/n_rise * gmax
        pulse = Pulse(rf=rf,gr=gr,dt=dt,device=device,name='example')
        return pulse
    @staticmethod
    def example_none(device=torch.device('cpu')):
        gr = torch.zeros((3,10))
        rf = torch.zeros((2,10))
        return Pulse(gr=gr,rf=rf,dt=1e-3,device=device)


class Spin:
    def __init__(self,T1=1000.0,T2=100.0,df=0.0,kappa=1.0,gamma=Gamma,loc=[0.,0.,0.],M=[0.,0.,1.],name='spin',device=torch.device("cpu")):
        """Initialize of a spin.

        Args:
            T1: 		(ms)
            T2: 		(ms)
            df: 		(Hz)			off-resonance
            kappa:						B1 transmit
            gamma: 		(MHz/Tesla) 	(gyromagnetic ratio normalized by 2*pi) 
            M: (1,3) or None			manetization vector (Tensor)
            loc: (1,3)	(cm) 			spatial location
        
        attribute:
            T1: 		(ms)
            T2: 		(ms)
            df: 		(Hz)			off-resonance
            kappa:						B1 transmit
            gamma: 		(MHz/Tesla) 	(gyromagnetic ratio normalized by 2*pi) 
            M: (1,3)					manetization vector (Tensor)
            loc: (1,3)	(cm) 			spatial location
        """
        self.device = device
        self.dtype = torch.float32
        self.name = name
        # --------------------------
        self.T1 = T1
        self.T2 = T2
        self.df = df            # off-resonance, (Hz)
        self.kappa = kappa
        self.gamma = gamma      # (MHz/Tesla)
        self.set_location(loc) 	# -> self.loc
        self.set_M(M)		    # -> self.M
    
    def set_location(self,loc):
        '''loc: (1*3), list or tensor'''
        if not isinstance(loc,torch.Tensor): loc = torch.tensor(loc)
        self.loc = loc.to(self.dtype).to(self.device)
    def set_M(self,M=None,normalize=True):
        '''set magnetization of the spin, allow the Mag can be zero'''
        if M==None:
            self.M = torch.tensor([0.,0.,1.],device=self.device)
        else:
            if not isinstance(M,torch.Tensor): M = torch.tensor(M)
            M = M.to(self.dtype)
            if normalize: M = torch.nn.functional.normalize(M,dim=0)
            self.M = M.to(self.device)
    # --------------------------------------------------------------
    # Methods for designing magnetization
    # --------------------------------------------------------------
    @staticmethod
    def calculate_target_excited_Mxy(flip,phase)->torch.Tensor:
        '''Return target transverse magnetization for given flip angle and phase.
        
        Args:
            flip: 0-180     (degree)
            phase: 0-360    (degree) if phase is 0, then the rf is applied along the x-axis
        
        Returns:
            Mxy:            (complex)
        '''
        theta = torch.tensor([flip])/180*torch.pi
        phi = torch.tensor([phase])/180*torch.pi     # the phase of Mxy after excitation
        Mxy = 1j*torch.sin(theta)*torch.exp(1j*phi)
        return Mxy
    @staticmethod
    def calculate_target_excited_M(flip,phase)->torch.Tensor:
        '''Return target magnetization after excitation.
        
        Args:
            flip: 0-180     (degree)
            phase: 0-360    (degree) along which direction the rf is applied
        
        Returns:
            M: (1,3)
        '''
        theta = torch.tensor([flip])/180*torch.pi
        Mxy = Spin.calculate_target_excited_Mxy(flip,phase)
        M = torch.zeros(3)
        M[0] = torch.real(Mxy)
        M[1] = torch.imag(Mxy)
        M[2] = torch.cos(theta)
        return M
    @staticmethod
    def calculate_target_spindomain_excitation(flip,phase)->tuple[torch.Tensor,torch.Tensor]:
        '''Return target spin-domain parameters for excitation.

        Args:
            flip: 0-180         (deg)
            phase: 0-360        (deg) along which direction the rf is applied
        
        Returns:
            alpha.conj()*beta:  (complex)
            |beta|:              (real)
        '''
        theta = torch.tensor([flip])/180*torch.pi
        phi = torch.tensor([phase])/180*torch.pi
        alphaConj_beta = 0.5*torch.exp(1j*phi)*torch.sin(theta)*1j
        beta_magnitude = torch.sqrt((1-torch.cos(theta))/2)
        return alphaConj_beta,beta_magnitude
    @staticmethod
    def calculate_target_spindomain_refocusing(phase) -> torch.Tensor:
        '''Return target spin-domain parameters for refocusing.

        Args:
            phase: 0-180        (deg) along which direction the rf is applied
        
        Returns: 
            beta^2:             (compelx)
        '''
        phi = torch.tensor([phase])/180*torch.pi
        betasquare = - torch.exp(-1j*2*phi)
        return betasquare
    @staticmethod
    def calculate_target_spindomain_inversion() -> torch.Tensor: #TODO
        ''''''
        return 'todo'
        '''inversion |beta|^2 = 1 in-slice, = 0 out-slice'''
        para_real = b_real**2 + b_imag**2
        para_imag = 0.0
    # --------------------------------------------------------------
    @staticmethod
    def calculate_flipangle(M)->torch.Tensor:
        '''Return flip angle for given excitated magnetization.'''
        Mxy = torch.linalg.norm(M[:2])
        flipangle = torch.asin(Mxy)/torch.pi*180
        flipangle = (flipangle-90)*torch.sign(M[2]) + 90 # this is for TODO
        return flipangle
    # --------------------------------------------------------------
    @staticmethod
    def calculate_rotated_M(flipz,rotxy,M=[0.,0.,1.]): #TODO
        '''Return M from given rotations'''
    @staticmethod
    def calculate_rotated_M_spindomain(alpha,beta,M=torch.tensor([0.,0.,1.])): #TODO tocheck
        '''Return M from spin-domain rotation parameters.'''
        if not isinstance(alpha,torch.Tensor): alpha = torch.tensor([alpha])
        if not isinstance(beta,torch.Tensor): beta = torch.tensor([beta])
        if not isinstance(M,torch.Tensor): M = torch.tensor([M])
        Mxy = M[0] + 1j*M[1]
        Mxynew = ((alpha.conj())**2)*Mxy - (beta**2)*(Mxy.conj()) + 2*alpha.conj()*beta * M[2]
        Mznew = -2*torch.abs(alpha*beta*Mxy) + (alpha.abs()**2 - beta.abs()**2)*M[2]
        Mnew = torch.zeros_like(M)
        Mnew[0] = Mxynew.real
        Mnew[1] = Mxynew.imag
        Mnew[2] = Mznew
        return Mnew

    # --------------------------------------------------------------
    # Methods
    # --------------------------------------------------------------
    # get transverse magnetization (for single spin it is easy to be done outside the class)

    '''Methods in progress or to be decided'''
    def _calculate_Beff(self,rf,gr): #TODO
        '''function that calculate the effective magnetic field'''
    def _rotation(): #TODO
        '''rotation of magnetization'''
        print('spin.rotation not implemented yet!')
        return
    def _rotation_spindomain(self,alpha,beta,Minit=None): #TODO
        '''compute the rotation result of spin-domain rotation parameters
        
        input:
            alpha:(compelx)
            beta: (complex)
        '''
        if Minit==None:
            Minit = self.M
        # calculate the rotated magnetization
        Mxy = Minit[0] + Minit[1]*1j
        Mend_z = (alpha*torch.conj(alpha) - beta*torch.conj(beta))*Minit[2] \
            - 2*torch.real(alpha*beta*torch.conj(Mxy))
        Mend_xy = (alpha.conj())**2*Mxy + (beta**2)*torch.conj(Mxy) + 2*torch.conj(alpha)*beta*Minit[2]
        # assign the magnetization after rotation
        Mend = torch.zeros_like(Minit)
        Mend[0] = torch.real(Mend_xy)
        Mend[1] = torch.imag(Mend_xy)
        Mend[2] = torch.real(Mend_z)
        return Mend

    
    # --------------------------------------------------------------
    # Methods that do operations to the spin
    # --------------------------------------------------------------

    # ------------------------------------------------------------------
    # Methods that display things
    # ------------------------------------------------------------------
    def show_info(self):
        print('>> Spin: {}, ({})'.format(self.name, self.device))
        print('\tname  | unit    | value ')
        print('\t------------------------------------------------')		
        print('\tgamma | (MHz/T) | {}'.format(self.gamma))
        print('\tT1    | (ms)    | {}'.format(self.T1))
        print('\tT2    | (ms)    | {}'.format(self.T2))
        print('\tdf    | (Hz)    | {}'.format(self.df))
        print('\tkappa | (1)     | {}'.format(self.kappa))
        print('\tloc   | (cm)    |',self.loc.tolist())
        print('\tMag   | ------- |',self.M.tolist())
        # print('\t'+''.center(20,'-'))
        return
    


def blochsim_spin(spin:Spin,Nt,dt,rf:torch.Tensor,gr:torch.Tensor,device=torch.device('cpu'),history=False):
    """Bloch simulation for a single Spin.
    
    Args:
        spin:					Spin object
        Nt:						number of timepints
        dt: 			(ms)	time resolution
        rf: (2,Nt)		(mT) 	rf waveform (real and imaginary)
        gr: (3,Nt)		(mT/m)	(three channel gradient waveform)
        details:
    
    Returns:
        M: (1,3) or (3,Nt+1) 	final magnetization / evolution of magn
    """
    display_details = False
    M = spin.M
    dt = torch.tensor(dt,device=device)
    # ------------------------------
    if history:
        M_hist = torch.zeros((3,Nt+1),device=device)
        M_hist[:,0] = M

    # Define some variables related to the relaxation
    # they are unitless
    # -----------------
    E1 = torch.exp(-dt/spin.T1) # mT/mT = 1
    E2 = torch.exp(-dt/spin.T2) # mT/mT = 1
    E = torch.tensor([[E2,0.,0.],
                [0.,E2,0.],
                [0.,0.,E1]],device=device)
    e = torch.tensor([0.,0.,1-E1],device=device)
    # print('e:',e.dtype)

    # Calculate the effective magnetic field [mT]
    # Beff = B1*kappa + B_offres + B_gradient
    # <r,G> = cm * mT/m * 1e-2 = mT
    # df[Hz] / gamma[MHz/T] * 1e-3 = mT
    # --------------------------------------
    Beff_hist = torch.zeros((3,Nt),device=device)*1.0
    Beff_hist[:2,:] = rf*spin.kappa # mT (rf pulse, w/ transmission factor)
    Beff_hist[2,:] = spin.loc@gr*1e-2 + spin.df/spin.gamma*1e-3
    # print('Beff_hist:',Beff_hist.shape)
    # print(Beff_hist[:,:5])
    # print(rf[:,:5])

    # Simulation by time steps
    for k in range(Nt):
        # Beff = torch.zeros(3,device=device)
        # Beff[0] = rf[0,k]
        # Beff[1] = rf[1,k]
        # Beff[2] = torch.dot(gr[:,k], spin.get_loc())*1e-2 + spin.df/spin.gamma

        # from effective field, obtain
        # 1) norm of effective field; 2) its unit directional vector;
        Beff = Beff_hist[:,k]
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
        
        # compute the magnetization after the rotation + relaxation
        # M_temp = R1@M_hist[:,k] + torch.sin(phi)*torch.cross(Beff_unit,M_hist[:,k])
        # M_hist[:,k+1] = E@M_temp + e
        R1 = torch.cos(phi)*torch.eye(3,device=device) + (1-torch.cos(phi))*torch.outer(Beff_unit,Beff_unit)
        M = R1@M + torch.sin(phi)*torch.linalg.cross(Beff_unit,M)

        # computed the effect of relaxation
        M = E@M + e
        if history:
            M_hist[:,k+1] = M

        if display_details:
            pass
            # if k%50==0:
            # 	print('k =',k)
            # 	print(M.shape)
            # 	print(M.norm())
    if history:
        return M_hist
    else:
        return M
def spinorsim_spin_singlestep(spin:Spin,Nt,dt,rf,gr,device=torch.device('cpu'),history=False):
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
    Beff_hist = torch.zeros((3,Nt),device=device)*1.0
    Beff_hist[:2,:] = rf*spin.kappa # mT
    Beff_hist[2,:] = spin.loc@gr*1e-2 + spin.df/spin.gamma*1e-3

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
def spinorsim_spin_seperatestep(spin:Spin,Nt,dt,rf,gr,device=torch.device('cpu'),history=False):
    '''Spinor simulation for a single spin, in spin-domain
    this simulation separate the rf, and gradient effects

    input:
        spin:
        Nt:
        dt:			(ms)
        rf:(2*Nt)	(mT)
        gr:(3*Nt)	(mT/m)
        device:		"cpu"/"gpu"
        history:	True/False
    output:
        alpha: (complex)
        beta: (complex)
    '''
    # compute for free-precession:
    Bz_hist = spin.loc@gr*1e-2 + spin.df/spin.gamma*1e-3 # (1*Nt) (mT)
    pre_phi = -2*torch.pi*spin.gamma*Bz_hist*dt # 2*pi*MHz/T*mT*ms = rad/s
    pre_aj_hist = torch.exp(-torch.tensor([0.+1.0j],device=device)*pre_phi/2)
    # compute for nutation:
    rf_norm = rf.norm(dim=0)
    rf_unit = torch.nn.functional.normalize(rf,dim=0)
    nut_phi = -2*torch.pi*spin.gamma*rf_norm*dt*spin.kappa
    nut_aj_hist = torch.cos(nut_phi/2)
    nut_bj_hist = -(torch.tensor([0.+1.0j],device=device)*rf_unit[0]-rf_unit[1])*torch.sin(nut_phi/2)

    # for recording:
    ahist = (1.0+0.0j)*torch.zeros(Nt+1,device=device)
    bhist = (1.0+0.0j)*torch.zeros(Nt+1,device=device)
    ahist[0] = 1.0+0.0j
    bhist[0] = 0.0+0.0j
    for t in range(Nt):
        # free precession period:
        atmp = pre_aj_hist[t]*ahist[t]
        btmp = pre_aj_hist[t].conj()*bhist[t]

        # nutation period:
        ahist[t+1] = nut_aj_hist[t]*atmp - nut_bj_hist[t].conj()*btmp
        bhist[t+1] = nut_bj_hist[t]*atmp + nut_aj_hist[t].conj()*btmp

    # get the final state
    a,b = ahist[-1],bhist[-1]
    if history:
        return ahist,bhist
    else:
        return a,b



class SpinArray:
    '''Group of spins'''
    def __init__(self,loc:torch.Tensor,T1=1000,T2=100,gamma=Gamma,df=0.0,kappa=1.0,M=[0.,0.,1.],name='spinarray',device=torch.device("cpu")):
        '''Initialize of spins.

        Args:
            loc:   (3,num)			(cm)        locations
            T1:    (1) or (1,num)	(ms)        T1 values
            T2:    (1) or (1,num)	(ms)        T2 values
            gamma: (1) or (1,num)	(MHz/T) 	(gyromagnetic ratio normalized by 2*pi) 
            df:    (1) or (1,num)	(Hz)        off-resonance
            kappa: (1) or (1,num)				transmit B1 scaling factor (>=0)
            M:     (3,num)                      magnetization vectors
            device:                             torch.device
        
        properties:
            num: 								total number of spins
        provided methods:
            set B0/B1 homogeneous
        '''
        self.device = device
        self.dtype = torch.float32
        self.name = name
        # ---------------------------------------
        self.loc =   loc.to(self.device)      # (3,num)(cm)
        self.T1 =    self._prepare_values(T1) # (ms)
        self.T2 =    self._prepare_values(T2) # (ms)
        self.gamma = self._prepare_values(gamma) # MHz/Tesla
        self.df =    self._prepare_values(df) # (Hz)
        self.kappa = self._prepare_values(kappa) 
        self.set_M(M)   # -> self.M
        # ----------------------
        self.mask = None
        # assert loc.shape[1] == len(T1)
        # assert len(T1)==len(T2)
        
    @property
    def num(self):
        '''number of spins''' # it is better to set this as a property
        return self.loc.shape[1]

    def _prepare_values(self,vals):
        '''prepare the input values. 

        Args:
            vals: a float/int, or a tensor array
        '''
        if not isinstance(vals,torch.Tensor): vals = torch.tensor([vals])
        if len(vals) == 1: vals = torch.ones(self.num)*vals
        # print('error in prepare spin properties !')
        vals = vals.to(self.dtype)
        return vals.to(self.device)
    def set_M(self,M=[0.,0.,1.],normalize=True):
        '''Set the magnetization of spins.

        Args:
            M: (1,3) or (3,num) (tensor|list|None)
        '''
        if not isinstance(M,torch.Tensor): M = torch.tensor(M)
        if torch.numel(M) == 3: M = M.reshape(3,1)*torch.ones(self.num)
        M = M.to(self.dtype)
        if normalize: M = torch.nn.functional.normalize(M,dim=0)
        self.M = M.to(self.device)
        return
    def set_B0_homogeneous(self):
        '''set spins df=0 (Hz)'''
        self.df = self.df*0.0
        return
    def set_B1_homogeneous(self):
        '''set spins kappa=1'''
        self.kappa = self.kappa*0.0 + 1.0
        return
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_Mxy(M:torch.Tensor)->torch.Tensor:
        '''Calculate transverse magnetization (complex). M:(3,num)'''
        return M[0]+1j*M[1]
    @staticmethod
    def calculate_rotated_M_spindomain(alpha:torch.Tensor,beta:torch.Tensor,M=torch.tensor([[0.],[0.],[1.]])): #TODO
        '''Calculate rotated magnetization for given spin-domain parameters.
        
        Args:
            alpha: (1,num) (complex)
            beta: (1,num) (complex)
            M: (3,num)

        Returns:
            Mend: (3,num) 
        '''
        # if not isinstance(alpha,torch.Tensor): alpha = torch.tensor([alpha])
        # if not isinstance(beta,torch.Tensor): beta = torch.tensor([beta])
        device = alpha.device
        num = len(alpha)
        if not isinstance(M,torch.Tensor): M = torch.tensor([M])
        if torch.numel(M) == 3: M = M*torch.ones(num)
        M = M.to(device)
        Mxy = M[0] + 1j*M[1]
        Mxynew = ((alpha.conj())**2)*Mxy - (beta**2)*(Mxy.conj()) + 2*alpha.conj()*beta * M[2]
        Mznew = -2*torch.abs(alpha*beta*Mxy) + (alpha.abs()**2 - beta.abs()**2)*M[2]
        Mnew = torch.zeros_like(M)
        Mnew[0] = Mxynew.real
        Mnew[1] = Mxynew.imag
        Mnew[2] = Mznew
        return Mnew
        if Minit==None:
            Minit = self.M
        Mxy = Minit[0] + Minit[1]*1j
        Mend_z = (alpha*torch.conj(alpha) - beta*torch.conj(beta))*Minit[2] \
            - 2*torch.real(alpha*beta*torch.conj(Mxy))
        Mend_xy = (alpha.conj())**2*Mxy + (beta**2)*torch.conj(Mxy) + 2*torch.conj(alpha)*beta*Minit[2]
        # Assign new magnetization
        Mend = torch.zeros_like(Minit)
        Mend[0] = torch.real(Mend_xy)
        Mend[1] = torch.imag(Mend_xy)
        Mend[2] = torch.real(Mend_z)
        return Mend
    @staticmethod
    def calculate_rotated_M(M): #TODO
        '''Calculate rotated magnetization for given rotation.'''
    @staticmethod
    def calculate_flipangles(M)->torch.Tensor:
        '''Return flip angles for given magnetization state.
        
        Args:
            M:(3,num)
        
        Returns:
            flipangle: (1,num) (deg)
        '''
        Mxy = torch.linalg.norm(M[:2,:],dim=0)
        flipangle = torch.asin(Mxy)/torch.pi*180
        flipangle = (flipangle-90)*torch.sign(M[2,:]) + 90
        return flipangle
    # -----------------------------------------------------------------
    @staticmethod
    def spinarray_Beffhist(spinarray,Nt,dt,rf:torch.Tensor,gr:torch.Tensor,device=torch.device('cpu'))->torch.Tensor:
        '''Get effective B-field for each spin at every timepoint.

        input:
            spinarray:
            Nt:
            dt:					(ms)
            rf:(2*Nt)			(mT)
            gr:(3*Nt)			(mT/m)
            device:
        needed info of spinarray:
            spinarray.loc: (3*num) (cm)
            spinarray.gamma: (1*num) 
            spinarray.df: (1*num) (Hz)
            spinarray.kappa: (1*num)
        output:
            Beff_hist:(3*num*Nt)	(mT)
        '''
        # starttime = time()
        num = spinarray.num
        # M_hist = torch.zeros((3,num,Nt+1),device=device)
        # M_hist[:,:,0] = 

        # location = spinarray.loc #(3*num)
        # df = torch.rand(num,device=device)
        # df = spinarray.df
        # print(gr.requires_grad)
        # print(gr[:,:8])

        # Test:
        # print('rf',rf[:,:5])
        # print('B1',spinarray.B1map[:2])

        # Calculate the effective magnetic field (mT)
        # Beff = B1*kappa + B_offres + B_grad
        # <r,G> = cm * mT/m * 1e-2 = mT
        # df[Hz] / gamma[MHz/T] * 1e3 = mT
        # -----------------------------------
        
        # Effective magnetic field given by: off-resonance
        offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num) Hz/(MHz/T) = mT
        offBeff = offBeff.reshape(num,1)
        # gradB = spinarray.loc.T@gr*1e-2

        # Effective magnetic field given by: 1)rf, 2)gradient,and 3)B1 transmission
        Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
        Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt) # if no B1 map
        Beff_hist[:2,:,:] = Beff_hist[:2,:,:]*spinarray.kappa.reshape(1,num,1) # consider with the B1 transmit map
        Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr*1e-2 + offBeff
        

        # print(spinarray.loc)
        # print(gradB)
        # loss = torch.sum(Beff_hist**2)
        # loss = torch.sum(Beff_hist)
        # print('loss:',loss)
        # Beff_hist.backward(torch.rand_like(Beff_hist))
        # print(gr.grad[:,:8])

        # normalization:
        # Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
        # Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
        # phi_hist = -(dt*2*torch.pi*spinarray.gamma)*Beff_norm_hist.T #(Nt*num)
        # phi_hist = phi_hist.T #(num*Nt)

        # Test of B1 map inclusion:
        # print('Beff',Beff_hist[:,0,:5])
        # print('Beff',Beff_hist[:,1,:5])

        return Beff_hist
    # -----------------------------------------------------------------
    # methods that selected spins
    # -----------------------------------------------------------------
    def get_index_all(self):
        '''return index of all the spins.'''
        return torch.arange(self.num,device=self.device)
    def get_index(self,xlim,ylim,zlim,inside=True):
        '''return spins index (cube)
        
        Args:
            xlim: [xmin,xmax](cm)
            ylim: [ymin,ymax](cm)
            zlim: [zmin,zmax](cm)
        
        Returns:
            index
        '''
        idx_x = (self.loc[0,:]>=xlim[0]) & (self.loc[0,:]<=xlim[1])
        idx_y = (self.loc[1,:]>=ylim[0]) & (self.loc[1,:]<=ylim[1])
        idx_z = (self.loc[2,:]>=zlim[0]) & (self.loc[2,:]<=zlim[1])
        # print(idx_x)
        idx = idx_x & idx_y
        idx = idx & idx_z
        # print(idx)
        idx = torch.nonzero(idx)
        idx = idx.reshape(-1)
        # print(idx)
        if inside==False:
            idxall = self.get_index_all()
            idx = index_subtract(idxall,idx)
        return idx
    def get_index_roi(self,roi,offset=[0,0,0]):
        '''return spin index of an cubic ROI.

        Args:
            roi: e.g., [10,10,10] (cm)
            offset: e.g., [0,0,0] (cm)
        '''
        x,y,z = offset[0],offset[1],offset[2]
        xlim = [-roi[0]/2+x,roi[0]/2+x]
        ylim = [-roi[1]/2+y,roi[1]/2+y]
        zlim = [-roi[2]/2+z,roi[2]/2+z]
        idx = self.get_index(xlim,ylim,zlim)
        return idx
    def get_index_roi_transition(self,roi,transition_width,offset=[0,0,0]):
        '''return spin index of transition region of an ROI 
        
        Args:
            roi: e.g., [10,10,10] (cm)
            offset: e.g., [0,0,0] (cm)
            transition_width: e.g., [1,1,1] (cm)
        '''
        x,y,z = offset[0],offset[1],offset[2]
        xw,yw,zw = transition_width[0]/2,transition_width[1]/2,transition_width[2]/2
        xlim_small = [-roi[0]/2+xw+x,roi[0]/2-xw+x]
        ylim_small = [-roi[1]/2+yw+y,roi[1]/2-yw+y]
        zlim_small = [-roi[2]/2+zw+z,roi[2]/2-zw+z]
        xlim_large = [-roi[0]/2-xw+x,roi[0]/2+xw+x]
        ylim_large = [-roi[1]/2-yw+y,roi[1]/2+yw+y]
        zlim_large = [-roi[2]/2-zw+z,roi[2]/2+zw+z]
        idx_small = self.get_index(xlim_small,ylim_small,zlim_small)
        idx_large = self.get_index(xlim_large,ylim_large,zlim_large)
        idx_transi = index_subtract(idx_large,idx_small)
        return idx_transi
    def get_index_roi_inside(self,roi,transition_width,offset=[0,0,0]):
        '''return spin index of roi without transition region
        
        Args:
            roi: e.g., [10,10,10]
            offset: e.g., [0,0,0]
            transition_width: e.g., [1,1,1]
        '''
        x,y,z = offset[0],offset[1],offset[2]
        xw,yw,zw = transition_width[0]/2,transition_width[1]/2,transition_width[2]/2
        xlim_small = [-roi[0]/2+xw+x,roi[0]/2-xw+x]
        ylim_small = [-roi[1]/2+yw+y,roi[1]/2-yw+y]
        zlim_small = [-roi[2]/2+zw+z,roi[2]/2-zw+z]
        idx_small = self.get_index(xlim_small,ylim_small,zlim_small)
        return idx_small
    def get_index_roi_stopband(self,roi,transition_width,offset=[0,0,0]):
        '''return spin index of stopband region of an ROI'''
        x,y,z = offset[0],offset[1],offset[2]
        xw,yw,zw = transition_width[0]/2,transition_width[1]/2,transition_width[2]/2
        xlim_large = [-roi[0]/2-xw+x,roi[0]/2+xw+x]
        ylim_large = [-roi[1]/2-yw+y,roi[1]/2+yw+y]
        zlim_large = [-roi[2]/2-zw+z,roi[2]/2+zw+z]
        idx_large = self.get_index(xlim_large,ylim_large,zlim_large)
        idx_stopband = index_subtract(self.get_index_all(),idx_large)
        return idx_stopband
    def get_index_circle(self,center=[0.,0.],radius=1.,dir='z'):
        '''return spins index (cylinder)
        
        Args:
            center: [x,y](cm), 
            radius:(cm)
        '''
        dis_squ = (self.loc[0,:]-center[0])**2 + (self.loc[1,:]-center[1])**2
        idx = dis_squ <= radius**2
        idx = torch.nonzero(idx)
        idx = idx.reshape(-1)
        return idx
    def get_index_ball(self,center=[0.,0.,0.],radius=1.,inside=True): #TODO
        '''return spins index (ball)
        
        Args:
            center:(3*)(cm)
            radius:(cm)
        '''
        dis_squ = (self.loc[0,:]-center[0])**2 + (self.loc[1,:]-center[1])**2 + (self.loc[2,:]-center[2])**2
        if inside:
            idx = dis_squ <= radius**2
        else:
            idx = dis_squ > radius**2
        idx = torch.nonzero(idx)
        idx = idx.reshape(-1)
        return idx
    def get_index_ellipsoid(self,center=[0.,0.,0.],abc=[1.,1.,1.],inside=True):
        '''get spins index (ellipsoid)
        https://en.wikipedia.org/wiki/Ellipsoid
        inpute:
            center: (3*)(cm)
            abc: (3*)(cm) ellipsoid parameters
        output:
            index
        '''
        a,b,c = abc[0],abc[1],abc[2]
        dis_squ = (self.loc[0,:]-center[0])**2/(a**2) + (self.loc[1,:]-center[1])**2/(b**2) + (self.loc[2,:]-center[2])**2/(c**2)
        if inside:
            idx = dis_squ <= 1.0
        else:
            idx = dis_squ > 1.0
        idx = torch.nonzero(idx)
        idx = idx.reshape(-1)
        return idx
    # -------------------------------------------------------------------
    # Methods for designing magnetization, rf pulses
    # -------------------------------------------------------------------
    def calculate_target_excited_M(self,flip,phase,roi,roi_offset=[0,0,0]):
        '''Return target magnetization after specified excitation.
        
        Args:
            flip: 0-180 (deg)
            phase: 0-360 (deg)     along which axis the rf is applied
            roi: [x,y,z] (cm)
            roi_offset: [x0,y0,z0] (cm)

        Returns:
            M: (3,num)
        '''
        roi_idx = self.get_index_roi(roi=roi,offset=roi_offset)
        M_exc = Spin.calculate_target_excited_M(flip=flip,phase=phase).to(self.device)
        Mtarget = torch.zeros_like(self.M)
        Mtarget[2,:] = 1.
        Mtarget[0,roi_idx] = M_exc[0]
        Mtarget[1,roi_idx] = M_exc[1]
        Mtarget[2,roi_idx] = M_exc[2]
        return Mtarget
    def calculate_target_excited_Mxy(self,flip,phase,roi,roi_offset=[0,0,0]):
        '''Return target transverse magnetization after desired excitation.

        Args:
            flip: 0-180 (deg)
            phase: 0-180 (deg)    along which axis the rf is applied
            roi: [x,y,z] (cm)
            roi_offset: [x0,y0,z0] (cm)

        Returns:
            M: (1,num)(complex)
        '''
        Mtarget = self.calculate_target_excited_M(flip=flip,phase=phase,roi=roi,roi_offset=roi_offset)
        Mxytarget = self.calculate_Mxy(Mtarget)
        return Mxytarget
    def calculate_target_spindomain_excitation(self,flip,phase,roi,roi_offset=[0,0,0]):
        '''Return target spin domain parameters for describing specified excitation.

        Args:
            flip: (deg)
            phase: (deg) along which axis the rf is applied

        Returns:
            alpha.conj()*beta: (1,num) (complex)
            beta_norm: (1,num) (real)
        '''
        roi_idx = self.get_index_roi(roi=roi,offset=roi_offset)

        alphaconj_beta_ref,beta_magnitude_ref = Spin.calculate_target_spindomain_excitation(
            flip=flip,phase=phase
        )
        alphaconj_beta_ref = alphaconj_beta_ref.to(self.device)
        beta_magnitude_ref = beta_magnitude_ref.to(self.device)

        alphaconj_beta = torch.zeros_like(self.df)*0.0j
        beta_magnitude = torch.zeros_like(self.df)
        
        alphaconj_beta[roi_idx] = alphaconj_beta_ref
        beta_magnitude[roi_idx] = beta_magnitude_ref
        return alphaconj_beta,beta_magnitude
    def calculate_target_spindomain_refocusing(self,phase,roi,roi_offset=[0,0,0]):
        '''Return target spin domain parameters for refocsuing design.

        Args:
            phase: (deg) along which axis the rf is applied

        Returns:
            beta^2: (1,num)(complex)
        '''
        roi_idx = self.get_index_roi(roi=roi,offset=roi_offset)
        betasquare_target = torch.zeros_like(self.df)*0.0j
        betasquare_ref = Spin.calculate_target_spindomain_refocusing(phase).to(self.device)
        betasquare_target[roi_idx] = betasquare_ref
        return betasquare_target
    # ---------------------------------------------------------------------
    # Methods which new object created:
    # ---------------------------------------------------------------------
    def get_spin(self,index=0):
        '''get a new Spin object from the SpinArray'''
        T1 = self.T1[index]
        T2 = self.T2[index]
        df = self.df[index]
        gamma = self.gamma[index]
        kappa = self.kappa[index]
        loc = self.loc[:,index]
        M = self.M[:,index]
        spin = Spin(T1=T1,T2=T2,df=df,gamma=gamma,kappa=kappa,loc=loc,M=M,device=self.device)
        return spin
    def get_spins(self,spin_idx:torch.Tensor,device=torch.device('cpu')) -> 'SpinArray':
        '''Return new spinarray with selected index, as a new SpinArray.

        Args:
            spin_idx: (tensor)
            device: 
        '''
        new_loc = self.loc[:,spin_idx]
        new_T1 = self.T1[spin_idx]
        new_T2 = self.T2[spin_idx]
        new_gamma = self.gamma[spin_idx]
        new_df = self.df[spin_idx]
        new_kappa = self.kappa[spin_idx]
        new_M = self.M[:,spin_idx]
        new_spinarray = SpinArray(
            loc=new_loc,
            T1=new_T1,T2=new_T2,gamma=new_gamma,
            df=new_df,kappa=new_kappa,
            M=new_M,
            device=device)
        return new_spinarray    
    # --------------------------------------------------------
    # Method display self info
    # --------------------------------------------------------
    def show_info(self):
        print('>> '+'SpinArray: (#spin={})'.format(self.num),self.device)
        # print('\tnum of spins:',self.num)
        print('\tname  | unit    | shape | mean | var | min ~ max')
        print('\t------------------------------------------------')	
        def get_display_appendstr(val:torch.Tensor,longformat=False):
            if longformat:
                ds = ' {} | {:.3f} | {:.3f} '.format(
                    list(val.shape), val.mean(), val.var(), 
                )
            else:
                ds = ' {} | {:.3f} | {:.3f} | {:.2f}~{:.2f} |'.format(
                    list(val.shape), val.mean(), val.var(),
                    val.min(), val.max() 
                )
            return ds

        s = '\tgamma | (MHz/T) |' + get_display_appendstr(self.gamma)
        print(s,self.gamma.device)

        s = '\tT1    | (ms)    |' + get_display_appendstr(self.T1)
        print(s,self.T1.device)
        
        s = '\tT2    | (ms)    |' + get_display_appendstr(self.T2)
        print(s,self.T2.device)

        s = '\tdf    | (Hz)    |' + get_display_appendstr(self.df)
        print(s,self.df.device)

        s = '\tkappa | (1)     |' + get_display_appendstr(self.kappa)
        print(s,self.kappa.device)
        
        # location
        s = '\tloc   | (cm)    | {} |'.format(list(self.loc.shape))
        s = s+' ({:.2f},{:.2f},{:.2f}) ~'.format(self.loc[0,:].min(), self.loc[1,:].min(),self.loc[2,:].min())
        s = s+' ({:.2f},{:.2f},{:.2f}) |'.format(self.loc[0,:].max(), self.loc[1,:].max(),self.loc[2,:].max())
        print(s,self.loc.device)

        print('\tM     | ------- | {} |'.format(
            list(self.M.shape)),self.M[:,0].tolist(), self.M.device)
        # print('\tdf(Hz): \tmean={:.4f}, var={:.4f}, {}~{}'.format(self.df.mean(),self.df.var(),
        # 									self.df.min(),self.df.max()))
        # print('\tmask: | {}/{} |'.format(torch.sum(self.mask),self.num))
        # print('\t>>\tas grid shape:',self._if_as_grid())
        # if self._if_as_grid():
        # 	print('\t\tFOV(cm):',self.fov,'| dim:',self.dim)
        print('\t'+''.center(20,'-'))
        return

class SpinGrid(SpinArray):
    '''A group spins located at 3d spatial grid.

    It can be assigned a mask to extract required spins,
    also it should be able to `unmask` the values to its 3d grid for illustration.
    '''
    def __init__(self, fov, dim, fov_offset=[0,0,0],
                 T1=1000, T2=100, B0map=0., B1map=1., 
                 gamma=Gamma, M=[0.,0.,1.], 
                 mask=None,
                 name='spingrid',
                 device=torch.device('cpu')):
        '''Initialize spin grid.

        Args:
            fov: e.g.,[10,10,10]            (cm)      field-of-view
            dim:   [nx,ny,nz]
            T1:    (nx,ny,nz) or one value     (ms)	  3d T1 values 
            T2:    (nx,ny,nz) or one value     (ms)	  3d T2 values
            B0map: (nx,ny,nz) or one value  (Hz)      3d b0 map, off-resonance
            B1map: (nx,ny,nz) or one value  (unit=1)  3d b1 map, transmit B1 scaling factor
            mask:  (nx,ny,nz)
            gamma: (nx,ny,nz) or one value  (MHz/T)	  gyromagnetic ratio
            M:     (3,num)(tensor) 					  magnetization vectors
            
        inherited attribute:
            loc:   (3,num)(tensor)	(cm)		locations 
            T1:    (1,num)(tensor)		(ms)		T1 values 
            T2:    (1,num)(tensor)		(ms)		T2 values
            gamma: (1,num)(tensor)	(MHz/T)		gyromagnetic ratio
            df:    (1,num)(tensor) 	(Hz)		off-resonance
            kappa: (1,num) 						transmit B1 scaling factor
            M: 

        added attribute:
            fov:  [Wx,Wy,Wz]		(cm) 		field-of-view
            dim:  [nx,ny,nz]					dimensions
            mask: (1,num)					    mask of useful spins
        '''
        self.device = device
        self.fov = fov
        self.dim = dim
        self.fov_offset = fov_offset
        # -----------------------------------
        loc =   self._build_locations().reshape(3,-1)
        T1 =    self._try_vectorize_3dtensor(T1)
        T2 =    self._try_vectorize_3dtensor(T2)
        gamma = self._try_vectorize_3dtensor(gamma)
        df =    self._try_vectorize_3dtensor(B0map)
        kappa = self._try_vectorize_3dtensor(B1map)
        super().__init__(loc, T1=T1, T2=T2, gamma=gamma, df=df, kappa=kappa, M=M, name=name, device=device)
        # ------------------------------------------
        # TODO how to handle a mask
        self.mask = mask # if no, let it be None

        # if mask==None: mask = torch.ones(self.dim).to(self.device)
        # self.mask = self._reshape_value_from_3dtensor(mask)

        # additional information
        # self.mask = self.set_mask(mask)
        # self._as_grid = True
    # -------------------------------------------------
    # Methods that get some properties as a 3D matrix
    # -------------------------------------------------
    def _build_locations(self):
        '''return locations of spins based on fov and dim, (3,nx,ny,nz)'''
        x = (torch.arange(self.dim[0])-self.dim[0]//2)*self.fov[0]/self.dim[0]
        y = (torch.arange(self.dim[1])-self.dim[1]//2)*self.fov[1]/self.dim[1]
        z = (torch.arange(self.dim[2])-self.dim[2]//2)*self.fov[2]/self.dim[2]
        x = x + self.fov_offset[0]
        y = y + self.fov_offset[1]
        z = z + self.fov_offset[2]
        # Construct locations for every spin
        # -----------------------------
        loc = torch.zeros((3,self.dim[0],self.dim[1],self.dim[2]))
        for i in range(self.dim[0]):
            loc[0,i,:,:] = x[i]
        for i in range(self.dim[1]):
            loc[1,:,i,:] = y[i]
        for i in range(self.dim[2]):
            loc[2,:,:,i] = z[i]
        return loc.to(self.device)
    def _try_vectorize_3dtensor(self,vals):
        '''vectorized the value matrix if it is tensor, input shape (nx,ny,nz)'''
        if isinstance(vals,torch.Tensor): vals = vals.view(-1)
        return vals
    # -----------------------------------------------
    # consider B0, B1, T1, T2 maps
    def set_B0map(self,B0map:torch.Tensor):
        '''set off-resonance map (df attribute)

        Args:
            B0map: (x,y,z) (Hz)
        '''
        B0map = B0map.contiguous().view(-1) if B0map != None else None
        self.df = B0map.to(self.dtype)
        return
    def set_B1map(self,B1map:torch.Tensor):
        '''set B1 map (kappa attribute) (x,y,z)
        
        Args: 
            B1map: (x,y,z) (scaling factor)
        '''
        B1map = B1map.contiguous().view(-1) if B1map != None else None
        self.kappa = B1map
        return
    def set_T1map(self,T1map:torch.Tensor):
        '''set off-resonance map (df attribute)

        Args:
            T1map: (x,y,z) (ms)
        '''
        T1map = T1map.contiguous().view(-1) if T1map != None else None
        self.T1 = T1map.to(self.dtype)
        return
    def set_T2map(self,T2map:torch.Tensor):
        '''set off-resonance map (df attribute)

        Args:
            T2map: (x,y,z) (ms)
        '''
        T2map = T2map.contiguous().view(-1) if T2map != None else None
        self.T2 = T2map.to(self.dtype)
        return
    # -------------------------------------------
    # consider mask of spins
    def get_mask(self,spinidx) -> torch.Tensor:
        '''return mask (nx,ny,nz)'''
        mask = torch.zeros_like(self.df)
        mask[spinidx] = 1
        mask = mask.reshape(self.dim)
        return mask
    def get_roi_mask(self,roi,offset) -> torch.Tensor:
        '''return ROI mask (nx,ny,nz)'''
        mask = torch.zeros_like(self.df)
        spinidx = self.get_index_roi(roi=roi,offset=offset)
        mask[spinidx] = 1
        mask = mask.reshape(self.dim)
        return mask
    
    def set_mask(self,mask:torch.Tensor):
        '''set mask of spins
        
        Args:
            mask: (x,y,z) (0 or 1)
        '''
        mask = mask.contiguous().view(-1) if mask != None else torch.ones(self.num)
        self.mask = mask.to(self.dtype)
        return
    def get_index_masked(self):
        '''spins selected by mask'''
        idx = self.mask>0
        idx = torch.nonzero(idx)
        return idx
    def get_masked_spins(self,device=torch.device("cpu")) -> SpinArray:
        '''return masked spinarrays'''
        idx = self.get_index_masked()
        return self.get_spins(idx,device=device)
    # --------------------------------------- 
    # Methods that consider the object as a 3D grid
    # --------------------------------------- 
    def match_3dgrid(self,value:torch.Tensor): #TODO
        '''if cube has mask, then match value list to 3D grid'''
        if torch.numel(value)==self.num:
            value3d = value.reshape(self.dim)
        else:
            # assume there it is masked values
            value3d = torch.nan*torch.zeros(self.num).to(value.dtype).to(value.device)
            idx_masked = self.get_index_masked()
            value3d[idx_masked] = value
            value3d = value3d.reshape(self.dim)
        return value3d
    def __map_interpolate_fn(self,ref_map,ref_x,ref_y,ref_z,padding_val=0):
        '''
        interpolate a map to right locations
            refermap: (nx*ny*nz) (numpy array)
            ref_x: (numpy array)
            ref_y: (numpy array)
            ref_z: (numpy array)
            padding_val: if outside the region
        output:
            newmap (num) (tensor)
        '''
        # first check if the target region within the input map
        ref_map = ref_map.cpu().numpy()
        ref_x = ref_x.cpu().numpy()
        ref_y = ref_y.cpu().numpy()
        ref_z = ref_z.cpu().numpy()

        nx,ny,nz = ref_map.shape
        # print(ref_x)
        # print(ref_y)
        # print(ref_z)
        def extend_loc(locs,ref_loc):
            llen = len(ref_loc)
            if locs.min().cpu().numpy() < np.min(ref_loc):
                lshift = 1
            else:
                lshift = 0
            if locs.max().cpu().numpy() > np.max(ref_loc):
                lappend = 1
            else:
                lappend = 0
            newloc = np.zeros(llen+lshift+lappend)
            newloc[lshift:lshift+llen] = ref_loc
            if lshift > 0:
                newloc[0] = locs.min().cpu().numpy() - 0.1
            if lappend > 0:
                newloc[-1] = locs.max().cpu().numpy() + 0.1
            return newloc,lshift,lappend
        ext_x, xshift, xappend = extend_loc(self.loc[0,:],ref_x)
        ext_y, yshift, yappend = extend_loc(self.loc[1,:],ref_y)
        ext_z, zshift, zappend = extend_loc(self.loc[2,:],ref_z)
        
        ext_map = np.ones((nx+xshift+xappend,ny+yshift+yappend,nz+zshift+zappend))*padding_val
        ext_map[xshift:xshift+nx, yshift:yshift+ny, zshift:zshift+nz] = ref_map
        # print(nx,ny,nz)
        # print(ext_x.shape,ext_y.shape,ext_z.shape)
        # print(ext_map.shape)

        if xshift+xappend+yshift+yappend+zshift+zappend > 0:
            print('[warning] padding while interpolation !')


        if True:
        # try:
            # Interpolation function
            interp_fn = interpolate.RegularGridInterpolator((ext_x,ext_y,ext_z),ext_map)
            # interpolations:
            loc = np.array(self.loc.tolist())  # (3*num)
            # print(loc.shape,loc.dtype)
            newmap = interp_fn(loc.T)
            newmap = torch.tensor(newmap, device=self.device)
            newmap = newmap.to(self.loc.dtype)

            # loc,loc_x,loc_y,loc_z = mri.Build_SpinArray_loc(fov=fov,dim=dim)
            # loc_x,loc_y,loc_z = loc_x.numpy(),loc_y.numpy(),loc_z.numpy()
            # # print(max(loc_x),max(loc_y),max(loc_z))
            # loc = loc.numpy().reshape(3,-1).T
            # B0map = interp_fn(loc)
            # B0map = torch.tensor(B0map,device=device).reshape(dim)

            # print('>> interpolate map',newmap.shape)
        # else:
        # except:
        # 	init_fail = True
        # 	print('>> [Error] interpolation fails !!')
        # 	newmap = None
        return newmap
    # ------------------------------------------------
    # Methods for plot
    # ------------------------------------------------
    def _cat_slices(self,valuemap:torch.Tensor)->torch.Tensor:
        '''concate the slices as one image, from (nx,ny,nslice),

        where it also rotate the image

        Args:
            img3d: (tensor)
        '''
        x,y,z = valuemap.shape
        # vmin = torch.min(img3d)
        vmax = torch.max(valuemap)
        # -------------------------------------------
        # make sure prepare enought number of subplots
        row = math.floor(np.sqrt(z))
        if row < 1: row = 1
        col = math.ceil(z/row)
        if row*col < z: col = col + 1
        # -----------------------------------
        # build collected big image
        ext_x = x + 0
        ext_y = y + 0
        im_coll = torch.zeros((row*ext_y,col*ext_x),device=valuemap.device)*vmax
        n = 0
        for i in range(row):
            for j in range(col):
                if n < z:
                    # im_coll[ext_x*i:ext_x*i+x,ext_y*j:ext_y*j+y] = img3d[:,:,n]
                    im_coll[ext_y*i:ext_y*i+y,ext_x*j:ext_x*j+x] = torch.rot90(valuemap[:,:,n])
                    n = n+1
                else:
                    # n = n+1
                    break
        im_coll = im_coll[:-1,:-1]
        return im_coll
    def plot(self,valuemap,vrange=None,title='',cmap='viridis',figname='tmp.png',savefig=False,ax=None): #TODO
        '''Plot function of values that displays as this cube slice-by-slice.

        Args:
            valuemap: (1,num)(tensor)
            vrange: e.g., [0,1]
            figname: 
            savefig:
            ax:
        ''' 
        if ax==None:
            fig,ax = plt.subplots(figsize=(12,9))
            newfig = True
        else:
            newfig = False
        imval = self.match_3dgrid(valuemap)
        nx,ny,nslice = imval.shape
        imval = self._cat_slices(imval)
        imval = imval.cpu().numpy()
        # plot the image
        if vrange==None:
            im = ax.imshow(imval,cmap=cmap)
        else:
            vmin,vmax = vrange[0],vrange[1]
            im = ax.imshow(imval,vmin=vmin,vmax=vmax,cmap=cmap)
        # plot the boundary
        ylen,xlen = imval.shape
        xlen = xlen - 1
        ylen = ylen - 1
        xbd_list = np.arange(0,xlen,nx)
        ybd_list = np.arange(0,ylen,ny)
        boundary_color = 'white'
        for xbd in xbd_list:
            ax.plot([xbd-0.5,xbd-0.5],[-0.5,ylen],color=boundary_color,lw=0.5)
        for ybd in ybd_list:
            ax.plot([-0.5,xlen],[ybd-0.5,ybd-0.5],color=boundary_color,lw=0.5)
        # cax
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size="10%",pad=0.1)
        # fig.colorbar(im,shrink=0.75)
        plt.colorbar(im,cax=cax)
        ax.set_title(title)

        # figure settings ----------
        # ax.axis('equal')
        # ax.axis('off')
        # plt.tight_layout()

        # check save: -------------
        if newfig:
            fig.patch.set_alpha(0.0)
            print('save fig: '+figname)
            plt.savefig(figname)
            plt.close(fig)
        return
    # --------------------------
    # may need another thinking functions
    def _plot_2group(self,map1:torch.tensor,map2:torch.tensor,vrange1=None,vrange2=None,
                  title1='',title2='',suptitle='',figname='tmp.png',savefig=False,ax=None):
        '''plot of two maps together
        
        input:
            map1: (1:num)
            map2: (1*num)
            vrange1:
            vrange2:
            title1:
            title2:
            figname:
            savefig:
        '''
        fig = plt.figure(figsize=(12,6))

        # Plot of map1 >>>>>>>>>>>>>>>>>>>>>
        ax = plt.subplot(1,2,1)
        imval = self.match_3dgrid(map1)
        imval = self._cat_slices(imval)
        imval = imval.cpu().numpy()
        if vrange1==None:
            im = ax.imshow(imval)
        else:
            vmin,vmax = vrange1[0],vrange1[1]
            im = ax.imshow(imval,vmin=vmin,vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)
        # fig.colorbar(im,shrink=0.75)
        fig.colorbar(im,cax=cax)
        ax.set_title(title1)

        # plot of map2 >>>>>>>>>>>>>
        ax = plt.subplot(1,2,2)
        imval = self.match_3dgrid(map2)
        imval = self._cat_slices(imval)
        imval = imval.cpu().numpy()
        if vrange2==None:
            im = ax.imshow(imval)
        else:
            vmin,vmax = vrange2[0],vrange2[1]
            im = ax.imshow(imval,vmin=vmin,vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)
        fig.colorbar(im,cax=cax)
        ax.set_title(title2)

        # super title
        fig.suptitle(suptitle)

        # figure settings ----------
        # ax.axis('equal')
        # fig.patch.set_alpha(0.0)
        # ax.axis('off')
        plt.tight_layout()

        # check save: -------------
        if savefig:
            print('save fig: '+figname)
            plt.savefig(figname)
            plt.close(fig)
        return
    def _plot_3group(self,map1,map2,map3,vrange1=None,vrange2=None,vrange3=None,
                  title1='',title2='',title3='',suptitle='',figname='tmp.png',savefig=False):
        '''plot of 3 group of maps in one figure
        
        input:
            map1:(1*num)
            map2:(1*num)
            map3:(1*num)
            vrange1: [vmin,vmax]
            vrange2: [vmin,vmax]
            vrange3: [vmin,vmax]
            title1:
            title2:
            title3:
            figname:
            savefig:
        '''
        fig = plt.figure(figsize=(12,4))
        # im = ax.imshow(im_coll,vmin=vmin,vmax=vmax,cmap='gray')
        
        # >> Plot of map 1
        ax = plt.subplot(1,3,1)
        imval = self.match_3dgrid(map1)
        imval = self._cat_slices(imval)
        imval = imval.cpu().numpy()
        if vrange1==None:
            im = ax.imshow(imval)
        else:
            vmin,vmax = vrange1[0],vrange1[1]
            im = ax.imshow(imval,vmin=vmin,vmax=vmax)
        # cax = 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)
        plt.colorbar(im,cax=cax)
        ax.set_title(title1)

        # >> Plot of map 2
        ax = plt.subplot(1,3,2)
        imval = torch.angle(map2)
        imval = self.match_3dgrid(imval)
        imval = self._cat_slices(imval)
        imval = imval.cpu().numpy()
        if vrange2==None:
            im = ax.imshow(imval)
        else:
            vmin,vmax = vrange2[0],vrange2[1]
            im = ax.imshow(imval,vmin=vmin,vmax=vmax)
        # cax = 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)
        plt.colorbar(im,cax=cax)
        ax.set_title(title2)

        # >> Plot of map 3
        ax = plt.subplot(1,3,3)
        imval = self.match_3dgrid(map3)
        imval = self._cat_slices(imval)
        imval = imval.cpu().numpy()
        if vrange3==None:
            im = ax.imshow(imval)
        else:
            vmin,vmax = vrange3[0],vrange3[1]
            im = ax.imshow(imval,vmin=vmin,vmax=vmax)
        # cax = 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)
        plt.colorbar(im,cax=cax)
        ax.set_title(title3)

        # 
        fig.suptitle(suptitle)

        # >> Some settings of the figure
        # ax.axis('equal')
        # fig.patch.set_alpha(0.0)
        # ax.axis('off')
        plt.tight_layout()

        # check save: --------------        
        if savefig:
            print('save fig: '+figname)
            plt.savefig(figname)
            plt.close(fig)
        return
    def _plot_4group(self,map1,map2,map3,map4,vrange1=None,vrange2=None,vrange3=None,vrange4=None,
                  title1='',title2='',title3='',title4='',suptitle='',figname='tmp.png',savefig=False):
        '''plot of 3 group of maps in one figure
        
        input:
            map1:(1*num)
            map2:(1*num)
            map3:(1*num)
            vrange1: [vmin,vmax]
            vrange2: [vmin,vmax]
            vrange3: [vmin,vmax]
            title1:
            title2:
            title3:
            figname:
            savefig:
        '''
        # fig = plt.figure(figsize=(12,12))
        fig = plt.figure(figsize=(20,6))
        # im = ax.imshow(im_coll,vmin=vmin,vmax=vmax,cmap='gray')
        vrange_list = [vrange1,vrange2,vrange3,vrange4]
        title_list = [title1,title2,title3,title4]

        for k,valmap in zip(range(4),[map1,map2,map3,map4]):
            ax = plt.subplot(1,4,k+1)
            imval = self.match_3dgrid(valmap)
            imval = self._cat_slices(imval)
            imval = imval.cpu().numpy()
            vrange = vrange_list[k]
            if vrange==None:
                im = ax.imshow(imval)
            else:
                vmin,vmax = vrange[0],vrange[1]
                im = ax.imshow(imval,vmin=vmin,vmax=vmax)
            # cax = 
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right",
                                        size="10%",
                                        pad=0.1)
            plt.colorbar(im,cax=cax)
            ax.set_title(title_list[k])
        
        fig.suptitle(suptitle)

        # >> Some settings of the figure
        # ax.axis('equal')
        # fig.patch.set_alpha(0.0)
        # ax.axis('off')
        plt.tight_layout()

        # check save: --------------
        if savefig:
            print('save fig: '+figname)
            plt.savefig(figname)
            plt.close(fig)
        return
    def plot_magnetization(self,mag=None,Mmagmax=None,figname='tmp.png',savefig=False):
        '''plot magnetization, display slice-by-slice

        input:
            mag: (3*num)(tensor) default using the self.Mag if not specified
            figname: 
            savefig:
        '''
        if mag == None:
            mag = self.Mag

        mxy = mag[0,:] + 1j*mag[1,:]
        mz = mag[2,:]

        if Mmagmax==None:
            vrange1 = None
            vrange3 = None
        else:
            vrange1 = [0,Mmagmax]
            vrange3 = [-Mmagmax,Mmagmax]
        
        self._plot_3group(
            map1=torch.abs(mxy), map2=torch.angle(mxy), map3=mz,
            vrange1=vrange1,vrange3=vrange3,
            title1=r'$|M_{xy}|$', title2=r'${\angle}M_{xy}$', title3=r'$M_z$',
            figname=figname,savefig=savefig
        )
        return
    def plot_complex(self,valuemap,vrange=None,figname='tmp.png',savefig=False):
        '''plot value with complex number, display slice-by-slice

        input:
            valuemap: (1*num)(tensor)(complex)
            vrange: e.g., [0,1]
            figname: 
            savefig:
        ''' 
        magn = torch.abs(valuemap)
        phase = torch.angle(valuemap)
        self._plot_2group(
            magn,phase,title1='magnitude',title2='phase',
            vrange1=vrange,vrange2=None,
            figname=figname,savefig=savefig
        )
        return
    # ---------------------------------
    # Method display some infos
    def show_info(self):
        print('>> '+'SpinArrayGrid: {} (#spin={}) | FOV:'.format(self.name,self.num),self.fov,'| dim:',self.dim)
        # print('\tnum of spins:',self.num)
        print('\tname  | unit    | shape | mean | min ~ max | var |')
        print('\t------------------------------------------------')
        def get_display_appendstr(val:torch.Tensor,longformat=False):
            if longformat:
                ds = ' {} | {:.3f} | {:.3f} '.format(
                    list(val.shape), val.mean(), val.var(), 
                )
            else:
                ds = ' {} | {:.3f} | {:.3f} | {:.2f}~{:.2f} |'.format(
                    list(val.shape), val.mean(), val.var(),
                    val.min(), val.max() 
                )
            return ds


        s = '\tgamma | (MHz/T) |' + get_display_appendstr(self.gamma)
        print(s,self.gamma.dtype)

        s = '\tT1    | (ms)    |' + get_display_appendstr(self.T1)
        print(s,self.T1.dtype)
        
        s = '\tT2    | (ms)    |' + get_display_appendstr(self.T2)
        print(s,self.T2.dtype)

        s = '\tdf    | (Hz)    |' + get_display_appendstr(self.df)
        print(s,self.df.dtype)

        s = '\tkappa | (1)     |' + get_display_appendstr(self.kappa.abs())
        print(s,self.kappa.dtype)

        # s = '\tdf    | (Hz)    | ({}) | {:.3f}~{:.3f} \t|'.format(len(self.df),self.df.min(),self.df.max())
        # s = s+' {:.4f} \t| {:.4f} |'.format(self.df.mean(),self.df.var())
        # print(s,self.df.dtype)

        # s = '\tkappa | (1)     | ({}) | {:.5f}~{:.5f} \t|'.format(len(self.kappa),self.kappa.min(),self.kappa.max())
        # s = s+' {:.4f} \t| {:.4f} |'.format(self.kappa.mean(),self.kappa.var())
        # print(s,self.kappa.dtype)

        # s = '\tkappa | (1) | {} | {} | {} |'.format(len(self.kappa),self.kappa.min(),self.kappa.max())
        # s = s+' {} | {} |'.format(self.kappa.mean(),self.kappa.var())
        # print(s)

        # location
        s = '\tloc   | (cm)    | {} |'.format(list(self.loc.shape))
        s = s+' ({:.2f},{:.2f},{:.2f})~'.format(self.loc[0,:].min(), self.loc[1,:].min(),self.loc[2,:].min())
        s = s+'({:.2f},{:.2f},{:.2f}) |'.format(self.loc[0,:].max(), self.loc[1,:].max(),self.loc[2,:].max())
        print(s,self.loc.dtype)
        # s = '\tloc x | (cm) | {} | {} |'.format(self.loc[0,:].min(), self.loc[0,:].max())
        # print(s)
        # s = '\tloc y | (cm) | {} | {} |'.format(self.loc[1,:].min(), self.loc[1,:].max())
        # print(s)
        # s = '\tloc z | (cm) | {} | {} |'.format(self.loc[2,:].min(), self.loc[2,:].max())
        # print(s)
        # print('\tloc(cm): {:.2f}~{:.2f}, {:.2f}~{:.2f}, {:.2f}~{:.2f}'.format(self.loc[0,:].min(),
        # 										self.loc[0,:].max(),
        # 										self.loc[1,:].min(),self.loc[1,:].max(),
        # 										self.loc[2,:].min(),self.loc[2,:].max()))

        print('\tM     | ------- | {} |'.format(list(self.M.shape)),self.M.mean(dim=1).tolist())
        print('\tdevcice:',self.device)
        print('\t'+''.center(20,'-'))
        return


'''From the following part, provides functions for simulation.'''

# Bloch simulation for spinarray:
# ----------------------------------------------------
def blochsim_(spinarray:SpinArray,Nt,dt,rf:torch.Tensor,gr:torch.Tensor,device=torch.device('cpu'),history=False):
    """Bloch simulation for spins (only forward computation is defined).

    Args:
        spinarray:			SpinArray object
        Nt:
        dt:			(ms)
        rf:(2,num)	(mT)
        gr:(3,num)	(mT/m)
        device:

    need info of spinarray:
        spinarray.loc:
        spinarray.gamma
        spinarray.df
        spinarray.kappa
    
    Returns:
        M: (3,num) or (3,num,Nt+1)[w/ history]
    """
    display_details = False
    num = spinarray.num

    # starttime = time()
    M = spinarray.M #(3,num) record the changing magn state
    
    # variable for recording the all magnetization states
    if history:
        M_hist = torch.zeros((3,num,Nt+1),device=device) # (3,num,Nt+1)
        M_hist[:,:,0] = M

    # M_average_hist = torch.zeros((3,Nt+1),device=device)
    # M_average_hist[:,0] = torch.sum(M,dim=1)

    # parameters relate to relaxation
    # ------------------------------------
    E1 = torch.exp(-dt/spinarray.T1).reshape(1,num) #(1,num)
    E2 = torch.exp(-dt/spinarray.T2).reshape(1,num)
    # change them into matrix
    E = torch.cat((E2,E2,E1),dim=0) #(3,num)
    Es = torch.cat((torch.zeros((1,num)).to(device),torch.zeros((1,num)).to(device),1-E1)) #(3,num)

    # calculate the Beff: (3,num,Nt) (mT)
    # Beff = B1*kappa + B_offres + B_gradient
    # (calculate with equations here)
    # --------------------------
    offBeff = spinarray.df/spinarray.gamma*1e-3 #(1,num)
    offBeff = offBeff.reshape(num,1)
    Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
    Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt)
    Beff_hist[:2,:,:] = Beff_hist[:2,:,:]*spinarray.kappa.reshape(1,num,1) # consider with the B1 transmit map
    Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr*1e-2 + offBeff

    # get the rotation axis (unit norm)
    # and rotation angle:
    Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
    Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
    phi_hist = -Beff_norm_hist*(spinarray.gamma.reshape(num,1))*dt*2*torch.pi #(num*Nt)
    sin_phi_hist = torch.sin(phi_hist)
    cos_phi_hist = torch.cos(phi_hist)

    # print('simulation loop start time:{}'.format(time()-starttime))
    # M_temp = torch.zeros((3,num),device=device)

    # Simulation by time steps
    # using:
    # R@m = cos(phi) m + (1-cos(phi))*u*u^T*m + sin(phi)*cross(u,m)
    for t in range(Nt):
        uCm = torch.linalg.cross(Beff_unit_hist[:,:,t],M,dim=0) #(3*num)
        uuTm = Beff_unit_hist[:,:,t]*torch.sum(Beff_unit_hist[:,:,t]*M,dim=0) #(1*num)
        M_temp = cos_phi_hist[:,t]*M + (1-cos_phi_hist[:,t])*uuTm + sin_phi_hist[:,t]*uCm

        # Relaxation effect
        M = E*M_temp + Es

        # record of all magn states
        if history: M_hist[:,:,t+1] = M

        # M_average_hist[:,t+1] = torch.sum(M,dim=1)

        if False:
            kk = int(Nt/10)
            if t%kk == 0:
                print('->',100*t/Nt,'%')
                # print('', end='')
            # if k%50 == 0:
            # 	print()
    # print('->stopped time:',time()-starttime)
    if history:
        return M_hist
    else:
        return M
class BlochSim_Array(torch.autograd.Function):
    @staticmethod
    def forward(ctx,spinarray,Nt,dt,Beff_unit_hist,phi_hist,device):
        """Bloch simulation for spin arrays

        input:
            spinarray:
            Nt:
            dt:							(ms)
            Beff_unit_hist: (3*num*Nt)	(mT)	effective rotation axis
            phi_hist: (num*Nt)			(rad)	effective rotation angles
            device:	
        output:
            M: 		
        """
        # starttime = time()
        num = spinarray.num
        M = spinarray.M # (3*num)
        M_hist = torch.zeros((3,num,Nt+1),device=device) # (3*num*(Nt+1))
        M_hist[:,:,0] = M

        # location = spinarray.loc #(3*num)
        # df = torch.rand(num,device=device)
        # df = spinarray.df
        # gamma = 42.48

        E1 = torch.exp(-dt/spinarray.T1).reshape(1,num) #(1*num)
        E2 = torch.exp(-dt/spinarray.T2).reshape(1,num)
        # change into matrix
        E = torch.cat((E2,E2,E1),dim=0) #(3*num)
        Es = torch.cat((torch.zeros((1,num),device=device),
            torch.zeros((1,num),device=device),1-E1))

        # calculate the Beff:
        # offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num)
        # offBeff = offBeff.reshape(num,1)
        # Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
        # Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt)
        # Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr + offBeff

        # normalization:
        # Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
        # Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
        # phi_hist = -(dt*2*torch.pi*spinarray.gamma)*Beff_norm_hist.T #(Nt*num)
        # phi_hist = phi_hist.T #(num*Nt)

        sin_phi_hist = torch.sin(phi_hist)
        cos_phi_hist = torch.cos(phi_hist)

        # print('simulation loop start time:{}'.format(time()-starttime))
        # M_temp = torch.zeros((3,num),device=device)
        for t in range(Nt):
            uCm = torch.linalg.cross(Beff_unit_hist[:,:,t],M,dim=0) #(3*num) sin(phi)*ut X mt
            uuTm = Beff_unit_hist[:,:,t]*torch.sum(Beff_unit_hist[:,:,t]*M,dim=0) #(1*num) ut^T*mt
            M_temp = cos_phi_hist[:,t]*M + (1-cos_phi_hist[:,t])*uuTm + sin_phi_hist[:,t]*uCm
            M = E*M_temp + Es # relaxation
            M_hist[:,:,t+1] = M

            if False:
                kk = int(Nt/10)
                # if t%kk == 0:
                # 	print('->',100*t/Nt,'%')
                    # print('', end='')
                if t%50 == 0:
                    print('t =',t)
                    print(M.norm(dim=0))
        # print('->stopped time:',time()-starttime)
        ctx.save_for_backward(E,Beff_unit_hist,phi_hist,M_hist)
        return M
    @staticmethod
    def backward(ctx,grad_output):
        grad_spinarray = grad_Nt = grad_dt = grad_Beff_unit_hist = grad_phi_hist = None
        grad_device = None
        needs_grad = ctx.needs_input_grad
        # print(needs_grad)
        # print(grad_output) #(3*num)
        # print('grad_output',grad_output.shape)

        E,Beff_unit_hist,phi_hist,M_hist = ctx.saved_tensors # M_hist:(3*num*Nt)
        Nt = Beff_unit_hist.shape[-1]
        # print('Nt=',Nt)
        # print(E) # (3*num)
        # print('M_hist',M_hist.shape) # (3*num*(Nt+1))
        # print(M_hist[:,:,-1])

        grad_phi_hist = torch.zeros_like(phi_hist) # (num*Nt)
        grad_Beff_unit_hist = torch.zeros_like(Beff_unit_hist) #(3*num*Nt)
        # print('initial grad tensors:')
        # print(grad_phi_hist.shape)
        # print(grad_Beff_unit_hist.shape)

        sin_phi_hist = torch.sin(phi_hist) # (num,Nt)
        cos_phi_hist = torch.cos(phi_hist) # (num,Nt)
        # print('sin_phi_hist.shape',sin_phi_hist.shape)

        pl_pmtt = grad_output
        for k in range(Nt):
            # currently partial gradient: pl_pmtt: p{L}/p{M_{t+1}}
            # ------------
            t = Nt - k - 1

            pl_pp = E*pl_pmtt # (3*num)
            # print(pl_pp.shape)
            # print(M_hist[:,:,t],sin_phi_hist[:,t])
            uCm = torch.linalg.cross(Beff_unit_hist[:,:,t],M_hist[:,:,t],dim=0) # (3*num)
            uTm = torch.sum(Beff_unit_hist[:,:,t]*M_hist[:,:,t],dim=0) # (1,num)
            # print('uTm:',uTm)
            # print('uCm:',uCm)
            pp_pphi = -M_hist[:,:,t]*sin_phi_hist[:,t] + sin_phi_hist[:,t]*uTm*Beff_unit_hist[:,:,t] + cos_phi_hist[:,t]*uCm
            # print(pp_pphi.shape) # (3*num)
            pl_pphi = torch.sum(pl_pp*pp_pphi, dim=0) #(1*num)
            # print(pl_pphi)
            grad_phi_hist[:,t] = pl_pphi #(num)
            # print(grad_phi_hist[:,-1])
            # print(grad_phi_hist[:,-2])

            uTplpp = torch.sum(Beff_unit_hist[:,:,t]*pl_pp, dim=0)
            mCplpp = torch.linalg.cross(M_hist[:,:,t],pl_pp,dim=0)
            pl_put = (1-cos_phi_hist[:,t])*(M_hist[:,:,t]*uTplpp + uTm*pl_pp) + sin_phi_hist[:,t]*mCplpp
            grad_Beff_unit_hist[:,:,t] = pl_put
            # print(grad_Beff_unit_hist[:,:,-1])
            # print(grad_Beff_unit_hist[:,:,-2])

            # compute the partial gradient w.r.t. the M_{t}:
            # pl/pm{t} = R^T E pl/pm{t+1}
            # R^T * p{l}/p{p}
            # ---------
            # pl_pp = pl_pmtt*E # (3*num)
            # uTplpp = torch.sum(Beff_unit_hist[:,:,t]*pl_pp, dim=0)
            uCplpp = torch.linalg.cross(Beff_unit_hist[:,:,t],pl_pp,dim=0)
            pl_pmt = cos_phi_hist[:,t]*pl_pp + (1-cos_phi_hist[:,t])*Beff_unit_hist[:,:,t]*uTplpp - sin_phi_hist[:,t]*uCplpp
            # print('pl_pmt.shape',pl_pmt.shape)
            # print(pl_pmt)
            pl_pmtt = pl_pmt
            # break
        # print('end test backward\n')
        return grad_spinarray,grad_Nt,grad_dt,grad_Beff_unit_hist,grad_phi_hist,grad_device
blochsim_array = BlochSim_Array.apply
def blochsim(spinarray:SpinArray,Nt,dt,rf:torch.Tensor,gr:torch.Tensor,device=torch.device('cpu'),details=False):
    """Bloch simulation (implemented with auto-diff for more efficient backward).

    The final bloch simulation function, with custom backward
    
    Args:
        rf:(2,Nt)(mT), 
        gr:(3,Nt)(mT/m), 
        dt:(ms)
    
    Returns: 
        Magnetization: (3,num)
    """

    # Some check first
    # make sure the rf and gr have the same length
    # --------------------------------------------
    assert rf.shape[0] == 2
    assert gr.shape[0] == 3
    assert rf.shape[1] == gr.shape[1]


    # compute effective B for all time points:
    # either calculate here, or use a function
    # ------------------------------------------------
    # >> write formula again: (not going to use this in the future)
    calculate_here = False
    if calculate_here:
        num = spinarray.num
        offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num)
        offBeff = offBeff.reshape(num,1)
        Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
        Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt)
        Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr*1e-2 + offBeff
    # >> or just by: (call built functions)
    Beff_hist = spinarray_Beffhist(spinarray,Nt,dt,rf,gr,device=device)


    # compute normalized B and phi, for all time points:
    # --------------------------------------------------
    Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
    Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
    phi_hist = -Beff_norm_hist*(spinarray.gamma.reshape(-1,1))*dt*2*torch.pi #(num*Nt)
    

    # compute the simulation
    # ----------------------
    M = blochsim_array(spinarray,Nt,dt,Beff_unit_hist,phi_hist,device)
    return M


# ################################################################################
def spinarray_Beffhist(spinarray:SpinArray,Nt,dt,rf:torch.tensor,gr:torch.tensor,device=torch.device('cpu')):
    '''Get effective B-field for each spin at every timepoint.

    input:
        spinarray:
        Nt:
        dt:					(ms)
        rf:(2*Nt)			(mT)
        gr:(3*Nt)			(mT/m)
        device:
    needed info of spinarray:
        spinarray.loc: (3*num) (cm)
        spinarray.gamma: (1*num) 
        spinarray.df: (1*num) (Hz)
        spinarray.kappa: (1*num)
    output:
        Beff_hist:(3*num*Nt)	(mT)
    '''
    # starttime = time()
    num = spinarray.num
    # M_hist = torch.zeros((3,num,Nt+1),device=device)
    # M_hist[:,:,0] = 

    # location = spinarray.loc #(3*num)
    # df = torch.rand(num,device=device)
    # df = spinarray.df
    # print(gr.requires_grad)
    # print(gr[:,:8])

    # Test:
    # print('rf',rf[:,:5])
    # print('B1',spinarray.B1map[:2])

    # Calculate the effective magnetic field (mT)
    # Beff = B1*kappa + B_offres + B_grad
    # <r,G> = cm * mT/m * 1e-2 = mT
    # df[Hz] / gamma[MHz/T] * 1e3 = mT
    # -----------------------------------
    
    # Effective magnetic field given by: off-resonance
    offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num) Hz/(MHz/T) = mT
    offBeff = offBeff.reshape(num,1)
    # gradB = spinarray.loc.T@gr*1e-2

    # Effective magnetic field given by: 1)rf, 2)gradient,and 3)B1 transmission
    Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
    Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt) # if no B1 map
    Beff_hist[:2,:,:] = Beff_hist[:2,:,:]*spinarray.kappa.reshape(1,num,1) # consider with the B1 transmit map
    Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr*1e-2 + offBeff
    

    # print(spinarray.loc)
    # print(gradB)
    # loss = torch.sum(Beff_hist**2)
    # loss = torch.sum(Beff_hist)
    # print('loss:',loss)
    # Beff_hist.backward(torch.rand_like(Beff_hist))
    # print(gr.grad[:,:8])

    # normalization:
    # Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
    # Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
    # phi_hist = -(dt*2*torch.pi*spinarray.gamma)*Beff_norm_hist.T #(Nt*num)
    # phi_hist = phi_hist.T #(num*Nt)

    # Test of B1 map inclusion:
    # print('Beff',Beff_hist[:,0,:5])
    # print('Beff',Beff_hist[:,1,:5])

    return Beff_hist
    # ---------------
# spin-domain simulation using complex numbers
def spinorsim_c_singlestep(spinarray:SpinArray,Nt,dt,rf,gr,device=torch.device('cpu'),details=False,history=False):
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
    Beff_hist = spinarray_Beffhist(spinarray,Nt,dt,rf,gr,device=device)

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
def spinorsim_c_seperatestep(spinarray,Nt,dt,rf,gr,device=torch.device('cpu'),details=False,history=False):
    '''Spin-domain simulation for spinarray, compute using complex number.
    Seperate the effect of rf and gradient as a 2-step simulation.
    
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

    # compute for free-precession:
    offBz = spinarray.df/spinarray.gamma*1e-3 #(1*num)
    offBz = offBz.reshape(num,1) #(num*1)
    Bz_hist = spinarray.loc.T@gr*1e-2 + offBz # mT/cm*cm = mT, Hz/(MHz/T) = 1e-3*mT  #(num*Nt)
    # pre_phi = -2*torch.pi*spinarray.gamma*Bz_hist
    # pre_aj_hist = torch.exp(-pre_phi/2)

    # compute for nutation:
    rf_norm_hist = rf.norm(dim=0) #(Nt)
    rf_unit_hist = torch.nn.functional.normalize(rf,dim=0) #(2*Nt)
    # nut_phi_hist = -dt*2*torch.pi*torch.outer(spinarray.gamma, rf_norm_hist) #(num*Nt)
    # nut_aj_hist = torch.cos(nut_phi/2)
    # nut_bj_hist = -(torch.tensor([0.+1.0j],device=device)*rf_unit[0]-rf_unit[1])*torch.sin(nut_phi/2)
    # print(nut_phi_hist.shape)

    # for recording:
    ahist = (1.0+0.0j)*torch.zeros((num,Nt+1),device=device)
    bhist = (1.0+0.0j)*torch.zeros((num,Nt+1),device=device)
    ahist[:,0] = 1.0+0.0j
    bhist[:,0] = 0.0+0.0j
    for t in range(Nt):
        # free precession period:
        Bz = Bz_hist[:,t] #(num)
        pre_phi = -dt*2*torch.pi*spinarray.gamma*Bz #(num)
        aj = torch.exp(-torch.tensor([0.+1.0j],device=device)*pre_phi/2) #(num)
        atmp = aj*ahist[:,t] #(num)
        btmp = (aj.conj())*bhist[:,t] #(num)

        # nutation period:
        # nut_phi = nut_phi_hist[:,t]
        nut_phi = -dt*2*torch.pi*spinarray.gamma*rf_norm_hist[t]*spinarray.kappa
        aj = torch.cos(nut_phi/2) #(num)
        bj = -(torch.tensor([0.+1.0j],device=device)*rf_unit_hist[0,t] - rf_unit_hist[1,t])*torch.sin(nut_phi/2) #(num)
        ahist[:,t+1] = aj*atmp - (bj.conj())*btmp
        bhist[:,t+1] = bj*atmp + (aj.conj())*btmp

    # return the final value
    a = ahist[:,-1]
    b = bhist[:,-1]
    # print('a,b:',a.shape,b.shape)
    return a,b
# spin-domain simulation avoid complex numbers, by using real numbers
def spinorsim_r2_nograd(spinarray,Nt,dt,rf,gr,device=torch.device('cpu'),details=False):
    '''
    treat all as real numbers, but not explict auto-diff
    Beff_hist:(3*num*Nt), phi_hist:(num*Nt)
    out: a_real:(num), a_imag:(num), b_real:(num), b_imag:(num)

    caution: when calling backward, there's a factor in this implementation!
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

    # compute for free-precession:
    offBz = spinarray.df/spinarray.gamma*1e-3 #(1*num)
    offBz = offBz.reshape(num,1) #(num*1)
    Bz_hist = spinarray.loc.T@gr*1e-2 + offBz # mT/cm*cm = mT, Hz/(MHz/T) = 1e-3*mT  #(num*Nt)
    # pre_phi = -2*torch.pi*spinarray.gamma*Bz_hist
    # pre_aj_hist = torch.exp(-pre_phi/2)

    # compute for nutation:
    rf_norm_hist = rf.norm(dim=0) #(Nt)
    rf_unit_hist = torch.nn.functional.normalize(rf,dim=0) #(2*Nt)
    nut_phi_hist = -dt*2*torch.pi*torch.outer(spinarray.gamma, rf_norm_hist) #(num*Nt)
    # nut_phi_hist = spinarray.B1map.reshape(-1,1)*nut_phi_hist
    # nut_aj_hist = torch.cos(nut_phi/2)
    # nut_bj_hist = -(torch.tensor([0.+1.0j],device=device)*rf_unit[0]-rf_unit[1])*torch.sin(nut_phi/2)
    # print(nut_phi_hist.shape)

    # for recording:
    a_real = torch.ones(num,device=device)
    a_imag = torch.zeros(num,device=device)
    b_real = torch.zeros(num,device=device)
    b_imag = torch.zeros(num,device=device)
    a = torch.tensor([1.,0.],device=device)
    b = torch.tensor([0.,0.],device=device)
    for t in range(Nt):
        # ----free precession period:
        Bz = Bz_hist[:,t] #(num)
        pre_phi = -dt*2*torch.pi*spinarray.gamma*Bz #(num)

        aj_real = torch.cos(pre_phi/2)
        aj_imag = -torch.sin(pre_phi/2)
        bj_real,bj_imag = 0.,0.

        atmp_real = aj_real*a_real - aj_imag*a_imag - bj_real*b_real - bj_imag*b_imag
        atmp_imag = aj_real*a_imag + aj_imag*a_real - bj_real*b_imag + bj_imag*b_real
        btmp_real = bj_real*a_real - bj_imag*a_imag + aj_real*b_real + aj_imag*b_imag
        btmp_imag = bj_real*a_imag + bj_imag*a_real + aj_real*b_imag - aj_imag*b_real

        # print(atmp.abs()**2 + btmp.abs()**2)

        # ----nutation period:
        # nut_phi = nut_phi_hist[:,t]
        nut_phi = -dt*2*torch.pi*spinarray.gamma*rf_norm_hist[t]*spinarray.kappa

        # aj = torch.cos(nut_phi/2) #(num)
        # bj = -(torch.tensor([0.+1.0j],device=device)*rf_unit_hist[0,t] - rf_unit_hist[1,t])*torch.sin(nut_phi/2) #(num)
        aj_real = torch.cos(nut_phi/2) #(num)
        aj_imag = 0.0
        bj_real = rf_unit_hist[1,t]*torch.sin(nut_phi/2)
        bj_imag = -rf_unit_hist[0,t]*torch.sin(nut_phi/2)

        anew_real = aj_real*atmp_real - aj_imag*atmp_imag - bj_real*btmp_real - bj_imag*btmp_imag
        anew_imag = aj_real*atmp_imag + aj_imag*atmp_real - bj_real*btmp_imag + bj_imag*btmp_real
        bnew_real = bj_real*atmp_real - bj_imag*atmp_imag + aj_real*btmp_real + aj_imag*btmp_imag
        bnew_imag = bj_real*atmp_imag + bj_imag*atmp_real + aj_real*btmp_imag - aj_imag*btmp_real

        a_real = anew_real
        a_imag = anew_imag
        b_real = bnew_real
        b_imag = bnew_imag
        
        # print(bj.abs())
        # print(aj.abs()**2 + bj.abs()**2)
        # print(ahist[:,t+1].abs()**2 + bhist[:,t+1].abs()**2)

        # print(ahist[:,t+1].abs().max())
        # print(bhist[:,t+1].abs().max())
        # print('-----------')
    # return the final value
    return a_real, a_imag, b_real, b_imag
def spinorsim_r_separatestep(spinarray,Nt,dt,rf,gr,device=torch.device('cpu'),details=False):
    '''Spin-domain simulation of spins. 
    Treat all values as real numbers. 
    Also used for testing computing gradients. 
    Consider effects of rf, gr separately, 2-step simulation. 

    intermediate values:
        Beff_hist:(3*num*Nt), phi_hist:(num*Nt)
    output: 
        a_real:(1*num)
        a_imag:(1*num)
        b_real:(1*num)
        b_imag:(1*num)
    '''
    # if masked:
    # 	spinarray = spinarray.get_unmasked()

    # if want to simulate smaller Nt
    Nt_p = min(rf.shape[1],gr.shape[1])
    if Nt > Nt_p:
        Nt = Nt_p
        if details:
            print('modified Nt to {}'.format(Nt))
    rf = rf[:,:Nt]
    gr = gr[:,:Nt]

    num = spinarray.num

    # test part:
    # gr.requires_grad = True

    # compute the precession: (generated by gradient and B0 map)
    offBz = spinarray.df/spinarray.gamma*1e-3 #(1*num)
    offBz = offBz.reshape(num,1) #(num*1)
    Bz_hist = spinarray.loc.T@gr*1e-2 + offBz # mT/cm*cm = mT, Hz/(MHz/T) = 1e-3*mT  #(num*Nt)
    pre_phi_hist = -dt*2*torch.pi*spinarray.gamma.reshape(num,-1)*Bz_hist #(num*Nt)
    pre_aj_real_hist = torch.cos(pre_phi_hist/2)
    pre_aj_imag_hist = -torch.sin(pre_phi_hist/2)
    # print(pre_phi_hist)

    # compute the nutation: (generated by rf)
    rf_norm_hist = rf.norm(dim=0) #(Nt)
    rf_unit_hist = torch.nn.functional.normalize(rf,dim=0) #(2*Nt)
    nut_phi_hist = -dt*2*torch.pi*torch.outer(spinarray.gamma, rf_norm_hist) #(num*Nt)
    nut_phi_hist = spinarray.kappa.reshape(-1,1)*nut_phi_hist
    nut_aj_real_hist = torch.cos(nut_phi_hist/2)
    nut_bj_real_hist = rf_unit_hist[1,:]*torch.sin(nut_phi_hist/2) #(num*Nt)
    nut_bj_imag_hist = -rf_unit_hist[0,:]*torch.sin(nut_phi_hist/2)

    # test:
    # pre_aj_real_hist.requires_grad = True
    # loss = (pre_aj_real_hist+pre_aj_imag_hist).sum()
    # loss.backward()
    # print(gr.grad[:,:4])


    # for recording:
    a_real = torch.ones(num,device=device)
    a_imag = torch.zeros(num,device=device)
    b_real = torch.zeros(num,device=device)
    b_imag = torch.zeros(num,device=device)
    for t in range(Nt):
        # ----free precession period:
        # Bz = Bz_hist[:,t] #(num)
        # pre_phi = -dt*2*torch.pi*spinarray.gamma*Bz #(num)

        # aj_real = torch.cos(pre_phi/2)
        # aj_imag = -torch.sin(pre_phi/2)
        aj_real = pre_aj_real_hist[:,t]
        aj_imag = pre_aj_imag_hist[:,t]
        bj_real,bj_imag = 0.,0.

        atmp_real = aj_real*a_real - aj_imag*a_imag - bj_real*b_real - bj_imag*b_imag
        atmp_imag = aj_real*a_imag + aj_imag*a_real - bj_real*b_imag + bj_imag*b_real
        btmp_real = bj_real*a_real - bj_imag*a_imag + aj_real*b_real + aj_imag*b_imag
        btmp_imag = bj_real*a_imag + bj_imag*a_real + aj_real*b_imag - aj_imag*b_real

        # print(atmp.abs()**2 + btmp.abs()**2)

        # ----nutation period:
        # nut_phi = nut_phi_hist[:,t]
        # nut_phi = -dt*2*torch.pi*spinarray.gamma*rf_norm_hist[t]

        # aj = torch.cos(nut_phi/2) #(num)
        # bj = -(torch.tensor([0.+1.0j],device=device)*rf_unit_hist[0,t] - rf_unit_hist[1,t])*torch.sin(nut_phi/2) #(num)
        # aj_real = torch.cos(nut_phi/2) #(num)
        # bj_real = rf_unit_hist[1,t]*torch.sin(nut_phi/2)
        # bj_imag = -rf_unit_hist[0,t]*torch.sin(nut_phi/2)
        aj_imag = 0.0
        aj_real = nut_aj_real_hist[:,t]
        bj_real = nut_bj_real_hist[:,t]
        bj_imag = nut_bj_imag_hist[:,t]

        anew_real = aj_real*atmp_real - aj_imag*atmp_imag - bj_real*btmp_real - bj_imag*btmp_imag
        anew_imag = aj_real*atmp_imag + aj_imag*atmp_real - bj_real*btmp_imag + bj_imag*btmp_real
        bnew_real = bj_real*atmp_real - bj_imag*atmp_imag + aj_real*btmp_real + aj_imag*btmp_imag
        bnew_imag = bj_real*atmp_imag + bj_imag*atmp_real + aj_real*btmp_imag - aj_imag*btmp_real

        a_real = anew_real
        a_imag = anew_imag
        b_real = bnew_real
        b_imag = bnew_imag
        
        # print(bj.abs())
        # print(aj.abs()**2 + bj.abs()**2)
        # print(ahist[:,t+1].abs()**2 + bhist[:,t+1].abs()**2)

        # print(ahist[:,t+1].abs().max())
        # print(bhist[:,t+1].abs().max())
        # print('-----------')
    # l = (a_real+a_imag+b_real+b_imag).sum()
    # l.backward()
    # print(pre_aj_real_hist.grad[:,-4:])
    # return the final value
    return a_real, a_imag, b_real, b_imag
def spinorsim_r_singlestep(spinarray:SpinArray,Nt,dt,rf,gr,device=torch.device('cpu'),details=False):
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
    Beff_hist = spinarray_Beffhist(spinarray,Nt,dt,rf,gr,device=device)

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
# Simulation separate stages of rf,gr. With explicit Jacobian.
class Spinorsim_SpinArray_SeparateStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx,spinarray,Nt,dt,
            pre_aj_real_hist,pre_aj_imag_hist,nut_aj_real_hist,
            nut_bj_real_hist,nut_bj_imag_hist,device):
        num = spinarray.num

        # compute for free-precession:
        # offBz = spinarray.df/spinarray.gamma*1e-3 #(1*num)
        # offBz = offBz.reshape(num,1) #(num*1)
        # Bz_hist = spinarray.loc.T@gr*1e-2 + offBz # mT/cm*cm = mT, Hz/(MHz/T) = 1e-3*mT  #(num*Nt)
        # pre_phi_hist = -dt*2*torch.pi*spinarray.gamma.reshape(num,-1)*Bz_hist #(num*Nt)
        # pre_aj_real_hist = torch.cos(pre_phi_hist/2)
        # pre_aj_imag_hist = -torch.sin(pre_phi_hist/2)

        # compute for nutation:
        # rf_norm_hist = rf.norm(dim=0) #(Nt)
        # rf_unit_hist = torch.nn.functional.normalize(rf,dim=0) #(2*Nt)
        # nut_phi_hist = -dt*2*torch.pi*torch.outer(spinarray.gamma, rf_norm_hist) #(num*Nt)
        # nut_aj_real_hist = torch.cos(nut_phi_hist/2)
        # nut_bj_real_hist = rf_unit_hist[1,:]*torch.sin(nut_phi_hist/2) #(num*Nt)
        # nut_bj_imag_hist = -rf_unit_hist[0,:]*torch.sin(nut_phi_hist/2)

        # ab = torch.zeros((4,num),device=device)
        # simulation:
        a_real = torch.ones(num,device=device)
        a_imag = torch.zeros(num,device=device)
        b_real = torch.zeros(num,device=device)
        b_imag = torch.zeros(num,device=device)

        a_real_hist = torch.zeros((2,num,Nt+1),device=device)
        a_imag_hist = torch.zeros((2,num,Nt+1),device=device)
        b_real_hist = torch.zeros((2,num,Nt+1),device=device)
        b_imag_hist = torch.zeros((2,num,Nt+1),device=device)
        a_real_hist[1,:,0] = a_real
        # where 2 means there are 2 separate state in the process

        for t in range(Nt):
            # ----free precession period:
            # Bz = Bz_hist[:,t] #(num)
            # pre_phi = -dt*2*torch.pi*spinarray.gamma*Bz #(num)

            # aj_real = torch.cos(pre_phi/2)
            # aj_imag = -torch.sin(pre_phi/2)
            aj_real = pre_aj_real_hist[:,t]
            aj_imag = pre_aj_imag_hist[:,t]
            bj_real,bj_imag = 0.,0.

            atmp_real = aj_real*a_real - aj_imag*a_imag - bj_real*b_real - bj_imag*b_imag
            atmp_imag = aj_real*a_imag + aj_imag*a_real - bj_real*b_imag + bj_imag*b_real
            btmp_real = bj_real*a_real - bj_imag*a_imag + aj_real*b_real + aj_imag*b_imag
            btmp_imag = bj_real*a_imag + bj_imag*a_real + aj_real*b_imag - aj_imag*b_real
            # atmp_real = aj_real*a_real - aj_imag*a_imag - bj_imag*b_imag
            # atmp_imag = aj_real*a_imag + aj_imag*a_real + bj_imag*b_real
            # btmp_real = bj_real*a_real + aj_real*b_real + aj_imag*b_imag
            # btmp_imag = bj_real*a_imag + aj_real*b_imag - aj_imag*b_real

            # record for backward:
            a_real_hist[0,:,t+1] = atmp_real
            a_imag_hist[0,:,t+1] = atmp_imag
            b_real_hist[0,:,t+1] = btmp_real
            b_imag_hist[0,:,t+1] = btmp_imag

            # print(atmp.abs()**2 + btmp.abs()**2)

            # ----nutation period:
            # nut_phi = nut_phi_hist[:,t]
            # nut_phi = -dt*2*torch.pi*spinarray.gamma*rf_norm_hist[t]

            # aj = torch.cos(nut_phi/2) #(num)
            # bj = -(torch.tensor([0.+1.0j],device=device)*rf_unit_hist[0,t] - rf_unit_hist[1,t])*torch.sin(nut_phi/2) #(num)

            # aj_real = torch.cos(nut_phi/2) #(num)
            aj_imag = 0.0
            # bj_real = rf_unit_hist[1,t]*torch.sin(nut_phi/2)
            # bj_imag = -rf_unit_hist[0,t]*torch.sin(nut_phi/2)
            aj_real = nut_aj_real_hist[:,t]
            bj_real = nut_bj_real_hist[:,t]
            bj_imag = nut_bj_imag_hist[:,t]

            anew_real = aj_real*atmp_real - aj_imag*atmp_imag - bj_real*btmp_real - bj_imag*btmp_imag
            anew_imag = aj_real*atmp_imag + aj_imag*atmp_real - bj_real*btmp_imag + bj_imag*btmp_real
            bnew_real = bj_real*atmp_real - bj_imag*atmp_imag + aj_real*btmp_real + aj_imag*btmp_imag
            bnew_imag = bj_real*atmp_imag + bj_imag*atmp_real + aj_real*btmp_imag - aj_imag*btmp_real
            # anew_real = aj_real*atmp_real - 0 - bj_real*btmp_real - bj_imag*btmp_imag
            # anew_imag = aj_real*atmp_imag + 0 - bj_real*btmp_imag + bj_imag*btmp_real
            # bnew_real = bj_real*atmp_real - bj_imag*atmp_imag + aj_real*btmp_real + 0
            # bnew_imag = bj_real*atmp_imag + bj_imag*atmp_real + aj_real*btmp_imag - 0

            a_real = anew_real
            a_imag = anew_imag
            b_real = bnew_real
            b_imag = bnew_imag

            a_real_hist[1,:,t+1] = a_real
            a_imag_hist[1,:,t+1] = a_imag
            b_real_hist[1,:,t+1] = b_real
            b_imag_hist[1,:,t+1] = b_imag
            
            # print(bj.abs())
            # print(aj.abs()**2 + bj.abs()**2)
            # print(ahist[:,t+1].abs()**2 + bhist[:,t+1].abs()**2)

            # print(ahist[:,t+1].abs().max())
            # print(bhist[:,t+1].abs().max())
            # print('-----------')
        # save variables for backward:
        ctx.save_for_backward(a_real_hist,a_imag_hist,b_real_hist,b_imag_hist,
            pre_aj_real_hist,pre_aj_imag_hist,nut_aj_real_hist,nut_bj_real_hist,nut_bj_imag_hist)
        # return the final value
        return a_real, a_imag, b_real, b_imag
    @staticmethod
    def backward(ctx,outputgrads1,outputgrads2,outputgrads3,outputgrads4):
        grad_spinarray = grad_Nt = grad_dt = grad_pre_aj_real_hist = grad_pre_aj_imag_hist = None
        grad_nut_aj_real_hist = grad_nut_bj_real_hist = grad_nut_bj_imag_hist = None
        grad_device = None
        needs_grad = ctx.needs_input_grad

        a_real_hist,a_imag_hist,b_real_hist,b_imag_hist, pre_aj_real_hist,pre_aj_imag_hist,nut_aj_real_hist,nut_bj_real_hist,nut_bj_imag_hist = ctx.saved_tensors #(num*Nt)
        Nt = pre_aj_real_hist.shape[1]

        grad_pre_aj_real_hist = torch.zeros_like(pre_aj_real_hist)
        grad_pre_aj_imag_hist = torch.zeros_like(pre_aj_imag_hist)
        grad_nut_aj_real_hist = torch.zeros_like(nut_aj_real_hist)
        grad_nut_bj_real_hist = torch.zeros_like(nut_bj_real_hist)
        grad_nut_bj_imag_hist = torch.zeros_like(nut_bj_imag_hist)

        # print('Nt =',Nt)
        # print(a_real_hist.shape)
        # print(type(outputgrads1))

        pl_pareal = outputgrads1
        pl_paimag = outputgrads2
        pl_pbreal = outputgrads3
        pl_pbimag = outputgrads4
        # -------
        # print('pl_pareal:')
        # print(pl_pareal[-4:])
        # print(pl_pareal.shape)

        for k in range(Nt):
            t = Nt - k - 1
            # print(t)
            
            # get previous state alpha and beta:
            ar = a_real_hist[0,:,t]
            ai = a_imag_hist[0,:,t]
            br = b_real_hist[0,:,t]
            bi = b_imag_hist[0,:,t]
            # partial derivative relating to rf:
            pl_pnutajreal = pl_pareal*ar + pl_paimag*ai + pl_pbreal*br + pl_pbimag*bi
            pl_pnutajimag = -pl_pareal*ai + pl_paimag*ar + pl_pbreal*bi - pl_pbimag*br
            pl_pnutbjreal = -pl_pareal*br - pl_paimag*bi + pl_pbreal*ar + pl_pbimag*ai
            pl_pnutbjimag = -pl_pareal*bi + pl_paimag*br - pl_pbreal*ai + pl_pbimag*ar
            # assign for the output gradients:
            grad_nut_aj_real_hist[:,t] = pl_pnutajreal
            grad_nut_bj_real_hist[:,t] = pl_pnutbjreal
            grad_nut_bj_imag_hist[:,t] = pl_pnutbjimag

            # update the inner partial gradients:
            ar = nut_aj_real_hist[:,t]
            ai = 0.0
            br = nut_bj_real_hist[:,t]
            bi = nut_bj_imag_hist[:,t]
            # partial deriative for inner state
            pl_pareal_tmp = ar*pl_pareal + ai*pl_paimag + br*pl_pbreal + bi*pl_pbimag
            pl_paimag_tmp = -ai*pl_pareal + ar*pl_paimag - bi*pl_pbreal + br*pl_pbimag
            pl_pbreal_tmp = -br*pl_pareal + bi*pl_paimag + ar*pl_pbreal - ai*pl_pbimag
            pl_pbimag_tmp = -bi*pl_pareal - br*pl_paimag + ai*pl_pbreal + ar*pl_pbimag

            # partial derivative relating to gradient:
            ar = a_real_hist[1,:,t]
            ai = a_imag_hist[1,:,t]
            br = b_real_hist[1,:,t]
            bi = b_imag_hist[1,:,t]
            pl_ppreajreal = ar*pl_pareal_tmp + ai*pl_paimag_tmp + br*pl_pbreal_tmp + bi*pl_pbimag_tmp
            pl_ppreajimag = -ai*pl_pareal_tmp + ar*pl_paimag_tmp + bi*pl_pbreal_tmp - br*pl_pbimag_tmp
            pl_pprebjreal = -br*pl_pareal_tmp - bi*pl_paimag_tmp + ar*pl_pbreal_tmp + ai*pl_pbimag_tmp
            pl_pprebjimag = -bi*pl_pareal_tmp + br*pl_paimag_tmp - ai*pl_pbreal_tmp + ar*pl_pbimag_tmp
            # assign for the output gradients:
            grad_pre_aj_real_hist[:,t] = pl_ppreajreal
            grad_pre_aj_imag_hist[:,t] = pl_ppreajimag


            # update the inner partial gradients:
            # ---------------
            ar = pre_aj_real_hist[:,t]
            ai = pre_aj_imag_hist[:,t]
            br = 0.0
            bi = 0.0
            pl_pareal = ar*pl_pareal_tmp + ai*pl_paimag_tmp + br*pl_pbreal_tmp + bi*pl_pbimag_tmp
            pl_paimag = -ai*pl_pareal_tmp + ar*pl_paimag_tmp - bi*pl_pbreal_tmp + br*pl_pbimag_tmp
            pl_pbreal = -br*pl_pareal_tmp + bi*pl_paimag_tmp + ar*pl_pbreal_tmp - ai*pl_pbimag_tmp
            pl_pbimag = -bi*pl_pareal_tmp - br*pl_paimag_tmp + ai*pl_pbreal_tmp + ar*pl_pbimag_tmp

        # output the grads:
        return grad_spinarray,grad_Nt,grad_dt,grad_pre_aj_real_hist,grad_pre_aj_imag_hist,grad_nut_aj_real_hist,grad_nut_bj_real_hist,grad_nut_bj_imag_hist,grad_device
spinorsim_spinarray_separatestep = Spinorsim_SpinArray_SeparateStep.apply
def spinorsim_separatestep(spinarray,Nt,dt,rf,gr,device=torch.device('cpu'),details=False):
    '''Spindomain simulation.'''
    num = spinarray.num


    # compute the precession: (generated by gradient and B0 map)
    offBz = spinarray.df/spinarray.gamma*1e-3 #(1*num)
    offBz = offBz.reshape(num,1) #(num*1)
    # print('== test:',offBz.dtype, gr.dtype, spinarray.loc.dtype)
    Bz_hist = spinarray.loc.T@gr*1e-2 + offBz # mT/cm*cm = mT, Hz/(MHz/T) = 1e-3*mT  #(num*Nt)
    pre_phi_hist = -dt*2*torch.pi*spinarray.gamma.reshape(num,-1)*Bz_hist #(num*Nt)
    pre_aj_real_hist = torch.cos(pre_phi_hist/2)
    pre_aj_imag_hist = -torch.sin(pre_phi_hist/2)
    # print(pre_phi_hist)

    # compute the nutation: (generated by rf)
    rf_norm_hist = rf.norm(dim=0) #(Nt)
    rf_unit_hist = torch.nn.functional.normalize(rf,dim=0) #(2*Nt)
    nut_phi_hist = -dt*2*torch.pi*torch.outer(spinarray.gamma, rf_norm_hist) #(num*Nt)
    nut_phi_hist = spinarray.kappa.reshape(-1,1)*nut_phi_hist
    nut_aj_real_hist = torch.cos(nut_phi_hist/2)
    nut_bj_real_hist = rf_unit_hist[1,:]*torch.sin(nut_phi_hist/2) #(num*Nt)
    nut_bj_imag_hist = -rf_unit_hist[0,:]*torch.sin(nut_phi_hist/2)


    # simulation:
    a_real,a_imag,b_real,b_imag = spinorsim_spinarray_separatestep(spinarray,Nt,dt,
        pre_aj_real_hist,pre_aj_imag_hist,nut_aj_real_hist,nut_bj_real_hist,nut_bj_imag_hist,device)

    return a_real,a_imag,b_real,b_imag

# ---------------------------------------------------
# Simulation use effective field from rf,gr. With explicit Jacobian.
class Spinorsim_SpinArray_SingleStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, spinarray,Nt,dt,
             aj_real_hist,aj_imag_hist,bj_real_hist,bj_imag_hist,
             device):
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
    def backward(ctx: torch.Any, outputgrads1, outputgrads2, outputgrads3, outputgrads4):
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
spinorsim_spinarray_singlestep = Spinorsim_SpinArray_SingleStep.apply
def spinorsim_singlestep(spinarray,Nt,dt,rf,gr,device=torch.device('cpu')):
    """Spin-domain simulation. With explicit Jacobian."""
    num = spinarray.num

    # Compute effective magnetic field
    # Beff_hist = spinarray_Beffhist(spinarray,Nt,dt,rf,gr,device=device)

    offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num) Hz/(MHz/T) = mT
    offBeff = offBeff.reshape(num,1)

    # Effective magnetic field given by: 1)rf, 2)gradient,and 3)B1 transmission
    Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
    Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt) # if no B1 map
    Beff_hist[:2,:,:] = Beff_hist[:2,:,:]*spinarray.kappa.reshape(1,num,1) # consider with the B1 transmit map
    Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr*1e-2 + offBeff


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
    a_real,a_imag,b_real,b_imag = spinorsim_spinarray_singlestep(spinarray,Nt,dt,
        aj_real_hist,aj_imag_hist,bj_real_hist,bj_imag_hist,device)
    return a_real, a_imag, b_real, b_imag

spinorsim = spinorsim_singlestep
spinorsim_c = spinorsim_c_singlestep

