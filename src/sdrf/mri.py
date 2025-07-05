'''Basic functions and definitions. 

Includes:
- Spin
- SpinArray
- SpinGrid
- Pulse

Info
- Author: jiayao
- Update date:   7/3/2025

Acknowledgement
this work has inspired and take reference of: 
- https://github.com/tianrluo/AutoDiffPulses
- https://github.com/tianrluo/MRphy.py
'''

import os
import math
import numpy as np
import torch
# from time import time
# import warnings

# packages for plot 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.io as spio
from scipy import interpolate
# from scipy.interpolate import interp1d



# -------------------------------------------------------------
# Define some constants
# -------------------------------------------------------------
Gamma = 42.48 # MHz/T (gyromagnetic ratio) (normalized by 2*pi)
# -------------------------------------------------------------


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
    def __init__(self,dt=1.0,rf=None,gr=None,device=torch.device("cpu"),name='pulse',rf_definition='real-imag'):
        '''Initialize of an instance of pulse.

        Args:
            dt:        (ms)
            rf: (2,Nt) (mT)     [real; imaginary](mT)(torch.tensor) or (amplitude, frequency)[mT, Hz]
            gr: (3,Nt) (mT/m)   three x-y-z channels (torch.tensor)
            device:             cpu or cuda
            name:               name for identification
            rf_definition:      real-imag, amp-freq

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
        
        # if rf==None:
        #     Nt = gr.shape[1]
        #     rf = torch.zeros(2,Nt)
        # elif gr==None:
        #     Nt = rf.shape[1]
        #     gr = torch.zeros(3,Nt)
        # else:
        #     assert rf.shape[1] == gr.shape[1]
        #     Nt = rf.shape[1]

        if rf_definition == 'real-imag':
            # determine the num of time points
            Nt_rf = 0 if (rf==None) else gr.shape[1]
            Nt_gr = 0 if (gr==None) else rf.shape[1]
            Nt = max(Nt_gr,Nt_rf)
            if Nt==0: raise BaseException

            self.rf_amp = None
            self.rf_freq = None

        elif rf_definition == 'amp-freq':
            Nt = len(rf[0])
            self.rf_amp = rf[0].to(self.dtype).to(self.device)
            self.rf_freq = rf[1].to(self.dtype).to(self.device)
            rf = self.calculate_rf_from_amp_freq(self.rf_amp, self.rf_freq, dt)

            # print(self.rf_amp.shape)
            # print(self.rf_freq.shape)
            
        else:
            raise BaseException
        
        if gr==None: gr = torch.zeros(3,Nt)
        if rf==None: rf = torch.zeros(2,Nt)

        self.rf = rf.to(self.dtype).to(self.device)
        self.gr = gr.to(self.dtype).to(self.device)
        self.dt = dt
        self.Nt = Nt
        
        self._rf_definition = rf_definition
        self._rf_unit = 'mT'
        self._gr_unit = 'mT/m'
        self._dt_unit = 'ms'
        
        self.sys_rfmax = float('Inf')         # system peak rf
        self.sys_gmax = float('Inf')          # system max gradient
        self.sys_slewratemax = float('Inf')   # system max slew-rate
        self.pnsTH_fn = lambda t: None

    # -------------------------------------------------------------
    
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
        excitation: $$k(t) = - gamma int_t^T G(s) ds$$
        imaging: $$k(t) = gamma int_0^t G(s) ds$$

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
    def rf_magnitude(self):
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
                $$k(t) propto gamma int G(t) dt$$

        input:
            gamma:(MHz/T)
        var used:
            self.gr:(3,Nt)(mT/m), 
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
    
    # ---------------------------------------------------------------
    # Methods make changes/operations to the pulse 
    # ---------------------------------------------------------------
    def change_dt(self,newdt):
        '''change of time resolution, (ms)
        
        Args:
            newdt: (ms) new time-step

        use method = 'linear', 'nearest'
        '''
        self._rf_definition = 'real-imag'
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
    
            
    # ---------------------------------------------------------------
    # Methods of plot function:
    # ---------------------------------------------
    
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
    def info(self):
        s = 'Pulse: {} | {} | dur={}ms, #pts={}, dt={}ms '.format(
            self.name,self.device,self.duration, self.Nt, self.dt) \
            + '| ({})'.format(self._rf_definition)
        print(s)
        return
    # ------------------------------
    # provided examples
    # ------------------------------
    @staticmethod
    def example_sinc_pulse(device=torch.device('cpu')):
        '''return naive sinc-shape pulse example for testing purpose'''
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
        pulse = Pulse(rf=rf,gr=gr,dt=dt,device=device,name='example-sinc')
        return pulse
    @staticmethod
    def example_rand_pulse(device=torch.device('cpu')):
        '''example'''
        pulse = Pulse(
            dt = 1e-3,
            rf = torch.rand(2,20),
            gr = torch.rand(3,20),
            device=device
        )
        return pulse



class Spin:
    def __init__(self,T1=1000.0,T2=100.0,df=0.0,kappa=1.0,gamma=Gamma,loc=[0.,0.,0.],M=[0.,0.,1.],name='spin',device=torch.device("cpu")):
        """Initialize of a spin.

        Args:
            T1: 		(ms)
            T2: 		(ms)
            df: 		(Hz)			off-resonance
            kappa:						B1 transmit factor
            gamma: 		(MHz/Tesla) 	(gyromagnetic ratio normalized by 2*pi) 
            M: (1,3) or None			manetization vector (Tensor)
            loc: (1,3)	(cm) 			spatial location
            name: 
            device: 
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
        '''loc: (1,3), list or tensor'''
        # if not isinstance(loc,torch.Tensor)
        if isinstance(loc,np.ndarray):
            loc = torch.from_numpy(loc)
        else:
            loc = torch.tensor(loc)
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
    # Methods
    # --------------------------------------------------------------
    def get_Beffective(self,rf:torch.Tensor,gr:torch.Tensor) -> torch.Tensor: 
        '''Return the effective magnetic field for this spin. (mT)
        
        Args:
            rf: (2,Nt), (mT)
            gr: (3,Nt), (mT/m)
        '''
        device = rf.device
        Nt     = rf.shape[1]

        # Calculate the effective magnetic field [mT]
        # Beff = B1*kappa + B_offres + B_gradient
        # <r,G> = cm * mT/m * 1e-2 = mT
        # df[Hz] / gamma[MHz/T] * 1e-3 = mT
        # --------------------------------------------
        Beff = torch.zeros((3,Nt),device=device)  # mT
        Beff[:2,:] = rf*self.kappa                # mT (rf pulse, w/ transmission factor)
        Beff[2,:]  = self.loc@gr*1e-2 + self.df*1e-3/self.gamma
        return Beff
    
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
    def calculate_target_spindomain_excitation_alphabeta(flip,phase) -> tuple[torch.Tensor,torch.Tensor]:
        '''Return the spin-domain parameters (complex) for excitation.
        
        Args:
            flip: 0-180         (deg)
            phase: 0-360        (deg) along which direction the rf is applied

        Returns:
            alpha:              (complex)
            beta:               (complex)
        '''
        theta = torch.tensor([flip])/180*torch.pi
        phi = torch.tensor([phase])/180*torch.pi
        # 
        alpha = torch.cos(theta/2)*(1.+0.j)
        beta = 1j*torch.exp(1j*phi)*torch.sin(theta/2)
        return alpha,beta
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
            name: 
            device:                             torch.device
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
    def set_B0(self,b0):
        '''set df (Hz)'''
        self.b0 = b0
        return
    def set_B1(self,b1):
        '''set kappa'''
        self.kappa = b1
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_Mxy(M:torch.Tensor)->torch.Tensor:
        '''Calculate transverse magnetization (complex). -> (1,num)
        
        Args:
            M:(3,num)
        '''
        return M[0]+1j*M[1]
    @staticmethod
    def calculate_flipangles(M:torch.Tensor)->torch.Tensor:
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
    def get_Beffective(self,Nt,rf:torch.Tensor,gr:torch.Tensor,device=torch.device('cpu')) -> torch.Tensor:
        '''Return effective B-field for each spin. -> (3, #spins, Nt)
        
        Args:
            Nt: 
            rf: (mT)
            gr: (mT/m)
        '''
        num = self.num

        # Calculate the effective magnetic field (mT)
        # Beff = B1*kappa + B_offres + B_grad
        # <r,G> = cm * mT/m * 1e-2 = mT
        # df[Hz] / gamma[MHz/T] * 1e3 = mT
        # -----------------------------------
        # Effective magnetic field given by: off-resonance
        offBeff = self.df/self.gamma*1e-3  #(1*num) Hz/(MHz/T) = mT
        offBeff = offBeff.reshape(num,1)

        # gradB = spinarray.loc.T@gr*1e-2

        # Effective magnetic field given by: 1)rf, 2)gradient,and 3)B1 transmission
        Beff = torch.zeros((3,num,Nt),device=device)*1.0
        Beff[:2,:,:] = Beff[:2,:,:] + rf.reshape(2,1,Nt)          # B1
        Beff[:2,:,:] = Beff[:2,:,:]*self.kappa.reshape(1,num,1)   # w/ B1 transmit map
        Beff[2,:,:] = Beff[2,:,:] + self.loc.T@gr*1e-2 + offBeff  # gradients, off-resonance effect

        return Beff
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
            alpha: (1,num) (complex but analytically real)
            beta:  (1,num) (compelx)
        '''
        roi_idx = self.get_index_roi(roi=roi,offset=roi_offset)

        # reference alpha, beta in the ROI
        alpha_ref,beta_ref = Spin.calculate_target_spindomain_excitation_alphabeta(
            flip=flip,phase=phase
        )
        alpha_ref = alpha_ref.to(self.device)
        beta_ref = beta_ref.to(self.device)

        # build parameters for the group of spins
        alpha_tar = torch.zeros_like(self.df)*0.j + 1.
        beta_tar = torch.zeros_like(self.df)*0.0j

        alpha_tar[roi_idx] = alpha_ref
        beta_tar[roi_idx] = beta_ref
        return alpha_tar,beta_tar
    def calculate_target_spindomain_excitation_v2(self,flip,phase,roi,roi_offset=[0,0,0]):
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
    def info(self):
        print('SpinArray: ')


    # ----------------------------------------------
    # provided examples
    # ----------------------------------------------
    @staticmethod
    def example_spins(device=torch.device('cpu')):
        '''example'''
        # print_name('Example of spin array:')
        n = 5
        loc = torch.rand((3,n),device=device)
        loc[0,:] = torch.tensor([0.,1.,0.,3.,4.],device=device)
        loc[1,:] = torch.tensor([0.,0.,0.,0.,0.],device=device)
        loc[2,:] = torch.tensor([0.,0.,0.,1.,0.],device=device)
        T1 = torch.ones(n,device=device)*1000.0
        T2 = torch.ones(n,device=device)*100.0
        df = torch.tensor([10.,5.,0.,0.,0.],device=device)

        spinarray = SpinArray(loc=loc,T1=T1,T2=T2,df=df,device=device)
        # spinarray.df = torch.tensor([10.,5.,0.,0.,0.],device=device)
        # spinarray.show_info()
        return spinarray



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
        
        if isinstance(valuemap,np.ndarray): valuemap = torch.from_numpy(valuemap)
        if len(valuemap.shape)==3:
            imval = valuemap
        else:
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
        ax.set_axis_off()

        # figure settings ----------
        # ax.axis('equal')
        # ax.axis('off')
        # plt.tight_layout()

        # check save: -------------
        if newfig:
            if savefig:
                fig.patch.set_alpha(0.0)
                print('save fig: '+figname)
                plt.savefig(figname)
                plt.close(fig)
            else:
                plt.show()
        return
    # ---------------------------------
    # Method display some infos
    def info(self):
        print('SpinArrayGrid: #spins={} | {}'.format(self.num, self.device))
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



    # ---------------------
    # Things may not needed
    # ---------------------
    @staticmethod
    def example_cube(device=torch.device('cpu')):
        # print_name('Example of a cube:')
        fov = [4,4,2] # cm
        dim = [3,3,5]
        cube = SpinGrid(fov=fov,dim=dim,device=device)
        return cube

