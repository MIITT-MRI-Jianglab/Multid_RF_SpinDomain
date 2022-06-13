# do bloch simulation
# author: jiayao

from importlib.metadata import MetadataPathFinder
from math import gamma
from pyclbr import Function
from re import M
# from turtle import end_fill
from matplotlib import lines
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time




# import matplotlib
# matplotlib.use('TkAgg')

# import mri_sim_pkone

print()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print('>> mri: using device:',device)


# some basic information
# ----------------------------------
def MR():
	ss = ['1 Gauss = 0.1 mT', 'H1: gamma = 4.26kHz/G = 42.48 MHz/T']
	ss.append('0.25 G = 0.025 mT')
	def table_print(x):
		print('\t'+x)
	print(''.center(50,'-'))
	for s in ss:
		table_print(s)
	# gamma = 42.48 #MHz/T
	# B0 = 3.0 #T
	# B1 = 0.02 # mT
	# omega0 = gamma*B1*2*torch.pi
	print(''.center(50,'-'))
	print('\tmri module:\n\trf: (mT), gr: (mT/cm)\n\ttime or dt: (ms)')
	print(''.center(50,'-'))
	return
	# ------------

# define rotation matrices
class RotationMatrix:
	def __init__(self):
		return
	def zrot(self,phi):
		'''from x to y'''
		Rz = np.array([[np.cos(phi),-np.sin(phi),0.],
						[np.sin(phi),np.cos(phi),0.],
						[0.,0.,1.]])
		return Rz
	def xrot(self,phi):
		'''from x to z'''
		Rx = np.array([[1.,0.,0.],
						[0.,np.cos(phi),-np.sin(phi)],
						[0.,np.sin(phi),np.cos(phi)]])
		return Rx
	def yrot(self,phi):
		'''from z to x'''
		Ry = np.array([[np.cos(phi),0.,np.sin(phi)],
						[0.,1.,0.],
						[-np.sin(phi),0.,np.cos(phi)]])
		return Ry
	def throt(self,phi,theta):
		'''todo'''
		Rz = self.zrot(-theta)
		Rx = self.xrot(phi)
		Rth = np.linalg.inv(Rz)@Rx@Rz
		return Rth
	def example(self):
		phi = 45/180*np.pi
		Rz = self.zrot(phi)
		Rx = self.xrot(phi)
		Ry = self.yrot(phi)
		Throt = self.throt(phi,phi)
		m = np.array([0,0,1])
		plt.figure()
		ax = plt.figure().add_subplot(projection='3d')
		ax.plot((0,m[0]),(0,m[1]),(0,m[2]),label='magnetization')
		p = Rz@m
		ax.plot((0,p[0]),(0,p[1]),(0,p[2]),linewidth=1,linestyle='--')
		ax.text(p[0],p[1],p[2],r'$R_z$',fontsize=8)
		p = Rx@m
		ax.plot((0,p[0]),(0,p[1]),(0,p[2]),linewidth=1,linestyle='--')
		ax.text(p[0],p[1],p[2],r'$R_x$',fontsize=8)
		p = Ry@m
		ax.plot((0,p[0]),(0,p[1]),(0,p[2]),linewidth=1,linestyle='--')
		ax.text(p[0],p[1],p[2],r'$R_y$',fontsize=8)
		p = Throt@m
		ax.plot((0,p[0]),(0,p[1]),(0,p[2]),linewidth=1,linestyle='--')
		ax.text(p[0],p[1],p[2],r'$R_th$',fontsize=8)
		# 
		ax.legend()
		ax.set_xlim(-1.1,1.1)
		ax.set_ylim(-1.1,1.1)
		ax.set_zlim(-1.1,1.1)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		picname = 'pictures/test_rotations.png'
		print('save fig...'+picname)
		plt.savefig(picname)
		# plt.show()
		return


class Pulse:
	def __init__(self,rf=None,gr=None,dt=1.0) -> None:
		self.Nt = 10 # time points
		self.dt = dt # ms
		self._set_rf(rf)
		self._set_gr(gr)
		self._set_Nt()
		pass
	def _set_rf(self,rf):
		if rf==None:
			self.rf = torch.zeros((2,self.Nt),device=device) # mT
		else:
			self.rf = rf
		return
	def _set_gr(self,gr):
		if gr == None:
			self.gr = torch.zeros((3,self.Nt),device=device) # mT/cm
		else:
			self.gr = gr
		return
	def _set_Nt(self):
		if self.rf.shape[1] == self.gr.shape[1]:
			self.Nt = self.rf.shape[1]
		else:
			print('length gr != rf')
		return
	def show_info(self):
		print('>> Pulse:')
		print('\tduration={}ms, time points={}, dt={}ms'.format(self.dt*self.Nt,self.Nt,self.dt))
		print('\trf:',self.rf.shape,'gr:',self.gr.shape)
		print('\t'+''.center(20,'-'))
		return
def example_pulse():
	Nt = 400 #100
	dt = 0.01 # ms
	t0 = 100 # time parameter for a sinc pulse
	rf = 0.0*torch.zeros((2,Nt),device=device)
	rf[0,:] = 6*1e-3*torch.special.sinc((torch.arange(Nt,device=device)-200)/t0)
	gr = 0.0*torch.zeros((3,Nt),device=device) # mT/cm
	gr[2,:] = 0.1*torch.ones(Nt,device=device) # mT/cm
	pulse = Pulse(rf=rf,gr=gr,dt=dt)
	pulse.show_info()
	return pulse

# define a spin class
# --------------------------------------
"""properties:
	location: x,y,z
	gamma, 
	T1, T2, 
	B0, B1
"""
class Spin:
	# gamma = 42.48 # MHz/Tesla
	# T1 = 1000.0 # ms
	# T2 = 100.0 # ms
	# x,y,z = 0.,0.,0. # (cm)
	def __init__(self,T1=1000.0,T2=100.0,df=0.0,gamma=42.48,loc=torch.tensor([0.,0.,0.]),
				mag=torch.tensor([0.,0.,1.],device=device)):
		self.T1 = T1
		self.T2 = T2
		self.df = df # off-resonance, (Hz)
		self.gamma = gamma #MHz/Tesla
		# self.loc = torch.tensor([0.,0.,0.],device=device)
		self.set_position(loc[0],loc[1],loc[2])
		self.set_Mag(mag)
	def set_position(self,x,y,z):
		self.x = x
		self.y = y
		self.z = z
		# self.loc = 1.*torch.tensor([x,y,z],device=device)
		return
	def set_Mag(self,M):
		self.Mag = torch.empty(3,device=device)
		M_norm = torch.norm(M)
		if M_norm == 0:
			print('error')
		else:
			M = M/M_norm
		self.Mag[0] = M[0]
		self.Mag[1] = M[1]
		self.Mag[2] = M[2]
		return
	def get_loc(self):
		p = torch.zeros(3,device=device)
		p[0] = self.x
		p[1] = self.y
		p[2] = self.z
		return p
	def show_info(self):
		print('>> Spin:')
		print('\tlocation(cm):',self.get_loc(),', df:',self.df,'Hz')
		print('\tT1:',self.T1,'ms, T2:',self.T2,'ms, gamma:',self.gamma,'MHz/Tesla')
		print('\tMag:',self.Mag)
		print('\t'+''.center(20,'-'))
		return

# define a spin array
# ------------------------------------
class SpinArray:
	def __init__(self,loc,T1,T2,gamma=42.48,M=None,df=0.0):
		self.loc = loc # (3*num)(cm)
		self.num = loc.shape[1]
		self.T1 = T1 # (n)(ms)
		self.T2 = T2 # (n)(ms)
		self.gamma = torch.ones(self.num,device=device)*gamma # MHz/Tesla
		# self.Mag = torch.zeros((3,num),device=device)
		self.set_Mag(M)
		# self.gamma = torch.ones(num,device=device)*gamma
		# self.loc = torch.zeros((3,num),device=device)
		self.df = df*torch.ones(self.num,device=device)
	def set_Mag(self,M=None):
		if M==None:
			self.Mag = torch.zeros((3,self.num),device=device)
			self.Mag[2,:] = 1.0
		else:
			self.Mag = M
		return
	def add_spin(self,spin): #[TODO]
		pass
	def get_index(self,xlim,ylim,zlim):
		idx_x = (self.loc[0,:]>=xlim[0]) & (self.loc[0,:]<=xlim[1])
		idx_y = (self.loc[1,:]>=ylim[0]) & (self.loc[1,:]<=ylim[1])
		idx_z = (self.loc[2,:]>=zlim[0]) & (self.loc[2,:]<=zlim[1])
		# print(idx_x)
		idx = idx_x & idx_y
		idx = idx & idx_z
		idx = torch.nonzero(idx)
		idx = idx.reshape(-1)
		# print(idx)
		return idx
	def get_spins(self,xlim,ylim,zlim):
		idx = self.get_index(xlim,ylim,zlim)
		new_loc = self.loc[:,idx]
		new_T1 = self.T1[idx]
		new_T2 = self.T2[idx]
		new_gamma = self.gamma[idx]
		new_Mag = self.Mag[:,idx]
		new_df = self.df[idx]
		new_spinarray = SpinArray(new_loc,new_T1,new_T2,new_gamma,new_Mag,new_df)
		return new_spinarray
	def get_spin(self,index=0):
		T1 = self.T1[index]
		T2 = self.T2[index]
		df = self.df[index]
		gamma = self.gamma[index]
		spin = Spin(T1=T1,T2=T2,df=df,gamma=gamma)
		spin.set_position()
	def show_info(self):
		print('>> '+'SpinArray:')
		print('\tnum:',self.num,', loc(cm):',self.loc.shape,', df(Hz):',self.df.shape)
		print('\tT1(ms):',self.T1.shape,',',self.T1.view(-1)[0])
		print('\tT2(ms):',self.T2.shape,',',self.T2.view(-1)[0])
		# print('\tgamma(MHz/T):',self.gamma.shape,',',self.gamma[0])
		print('\tMag:',self.Mag.shape,',',self.Mag[:,0])
		# print('\tdf:',self.df)
		# print('\tMag:',self.Mag)
		print('\t'+''.center(20,'-'))
		return

# class SpinCube(SpinArray):
# 	def __init__(self, loc, T1, T2, gamma=42.48, M=None, df=0):
# 		super().__init__(loc, T1, T2, gamma, M, df)

# function: build a spin array
# ---------------------------------------------------
def Build_SpinArray(fov=[4,4,4],dim=[3,3,3]):
	# fov = [4,4,4] # cm
	# dim = [2,2,3] # numbers
	num = dim[0]*dim[1]*dim[2]
	def linespace(L,n):
		if n == 1:
			p = torch.tensor([0.],device=device)
		elif n%2 == 1:
			p = torch.linspace(-L/2,L/2,n,device=device)
		else:
			p = torch.linspace(-L/2,L/2,n+1,device=device)[1:]
		return p
	# a test
	# for i in range(4):
	# 	print(linespace(6,i))

	x = linespace(fov[0],dim[0])
	y = linespace(fov[1],dim[1])
	z = linespace(fov[2],dim[2])
	# print(x)
	# print(y)
	# print(z)
	loc = torch.zeros(3,num,device=device)
	i = 0
	for ix in range(dim[0]):
		for iy in range(dim[1]):
			for iz in range(dim[2]):
				loc[0,i] = x[ix]
				loc[1,i] = y[iy]
				loc[2,i] = z[iz]
				i += 1	
	# print(loc)

	T1 = torch.ones(num,device=device)*1000.0 #1470 # ms
	T2 = torch.ones(num,device=device)*100.0 #70 # ms
	# df = torch.ones(num,device=device)*0.0 # Hz
	spinarray = SpinArray(loc=loc,T1=T1,T2=T2)
	# spinarray.show_info()

	return spinarray
	# -----------------------








# bloch simulation of a spin
# -------------------------------------
def blochsim_spin(spin, Nt, dt, rf=0, gr=0):
	"""
	Bloch simulation for one spin
	rf (2*N)(mT), gr (3*N)(mT/cm), unit: mT
	dt: (ms)
	"""
	dt = torch.tensor(dt,device=device)

	M_hist = torch.zeros((3,Nt+1),device=device)
	# M[:,0] = torch.tensor([1.,0.,0.],device=device)
	M = spin.Mag
	M_hist[:,0] = M

	E1 = torch.exp(-dt/spin.T1)
	E2 = torch.exp(-dt/spin.T2)
	E = torch.tensor([[E2,0.,0.],
				[0.,E2,0.],
				[0.,0.,E1]],device=device)
	e = torch.tensor([0.,0.,1-E1],device=device)

	Beff_hist = torch.zeros((3,Nt),device=device)*1.0
	Beff_hist[0:2,:] = rf # mT
	Beff_hist[2,:] = spin.get_loc()@gr + spin.df/spin.gamma*1e-3 # mT/cm*cm = mT, Hz/(MHz/T)=1/1e6*1e3mT=1e-3*mT
	# print('Beff_hist:',Beff_hist.shape)

	for k in range(Nt):
		# Beff = torch.zeros(3,device=device)
		# Beff[0] = rf[0,k]
		# Beff[1] = rf[1,k]
		# Beff[2] = torch.dot(gr[:,k], spin.get_loc()) + spin.df/spin.gamma
		Beff = Beff_hist[:,k]
		Beff_norm = torch.linalg.norm(Beff,2)
		# print(Beff)
		if Beff_norm == 0:
			Beff_unit = torch.zeros(3,device=device)
		else:
			Beff_unit = Beff/torch.linalg.norm(Beff,2)

		# the rotation
		phi = -(2*torch.pi*spin.gamma)*Beff_norm*dt # Caution: what is the sign here!> 2*pi*MHz/T*mT*ms=rad
		R1 = torch.cos(phi)*torch.eye(3,device=device) + (1-torch.cos(phi))*torch.outer(Beff_unit,Beff_unit)
		# print(phi)
		# print(R1)

		# compute the magnetization:
		# M_temp = R1@M_hist[:,k] + torch.sin(phi)*torch.cross(Beff_unit,M_hist[:,k])
		# print(M_temp)
		# M_hist[:,k+1] = E@M_temp + e
		M = R1@M + torch.sin(phi)*torch.cross(Beff_unit,M)
		M = E@M + e
		M_hist[:,k+1] = M

		# if k%50==0:
		# 	print('k =',k)
		# 	print(M.shape)
		# 	print(M.norm())

	return M, M_hist
# compute the effective B at each time, for every spin
# ------------------------------------------------------
def spinarray_Beffhist(spinarray,Nt,dt,rf,gr):
	# starttime = time()
	num = spinarray.num
	M_hist = torch.zeros((3,num,Nt+1),device=device)
	M_hist[:,:,0] = M

	# location = spinarray.loc #(3*num)
	# df = torch.rand(num,device=device)
	# df = spinarray.df

	# calculate the Beff:
	offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num)
	offBeff = offBeff.reshape(num,1)
	Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
	Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt)
	Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr + offBeff

	# normalization:
	# Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
	# Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
	# phi_hist = -(dt*2*torch.pi*spinarray.gamma)*Beff_norm_hist.T #(Nt*num)
	# phi_hist = phi_hist.T #(num*Nt)

	return Beff_hist
	# ---------------
# Bloch simulation for spinarray:
# ----------------------------------------------------
class BlochSim_Array(torch.autograd.Function):
	@staticmethod
	def forward(ctx,spinarray,Nt,dt,Beff_unit_hist,phi_hist):
		"""
		Bloch simulation for spin arrays
		rf:(2*Nt)(mT), gr:(3*Nt)(mT/cm), dt:(ms)
		Beff_unit_hist:(3*num*Nt)(mT), phi_hist:(num*Nt)(rad)
		"""
		# starttime = time()
		num = spinarray.num
		M = spinarray.Mag # (3*num)
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
		Es = torch.cat((torch.zeros((1,num),device=device),torch.zeros((1,num),device=device),1-E1))

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
			suCm = sin_phi_hist[:,t]*torch.cross(Beff_unit_hist[:,:,t],M,dim=0) #(3*num) sin(phi)*ut X mt
			uTm = torch.sum(Beff_unit_hist[:,:,t]*M,dim=0) #(1*num) ut^T*mt
			M_temp = cos_phi_hist[:,t]*M + (1-cos_phi_hist[:,t])*uTm*Beff_unit_hist[:,:,t] + suCm
			M = E*M_temp + Es
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
			uCm = torch.cross(Beff_unit_hist[:,:,t],M_hist[:,:,t],dim=0) # (3*num)
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
			mCplpp = torch.cross(M_hist[:,:,t],pl_pp,dim=0)
			pl_put = (1-cos_phi_hist[:,t])*(M_hist[:,:,t]*uTplpp + uTm*pl_pp) + sin_phi_hist[:,t]*mCplpp
			grad_Beff_unit_hist[:,:,t] = pl_put
			# print(grad_Beff_unit_hist[:,:,-1])
			# print(grad_Beff_unit_hist[:,:,-2])

			# compute the partial gradient w.r.t. the M_{t}:
			# pl/pm{t} = R^T E pl/pm{t+1}
			# R^T * p{l}/p{p}
			# ---------
			pl_pp = pl_pmtt*E # (3*num)
			uTplpp = torch.sum(Beff_unit_hist[:,:,t]*pl_pp, dim=0)
			uCplpp = torch.cross(Beff_unit_hist[:,:,t],pl_pp)
			pl_pmt = cos_phi_hist[:,t]*pl_pp + (1-cos_phi_hist[:,t])*Beff_unit_hist[:,:,t]*uTplpp - sin_phi_hist[:,t]*uCplpp
			# print('pl_pmt.shape',pl_pmt.shape)
			# print(pl_pmt)
			pl_pmtt = pl_pmt
			# break
		# print('end test backward\n')
		return grad_spinarray,grad_Nt,grad_dt,grad_Beff_unit_hist,grad_phi_hist
# apply the function
blochsim_array = BlochSim_Array.apply

# def blochsim_array__(spinarray,Nt,dt,Beff_hist): #pause
# 	Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
# 	Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
# 	phi_hist = -(dt*2*torch.pi*spinarray.gamma)*Beff_norm_hist.T #(Nt*num)
# 	phi_hist = phi_hist.T #(num*Nt)
# 	# apply function with custom backward:
# 	M = blochsim_array_eq(spinarray,Nt,dt,Beff_unit_hist,phi_hist)
# 	return M

# the final bloch simulation function, with custom backward
# --------------------------------------------
def blochsim(spinarray,Nt,dt,rf,gr):
	# compute effective B for all time points:
	Beff_hist = spinarray_Beffhist(spinarray,Nt,dt,rf,gr)
	# compute normalized B and phi, for all time points:
	Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
	Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
	phi_hist = -(dt*2*torch.pi*spinarray.gamma)*Beff_norm_hist.T #(Nt*num)
	phi_hist = phi_hist.T #(num*Nt)
	# compute the simulation
	M = blochsim_array(spinarray,Nt,dt,Beff_unit_hist,phi_hist)
	return M
# Bloch simulation, only simulation, no cunstom backward
# ----------------------------------
def blochsim_(spinarray,Nt,dt,rf,gr):
	"""
	Bloch simulation for spin arrays
	rf:(2*N)(mT), gr:(3*N)(mT/cm), dt:(ms)
	"""
	# starttime = time()
	num = spinarray.num
	M = spinarray.Mag #(3*num)
	M_hist = torch.zeros((3,num,Nt+1),device=device)
	M_hist[:,:,0] = M

	E1 = torch.exp(-dt/spinarray.T1).reshape(1,num) #(1*num)
	E2 = torch.exp(-dt/spinarray.T2).reshape(1,num)
	# change into matrix
	E = torch.cat((E2,E2,E1),dim=0)
	Es = torch.cat((torch.zeros((1,num),device=device),torch.zeros((1,num),device=device),1-E1))

	# calculate the Beff:
	offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num)
	offBeff = offBeff.reshape(num,1)
	Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
	Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt)
	Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr + offBeff

	# normalization:
	Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
	Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
	phi_hist = -(dt*2*torch.pi*spinarray.gamma)*Beff_norm_hist.T #(Nt*num)
	phi_hist = phi_hist.T #(num*Nt)

	sin_phi_hist = torch.sin(phi_hist)
	cos_phi_hist = torch.cos(phi_hist)

	# print('simulation loop start time:{}'.format(time()-starttime))
	# M_temp = torch.zeros((3,num),device=device)
	for t in range(Nt):
		suCm = sin_phi_hist[:,t]*torch.cross(Beff_unit_hist[:,:,t],M,dim=0) #(3*num)
		uTm = torch.sum(Beff_unit_hist[:,:,t]*M,dim=0) #(1*num)
		M_temp = cos_phi_hist[:,t]*M + (1-cos_phi_hist[:,t])*uTm*Beff_unit_hist[:,:,t] + suCm
		M = E*M_temp + Es
		M_hist[:,:,t+1] = M

		if False:
			kk = int(Nt/10)
			if t%kk == 0:
				print('->',100*t/Nt,'%')
				# print('', end='')
			# if k%50 == 0:
			# 	print()
	# print('->stopped time:',time()-starttime)
	return M, M_hist
	# ------------------
# Shinnar-Le Roux relation simulation
# -------------------------------
def slrsim_spin(spin,Nt,dt,rf,gr):
	"""
	SLR simulation for one spin
	Beff_hist:(3*Nt)
	"""
	Beff_hist = torch.zeros((3,Nt),device=device)*1.0
	Beff_hist[0:2,:] = rf
	Beff_hist[2,:] = spin.get_loc()@gr + spin.df/spin.gamma*1e-3 # mT/cm*cm = mT, Hz/(MHz/T) = 1e-3*mT
	# 
	Beff_norm_hist = Beff_hist.norm(dim=0)
	Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0)
	phi_hist = -2*torch.pi*dt*spin.gamma*Beff_norm_hist
	# print(phi_hist)
	#
	ahist = (1.0+0.0j)*torch.zeros(Nt+1,device=device)
	bhist = (1.0+0.0j)*torch.zeros(Nt+1,device=device)
	ahist[0] = 1.0+0.0j
	bhist[0] = 0.0+0.0j
	for t in range(Nt):
		# Beff = Beff_hist[:,k]
		# Beff_norm = torch.linalg.norm(Beff,2)
		# if Beff_norm == 0:
		# 	Beff_unit = torch.tensor([0.,0.,1.],device=device)
		# else:
		# 	Beff_unit = Beff/Beff_norm
		phi = phi_hist[t]  # 2pi*MHz/Tesla*mT*ms
		Beff_unit = Beff_unit_hist[:,t]
		aj = torch.cos(phi/2) - torch.tensor([0.+1.0j],device=device)*Beff_unit[2]*torch.sin(phi/2)
		bj = -(torch.tensor([0.+1.0j],device=device)*Beff_unit[0]-Beff_unit[1])*torch.sin(phi/2)
		# _ = print(t,':',aj.item(),bj.item()) if t in [2,3] else None
		ahist[t+1] = aj*ahist[t] - (bj.conj())*bhist[t]
		bhist[t+1] = bj*ahist[t] + (aj.conj())*bhist[t]
		# Cj = torch.cos(phi/2)
		# Sj = torch.exp()*torch.sin(phi/2)
	# print(ahist.abs()**2 + bhist.abs()**2)
	a,b = ahist[-1],bhist[-1]
	# print(ahist[:5].tolist())
	# print(bhist[:5].tolist())
	# print(a)
	# print(b)
	return a,b
# SLR simulation, as complex number, cannot do backward
# --------------------------------------------
def slrsim_c(spinarray,Nt,dt,rf,gr):
	"""
	SLR simulation for spinarray, treat as complex numbers
	Beff_hist:(3*num*Nt), phi_hist:(num*Nt)
	"""
	# print('slr spinarray simulation')
	# print(spinarray.loc)
	# print(spinarray.df)
	# print(spinarray.gamma)
	# print(spinarray.T1)
	# print(spinarray.T2)
	# print(spinarray.Mag)
	num = spinarray.num
	Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
	offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num)
	offBeff = offBeff.reshape(num,1)
	Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt)
	Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr + offBeff
	# the rotations
	Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
	Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
	phi_hist = -(dt*2*torch.pi*spinarray.gamma)*(Beff_norm_hist.T)
	phi_hist = phi_hist.T
	# print(phi_hist)
	#
	ahist = (1.0+0.0j)*torch.zeros((num,Nt+1),device=device)
	bhist = (1.0+0.0j)*torch.zeros((num,Nt+1),device=device)
	ahist[:,0] = 1.0+0.0j
	bhist[:,0] = 0.0+0.0j
	for t in range(Nt):
		phi = phi_hist[:,t]
		aj = torch.cos(phi/2) - torch.tensor([0.+1.0j],device=device)*Beff_unit_hist[2,:,t]*torch.sin(phi/2)
		bj = -(torch.tensor([0.+1.0j],device=device)*Beff_unit_hist[0,:,t] - Beff_unit_hist[1,:,t])*torch.sin(phi/2)
		# _ = print(t,':',aj.tolist(),'\n\n',bj.tolist()) if t in [2,3] else None
		# print(aj*ahist[:,t])
		# print(ahist[:,:5])
		# if t < 5:
		# 	print(t, end=', ')
		# 	print(bhist[0,t])
		# 	print(aj[0])
		# 	print(bj[0])
			# print(bj*ahist[:,t] - (aj.conj())*bhist[:,t])
		# _ = print(t,',',ahist[t]) if t<30 else None
		# print('t:',t)
		# print(ahist[:,t].abs())
		# print(bhist[:,t].abs())
		# print()
		# if ahist[0,t].abs()<1e-6:
		# 	break
		ahist[:,t+1] = aj*ahist[:,t] - (bj.conj())*bhist[:,t]
		bhist[:,t+1] = bj*ahist[:,t] + (aj.conj())*bhist[:,t]
	a = ahist[:,-1]
	b = bhist[:,-1]
	# print(ahist[1,:5].tolist())
	# print(bhist[1,:5].tolist())
	# print('a,b')
	# print(a)
	# print(b)
	return a,b
# function simulate SLR as real numbers
# ------------------------------
class SLRSim_SpinArray(torch.autograd.Function):
	# def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
	# 	return super().forward(ctx, *args, **kwargs)
	@staticmethod
	def forward(ctx,spinarray,Nt,dt,Beff_unit_hist,phi_hist):
		"""
		SLR simulation for spinarray, but treat all as real numbers
		Beff_hist:(3*num*Nt), phi_hist:(num*Nt)
		"""
		# print('slr spinarray simulation')
		# print(spinarray.loc)
		# print(spinarray.df)
		# print(spinarray.gamma)
		# print(spinarray.T1)
		# print(spinarray.T2)
		# print(spinarray.Mag)

		num = spinarray.num

		# Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
		# offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num)
		# offBeff = offBeff.reshape(num,1)
		# Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt)
		# Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr + offBeff

		# the rotations
		# Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
		# Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
		# phi_hist = -(dt*2*torch.pi*spinarray.gamma)*(Beff_norm_hist.T)
		# phi_hist = phi_hist.T #(num*Nt)
		#
		ahist_real = torch.zeros((num,Nt+1),device=device)
		ahist_imag = torch.zeros((num,Nt+1),device=device)
		bhist_real = torch.zeros((num,Nt+1),device=device)
		bhist_imag = torch.zeros((num,Nt+1),device=device)
		ahist_real[:,0] = 1.0
		for t in range(Nt):
			phi = phi_hist[:,t]
			aj_real = torch.cos(phi/2)
			aj_imag = Beff_unit_hist[2,:,t]*torch.sin(phi/2)
			bj_real = Beff_unit_hist[1,:,t]*torch.sin(phi/2)
			bj_imag = Beff_unit_hist[0,:,t]*torch.sin(phi/2)
			ahist_real[:,t+1] = aj_real*ahist_real[:,t] - aj_imag*ahist_imag[:,t] - bj_real*bhist_real[:,t] - bj_imag*bhist_imag[:,t]
			ahist_imag[:,t+1] = aj_real*ahist_imag[:,t] + aj_imag*ahist_real[:,t] - bj_real*bhist_imag[:,t] + bj_imag*bhist_real[:,t]
			bhist_real[:,t+1] = bj_real*ahist_real[:,t] - bj_imag*ahist_imag[:,t] + aj_real*bhist_real[:,t] + aj_imag*bhist_imag[:,t]
			bhist_imag[:,t+1] = bj_real*ahist_imag[:,t] + bj_imag*ahist_real[:,t] + aj_real*bhist_imag[:,t] - aj_imag*bhist_real[:,t]
		a_real = ahist_real[:,-1] 
		a_imag = ahist_imag[:,-1]
		b_real = bhist_real[:,-1]
		b_imag = bhist_imag[:,-1]
		a = ahist_real[:,-1] + ahist_imag[:,-1]*torch.tensor([0.0+1.0j],device=device)
		b = bhist_real[:,-1] + bhist_imag[:,-1]*torch.tensor([0.0+1.0j],device=device)
		ctx.save_for_backward(Beff_unit_hist,phi_hist)
		return a_real,a_imag,b_real,b_imag
	@staticmethod
	def backward(ctx, *grad_outputs): #(ctx,grad_output1,grad_output2,grad_output3,grad_output4):
		print('test backward')
		# print(grad_output1)
		# print(grad_output2)
		# print(grad_output3)
		# print(grad_output4)
		# print(grad_outputs)

		Beff_unit_hist,phi_hist = ctx.saved_tensors # M_hist:(3*num*Nt)
		Nt = Beff_unit_hist.shape[-1]
		print('Nt =',Nt)

		grad_spinarray = grad_Nt = grad_dt = grad_Beff_unit_hist = grad_phi_hist = None
		needs_grad = ctx.needs_input_grad

		

		for k in range(Nt):
			t = Nt - k - 1
			# pl_pa = 



		print('end test backward')

		return grad_spinarray,grad_Nt,grad_dt,grad_Beff_unit_hist,grad_phi_hist
	# def backward(ctx: Any, *grad_outputs: Any) -> Any:
	# 	return super().backward(ctx, *grad_outputs)
# apply the simulate function:
slrsim_spinarray = SLRSim_SpinArray.apply
# SLR simulation, as all real numbers, custom backward
# ----------------------------------------------
def slrsim(spinarray,Nt,dt,rf,gr):
	# compute effective B for all time points:
	Beff_hist = spinarray_Beffhist(spinarray,Nt,dt,rf,gr)
	# compute normalized B and phi, for all time points:
	Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
	Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
	phi_hist = -(dt*2*torch.pi*spinarray.gamma)*Beff_norm_hist.T #(Nt*num)
	phi_hist = phi_hist.T #(num*Nt)
	# compute the SLR simulation:
	a_real,a_imag,b_real,b_imag = slrsim_spinarray(spinarray,Nt,dt,Beff_unit_hist,phi_hist)
	return a_real,a_imag,b_real,b_imag
	# ---------------
# SLR simulation as real numbers, only simulation
# -------------------------------------
def slrsim_(spinarray,Nt,dt,rf,gr):
	"""
	SLR simulation for spinarray, but treat all as real numbers
	Beff_hist:(3*num*Nt), phi_hist:(num*Nt)
	out: a_real:(num), a_imag:(num), b_real:(num), b_imag:(num)
	"""
	# print('slr spinarray simulation')
	# print(spinarray.loc)
	# print(spinarray.df)
	# print(spinarray.gamma)
	# print(spinarray.T1)
	# print(spinarray.T2)
	# print(spinarray.Mag)
	num = spinarray.num
	Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
	offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num)
	offBeff = offBeff.reshape(num,1)
	Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt)
	Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr + offBeff
	# the rotations
	Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
	Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
	phi_hist = -(dt*2*torch.pi*spinarray.gamma)*(Beff_norm_hist.T) #()
	phi_hist = phi_hist.T #(num*Nt)
	# print('phi_hist',phi_hist.shape)
	
	ahist_real = torch.zeros((num,Nt+1),device=device)
	ahist_imag = torch.zeros((num,Nt+1),device=device)
	bhist_real = torch.zeros((num,Nt+1),device=device)
	bhist_imag = torch.zeros((num,Nt+1),device=device)
	ahist_real[:,0] = 1.0
	# ab_hist = torch.zeros((4,num,Nt),device=device)
	# ab = torch.tensor([1.,0.,0.,0.],device=device)

	a_real = torch.ones(num,device=device)
	a_imag = torch.zeros(num,device=device)
	b_real = torch.zeros(num,device=device)
	b_imag = torch.zeros(num,device=device)
	a = torch.tensor([1.,0.],device=device)
	b = torch.tensor([0.,0.],device=device)
	# Beff_unit_hist.requires_grad = True
	# sinBeff_unit_hist = Beff_unit_hist*torch.sin(phi_hist/2....)
	# phi_hist.requires_grad = True
	for t in range(Nt):
		aj_real = torch.cos(phi_hist[:,t]/2)
		aj_imag = Beff_unit_hist[2,:,t]*torch.sin(phi_hist[:,t]/2)
		bj_real = Beff_unit_hist[1,:,t]*torch.sin(phi_hist[:,t]/2)
		bj_imag = Beff_unit_hist[0,:,t]*torch.sin(phi_hist[:,t]/2)

		# phi = phi_hist[:,t]
		# aj_real = torch.cos(phi/2)
		# aj_imag = Beff_unit_hist[2,:,t]*torch.sin(phi/2)
		# bj_real = Beff_unit_hist[1,:,t]*torch.sin(phi/2)
		# bj_imag = Beff_unit_hist[0,:,t]*torch.sin(phi/2)

		# _ = print(t,':',aj.tolist(),'\n\n',bj.tolist()) if t in [2,3] else None
		# print(aj*ahist[:,t])
		# print(ahist[:,:5])
		# if t < 5:
		# 	print(t, end=', ')
		# 	print(bhist[0,t])
		# 	print(aj[0])
		# 	print(bj[0])
			# print(bj*ahist[:,t] - (aj.conj())*bhist[:,t])
		# _ = print(t,',',ahist[t]) if t<30 else None
		# print('t:',t)
		# print(ahist[:,t].abs())
		# print(bhist[:,t].abs())
		# print()
		# if ahist[0,t].abs()<1e-6:
		# 	break

		anew_real = aj_real*a_real - aj_imag*a_imag - bj_real*b_real - bj_imag*b_imag
		anew_imag = aj_real*a_imag + aj_imag*a_real - bj_real*b_imag + bj_imag*b_real
		bnew_real = bj_real*a_real - bj_imag*a_imag + aj_real*b_real + aj_imag*b_imag
		bnew_imag = bj_real*a_imag + bj_imag*a_real + aj_real*b_imag - aj_imag*b_real
		
		a_real = anew_real
		a_imag = anew_imag
		b_real = bnew_real
		b_imag = bnew_imag
	
		ahist_real[:,t+1] = aj_real*ahist_real[:,t] - aj_imag*ahist_imag[:,t] - bj_real*bhist_real[:,t] - bj_imag*bhist_imag[:,t]
		ahist_imag[:,t+1] = aj_real*ahist_imag[:,t] + aj_imag*ahist_real[:,t] - bj_real*bhist_imag[:,t] + bj_imag*bhist_real[:,t]
		bhist_real[:,t+1] = bj_real*ahist_real[:,t] - bj_imag*ahist_imag[:,t] + aj_real*bhist_real[:,t] + aj_imag*bhist_imag[:,t]
		bhist_imag[:,t+1] = bj_real*ahist_imag[:,t] + bj_imag*ahist_real[:,t] + aj_real*bhist_imag[:,t] - aj_imag*bhist_real[:,t]
	# a_real = ahist_real[:,-1] 
	# a_imag = ahist_imag[:,-1]
	# b_real = bhist_real[:,-1]
	# b_imag = bhist_imag[:,-1]

	# a = ahist_real[:,-1] + ahist_imag[:,-1]*torch.tensor([0.0+1.0j],device=device)
	# b = bhist_real[:,-1] + bhist_imag[:,-1]*torch.tensor([0.0+1.0j],device=device)
	# print(ahist[1,:5].tolist())
	# print(bhist[1,:5].tolist())
	# print('a,b')
	# print(a)
	# print(b)
	# in-function test backward:
	# loss = torch.sum(b_real**2+b_imag**2)
	# loss.backward()
	# print('grads:')
	# print(phi_hist.grad.shape)
	# print(Beff_unit_hist.grad.shape)
	return a_real, a_imag, b_real, b_imag



# SLR transform
# --------------------------------------
def slr_transform_spin(spin,Nt,dt,rf,g):
	"""assume g is constant"""
	gamma = spin.gamma
	#
	# rf_norm_hist = rf.norm(dim=0) #(1*Nt)
	rf = rf[0,:] + (0.0+1.0j)*rf[1,:]
	rf_norm_hist = rf.abs()
	phi_hist = gamma*rf_norm_hist*dt
	A = (1.0+0.0j)*torch.zeros(Nt,device=device)
	B = (1.0+0.0j)*torch.zeros(Nt,device=device)
	A[0] = 1.0
	B[0] = 0.0
	Cj = torch.cos(phi_hist/2)
	Sj = -(0.0+1.0j)*(torch.exp((0.0+1.0j)*rf.angle()))*torch.sin(phi_hist/2)
	for t in range(Nt):
		A_tmp1 = Cj[t]*A
		A_tmp2 = -Sj[t].conj()*B
		B_tmp1 = Sj[t]*A
		B_tmp2 = Cj*B
		B = B_tmp1
		B[1:] = B[1:] + B_tmp2[:-1]
		A = A_tmp1
		A[1:] = A[1:] + A_tmp2[:-1]
	return A,B
	# -------------
# SLR transform along different locations
# --------------------------------
def slr_transform_1d(loc,Nt,dt,rf,g):
	return
	# -------











# =================================================================
# simulator backups and some old versions
# --------------------------------------------------------
# def blochsim_test(spin, Nt, dt, rf=0, gr=0):
# 	"""
# 	rf (2*N)(mT), gr (3*N)(mT/cm), unit: mT
# 	dt: (ms)
# 	"""
# 	# N = rf.shape[1]
# 	print(rf.requires_grad)
# 	dt = torch.tensor(dt,device=device)

# 	M_hist = torch.zeros((3,Nt+1),device=device)
# 	# M[:,0] = torch.tensor([1.,0.,0.],device=device)
# 	M = spin.Mag
# 	M_hist[:,0] = M


# 	E1 = torch.exp(-dt/spin.T1)
# 	E2 = torch.exp(-dt/spin.T2)
# 	E = torch.tensor([[E2,0.,0.],
# 				[0.,E2,0.],
# 				[0.,0.,E1]],device=device)
# 	e = torch.tensor([0.,0.,1-E1],device=device)

# 	Beff_hist = torch.zeros((3,Nt))*1.0
# 	Beff_hist[0:2,:] = rf
# 	Beff_hist[2,:] = spin.get_position()@gr + spin.df/spin.gamma
# 	print('Beff_hist:',Beff_hist.shape)

# 	print(Beff_hist.grad_fn)

# 	for k in range(Nt):
# 		# Beff = torch.zeros(3,device=device)
# 		# Beff[0] = rf[0,k]
# 		# Beff[1] = rf[1,k]
# 		# Beff[2] = torch.dot(gr[:,k], spin.get_position()) + spin.df/spin.gamma
# 		Beff = Beff_hist[:,k]
# 		Beff_norm = torch.linalg.norm(Beff,2)
# 		# print(Beff)
# 		if Beff_norm == 0:
# 			Beff_unit = torch.zeros(3,device=device)
# 		else:
# 			Beff_unit = Beff/torch.linalg.norm(Beff,2)
# 		# the rotation
# 		phi = -(2*torch.pi*spin.gamma)*Beff_norm*dt/1000  # Caution: what is the sign here!>
# 		# print(phi)
# 		R1 = torch.cos(phi)*torch.eye(3,device=device) + (1-torch.cos(phi))*torch.outer(Beff_unit,Beff_unit)
# 		# print(R1)
# 		# compute the magnetization
# 		M_temp = R1@M_hist[:,k] + torch.sin(phi)*torch.cross(Beff_unit,M_hist[:,k])
# 		M = R1@M + torch.sin(phi)*torch.cross(Beff_unit,M)
# 		# print(M_temp)
# 		# M_hist[:,k+1] = E@M_temp + e
# 		M = E@M + e
# 		M_hist[:,k+1] = M

# 	# return M, M_hist
# 	return M, M_hist

def blochsim_array_v1(spinarray,Nt,dt,rf,gr):
	"""
	not fully adopt the matrix computation, is slow

	for spin arrays
	rf:(2*N)(mT), gr:(3*N)(mT/cm), dt:(ms)
	"""
	starttime = time()
	# Nt = rf.shape[1]
	# num = 20
	num = spinarray.num

	# M = torch.zeros((3,num),device=device)
	# M[0,:] = 1.0
	M = spinarray.Mag
	M_hist = torch.zeros((3,num,Nt+1),device=device)
	M_hist[:,:,0] = M

	# loction = torch.rand((3,num),device=device)
	location = spinarray.loc
	# df = torch.rand(num,device=device)
	# df = spinarray.df
	# gamma = 42.48

	E1 = torch.exp(-dt/spinarray.T1)
	E2 = torch.exp(-dt/spinarray.T2)
	# print('E2:',E2.shape)
	# print(E2)
	# E = torch.tensor([[E2,0.,0.],
	# 			[0.,E2,0.],
	# 			[0.,0.,E1]],device=device)
	# e = torch.zeros((3,num),device=device)
	# e[2,:] = 1-E1
	# print((1-E1).shape)

	# Beff_hist = torch.zeros((3,Nt),device=device)*1.0
	# Beff_hist[0:2,:] = rf
	# Beff_hist[2,:] = spin.get_loc()@gr + spin.df/spin.gamma

	print('simulation time:{}'.format(time()-starttime))

	for k in range(Nt):
		# Beff = Beff_hist[:,k]
		Beff = torch.zeros((3,num),device=device)
		Beff[0,:] = torch.ones(num,device=device)*rf[0,k]
		Beff[1,:] = torch.ones(num,device=device)*rf[1,k]
		Beff[2,:] = gr[:,k]@location + spinarray.df/spinarray.gamma*1e-3 # Hz/(MHz/T)=1e-6*T=1e-3*mT
		Beff_norm = torch.norm(Beff,dim=0)
		Beff_unit = torch.nn.functional.normalize(Beff,dim=0)
		# the rotation
		phi = -(2*torch.pi*spinarray.gamma)*Beff_norm*dt # caution: the sign !!
		M_temp = torch.zeros((3,num),device=device)
		for n in range(num):
			R1 = torch.cos(phi[n])*torch.eye(3,device=device) + (1-torch.cos(phi[n]))*torch.outer(Beff_unit[:,n],Beff_unit[:,n])
			M_temp[:,n] = R1@M[:,n] + torch.sin(phi[n])*torch.cross(Beff_unit[:,n],M[:,n])
		M[0,:] = E2*M_temp[0,:]
		M[1,:] = E2*M_temp[1,:]
		M[2,:] = E1*M_temp[2,:] + (1-E1)
		# M = E@M_temp + e
		M_hist[:,:,k+1] = M
		# print('k:',k,'spin1:',M[:,0])

		if True:
			kk = int(Nt/10)
			if k%kk == 0:
				print('->',100*k/Nt,'%')
				# print('', end='')
			# if k%50 == 0:
			# 	print()
	print('->stopped time:',time()-starttime)

	return M, M_hist
def blochsim_array_v2(spinarray,Nt,dt,rf,gr):
	"""
	Bloch simulation for spin arrays, 
	rf:(2*N)(mT), gr:(3*N)(mT/cm), dt:(ms)
	"""
	starttime = time()
	num = spinarray.num
	M = spinarray.Mag #(3*num)
	M_hist = torch.zeros((3,num,Nt+1),device=device)
	M_hist[:,:,0] = M

	# location = spinarray.loc #(3*num)
	# df = torch.rand(num,device=device)
	# df = spinarray.df
	# gamma = 42.48

	E1 = torch.exp(-dt/spinarray.T1) #(num)
	E2 = torch.exp(-dt/spinarray.T2)

	# calculate the Beff:
	offBeff = spinarray.df/spinarray.gamma*1e-3 #(1*num)
	offBeff = offBeff.reshape(num,1)
	Beff_hist = torch.zeros((3,num,Nt),device=device)*1.0
	Beff_hist[:2,:,:] = Beff_hist[:2,:,:] + rf.reshape(2,1,Nt)
	Beff_hist[2,:,:] = Beff_hist[2,:,:] + spinarray.loc.T@gr + offBeff

	# normalization:
	Beff_norm_hist = Beff_hist.norm(dim=0) #(num*Nt)
	Beff_unit_hist = torch.nn.functional.normalize(Beff_hist,dim=0) #(3*num*Nt)
	phi_hist = -(dt*2*torch.pi*spinarray.gamma)*Beff_norm_hist.T #(Nt*num)
	phi_hist = phi_hist.T #(num*Nt)

	print('simulation loop start time:{}'.format(time()-starttime))
	# M_temp = torch.zeros((3,num),device=device)
	for t in range(Nt):
		Beff_unit = Beff_unit_hist[:,:,t] #(3*num)
		phi = phi_hist[:,t]
		#
		suCm = torch.sin(phi)*torch.cross(Beff_unit,M,dim=0) #(3*num)
		uTm = torch.sum(Beff_unit*M,dim=0) #(1*num)
		M_temp = torch.cos(phi)*M + (1-torch.cos(phi))*uTm*Beff_unit + suCm
		M[0,:] = E2*M_temp[0,:]
		M[1,:] = E2*M_temp[1,:]
		M[2,:] = E1*M_temp[2,:] + (1-E1)
		# M = E@M_temp + e
		M_hist[:,:,t+1] = M

		if False:
			kk = int(Nt/10)
			if t%kk == 0:
				print('->',100*t/Nt,'%')
				# print('', end='')
			# if k%50 == 0:
			# 	print()
	# print('->stopped time:',time()-starttime)
	return M, M_hist
# =================================================================










# ------------------------------------------------------------------
# some dependent functions
# ----------------------------------------------
def get_transverse_signal(M_hist):
	"""
	M_hist:(3*num*Nt)
	"""
	M_tr = M_hist[0:2,:,:]
	sig_tr = torch.sum(M_tr,dim=1)
	return sig_tr
def get_summation_signal(M_hist):
	"""
	M_hist:(3*num*Nt)
	"""
	sig = torch.sum(M_hist,dim=1)
	return sig
# ---------------------------------------------------------------
# plot function
# ----------------------------------
def plot_pulse(rf,gr,dt,picname='pictures/mri_pic_pulse.png'):
	'''
	rf:(2*Nt)(mT/cm), gr:(3*Nt)(mT/cm), dt:(ms)
	'''
	N = rf.shape[1]
	rf = np.array(rf.tolist())
	gr = np.array(gr.tolist())
	time = np.arange(N)*dt
	fig, ax = plt.subplots(2)
	# fig.suptitle('')
	ax[0].plot(time,rf[0,:],label='rf real')
	ax[0].plot(time,rf[1,:],label='rf imag')
	ax[1].plot(time,gr[0,:],label='gr,x')
	ax[1].plot(time,gr[1,:],label='gr,y')
	ax[1].plot(time,gr[2,:],label='gr,z')
	ax[0].set_ylabel('mT')
	ax[1].set_ylabel('mT/cm')
	plt.xlabel('time(ms)')
	ax[0].legend()
	ax[1].legend()
	print('save fig...'+picname)
	# plt.savefig(picname)
	plt.show()
	return
def plot_magnetization(M_hist, dt, picname='pictures/mri_pic_mag.png'):
	"""
	how magnetization changes with time
	M_hist:(3*Nt), dt:(ms)
	"""
	N = M_hist.shape[1]
	M = np.array(M_hist.tolist())
	time = np.arange(N)*dt
	plt.figure()
	plt.plot(time,M[0,:],label='x')
	plt.plot(time,M[1,:],label='y')
	plt.plot(time,M[2,:],label='z')
	plt.legend()
	plt.xlabel('time(ms)')
	print('save fig...'+picname)
	# plt.savefig(picname)
	plt.show()
	return
def plot_pulse_and_magnetization(rf, gr, M_hist, dt, picname='pictures/mri_pic_pulse_and_mag.png'):
	"""
	M_hist:(3*Nt), rf:(2*Nt)(mT/cm), gr:(3*Nt)(mT/cm), dt:(ms)
	"""
	N = rf.shape[1]
	rf = np.array(rf.tolist())
	gr = np.array(gr.tolist())
	time = np.arange(N)*dt
	fig, ax = plt.subplots(3)
	# fig.suptitle('')
	ax[0].plot(time,rf[0,:],label='rf real')
	ax[0].plot(time,rf[1,:],label='rf imag')
	ax[1].plot(time,gr[0,:],label='gr,x')
	ax[1].plot(time,gr[1,:],label='gr,y')
	ax[1].plot(time,gr[2,:],label='gr,z')
	N = M_hist.shape[1]
	M = np.array(M_hist.tolist())
	time = np.arange(N)*dt
	ax[2].plot(time,M[0,:],label='x')
	ax[2].plot(time,M[1,:],label='y')
	ax[2].plot(time,M[2,:],label='z')
	ax[0].set_ylabel('mT')
	ax[1].set_ylabel('mT')
	ax[2].set_ylabel('Magnetization')
	plt.xlabel('time(ms)')
	ax[0].legend()
	ax[1].legend()
	ax[2].legend()
	print('save fig...'+picname)
	# plt.savefig(picname)
	plt.show()
	return
def plot_magnetization_3d_rotation(M_hist,dt,picname='pictures/mri_pic_mag_3d_rotation.png'):
	"""
	plot of how magnetization changes in 3D
	"""
	M = np.array(M_hist.tolist())
	ax = plt.figure().add_subplot(projection='3d')
	ax.plot(M[0,:],M[1,:],M[2,:],label='magnetization')
	ax.plot((0,M[0,0]),(0,M[1,0]),(0,M[2,0]),linewidth=1,linestyle='--')
	ax.plot((0,M[0,-1]),(0,M[1,-1]),(0,M[2,-1]),linewidth=1,linestyle='--')
	ax.text(M[0,0],M[1,0],M[2,0],r'$M_0$',fontsize=8)
	ax.text(M[0,-1],M[1,-1],M[2,-1],r'end',fontsize=8)
	ax.legend()
	ax.set_xlim(-1.1,1.1)
	ax.set_ylim(-1.1,1.1)
	ax.set_zlim(-1.1,1.1)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	print('save fig...'+picname)
	# plt.savefig(picname)
	plt.show()
	return
def plot_transverse_signal(M_hist,dt,picname='pictures/mri_pic_signal.png'):
	"""
	plot the sum of transverse magnetization
	M_hist:(3*num*Nt), dt:(ms)
	"""
	Nt = M_hist.shape[2]
	time = np.arange(Nt)*dt
	signal_transverse = np.array(get_transverse_signal(M_hist).tolist())
	magnitude = np.sqrt(signal_transverse[0,:]**2 + signal_transverse[1,:]**2)
	plt.figure()
	plt.plot(time,signal_transverse[0,:],label='M_x sum',alpha=0.4)
	plt.plot(time,signal_transverse[1,:],label='M_y sum',alpha=0.4)
	plt.plot(time,magnitude,label='Mxy (magnitude)',ls='-',lw=1)
	plt.legend()
	plt.xlabel('time(ms)')
	plt.ylabel('signal')
	print('save fig...'+picname)
	# plt.savefig(picname)
	plt.show()
	return
def plot_slr_profile(locations,a,b,picname='pictures/mri_pic_slr_mag_profile.png'):
	"""
	M1d:(1*num), locations:(cm)
	"""
	loc = np.array(locations.tolist())
	Mxy = 2*(a.conj())*b # excitation case
	# Mz = a*(a.conj()) - b*(b.conj()) # excitation case
	Mz = a.abs()**2 - b.abs()**2
	# Mxy = (a.conj())**2 + b**2 # refocusing case
	plt.figure()
	# Mxy_mag = Mxy.abs()
	# M = np.array(Mxy_mag.tolist())
	# plt.plot(loc,M,label='Mxy')
	#
	M = np.array(Mxy.real.tolist())
	plt.plot(loc,M,label='Mx')
	M = np.array(Mxy.imag.tolist())
	plt.plot(loc,M,label='My')
	M = np.array(Mz.tolist())
	plt.plot(loc,M,label='Mz',ls='--')
	#
	plt.legend()
	plt.ylim(-1.05,1.05)
	plt.ylabel('M')
	plt.xlabel('cm')
	print('save fig...'+picname)
	# plt.savefig(picname)
	plt.show()
	return
def plot_magnetization_profile(locations,M_dis,picname='pictures/mri_pic_mag_profile.png'):
	"""
	M_dis:(3,num), locations:(num)(cm)
	"""
	M = np.array(M_dis.tolist())
	loc = np.array(locations.tolist())
	plt.figure()
	plt.plot(loc,M[0,:],label='Mx')
	plt.plot(loc,M[1,:],label='My')
	plt.plot(loc,M[2,:],label='Mz',ls='--')
	plt.legend()
	plt.ylim(-1.05,1.05)
	plt.xlabel('cm')
	print('save fig...'+picname)
	# plt.savefig(picname)
	plt.show()
	return



# function that save the pulse and other information
# ------------------------------------------------
import scipy.io as spio
def save_infos(Nt,dt,rf,gr,logname,otherinfodic={}):
	'''save the informations'''
	rf = np.array(rf.tolist())
	gr = np.array(gr.tolist())
	info = {}
	info['rf'] = rf
	info['gr'] = gr
	info['dt'] = dt
	info['Nt'] = Nt
	info['info'] = 'rf:mT, gr:mT/cm, dt:ms'
	for k in otherinfodic.keys():
		info[k] = otherinfodic[k]
	print('saved infos:',info.keys())
	spio.savemat(logname,info)
	return
	# ----------------
def read_data(filename):
	try:
		data = spio.loadmat(filename)
	except:
		print('fail to load data')
		return None
	return data
# read in spin array from mrphy cube obj:
# ------------------------------------------
def spinarray_from_mrphy_cube(cube):
	num = cube.loc_.shape[1]
	location = cube.loc_.reshape(num,3)
	location = location.T
	T1 = cube.T1_*1000
	T2 = cube.T2_*1000
	# print(dir(cube))
	# print(cube.γ_.shape)
	# print(cube.M_.shape)
	M0 = cube.M_.T
	M0 = M0.reshape(3,num)
	spinarray = SpinArray(loc=location,T1=T1,T2=T2)
	spinarray.gamma = cube.γ_/100
	spinarray.set_Mag(M0)
	return spinarray
	# -----------------
# read in pulse from mrphy pulse obj:
# -------------------------------------
def pulse_from_mrphy(mrphy_pulse):
	Nt = mrphy_pulse.rf.shape[2]
	dt = mrphy_pulse.dt.item()*1e3
	# print(dt)
	extract_rf = mrphy_pulse.rf.reshape(2,Nt)
	extract_gr = mrphy_pulse.gr.reshape(3,Nt)
	extract_rf = extract_rf*0.1
	extract_gr = extract_gr*0.1
	return Nt, dt, extract_rf, extract_gr
	# -----------------------




# ============================================================================
# ============================================================================
#                              parts for test
# ============================================================================
# ============================================================================
def test_freeprecession():
	N = 1000
	rf = 1.0*torch.zeros((2,N),device=device)
	gr = torch.zeros((3,N),device=device)
	gr[2,:] = torch.ones(N,device=device)*0.1
	#
	spin = Spin(df=0., loc=[0.,0.,1.])
	spin.set_Mag(torch.tensor([1.,0.,0.]))
	# simulation
	rf.requires_grad = True
	M,M_hist = blochsim(spin,N,1,rf,gr)
	plot_magnetization(M_hist,1)
	# plot_pulse(rf,gr,1)
	loss = torch.sum(M**2)
	loss.backward()
	return
def test2():
	g = torch.tensor([1.,1.,1.])
	x = torch.tensor([[1.,2.,3.,4.,0.],[1.,1.,1.,1.,0.],[1.,3.,4.,5.,0.]])
	print(x)
	print(g@x)
	print((g@x).shape)

	y = torch.nn.functional.normalize(x,dim=0)
	print(y)

	print('norm:',torch.norm(x,dim=0))
	print(np.sqrt(3))

	x = 1.*torch.tensor([1,2,3])
	print(x)

	return
def test3():
	N = 1000
	rf = torch.zeros((2,N),device=device)
	gr = torch.zeros((3,N),device=device)
	# blochsim_array(None,rf,gr,1)
	x = rf
	x = [1,2,3]
	x = np.array([1,2,3.])
	print(torch.tensor(x))
	return
def test_slr():
	n = 5
	loc = torch.rand((3,n),device=device)
	loc[0,:] = torch.tensor([0.,1.,0.,3.,4.],device=device)
	loc[1,:] = torch.tensor([0.,0.,0.,0.,0.],device=device)
	loc[2,:] = torch.tensor([0.,0.,0.,1.,0.],device=device)
	T1 = torch.ones(n,device=device)*1000.0
	T2 = torch.ones(n,device=device)*100.0
	#
	spinarray = SpinArray(loc=loc,T1=T1,T2=T2)
	spinarray.df = torch.tensor([10.,5.,0.,0.,0.],device=device)
	#
	Nt = 1000
	dt = 1.0 # ms
	rf = 1.0*torch.rand((2,Nt),device=device)
	gr = 0.0*torch.zeros((3,Nt),device=device)
	gr[2,:] = torch.ones(Nt,device=device)*0.05

	a,b = slrsim_spinarray(spinarray,Nt,dt,rf,gr)
	print(a)
	print(b)

	print('--------------')

	spin = Spin(df=10., loc=[0.,0.,0.])
	spin.show_info()
	a,b = slrsim_spin(spin,Nt,dt,rf,gr)
	print(a)
	print(b)
	return
def test4():
	def obj(x):
		t = torch.rand(3,2,2)
		g = x.reshape(3,1,2)
		print(g)
		p = t + g
		print(p)
		out = torch.sum(p)
		return out
	x = torch.tensor([[2.0,3.0,2.],[1.,1.,2.]])
	x.requires_grad = True
	loss = obj(x)
	print(loss)
	loss.backward()
	print(x.grad)
	return



# ============================================================================
# ============================================================================
#                              examples
# ============================================================================
# ============================================================================
def example_4_spin_freeprecession():
	print('\nExample of spin free precession')
	spin = Spin(df=10., loc=[0.,0.,1.5])
	spin.set_Mag(torch.tensor([0.,1.,0.]))
	spin.show_info()
	Nt = 1000
	rf = 0.0*torch.zeros((2,Nt),device=device)
	gr = 0.0*torch.zeros((3,Nt),device=device)
	dt = 1.0 # ms
	M,M_hist = blochsim_spin(spin,Nt,dt,rf,gr)
	print('stopped magnetization:\n\t',M)
	print('plot magnetization')
	plot_magnetization(M_hist,dt)
	print('plot 3d magnetization')
	plot_magnetization_3d_rotation(M_hist,dt)
	return
def example_5_spinarray_sim():
	print('\nExample of spin array simulation:')
	n = 5
	loc = torch.rand((3,n),device=device)
	loc[0,:] = torch.tensor([0.,1.,0.,3.,4.],device=device)
	loc[1,:] = torch.tensor([0.,0.,0.,0.,0.],device=device)
	loc[2,:] = torch.tensor([0.,0.,0.,1.,0.],device=device)
	T1 = torch.ones(n,device=device)*1000.0
	T2 = torch.ones(n,device=device)*100.0
	#
	spinarray = SpinArray(loc=loc,T1=T1,T2=T2)
	spinarray.df = torch.tensor([10.,5.,0.,0.,0.],device=device)
	#
	Nt = 1000
	dt = 1.0 # ms
	rf = 1.0*torch.ones((2,Nt),device=device)
	gr = 0.0*torch.zeros((3,Nt),device=device)
	gr[2,:] = torch.ones(Nt,device=device)*0.05
	#
	M,M_hist = blochsim_(spinarray,Nt,dt,rf,gr)
	print(M)
	M = blochsim(spinarray,Nt,dt,rf,gr)
	print(M)
	# plot_magnetization(M_hist[:,0,:],dt)
	# print('plot of transverse signal')
	# plot_transverse_signal(M_hist,dt)
	return
def example_6_arraysim_1d():
	print('\nExample of magnetization profile along one direction:')
	Nt = 400 #100
	dt = 0.01 # ms
	t0 = 100
	rf = 0.0*torch.zeros((2,Nt),device=device)
	rf[0,:] = 6*1e-3*torch.special.sinc((torch.arange(Nt,device=device)-200)/t0)
	gr = 0.0*torch.zeros((3,Nt),device=device) # mT/cm
	gr[2,:] = 0.1*torch.ones(Nt,device=device) # mT/cm
	# build a cube
	fov = [4,4,2]
	dim = [3,3,100]
	cube = Build_SpinArray(fov=fov,dim=dim)
	cube_seg = cube.get_spins([-0.1,0.1],[-0.1,0.1],[-2.,2.])
	loc = cube_seg.loc[2,:]
	M = blochsim(cube_seg,Nt,dt,rf,gr)
	print('plot the profile along z direction')
	plot_magnetization_profile(loc,M)
	return
def example_6_arraysim_1d_old():
	"""using older simulator, is much slower"""
	print('\nExample of magnetization profile along one direction:')
	Nt = 400 #100
	dt = 0.01 # ms
	t0 = 100
	rf = 0.0*torch.zeros((2,Nt),device=device)
	rf[0,:] = 6*1e-3*torch.special.sinc((torch.arange(Nt,device=device)-200)/t0)
	gr = 0.0*torch.zeros((3,Nt),device=device) # mT/cm
	gr[2,:] = 0.1*torch.ones(Nt,device=device) # mT/cm
	# build a cube
	fov = [4,4,2]
	dim = [3,3,100]
	cube = Build_SpinArray(fov=fov,dim=dim)
	cube_seg = cube.get_spins([-0.1,0.1],[-0.1,0.1],[-2.,2.])
	loc = cube_seg.loc[2,:]
	M,M_hist = blochsim_array_v1(cube_seg,Nt,dt,rf,gr)
	print('plot the profile along z direction')
	plot_magnetization_profile(loc,M)
	return
def example_7_spin_SLR():
	print('\nExample of SLR simulation of a spin:')
	spin = Spin(loc=torch.tensor([0.,0.,0.],device=device))
	spin.show_info()
	#
	Nt = 400 #100
	dt = 0.01 # ms
	t0 = 100
	rf = 0.0*torch.zeros((2,Nt),device=device)
	rf[0,:] = 6*1e-3*torch.special.sinc((torch.arange(Nt,device=device)-200)/t0)
	gr = 0.0*torch.zeros((3,Nt),device=device) # mT/cm
	gr[2,:] = 0.1*torch.ones(Nt,device=device) # mT/cm
	#
	a,b = slrsim_spin(spin,Nt,dt,rf,gr)
	Mxy = 2*(a.conj())*b # excitation case
	Mz = a*(a.conj()) - b*(b.conj()) # excitation case
	# Mxy = (a.conj())**2 + b**2 # refocusing case
	print('a,b:',a,b)
	print('Mxy:',Mxy)
	return
def example_8_array_slr_z():
	print('\nExample of SLR simulation along z-direction:')
	Nt = 400 #100
	dt = 0.01 # ms
	t0 = 100
	rf = 0.0*torch.zeros((2,Nt),device=device)
	rf[0,:] = 6*1e-3*torch.special.sinc((torch.arange(Nt,device=device)-200)/t0)
	# rf[0,:] = 58.7*1e-3*torch.ones(Nt,device=device)
	gr = 0.0*torch.zeros((3,Nt),device=device) # mT/cm
	gr[2,:] = 0.1*torch.ones(Nt,device=device) # mT/cm
	plot_pulse(rf,gr,dt) 
	# choose how to do simulation: spin by spin (slow), or as array (fast)
	sim_eachspin = False 
	if sim_eachspin: # do simulation spin by spin
		spin = Spin(loc=torch.tensor([0.,0.,0.],device=device))
		z = torch.linspace(-1,1,100) # location
		a = torch.zeros(len(z),device=device) + 0.0j*torch.zeros(len(z),device=device)
		b = torch.zeros(len(z),device=device) + 0.0j*torch.zeros(len(z),device=device)
		for k in range(len(z)):
			spin.set_position(0.,0.,z[k])
			a[k],b[k] = slrsim_spin(spin,Nt,dt,rf,gr)
	else: # do simulation over spin array
		spinarray = Build_SpinArray(fov=[1,1,1],dim=[1,1,100])
		spinarray.show_info()
		# print(spinarray.loc)
		z = spinarray.loc[2,:]
		a,b = slrsim_c(spinarray,Nt,dt,rf,gr)
		anew_real,anew_imag,bnew_real,bnew_imag = slrsim(spinarray,Nt,dt,rf,gr)
		print('old:',a[3:5])
		print('new:',anew_real[3:5])
		print('new:',anew_imag[3:5])
		# print('a:',a)
		# print('b:',b)
	# profile along z direction
	Mxy = 2*(a.conj())*b # excitation case
	Mz = a*(a.conj()) - b*(b.conj()) # excitation case
	plot_slr_profile(z, a, b)
	return
# def example_9_backward():
# 	"""using older simulator, is much slower"""
# 	print('\nExample of magnetization profile along one direction:')
# 	Nt = 400 #100
# 	dt = 0.01 # ms
# 	t0 = 100
# 	rf = 0.0*torch.zeros((2,Nt),device=device)
# 	rf[0,:] = 6*1e-3*torch.special.sinc((torch.arange(Nt,device=device)-200)/t0)
# 	gr = 0.0*torch.zeros((3,Nt),device=device) # mT/cm
# 	gr[2,:] = 0.1*torch.ones(Nt,device=device) # mT/cm
# 	# build a cube
# 	fov = [4,4,2]
# 	dim = [3,3,100]
# 	cube = Build_SpinArray(fov=fov,dim=dim)
# 	cube_seg = cube.get_spins([-0.1,0.1],[-0.1,0.1],[-2.,2.])
# 	loc = cube_seg.loc[2,:]

# 	rf.requires_grad = True
# 	gr.requires_grad = True
# 	M,M_hist = blochsim_array(cube_seg,Nt,dt,rf,gr)
# 	loss = torch.sum(M)
# 	print('loss:',loss)
# 	loss.backward()
# 	print('rf.grad:',rf.grad.shape)
# 	print('gr.grad:',gr.grad.shape)
# 	return
# ---------------------
# show possible examples:
def example():
	print()
	print(''.center(40,'-'))
	print('`mri` module examples:')
	print(''.center(40,'-'))

	# > Example 1: of building a spin
	print('Example of a spin:')
	spin = Spin(df=10., loc=[0.,0.,1.5])
	spin.set_Mag(torch.tensor([0.,1.,0.]))
	spin.show_info()
	print(''.center(40,'-'))

	# > Example 2: build a spin array
	print('\nExample of spin array:')
	n = 5
	loc = torch.rand((3,n),device=device)
	loc[0,:] = torch.tensor([0.,1.,0.,3.,4.],device=device)
	loc[1,:] = torch.tensor([0.,0.,0.,0.,0.],device=device)
	loc[2,:] = torch.tensor([0.,0.,0.,1.,0.],device=device)
	T1 = torch.ones(n,device=device)*1000.0
	T2 = torch.ones(n,device=device)*100.0
	spinarray = SpinArray(loc=loc,T1=T1,T2=T2)
	spinarray.df = torch.tensor([10.,5.,0.,0.,0.],device=device)
	spinarray.show_info()
	print('select spins by locations:\n\t',spinarray.get_index([3,6],[0,1],[0,1]))
	print(''.center(40,'-'))

	# > Example 3: build spin cube array using function
	print('\nExample of a cube:')
	print('build a cube:')
	fov = [4,4,2]
	dim = [3,3,500]
	cube = Build_SpinArray(fov=fov,dim=dim)
	cube.show_info()
	print('select some spins within the cube:')
	cube_seg = cube.get_spins([-0.1,0.1],[-0.1,0.1],[-2.,2.])
	cube_seg.show_info()
	print(''.center(40,'-'))

	# > Example 4: spin free precession
	example_4_spin_freeprecession()
	print(''.center(40,'-'))

	# > Example 5: Spin array bloch simulation
	example_5_spinarray_sim()
	print(''.center(40,'-'))

	# > Example 6: Example of magnetization profile along one direction
	example_6_arraysim_1d()
	print(''.center(40,'-'))

	# > Example 7: SLR simulation of a spin
	example_7_spin_SLR()
	print(''.center(40,'-'))

	# > Example 8: SLR simulation of spin array along z-direction
	example_8_array_slr_z()
	print(''.center(40,'-'))

	return
	# ------------
# example show form the loss and compute the gradient:
def example_backward():
	print('\nExample of spin array simulation:')
    # spin array:
	n = 5
	loc = torch.rand((3,n),device=device)
	loc[0,:] = torch.tensor([0.,1.,0.,3.,4.],device=device)
	loc[1,:] = torch.tensor([0.,0.,0.,0.,0.],device=device)
	loc[2,:] = torch.tensor([0.,0.,0.,1.,0.],device=device)
	T1 = torch.ones(n,device=device)*1000.0
	T2 = torch.ones(n,device=device)*100.0
	spinarray = SpinArray(loc=loc,T1=T1,T2=T2)
	spinarray.df = torch.tensor([10.,5.,0.,0.,0.],device=device)
	spinarray.show_info()

	# pulse:
	Nt = 1000
	dt = 1.0 # ms
	rf = 0.1*torch.ones((2,Nt),device=device)
	gr = 0.0*torch.zeros((3,Nt),device=device)
	gr[2,:] = torch.ones(Nt,device=device)*0.05

	# target for test
	Md = torch.zeros_like(spinarray.Mag,device=device)
	Md[1,:] = 1.0
	# print(Md)

	# backward of bloch simulation:
	if 1: # compute the true gradient
		rf.requires_grad = True
		# gr.requires_grad = True
		M,M_hist = blochsim_(spinarray,Nt,dt,rf,gr)
		print(M)
		print(Md)
		loss = torch.norm(M-Md)**2
		print('loss =',loss)
		loss.backward()
		# print(rf.grad.shape)
		print(rf.grad[:,:4])
		print()
	if 1: # test my gradient function
		rf.grad = None
		rf.requires_grad = True
		# gr.requires_grad = True
		# Beffhist = spinarray_Beffhist(spinarray,Nt,dt,rf,gr) # (3*num*Nt)
		# Beffhist.requires_grad = True
		M = blochsim(spinarray,Nt,dt,rf,gr)
		print(M)
		print(Md)
		loss = torch.norm(M-Md)**2
		print('loss =',loss)
		loss.backward()
		# print(M.grad.shape)
		# print(Beffhist.grad.shape)
		# print(Beffhist.grad)
		# print(rf.grad.shape)
		print(rf.grad[:,:4])
		print()

	# try backward of slr:
	if 1: # comute true slr simulation
		a,b = slrsim_c(spinarray,Nt,dt,rf,gr)
		loss = torch.sum(b.abs()**2)
		print('a:',a)
		print('b:',b)
		print('loss:',loss)
		print()
	if 1: # compute true gradient for slr
		rf.grad = None
		rf.requires_grad = True
		a_real,a_imag,b_real,b_imag = slrsim_(spinarray,Nt,dt,rf,gr)
		print('a real:',a_real)
		print('a imag:',a_imag)
		print('b real:',b_real)
		print('b imag:',b_imag)
		loss = torch.sum(b_real**2+b_imag**2)
		print('loss:',loss)
		loss.backward()
		print(rf.grad[:,:4])
		print()
	if 1: # test my slr gradient #todo
		rf.grad = None
		rf.requires_grad = True
		a_real,a_imag,b_real,b_imag = slrsim(spinarray,Nt,dt,rf,gr)
		print('a real:',a_real)
		print('a imag:',a_imag)
		print('b real:',b_real)
		print('b imag:',b_imag)
		loss = torch.sum(b_real**2+b_imag**2)
		print('loss:',loss)
		loss.backward()
		# print(rf.grad.shape)
		# print(rf.grad[:,:4])
		print()

	# save_infos(Nt,dt,rf,gr,'logs/mri_log.mat')
	return



if __name__ == "__main__":
	MR()
	example()
	example_backward()

	# example_pulse()
	# R = RotationMatrix()
	# R.example()

	# -------------------------
	# test the examples in function example():
	# example_4_spin_freeprecession()
	# example_5_spinarray_sim()
	# example_6_arraysim_1d()
	# example_7_spin_SLR()
	# example_8_array_slr_z()

	# some test function while coding: (not important)
	# test_freeprecession()
	# test2()
	# test3()
	# test_slr()
