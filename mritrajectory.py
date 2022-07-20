# mri trajetories
# author: jiayao

import numpy as np
import torch
import matplotlib.pyplot as plt



device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
# print('>> mri: using device:',device)
SAVE_FIG = True


# ----------------------------------------------------------------
# different functions design the shape of the gradient
# ----------------------------------------------------------------
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
    return
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
def sin_shape(Nt,dt=1.0,periods=2):
    '''Nt, dt:(ms)
    maximum value is 1.0 in z direction'''
    gr = torch.zeros((3,Nt),device=device)
    T = Nt/2
    omega = 2*torch.pi/T
    t = torch.arange(Nt,device=device)
    gr[2,:] = torch.sin(omega*t)
    return gr
def spiral(Nt,dt=1.0):
    '''Nt:, dt: (ms)'''
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





# --------------------------------------------------------
# plot functions
# ---------------------------------------------------------
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
def plot_kspace(gr,dt=1.0,case='excite',picname='pictures/mri_tmp_pic_kspace.png',save_fig=False):
    '''gr:(3*Nt)(mT/cm), dt:(1)(ms)'''
    if case == 'excite':
        kt = torch.cumsum(gr,dim=1)
        kt = np.array(kt.tolist())
        kt = np.concatenate((np.array([[0],[0],[0]]),kt),axis=1)
        kt = kt - kt[:,-1].reshape(3,1)
    elif case == 'image':
        kt = torch.cumsum(gr,dim=1)
        kt = np.array(kt.tolist())
        kt = np.concatenate((np.array([[0],[0],[0]]),kt),axis=1)
    maxk = np.amax(np.absolute(kt),axis=1)
    maxkx,maxky,maxkz = max(maxk),max(maxk),max(maxk)
    # print(maxx)
    # plot
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(kt[0,:],kt[1,:],kt[2,:],label='k-trajectory')
    # ax.plot((0,M[0,0]),(0,M[1,0]),(0,M[2,0]),linewidth=1,linestyle='--')
    # ax.plot((0,M[0,-1]),(0,M[1,-1]),(0,M[2,-1]),linewidth=1,linestyle='--')
    # ax.text(M[0,0],M[1,0],M[2,0],r'$k_0$',fontsize=8)
    # ax.text(M[0,-1],M[1,-1],M[2,-1],r'end',fontsize=8)
    ax.text(kt[0,0],kt[1,0],kt[2,0],r'start',fontsize=8)
    ax.legend()
    ax.set_xlim(-1.1*maxkx,1.1*maxkx)
    ax.set_ylim(-1.1*maxky,1.1*maxky)
    ax.set_zlim(-1.1*maxkz,1.1*maxkz)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if save_fig:
        print('save fig...'+picname)
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
    print('mritrajectory.py\n')
    # gr = spiral(1000)
    # gr = sin_shape(1000,periods=2)
    gr = squares(1000,periods=2)
    plot_G(gr,1,save_fig=SAVE_FIG)
    plot_kspace(gr,save_fig=SAVE_FIG)