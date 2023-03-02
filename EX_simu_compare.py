# compare the efficiency of different simulations, 
# the time used for calculating the derivatives
# author: jiayao
'''
use number of time points for RF is 1000
change of number of spins in simulation, and use different simulation functions
'''

import torch
from time import time
import matplotlib.pyplot as plt

import mri

device = torch.device('cuda:0')
# device = torch.device('cpu')

print('>> using device:', device)

def time_spinorsim_noautodiff(num=10, seed=0):
    '''
    num: number of spins used in the simulation
    '''
    torch.manual_seed(seed)

    loc = torch.randn((3,num),device=device)*10
    T1 = torch.ones(num,device=device)*1000.0
    T2 = torch.ones(num,device=device)*100.0
    spinarray = mri.SpinArray(loc=loc,T1=T1,T2=T2,device=device)
    # spinarray.show_info()

	# pulse:
    Nt = 1000
    dt = 1.0 # ms
    rf = 0.1*torch.randn((2,Nt),device=device) # mT
    gr = 5.0*torch.randn((3,Nt),device=device) # mT/m
    # print(gr[:,:5])

    rf.requires_grad = True
    gr.requires_grad = True
    ar,ai,br,bi = mri.spinorsim_(spinarray,Nt,dt,rf,gr,device=device)
    loss = torch.sum(ar+ai+br+bi)
    print('loss:',loss)

    starttime = time()
    loss.backward()
    t = time()-starttime
    print('print some value of computed derivative:')
    print(gr.grad[:,-4:])

    return t

def time_spinorsim(num=10, seed=0):
    '''
    num: number of spins used in the simulation
    '''
    torch.manual_seed(seed)

    loc = torch.randn((3,num),device=device)*10
    T1 = torch.ones(num,device=device)*1000.0
    T2 = torch.ones(num,device=device)*100.0
    spinarray = mri.SpinArray(loc=loc,T1=T1,T2=T2,device=device)
    # spinarray.show_info()

	# pulse:
    Nt = 1000
    dt = 1.0 # ms
    rf = 0.1*torch.randn((2,Nt),device=device) # mT
    gr = 5.0*torch.randn((3,Nt),device=device) # mT/m
    # print(gr[:,:5])

    rf.requires_grad = True
    gr.requires_grad = True
    ar,ai,br,bi = mri.spinorsim(spinarray,Nt,dt,rf,gr,device=device)
    loss = torch.sum(ar+ai+br+bi)
    print('loss:',loss)

    starttime = time()
    loss.backward()
    t = time()-starttime
    print('print some value of computed derivative:')
    print(gr.grad[:,-4:])

    return t

def main():
    time_1 = []
    time_2 = []
    # num_list = [10,100,500,1000,3000]
    num_list = [10,100,500,1000,3000,6000,10000,30000,50000]
    for n in num_list:
        print('num =',n)
        t1 = time_spinorsim_noautodiff(n)
        t2 = time_spinorsim(n)
        time_1.append(t1)
        time_2.append(t2)
        print()
    print(''.center(30,'-'))
    print('num of spins:',num_list)
    print('default in pytorch:',time_1)
    print('implement using our function:',time_2)

    # plot:
    picname = 'pictures/EX_simu_compare_speed.png'
    plt.figure()
    plt.plot(num_list,time_1,label='using default auto-diff in Pytorch',marker='o')
    plt.plot(num_list,time_2,label='our implementation using auto-diff',marker='o')
    plt.xlabel('simulated number of spins')
    plt.ylabel('time for computing derivative (unit: s)')
    plt.legend()
    plt.savefig(picname)
    print('save fig...'+picname)

    # Save:
    outputs = {'num_list':num_list,
        'info': 'compare backward with/out explicit Jacobain',
        'time_us': time_1,
        'time_py': time_2,
    }
    mri.save_infos(pulse=None,logname='ex_simu_compare.mat',otherinfodic=outputs)
    return

if __name__ == '__main__':
    main()