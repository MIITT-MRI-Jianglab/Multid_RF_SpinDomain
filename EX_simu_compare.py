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
import tracemalloc

import mri

device = torch.device('cuda:0')
# device = torch.device('cpu')

print('>> using device:', device)

def time_spinorsim_noautodiff(num=10, seed=0, Nt=1000):
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
    # Nt = 1000
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

def time_spinorsim(num=10, seed=0, Nt=1000):
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
    # Nt = 1000
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
    # First experiment, when change the number of spins
    ex1_time_py = []
    ex1_time_us = []
    ex1_memo_sizelist_py = []
    ex1_memo_peaklist_py = []
    ex1_memo_sizelist_us = []
    ex1_memo_peaklist_us = []
    # num_list = [10,100,500,1000,3000]
    num_list = [10,100,500,1000,3000,6000,10000,30000,50000]
    tracemalloc.start() # starting monitoring the memory
    for n in num_list:
        nt = 1000
        print('num =',n, 'Nt =',nt)

        print('---- w/o explicit jacobian')
        tracemalloc.reset_peak()
        t1 = time_spinorsim_noautodiff(num=n,Nt=nt)
        cursize,peaksize = tracemalloc.get_traced_memory()
        print('current size = {}, peak = {}'.format(cursize,peaksize))
        ex1_time_py.append(t1)
        ex1_memo_sizelist_py.append(cursize)
        ex1_memo_peaklist_py.append(peaksize)

        print('---- w/ explicit jacobian')
        tracemalloc.reset_peak()
        t2 = time_spinorsim(num=n,Nt=nt)
        cursize,peaksize = tracemalloc.get_traced_memory()
        print('current size = {}, peak = {}'.format(cursize,peaksize))
        ex1_memo_sizelist_us.append(cursize)
        ex1_memo_peaklist_us.append(peaksize)
        ex1_time_us.append(t2)

        print()
    tracemalloc.stop()
    
    print(''.center(30,'-'))
    print('num of spins:',num_list)
    print('default in pytorch:',ex1_time_py)
    print('implement using our function:',ex1_time_us)
    print(ex1_memo_sizelist_py)
    print(ex1_memo_sizelist_us)
    print(ex1_memo_peaklist_py)
    print(ex1_memo_peaklist_us)


    # Second experiment, when increase number of timepoints
    ex2_time_py = []
    ex2_time_us = []
    ex2_memo_sizelist_py = []
    ex2_memo_peaklist_py = []
    ex2_memo_sizelist_us = []
    ex2_memo_peaklist_us = []
    Nt_list = [10,100,500,1000,2000,4000]
    tracemalloc.start() # starting monitoring the memory
    for nt in Nt_list:
        num = 1000
        print('num =',num, 'Nt =',nt)

        print('---- w/o explicit jacobian')
        tracemalloc.reset_peak()
        t1 = time_spinorsim_noautodiff(num=num,Nt=nt)
        cursize,peaksize = tracemalloc.get_traced_memory()
        print('current size = {}, peak = {}'.format(cursize,peaksize))
        ex2_time_py.append(t1)
        ex2_memo_sizelist_py.append(cursize)
        ex2_memo_peaklist_py.append(peaksize)

        print('---- w/ explicit jacobian')
        tracemalloc.reset_peak()
        t2 = time_spinorsim(num=num,Nt=nt)
        cursize,peaksize = tracemalloc.get_traced_memory()
        print('current size = {}, peak = {}'.format(cursize,peaksize))
        ex2_memo_sizelist_us.append(cursize)
        ex2_memo_peaklist_us.append(peaksize)
        ex2_time_us.append(t2)

        print()
    tracemalloc.stop()
    
    print(''.center(30,'-'))
    print('Nt of timepoints:',Nt_list)
    print('default in pytorch:',ex2_time_py)
    print('implement using our function:',ex2_time_us)
    print(ex2_memo_sizelist_py)
    print(ex2_memo_sizelist_us)
    print(ex2_memo_peaklist_py)
    print(ex2_memo_peaklist_us)

    
    # plot of the results -------------------------------
    picname = 'EX_simu_compare_speed.png'
    plt.figure(figsize=(12,10))
    ax1 = plt.subplot(2,2,1)
    ax1.plot(num_list,ex1_time_py,label='default auto-differentiation in PyTorch',color='blue',marker='o')
    ax1.plot(num_list,ex1_time_us,label='our simulator with explicit Jacobian',color='red',marker='o')
    ax1.set_ylabel('time calculating derivative (s)')
    ax1.set_xlabel('number of spins')
    ax1.legend()
    ax1.set_title('when increase num of spins (#timepoints=1000)')
    
    ax2 = plt.subplot(2,2,2)
    ax2.plot(Nt_list,ex2_time_py,label='default auto-differentiation in PyTorch',color='blue',marker='o')
    ax2.plot(Nt_list,ex2_time_us,label='our simulator with explicit Jacobian',color='red',marker='o')
    ax2.set_ylabel('time culculating derivative (s)')
    ax2.set_xlabel('number of timepoints')
    ax2.legend()
    ax2.set_title('when increase num of timepoints (#spins=1000)')

    ax3 = plt.subplot(2,2,3)
    ax3.plot(num_list,ex1_memo_peaklist_py,label='default auto-differentiation in PyTorch',color='blue',marker='o')
    ax3.plot(num_list,ex1_memo_peaklist_us,label='our simulator with explicit Jacobian',color='red',marker='o')
    ax3.set_ylabel('monitored peak memory (byte)')
    ax3.set_xlabel('number of spins')
    ax3.legend()

    ax4 = plt.subplot(2,2,4)
    ax4.plot(Nt_list,ex2_memo_peaklist_py,label='default auto-differentiation in PyTorch',color='blue',marker='o')
    ax4.plot(Nt_list,ex2_memo_peaklist_us,label='our simulator with explicit Jacobian',color='red',marker='o')
    ax4.set_ylabel('monitored peak memory (byte)')
    ax4.set_xlabel('number of timepoints')
    ax4.legend()
    
    plt.savefig(picname)
    print('save fig...'+picname)

    # Save results --------------------------------------------
    outputs = {'info': 'compare backward with or without explicit Jacobain',
        'num_list':num_list,
        'ex1_time_py': ex1_time_py,
        'ex1_time_us': ex1_time_us,
        'ex1_memosize_py': ex1_memo_sizelist_py,
        'ex1_memosize_us': ex1_memo_sizelist_us,
        'ex1_memopeak_py': ex1_memo_peaklist_py,
        'ex1_memopeak_us': ex1_memo_peaklist_us,
        'Nt_list': Nt_list,
        'ex2_time_py': ex2_time_py,
        'ex2_time_us': ex2_time_us,
        'ex2_memosize_py': ex2_memo_sizelist_py,
        'ex2_memosize_us': ex2_memo_sizelist_us,
        'ex2_memopeak_py': ex2_memo_peaklist_py,
        'ex2_memopeak_us': ex2_memo_peaklist_us,
    }
    mri.save_pulse(pulse=None,logname='ex_simu_compare.mat',otherinfodic=outputs)

    

    return

if __name__ == '__main__':
    main()
    