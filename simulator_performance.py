'''evaluate the performance of the simulator w/ explicit jacobians'''
import torch
import numpy as np
import pandas as pd
from time import time
# import matplotlib.pyplot as plt
# import tracemalloc
import os
import argparse
from sdrf import mri
from sdrf.mrsim import SpinDomain


outdir = 'outputs/simulator'
if not os.path.exists(outdir):
    os.mkdir(outdir)


device = torch.device('cuda:0')
spinnum_list = [10,100,500,1000,2000,5000,10000] # num of spins
Nt_list = [10,100,500,1000,2000,3000] # num of time points
numX, NtX = np.meshgrid(np.array(spinnum_list),np.array(Nt_list))
# --------------------------------
len_spinnum = len(spinnum_list)
len_Nt = len(Nt_list)
# print(len(spinnum_list),len(Nt_list))

def the_scripts():
    for num in spinnum_list:
        for nt in Nt_list:
            print('python simulator_performance.py --num {} --nt {} --simulator with --append 1'.format(num,nt))
            print('python simulator_performance.py --num {} --nt {} --simulator without --append 1'.format(num,nt))
    return



# comparison of speed and peak memory for evaluating the derivatives
def timing_simulation(simfun, num=10, seed=0, Nt=1000):
    '''Return running time and test derivative for a simulator.

    Args:
        simfun:     simulation function
        num:        number of spins used in the simulation
        seed:       randomness
        Nt:         number of times points
    '''
    print('using device:',device)
    torch.manual_seed(seed)

    loc = torch.randn((3,num),device=device)*10
    T1 = torch.ones(num,device=device)*1000.0
    T2 = torch.ones(num,device=device)*100.0
    spinarray = mri.SpinArray(loc=loc,T1=T1,T2=T2,device=device)
    # spinarray.show_info()

	# pulse:
    dt = 1.0 # ms
    rf = 0.1*torch.randn((2,Nt),device=device) # mT
    gr = 5.0*torch.randn((3,Nt),device=device) # mT/m

    rf.grad = gr.grad = None
    rf.requires_grad = True
    gr.requires_grad = True
    ar,ai,br,bi = simfun(spinarray,Nt,dt,rf,gr,device=device)
    loss = torch.sum(ar+ai+br+bi)
    print('loss:',loss)

    starttime = time()
    loss.backward()
    t = time()-starttime
    
    # print('print some value of computed derivative:')
    # print(gr.grad[:,-4:])
    testgrad = gr.grad

    return t,testgrad

def main(num,nt,simulator='without-explicit-jacobian',append_result=True):
    ''''''    
    # num = numX[nt_idx,num_idx]
    # nt = NtX[nt_idx,num_idx]
    # print('num = {}, Nt = {}'.format(num,nt))
    outputcsv = os.path.join(outdir,'perf.csv')


    # torch.cuda.memory._record_memory_history()
    if simulator=='without-explicit-jacobian':
        print('---- w/o explicit jacobian')
        t,grad = timing_simulation(SpinDomain._spinorsim_r_singlestep,num=num,Nt=nt)
        sim = 'wo'

    elif simulator=='with-explicit-jacobian':
        print('---- w/ explicit jacobian')
        t,grad = timing_simulation(SpinDomain.spinorsim,num=num,Nt=nt)
        sim = 'w'

    else:
        raise BaseException

    cuda_max_memory = torch.cuda.max_memory_allocated()
    print(torch.cuda.max_memory_allocated()) # bytes -> Gb
    # torch.cuda.memory_stats()
    # torch.cuda.memory._dump_snapshot(os.path.join(outdir,"my_snapshot.pickle"))
    # print(torch.cuda.max_memory_reserved()*1e-9)


    # save the results
    x = {
        'n_spins': [num],
        'n_timepoints': [nt],
        'timecost': [t],
        'cuda_max_mem': [cuda_max_memory],
        'simulator': [sim]
    }
    df = pd.DataFrame(x)
    if append_result:
        dfext = pd.read_csv(outputcsv)
        dfnew = pd.concat([dfext,df],ignore_index=True)
        dfnew.to_csv(outputcsv,index=False)
    else:
        df.to_csv(outputcsv,index=False)


    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Simulation comparson.')
    parser.add_argument('--num',default=-1,type=int)
    parser.add_argument('--nt',default=-1,type=int)
    parser.add_argument('--simulator',default='with',type=str,help='with/out jacobian implementation')
    parser.add_argument('--append',default=1,type=int,help='append the results')
    args = parser.parse_args()
    num = args.num
    nt = args.nt
    if args.simulator=='with':
        simulator = 'with-explicit-jacobian'
    elif args.simulator=='without':
        simulator = 'without-explicit-jacobian'
    else:
        raise BaseException
    if args.append > 0:
        append = True
    else:
        append = False

    # the_scripts()
    main(num=num,nt=nt,simulator=simulator,append_result=append)