import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


spinnum_list = [10,100,500,1000,2000,5000,10000] # num of spins
Nt_list = [10,100,500,1000,2000,3000] # num of time points
numX, NtX = np.meshgrid(np.array(spinnum_list),np.array(Nt_list))


if __name__=='__main__':
    outputcsv = 'outputs/simulator/perf.csv'
    # outputcsv = 'outputs/exampleResults_simulator/perf.csv'

    df = pd.read_csv(outputcsv)
    # print(df)
    # print(df['n_spins'])
    # print(numX.shape)

    tcost_default = np.zeros_like(numX)
    tcost_ours = np.zeros_like(numX)
    gpucost_default = np.zeros_like(numX)
    gpucost_ours = np.zeros_like(numX)
    for n in range(len(spinnum_list)):
        for m in range(len(Nt_list)):
            num = spinnum_list[n]
            nt = Nt_list[m]
            d = df[df['n_spins']==num]
            d = d[d['n_timepoints']==nt]
            # print(d)
            
            d_default = d[d['simulator']=='wo']
            tcost_default[m,n] = d_default.iloc[0]['timecost']
            gpucost_default[m,n] = d_default.iloc[0]['cuda_max_mem']

            d_ours = d[d['simulator']=='w']
            tcost_ours[m,n] = d_ours.iloc[0]['timecost']
            gpucost_ours[m,n] = d_ours.iloc[0]['cuda_max_mem']


    fig = plt.figure(figsize=(8,4))

    # Time cost
    ax = plt.subplot(121,projection='3d')
    ax.plot_surface(numX, NtX, tcost_default, alpha=0.5, color='blue', label='default')
    ax.plot_surface(numX, NtX, tcost_ours, alpha=0.5, color='red', label='with explicit Jacobian')
    ax.set_title('(A) Time cost for calculating derivatives')
    ax.set_ylabel('number of time points')
    ax.set_xlabel('number of spins')
    ax.legend()
    
    
    # GPU usage
    ax = plt.subplot(122,projection='3d')
    ax.plot_surface(numX, NtX, gpucost_default, alpha=0.5, color='blue', label='default')
    ax.plot_surface(numX, NtX, gpucost_ours, alpha=0.5, color='red', label='with explicit Jacobian')
    ax.set_title('(B) Monitored GPU usage for calculating derivatives')
    ax.set_ylabel('number of time points')
    ax.set_xlabel('number of spins')

    plt.tight_layout()
    plt.savefig('outputs/simulator_perf.png')
    plt.close(fig)


    # ------------------------------

    print('save figure to:')
    print('outputs/simulator_perf.png')