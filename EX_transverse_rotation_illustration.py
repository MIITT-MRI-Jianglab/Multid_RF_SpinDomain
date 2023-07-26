# Illustration of transverse rotation using spin-domain parameters
# author: jiayao
'''
how transverse magnetization in mri rotates with different spin-domain parameters
'''

import numpy as np
import matplotlib.pyplot as plt


def rotation_case(M, beta_square=-1):
    Mnew = -beta_square*np.conj(M)
    mag = np.abs(Mnew)
    angle = np.angle(Mnew)
    return mag,angle

def main():
    '''
    with all |beta|=1, alpha=0
    '''
    n = 2
    # theta = np.arange(n)*2*np.pi/n
    # beta_square_list = np.exp((0+1j)*theta)
    # print(beta_square_list)
    beta_square_list = np.array([-1,-np.exp((0+1j)*np.pi/6)])

    num = 8
    theta = np.arange(num)
    M_list = np.exp((0+1j)*np.pi/16*theta)
    M = np.exp((0+1j)*np.pi/4)

    # plot:
    picname = 'EX_transverse_rotation_illustration.png'
    fig,axs = plt.subplots(n,num,figsize=(2*num,2*n),subplot_kw={'projection': 'polar'})
    for i in range(n):
        for j in range(num):
            beta_square = beta_square_list[i]
            if j==0:
                axs[i,j].set_title(r'$\beta^2=$'+'{:.2f}+({:.2f})i'.format(np.real(beta_square),np.imag(beta_square)))
            M = M_list[j]
            # initial:
            mag_init,angle_init = np.abs(M),np.angle(M)
            axs[i,j].plot([0,angle_init],[0,mag_init],ls='--',color='green')
            axs[i,j].text(angle_init,mag_init+0.1,'-',color='green')
            # end:
            mag_end,angle_end = rotation_case(M,beta_square)
            axs[i,j].plot([0,angle_end],[0,mag_end],color='red')
            axs[i,j].text(angle_end,mag_end+0.1,'+',color='red')
            # rotation axis:
            mag,angle = 1,(angle_init+angle_end)/2
            axs[i,j].plot([angle,angle],[0,mag],ls='-.',color='blue')
            # 
            axs[i,j].set_rticks([])  # Less radial ticks
    fig.tight_layout()
    plt.savefig(picname)
    print('save fig...'+picname)
    return

if __name__ == '__main__':
    main()