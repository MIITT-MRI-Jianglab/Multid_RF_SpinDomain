# example that design 3d refocusing pulse using spin-domain representation
# author: jiayao

import argparse
import torch
import os
from mritools import MRData
from mritools import mri
from mritools import mripulseOPT

import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='3d excitation pulse design'
    )
    parser.add_argument('--folder',type=str,default='test',help='folder name for results')
    parser.add_argument('--optimization',type=int,default=-1,help='optimization')
    parser.add_argument('--evaluation',type=int,default=-1,help='evaluation')
    parser.add_argument('--b0map',type=int,default=0,help='which b0 map')
    parser.add_argument('--b1map',type=int,default=0,help='which b1 map')
    args = parser.parse_args()
    # -----
    foldername = args.folder
    run_optimization = True if args.optimization > 0 else False
    run_evaluation = True if args.evaluation > 0 else False
    out_simulation_name = 'simulation_{}{}.mat'.format(args.b0map,args.b1map)
    # 
    outputdir = 'outputs'
    init_pulse_file = 'data/pulse_init_3d.mat'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    '''initial pulse'''
    pulse_init = mri.Pulse.load(init_pulse_file,device=device)
    pulse_init.change_dt(5e-3)
    
    '''b0 map'''
    if args.b0map > 0:
        b0mapfile = utils.b0mapfile_list[args.b0map]
        b0data = MRData.MRData.load(b0mapfile)
        b0map = b0data.image
        # b0data.info()
    else:
        b0map = torch.zeros(40,40,28)
    
    '''b1 map'''
    if args.b1map > 0:
        b1mapfile = utils.b1mapfile_list[args.b1map]
        b1data = MRData.MRData.load(b1mapfile)
        b1map = b1data.image.abs()
        # b1data.info()
    else:
        b1map = torch.ones(40,40,28)
    
    '''mask'''
    maskdata = MRData.MRData.load(utils.phantom_maskfile)
    mask = maskdata.image




    '''optimization spins target'''
    fov = [30,30,10]
    dim = [40,40,28]
    cube = mri.SpinGrid(fov=fov,dim=dim,device=device)
    cube.set_mask(mask.to(device))
    if args.b0map >= 0: cube.set_B0map(b0map.to(device))
    if args.b1map >= 0: cube.set_B1map(b1map.to(device))
    # cube.show_info()
    


    '''pulse optimization'''
    opti_roi = [8,8,6]
    opti_transition_width=[1.6,1.6,1.2]
    mripulseOPT.optimize_excitation(
        pulse_init,cube,
        opti_roi=opti_roi, 
        opti_transition_width=opti_transition_width, 
        opti_roi_flip=90, opti_roi_phase=270,
        rfmax=0.014, grmax=24, slewrate=120, 
        niter=10, rf_niter=5, gr_niter=5, lam=2e4, 
        output_dir=os.path.join(outputdir,foldername),
        output_pulsename='rf_excitation.mat', 
        output_simulation_name=out_simulation_name,
        run_optimization=run_optimization, run_evaluation=run_evaluation,
        device=device
    )
    
    print()