# MRIPulse
**Multidimensional RF pulse design in spin-domain using auto-differentiation**

author: Jiayao

This repository provides code to simulate the spin-domain rotation and design multidimensional RF pulse in spin-domain for magnetic resonance imaging. The simulation is built using auto-differentiation in Pytorch. And demo for designing a 3D refocusing pulse is provided. For running large experiments, it would be better to use GPU and have enough memory.

## Environment
My running environment
- Ubuntu 20.04.4 LTS
- Python 3.9.10
- Pytorch 1.10.2+cu113
- Cuda 11.3 (for GPU)

## Preliminaries


## General description
The basic functions for describing the spin behaviors and simulations including spin-domain simulation are provided in `mri.py`. Other files provided different utilities, and a breif description is as follows.
- `mri.py`: provides definition of basic class, e.g. `Spin`, `SpinArray`, `Pulse`, `Signal`. It also provides simulation functions with autodifferentiation.
- `mriopt.py`: provides solver for optimizing the object loss function. 



## Demos
Demos

**Demo: design of 3D refocusing pulse**
1. Running design 3D refocusing pulse with inhomo B0 and B1 maps
2. Running design 3D refocusing pulse without B0 and B1 maps (assuming homogeneous)
3. Final simulation of the designed pulses

## Additional experiments
- Compare methods for calculating derivative (for spin-domain simulation): 
  - Two methods are compared, one is only build forward computation, another is our implementation with both forward and backward function. The backward function is explicitly implemented in Pytorch using auto-differentiation. 
  - The experiment is `EX_simu_compare.py`
  - The comparison of the speed: (1000 time points for RF pulse) ![](https://github.com/MIITT-MRI-Jianglab/MRIPulse/blob/main/EX_results/EX_simu_compare_speed.png)
- How transverse magnetization rotate with different spin-domain parameters:
  - The experiment is `EX_transverse_rotation_illustration.py`. 
  - Considered different value of $\beta^2$ with norm equals 1 produces the rotation in transverse plane. For example, green denotes the initial transverse magnetization, red denotes the rotated magnetization, and blue denotes the computed rotation axis from them. ![](https://github.com/MIITT-MRI-Jianglab/MRIPulse/blob/main/EX_results/EX_transverse_rotation_illustration.png) 


## Acknowledgements
This work is inspired by and takes reference of
- https://github.com/tianrluo/AutoDiffPulses
- https://github.com/mikgroup/sigpy
