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
- `mri.py`: 
- `mriopt.py`:



## Demos
Demos
- `EX_pulse_design_3d.py`

## Additional experiments
- Compare methods of calculating derivative
