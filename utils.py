# some functions
import os

maindir = ''
b0mapfile_list = ['homogeneous',
                  os.path.join(maindir,'data/phantom_b0_1.mat'),
                  os.path.join(maindir,'data/phantom_b0_2.mat'),
                  os.path.join(maindir,'data/phantom_b0_3.mat'),
                  os.path.join(maindir,'data/phantom_b0_4.mat'),]
b1mapfile_list = ['homogeneous',
                  os.path.join(maindir,'data/phantom_b1_1.mat'),
                  os.path.join(maindir,'data/phantom_b1_2sim.mat'),
                  os.path.join(maindir,'data/phantom_b1_3sim.mat')]
phantom_maskfile = os.path.join(maindir,'data/phantom_mask.mat')
