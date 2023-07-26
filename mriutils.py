# Some random stuffs 
# author: jiayao


import torch
import scipy.io as spio
from scipy import interpolate
import numpy as np




# ###########################################################

def save_variables(var_dic,outputpath=None):
    '''save the variables, to .mat datafile'''
    if outputpath:
        try:
            spio.savemat(outputpath,var_dic)
            print('| save variables...'+outputpath)
            print('| -- saved variables:',var_dic.keys())
        except:
            print('error in saving variables!')
    else:
        print('no outputpath!')
    return

# ###########################################################
def load_initial_b0map(filepath):
    '''
    function load b0 map

    input: file path
    return (numpy array): b0map,loc_x,loc_y,loc_z (assume unit: cm)

    the input file is .mat file, and contains variables 
        - b0map: 3d matrix
        - loc_x
        - loc_y
        - loc_z
    '''
    # read in data file
    try:
        data = spio.loadmat(filepath)
        B0map = data['b0map']
        loc_x = data.get('loc_x')
        loc_y = data.get('loc_y')
        loc_z = data.get('loc_z')
        # test = data.get('test')
        try: 
            loc_x = loc_x.flatten()
            loc_y = loc_y.flatten()
            loc_z = loc_z.flatten()
        except:
            pass
        # print(x.shape, max(x),max(y),max(z))
        print('| read in measured B0map | '+filepath, B0map.shape, B0map.dtype, loc_x.dtype)
        # print(B0map.shape)
        init_fail = False
    except:
        init_fail = True
        print('| read in B0 map file fails!!')
        # print('>> initial B0 maps does not exist')
        
    return B0map,loc_x,loc_y,loc_z

def load_initial_b1map(filepath):
    '''
    function load b1 map

    input: file path
    return (numpy array): b1map,loc_x,loc_y,loc_z 

    the input file is .mat file, and contains variables 
        - b1map: 3d matrix
        - loc_x
        - loc_y
        - loc_z
    '''
    # read in data file
    try:
        data = spio.loadmat(filepath)
        B1map = data['b1map']
        loc_x = data.get('loc_x')
        loc_y = data.get('loc_y')
        loc_z = data.get('loc_z')
        # test = data.get('test')
        try: 
            loc_x = loc_x.flatten()
            loc_y = loc_y.flatten()
            loc_z = loc_z.flatten()
        except:
            pass
        # print(x.shape, max(x),max(y),max(z))
        print('| read in measured B1map | '+filepath, B1map.shape, B1map.dtype)
        # print(B0map.shape)
        init_fail = False
    except:
        init_fail = True
        print('| read in B1 map file fails!!')
        # print('>> initial B0 maps does not exist')
        
    return B1map,loc_x,loc_y,loc_z


def load_object_mask(filepath):
    '''
    function load b1 map

    input: file path
    return (numpy array): mask,loc_x,loc_y,loc_z 

    the input file is .mat file, and contains variables 
        - mask: 3d matrix
        - loc_x
        - loc_y
        - loc_z
    '''
    # read in data file
    try:
        data = spio.loadmat(filepath)
        mask = data['mask']
        loc_x = data.get('loc_x')
        loc_y = data.get('loc_y')
        loc_z = data.get('loc_z')
        # test = data.get('test')
        try: 
            loc_x = loc_x.flatten()
            loc_y = loc_y.flatten()
            loc_z = loc_z.flatten()
        except:
            pass
        # print(x.shape, max(x),max(y),max(z))
        print('| read in mask | '+filepath, mask.shape, mask.dtype, loc_x.dtype)
        # print(B0map.shape)
        init_fail = False
    except:
        init_fail = True
        print('| read in mask fails!!')
        # print('>> initial B0 maps does not exist')
        
    return mask,loc_x,loc_y,loc_z

# def map3d_interpolate():
#     '''
#     interpolate 3d map matrix grid to desired spatial locations 
#     input:
#     - 
#     output:
#     - interplated map (3d numpy matrix)
#     '''
#     # if True:
#     # try:
#     #     # Interpolation
#     #     interp_fn = interpolate.RegularGridInterpolator((x,y,z),B0map)
#     #     loc,loc_x,loc_y,loc_z = mri.Build_SpinArray_loc(fov=fov,dim=dim)
#     #     loc_x,loc_y,loc_z = loc_x.numpy(),loc_y.numpy(),loc_z.numpy()
#     #     # print(max(loc_x),max(loc_y),max(loc_z))
#     #     loc = loc.numpy().reshape(3,-1).T
#     #     B0map = interp_fn(loc)
#     #     B0map = torch.tensor(B0map,device=device).reshape(dim)
#     #     print('>> interpolate B0',B0map.shape)
#     #     # print(B0map.shape)
#     # except:
#     #     init_fail = True
#     #     print('>> interpolation fails !!')
#     # # ----------------------------------------------
#     # if init_fail:
#     #     B0map = torch.zeros(dim,device=device) # (Hz)
#     # print('>> adding B0 maps... max =',torch.max(B0map.abs()).item())

#     return