# generate some fake maps
# author: jiayao

import numpy as np
import torch

import mri


if __name__ == '__main__':
    device = torch.device('cuda:0')
    fov = [30,30,20]
    dim = [70,70,40]
    cube = mri.Build_SpinArray(fov=fov,dim=dim,device=device)

    # build B0 map
    loc = cube.loc
    dis_square = (loc[0,:]-15)**2 + 2*(loc[1,:]-15)**2 + (loc[2,:]-10)**2
    offres = 0.2*dis_square - 300
    dis_square = (loc[0,:]+2)**2 + (loc[1,:]+2)**2 + (loc[2,:]-1)**2
    kappa = 1.2-0.6*dis_square/dis_square.max()

    cube.kappa = kappa
    cube.df = offres

    mri.plot_cube_slices(cube,kappa,picname='pictures/tmppic_fake_B1.png',savefig=True)
    mri.plot_cube_slices(cube,offres,picname='pictures/tmppic_fake_B0.png',savefig=True)


    # save my fake maps
    # change to numpy arrays
    b1map = np.array(cube.get_kappagrid().tolist())
    b0map = np.array(cube.get_dfgrid().tolist())
    cube_loc = np.array(cube.get_locgrid().tolist())
    loc = np.zeros((dim[0],dim[1],dim[2],3))
    for x in range(dim[0]):
        for y in range(dim[1]):
            for z in range(dim[2]):
                loc[x,y,z,:] = cube_loc[:,x,y,z]
    loc_x = np.array(cube.grid_x.tolist())
    loc_y = np.array(cube.grid_y.tolist())
    loc_z = np.array(cube.grid_z.tolist())
    print(b0map.shape,b1map.shape,loc.shape)
    print(loc_x.shape)
    print(loc_y.shape)
    print(loc_z.shape)

    # save B0 map
    datadic = {
        'b0map': b0map,
        'loc':loc,
        'loc_x':loc_x,
        'loc_y':loc_y,
        'loc_z':loc_z,
    }
    mri.save_matlabdata('/scratch/JY_ImageRecon/Recon_Images/init_fake_B0map.mat', datadic)

    # save B1 map
    datadic = {
        'b1map': b1map,
        'loc':loc,
        'loc_x':loc_x,
        'loc_y':loc_y,
        'loc_z':loc_z,
    }
    mri.save_matlabdata('/scratch/JY_ImageRecon/Recon_Images/init_fake_B1map.mat', datadic)

