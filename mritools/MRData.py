# class of data, for easier operations
import numpy as np
import torch
import math
import os

from scipy import interpolate
import scipy.io as spio
import mat73

# # use pydicom to read DICOM files
# try:
#     from pydicom import dcmread
# except:
#     print('MRData import pydicom failed!')

# from mritools.callMatlab import readtwixMatlab




def calculate_slice_position(n_slice,slicethickness,distance_factor)->torch.Tensor:
    '''Return array of slice position. -> tensor
    
    Also a consideration of acquired images from Siemens.
    '''
    return (torch.arange(n_slice)-n_slice//2)*slicethickness*(1+distance_factor)
def calculate_effective_slice_fov(n_slice,slicethickness,distance_factor):
    '''Return the fov along slice direction considering distance factor'''
    return n_slice*slicethickness*(1+distance_factor)
def calculate_fovpixel_position(fov,npoints)->torch.Tensor:
    '''Return 1D array of pixel locations. (unit the same as fov)
    
    The location = 0 is always included.
    '''
    return (torch.arange(npoints)-npoints//2)*fov/npoints
def get_xyz_matrix(locx,locy,locz) -> torch.Tensor:
    '''Return location matrix of size (3,x,y,z).'''
    nx,ny,nz = len(locx),len(locy),len(locz)
    loc = torch.zeros((3,nx,ny,nz))
    for i in range(nx):
        loc[0,i,:,:] = locx[i]
    for i in range(ny):
        loc[1,:,i,:] = locy[i]
    for i in range(nz):
        loc[2,:,:,i] = locz[i]
    return loc
def get_xyz_matrix_from_fov(fov,dim) -> torch.Tensor:
    '''Return location matrix of size (3,x,y,z).'''
    locx = calculate_fovpixel_position(fov[0],dim[0])
    locy = calculate_fovpixel_position(fov[1],dim[1])
    locz = calculate_fovpixel_position(fov[2],dim[2])
    return get_xyz_matrix(locx,locy,locz)
def image_interpolate(img:torch.Tensor,fov,tarfov,tardim=None,taroffset=[0,0,0],details=False)->torch.Tensor:
    '''Interpolate 3D image to new FOV and dimensions.
    
    inputs:
        img: (nx,ny,nz)           reference image
        fov: e.g., [1,1,1]        reference FOV
        tarfov: e.g., [5,5,5]     target FOV
        tardim: e.g., [8,8,8]     target dimensions
        taroffset:                target shift of center
    '''
    # first make sure the image shape is (nx,ny,nz)
    # if len(img.shape) == 2: 
    #     nx,ny = img.shape
    #     img = img.reshape(nx,ny,1)
    # print('(img interpolation) fov: {}->{}'.format(fov,tarfov))
    # print('(img interpolation) img: {}->{}'.format(list(img.shape),tardim))

    device = img.device
    dim = img.shape
    
    # Change data to numpy for interpolation
    # and get locations for reference image (for building interpolation function)
    if tardim==None: tardim=list(img.shape)
    img = img.cpu().numpy()
    loc_x = calculate_fovpixel_position(fov[0],dim[0]).cpu().numpy()
    loc_y = calculate_fovpixel_position(fov[1],dim[1]).cpu().numpy()
    loc_z = calculate_fovpixel_position(fov[2],dim[2]).cpu().numpy()

    # if len(dim)==2:
    #     # make sure the image is of shape (nx*ny*nz)
    #     X = X.reshape(self.nx,self.ny,self.nz)
    

    # Build interpolation function
    # use 'nearest' method, which is acceptable when interpolate to higher resolution
    # or use 'linear'
    # -------------------------------
    interp_fn = interpolate.RegularGridInterpolator(
        (loc_x,loc_y,loc_z), img, method='nearest'
    )
    # ----------------------------------------------------
    # Get the coordinates for target new image
    loc_target = get_xyz_matrix_from_fov(tarfov,tardim).cpu().numpy().reshape(3,-1) # (3*num)
    loc_target[0] = loc_target[0] + taroffset[0]
    loc_target[1] = loc_target[1] + taroffset[1]
    loc_target[2] = loc_target[2] + taroffset[2]
    # Considering the locations that can not be handled in the interpolation function
    # project its location to the nearest one
    loc_target[0] = np.clip(loc_target[0],np.min(loc_x),np.max(loc_x))
    loc_target[1] = np.clip(loc_target[1],np.min(loc_y),np.max(loc_y))
    loc_target[2] = np.clip(loc_target[2],np.min(loc_z),np.max(loc_z))
    # -------------------------------------------
    Xnew = interp_fn(loc_target.T)                           # Interpolation
    Xnew = torch.from_numpy(Xnew).to(device).reshape(tardim) # move to torch, and Reshape the new image

    if details:
        print('(image interpolation) fov:{}->{}, dim:{}->{}'.format(fov,tarfov,list(img.shape),list(Xnew.shape)))

    return Xnew

class MRData:
    """Class for data, raw data and possibly its corresponding images."""
    # twix_datadims = ['Col', 'Cha', 'Lin', 'Par', 'Sli', 'Ave', 'Phs', 'Eco', 
    #                 'Rep', 'Set', 'Seg', 'Ida', 'Idb', 'Idc', 'Idd', 'Ide']
    def __init__(self,*,
        data=None, datainfo:dict={}, datadims=[],
        coilmaps=None, imageall=None,
        image=None, imagemask=None, 
        fov=[0.,0.,0.],
        slice_thickness=0,slice_dist_factor=0,
        fov_x=0, fov_y=0,
        description='',
        # device=torch.device("cpu")
        ) -> None:
        '''MRdata
        
        Args:
            data: raw k-space data
            datainfo:
            datadims:
            coilmaps:
            imageall:
            image:
            imagemask:
            fov:
            slice_thickness:
            slice_dist_factor:
            description:
        '''
        # self.device = device

        # ------------------- relate to raw data
        self.data = data             # raw data
        self.datainfo = datainfo
        self.datadims = datadims
        # ------------- relate to images and recon details
        self.coilmaps = coilmaps
        self.imageall = imageall
        self.image = image           # recon image
        self.imagemask = imagemask   # image mask if needed
        # ----------------
        self.description = description

        # Other infos
        self.slice_thickness = slice_thickness # (mm)
        self.slice_dist_factor = slice_dist_factor # (mm)
        self.fov = fov
        self.fov_unit = 'mm'

        # check values
        if self.slice_thickness == None: self.slice_thickness = 0
        if self.slice_dist_factor == None: self.slice_dist_factor = 0
        if self.fov == None: self.fov = 0
        self.fov = np.array(self.fov)
        self.fov = np.squeeze(self.fov)



        '''TODO part'''
        # make last three dimensions of the image be
        # [..., Nx, Ny, Nz]
        self.imagedims = []
        self.image_multicoil = False

        # Other properties
        # self.fov = fov if (fov != None) else [0.,0.,0.] 

        # some pre-defined properties
        self.twix_datadims = ['Col', 'Cha', 'Lin', 'Par', 'Sli', 'Ave', 'Phs', 'Eco', 
                              'Rep', 'Set', 'Seg', 'Ida', 'Idb', 'Idc', 'Idd', 'Ide']
        self.prefer_datadims = []
    # --------------------------------------------------------------
    # save the data and load the data
    # -------------------------------------------------------------
    def save(self,datapath='mrd_tmp.mat',
             data=False,image=False,imageall=False,coilmaps=False,imagemask=False,
             saveall=False):
        """Save the data with required fields.
        
        suggest to use name, mrd_xxxxx.mat
        """
        if saveall:
            data,image,imageall,coilmaps,imagemask = True,True,True,True,True
        txt = []
        if data: txt.append('data')
        if image: txt.append('image')
        if imageall: txt.append('imageall')
        if coilmaps: txt.append('coilmaps')
        if imagemask: txt.append('imagemask')
        def data_to_np(x):
            if x==None:
                return []
            elif isinstance(x,torch.Tensor):
                return x.cpu().numpy()
            else:
                return x
        var_dic = {
            'data': data_to_np(self.data) if data else [],
            'coilmaps': data_to_np(self.coilmaps) if coilmaps else [],
            'image': data_to_np(self.image) if image else [],
            'imageall': data_to_np(self.imageall) if imageall else [],
            'imagemask': data_to_np(self.imagemask) if imagemask else [],
            'fov': self.fov,
            'slicethickness': self.slice_thickness,
            'slicedistfactor': self.slice_dist_factor,
            'description': self.description
        }
        if datapath[-4:] != '.mat':
            datapath = datapath + '.mat'
        spio.savemat(datapath,var_dic)
        print('save: '+datapath)
        print('---w/:{}'.format(txt))
        return
    @staticmethod
    def load(datapath,details=True):
        '''Load data from saved data by this class.'''
        if details: print('load: '+datapath)
        mdata = spio.loadmat(datapath)
        # print(mdata)
        def try_convert_to_torch(v):
            if isinstance(v,np.ndarray):
                v = np.squeeze(v)
                v = torch.from_numpy(v)
            return v
        # ---------------------------------
        data = try_convert_to_torch(mdata.get('data'))
        image = try_convert_to_torch(mdata.get('image'))
        imagemask = try_convert_to_torch(mdata.get('imagemask'))
        imageall = try_convert_to_torch(mdata.get('imageall'))
        coilmaps = try_convert_to_torch(mdata.get('coilmaps'))
        fov = mdata.get('fov')
        try:
            fov = np.squeeze(fov).tolist()
        except:
            fov = None # ??????????? TODO
        slicethickness = np.squeeze(mdata.get('slicethickness'))
        slicedistfactor = np.squeeze(mdata.get('slicedistfactor'))
        description = np.squeeze(mdata.get('description'))
        # ---------------------------------
        mrdata = MRData(
            data=data,
            image=image,
            imagemask=imagemask,
            imageall=imageall,
            coilmaps=coilmaps,
            fov=fov,
            slice_thickness=slicethickness,
            slice_dist_factor=slicedistfactor,
            description=description
            )
        # print('---- image:{}, fov:{}'.format(mrdata.get_shape_of(mrdata.image),mrdata.fov3d))
        return mrdata
    # --------------------------------------
    def add_description(self,txt):
        self.description = txt
        return
    def get_shape_of(self,x):
        '''Return x.shape if it is an property'''
        if hasattr(x,'shape'):
            return list(x.shape)
        else:
            return []
    def get_dtype_of(self,x):
        '''Return x.dtype if it is an attribute'''
        if hasattr(x,'dtype'):
            return x.dtype
        else:
            return None
    # ------------------------------------------
    # providing some general functions
    @staticmethod
    def calculate_slice_position(n_slice,slicethickness,distance_factor)->torch.Tensor:
        '''Return array of slice position. -> tensor
        
        Also a consideration of acquired images from Siemens.
        '''
        return (torch.arange(n_slice)-n_slice//2)*slicethickness*(1+distance_factor)
    @staticmethod
    def calculate_fovpixel_position(fov,npoints)->torch.Tensor:
        '''Return 1D array of pixel locations. (unit the same as fov)
        
        The location = 0 is always included.
        '''
        return (torch.arange(npoints)-npoints//2)*fov/npoints


    @property
    def sliceFOV(self):
        return calculate_effective_slice_fov(self.Nslice,self.slice_thickness,self.slice_dist_factor)
    
    # ------------------------------------------
    # Debuggin' part



    # dimension of the data
    def _try_get_image_dimension_of(self,idx):
        '''Return dimension (if exists) or 0.'''
        try:
            N = self.image.shape[idx]
        except:
            N = 0
        return N
    @property
    def Nx(self):
        return self._try_get_image_dimension_of(-3)
    @property
    def Ny(self):
        return self._try_get_image_dimension_of(-2)
    @property
    def Nz(self):
        return self._try_get_image_dimension_of(-1)
    @property
    def Nslice(self):
        return self._try_get_image_dimension_of(-1)
    # --------------------------------------
    @property
    def resolution(self): #TODO
        return
    
    @property
    def loc_x(self):
        return self.fov_linspace(self.fov[0],self.Nx)
    @property
    def loc_y(self):
        return self.fov_linspace(self.fov[1],self.Ny)
    @property
    def loc_z(self):
        return self.fov_linspace(self.fov[2],self.Nz)
    
    # ---------------------------
    # Methods for more operations
    # ---------------------------
    # def image_interpolate(self,fov,dim,offset=[0,0,0])->torch.Tensor:
    #     '''Interpolate image to new FOV and dimensions.'''
    #     # calculate the current image's FOV

    #     # interpolate to the new fov

    #     return


    # def interpolate_image_to(self,newfov,newdim,offset=[0,0,0],multichannle=False):
    #     '''Interpolate image to new FOV and dimensions.'''
    #     # use image interpolation function provided in this module
    

    def single_image_interpolate(self,fov,dim,offset=[0,0,0]) -> torch.tensor:
        '''Interpolate single image to new FOV and dimensions.
        
        inputs:
            fov: e.g., [1,1,1]
            dim: e.g., [5,5,5]
            offset: 
        '''
        return image_interpolate(self.image,self.fov,tarfov=fov,tardim=dim,taroffset=offset)

    
    # TODO LISTS
    
    # @property
    # def image_Ndim(self):
    #     return len(self.image.shape)
    
    # @property
    # def Ncoils(self):
    #     return
    

    # def initialize(self):
    #     return
    # def __permute_data_in_preferred_order(self):
    #     """Permute data dimensions."""
    # def __get_image_location_matrix(self):
    #     """todo"""
    # def __get_image_location_axis(self):
    #     """todo"""
    

    
    
    # DISPLAY OF SOME INFORMATION
    def info(self):
        '''display of the basic information'''
        # -----------------------------
        print('MRData: '+self.description)
        print('   data: {}'.format(self.get_shape_of(self.data)))
        print('   data dim:',self.datadims)
        print('   image: {}'.format(self.get_shape_of(self.image)),self.get_dtype_of(self.image))
        print('   imageall: {}'.format(self.get_shape_of(self.imageall)),self.get_dtype_of(self.imageall))
        print('   coilmaps: {}'.format(self.get_shape_of(self.coilmaps)),self.get_dtype_of(self.coilmaps))
        print('   imagemask: {}'.format(self.get_shape_of(self.imagemask)),self.get_dtype_of(self.imagemask))
        # print('   fov: {}mm, slice={}mm(dist factor={})'.format(self.fov,self.slice_thickness,self.slice_dist_factor))
        # print('   ----')
        # print('   ')
        # print('    Nx={} | Ny={} | Nz={} |'.format(self.Nx,self.Ny,self.Nz),'FOV =',self.fov)
        # print(self.datainfo)
        return
    # ----------------------------------------------------------

