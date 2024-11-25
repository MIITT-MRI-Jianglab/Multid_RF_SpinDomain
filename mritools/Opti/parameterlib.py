'''calculate the parameters for pulse optimization'''

import torch
from mritools import mri


def get_target_spindomain_parameters_list(cube:mri.SpinArray,roi,pulse_function:str,phase,flip=0,roi_offset=[0,0,0]):
    r'''Return the parameters and transform function for RF optimization in spin-domain.
    
    Two cases are considered:
    1. refocusing pulse design $\beta^2$
    2. excitation pulse design

    Args:
        phase: (deg) along which axis the rf is applied
    '''
    if pulse_function=='refocusing':
        betasquare = cube.calculate_target_spindomain_refocusing(
            phase=phase,roi=roi,roi_offset=roi_offset)
        para_1 = torch.real(betasquare)
        para_2 = torch.imag(betasquare)

        def para_fn(ar,ai,br,bi):
            para_real = br**2 - bi**2   # real part of beta^2
            para_imag = 2*br*bi         # imag part of beta^2
            return para_real,para_imag
        
        return [para_1,para_2],para_fn
    
    elif pulse_function=='excitation':
        alphaconj_beta,betanorm = cube.calculate_target_spindomain_excitation(
            flip=flip,phase=phase,roi=roi)
        para_1 = torch.real(alphaconj_beta)
        para_2 = torch.imag(alphaconj_beta)
        # para_3 = betanorm

        def para_fn(ar,ai,br,bi):
            para_1 = ar*br + ai*bi              # real part of $\alpha^{*} \beta$
            para_2 = ar*bi - ai*br              # imag part of $\alpha^{*} \beta$
            para_3 = torch.sqrt(br**2 + bi**2)  # the magnitude of beta
            return para_1,para_2,para_3
        
        return [para_1,para_2,betanorm],para_fn
    
    elif pulse_function=='inversion':
        print('not implemented !')
        return
    
    else:
        print('error !')
        return