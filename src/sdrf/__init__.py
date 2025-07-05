'''Spin-domain optimization for multidimensional RF pulse optimization.

- reference: Yang, Jiayao, et al. Multidimensional RF pulse design using auto-differentiable spin-domain optimization and its application to reduced field-of-view imaging. Magnetic Resonance in Medicine (2025).
- reference doi: https://onlinelibrary.wiley.com/doi/10.1002/mrm.30607
- author: Jiayao Yang
'''

from . import mri
from .mrsim import SpinDomain
from .mrpulse import spDmPulseOpt

__author__ = "jiayao"
__version__ = "0.0.2"

# print('import mritools     ----succeed!')