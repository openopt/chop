"""chop: constrained optimization for PyTorch"""
__version__ = "0.0.1"

from . import stochastic
from . import optim
from . import constraints
from . import penalties

from .utils import data
from .utils import logging
from .utils import image
from .utils import utils
from . import adversary
from .adversary import Adversary
