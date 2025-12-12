from cup.core.registry import register_binning
from typing import Tuple
import numpy as np
import hist 

@register_binning('log')
def binning_logscale(bins: int, limits: Tuple[int, int], flow: bool, name: str):
    _min, _max = limits
    return hist.axis.Variable(np.logspace(_min, _max, bins), name=name, flow=flow)

@register_binning('linear')
def binning_default(bins: int, limits: Tuple[int, int], flow: bool, name: str):
    _min, _max = limits
    return hist.axis.Regular(bins=bins, start=_min, stop=_max, name=name, flow=flow)