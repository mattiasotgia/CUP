import hist
import numpy as np
from typing import Tuple

from cup.core.registry import register_binning

@register_binning('log')
def binning_logscale(bins: int, limits: Tuple[int, int], flow: bool, name: str):
    _min, _max = limits
    logMin = np.min([np.log10(_min), np.log10(_max)])
    logMax = np.max([np.log10(_min), np.log10(_max)])
    return hist.axis.Variable(
        np.logspace(logMin, logMax, bins), name=name, flow=flow
    )

@register_binning('linear')
def binning_default(bins: int, limits: Tuple[int, int], flow: bool, name: str):
    _min, _max = limits
    return hist.axis.Regular(bins=bins, start=_min, stop=_max, name=name, flow=flow)
