"""
cup.core.binnings
=================
Built-in binning / axis factories registered via ``@register_binning``.

To add a new scale (e.g. ``"sqrt"``) create a function with the signature::

    def my_scale(bins: int, limits, flow: bool, name: str) -> hist.axis.*:
        ...

and decorate it with ``@register_binning("sqrt")``.
"""

from typing import Tuple

import numpy as np
import hist

from cup.core.registry import register_binning


@register_binning("linear")
def binning_linear(
    bins: int,
    limits: Tuple[float, float],
    flow: bool,
    name: str,
) -> hist.axis.Regular:
    """Uniformly-spaced bins."""
    lo, hi = sorted(limits)
    return hist.axis.Regular(bins=bins, start=lo, stop=hi, name=name, flow=flow)


@register_binning("log")
def binning_log(
    bins: int,
    limits: Tuple[float, float],
    flow: bool,
    name: str,
) -> hist.axis.Variable:
    """Logarithmically-spaced bins."""
    lo, hi = sorted(np.log10(limits))
    edges = np.logspace(lo, hi, bins)
    return hist.axis.Variable(edges, name=name, flow=flow)