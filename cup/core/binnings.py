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
    bins: int | None = None,
    limits: Tuple[float, float] | None = None,
    flow: bool = False,
    name: str = "",
    edges: list | None = None,
) -> hist.axis.Regular | hist.axis.Variable:
    if edges is not None:
        return hist.axis.Variable(sorted(edges), name=name, flow=flow)
    lo, hi = sorted(limits)
    return hist.axis.Regular(bins=bins, start=lo, stop=hi, name=name, flow=flow)


@register_binning("log")
def binning_log(
    bins: int | None = None,
    limits: Tuple[float, float] | None = None,
    flow: bool = False,
    name: str = "",
    edges: list | None = None,
) -> hist.axis.Variable:
    if edges is not None:
        return hist.axis.Variable(sorted(edges), name=name, flow=flow)
    lo, hi = sorted(np.log10(limits))
    edges = np.logspace(lo, hi, bins + 1)
    return hist.axis.Variable(edges, name=name, flow=flow)