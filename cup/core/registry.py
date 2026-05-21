"""
cup.core.registry
=================
Central registries for filters and binning strategies.

Both registries use a decorator-based API so new entries can be added
from external packages without touching this file::

    from cup.core.registry import register_filter, register_binning

    @register_filter("my_filter")
    def my_filter(df, **params):
        ...

    @register_binning("sqrt")
    def sqrt_binning(bins, limits, flow, name):
        ...
"""

from typing import Callable, Dict, Optional

# ---------------------------------------------------------------------------
# Filter registry
# ---------------------------------------------------------------------------

FILTER_REGISTRY: Dict[str, Callable] = {}
FILTER_DESCRIBE_REGISTRY: Dict[str, Callable] = {}


def register_filter(name: str, describe: Optional[Callable] = None):
    """Decorator that registers a filter function (and optional describe fn)."""

    def wrapper(func: Callable) -> Callable:
        FILTER_REGISTRY[name] = func
        if describe is not None:
            FILTER_DESCRIBE_REGISTRY[name] = describe
        return func

    return wrapper


# ---------------------------------------------------------------------------
# Binning / axis registry
# ---------------------------------------------------------------------------

BINSCALE_REGISTRY: Dict[str, Callable] = {}


def register_binning(name: str):
    """Decorator that registers a binning factory function."""

    def wrapper(func: Callable) -> Callable:
        BINSCALE_REGISTRY[name] = func
        return func

    return wrapper