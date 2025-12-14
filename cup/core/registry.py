'''
Docstring for cup.core.registry
'''

from typing import Dict, Callable

import cup.core

FILTER_REGISTRY: Dict[str, Callable] = {}

def register_filter(name):
    def wrapper(func):
        FILTER_REGISTRY[name] = func
        return func
    return wrapper


BINSCALE_REGISTRY: Dict[str, Callable] = {}

def register_binning(name):
    def wrapper(func):
        BINSCALE_REGISTRY[name] = func
        return func
    return wrapper