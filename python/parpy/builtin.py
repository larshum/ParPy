from numpy import inf
import numpy as np
import contextlib

gpu = contextlib.nullcontext()

def max(x):
    return np.max(x)

def min(x):
    return np.min(x)

def sum(x):
    return np.sum(x)

def prod(x):
    return np.prod(x)

def convert(e, ty):
    return e

def label(x):
    assert x is not None, "parpy.label expects one argument"

def static_backend_eq(x):
    return False

def static_types_eq(l, r):
    return l == r

def static_fail(s):
    raise RuntimeError(s)
