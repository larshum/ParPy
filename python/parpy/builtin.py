from numpy import inf
import numpy as np
import contextlib

# Binary operators

def maximum(x, y):
    return np.maximum(x, y)

def minimum(x, y):
    return np.minimum(x, y)

# Unary reduction operators

def max(x):
    return np.max(x)

def min(x):
    return np.min(x)

def sum(x):
    return np.sum(x)

def prod(x):
    return np.prod(x)

# Built-in utility functions for controlling the generated code

gpu = contextlib.nullcontext()

def convert(e, ty):
    return e

def label(x):
    assert x is not None, "parpy.label expects one argument"

def inline(e):
    pass

def static_backend_eq(x):
    return False

def static_types_eq(l, r):
    return l == r

def static_fail(s):
    raise RuntimeError(s)
