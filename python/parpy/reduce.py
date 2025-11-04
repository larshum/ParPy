import numpy as np

# The below reductions are implemented in the compiler. References to these
# functions are recognized by the compiler and replaced by a proper
# implementation.

def max(x):
    return np.max(x)

def min(x):
    return np.min(x)

def sum(x):
    return np.sum(x)

def prod(x):
    return np.prod(x)
