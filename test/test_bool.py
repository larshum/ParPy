import parpy
import pytest
import torch

from common import *

torch.manual_seed(1234)

def parpy_and(x, y):
    with parpy.gpu:
        x[0] = x[0] and y[0]

def parpy_or(x, y):
    with parpy.gpu:
        x[0] = x[0] or y[0]

def parpy_not(x, y):
    with parpy.gpu:
        x[0] = not y[0]

boolean_ops = [parpy_and, parpy_or, parpy_not]

def bool_wrap(fn, x, opts=None):
    out = torch.zeros(1, dtype=torch.bool)
    if opts is None:
        fn(out, x)
    else:
        parpy.jit(fn)(out, x, opts=opts)
    return out

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('fn', boolean_ops)
def test_bool_gpu(backend, fn):
    def helper():
        x = torch.tensor([True], dtype=torch.bool)
        expected = bool_wrap(fn, x)
        actual = bool_wrap(fn, x, par_opts(backend, {}))
        assert torch.allclose(expected, actual)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('fn', boolean_ops)
def test_bool_compiles(backend, fn):
    x = torch.tensor([True], dtype=torch.bool)
    res = torch.zeros(1, dtype=torch.bool)
    s = parpy.print_compiled(fn, [x, res], par_opts(backend, {}))
    assert len(s) != 0
