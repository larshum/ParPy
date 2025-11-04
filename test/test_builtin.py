import numpy as np
import parpy
import torch

from common import *

@parpy.jit
def parpy_maximum(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.maximum(dst[0], x[0])

@parpy.jit
def parpy_minimum(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.minimum(dst[0], x[0])

data_types = [
    parpy.types.I8,
    parpy.types.I16,
    parpy.types.I32,
    parpy.types.I64,
    parpy.types.U8,
    parpy.types.U16,
    parpy.types.U32,
    parpy.types.U64,
    parpy.types.F16,
    parpy.types.F32,
    parpy.types.F64,
]

binary_tests = [
    (parpy_maximum, np.maximum),
    (parpy_minimum, np.minimum),
]

def binop_should_fail(backend, dtype):
    if backend == parpy.CompileBackend.Cuda:
        return False
    elif backend == parpy.CompileBackend.Metal:
        return dtype == parpy.types.F64

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', binary_tests)
@pytest.mark.parametrize('dtype', data_types)
def test_run_binary_operation(backend, test_data, dtype):
    def helper():
        parpy_fn, seq_fn = test_data
        x = np.ndarray((1,), dtype=dtype.to_numpy())
        x[0] = 1
        dst = np.zeros((1,), dtype=dtype.to_numpy())
        opts = par_opts(backend, {})
        if binop_should_fail(backend, dtype):
            with pytest.raises(TypeError):
                parpy_fn(dst, x, opts=opts)
        else:
            parpy_fn(dst, x, opts=opts)
            zero = np.zeros((1,), dtype=dtype.to_numpy())
            expected = seq_fn(zero, x)
            assert np.allclose(dst, expected, atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', binary_tests)
@pytest.mark.parametrize('dtype', data_types)
def test_compile_binary_operation(backend, test_data, dtype):
    parpy_fn, _ = test_data
    x = np.ndarray((1,), dtype=dtype.to_numpy())
    x[0] = 1
    dst = np.zeros((1,), dtype=dtype.to_numpy())
    opts = par_opts(backend, {})
    if binop_should_fail(backend, dtype):
        with pytest.raises(TypeError):
            parpy.print_compiled(parpy_fn, [dst, x], opts)
    else:
        code = parpy.print_compiled(parpy_fn, [dst, x], opts)
        assert len(code) != 0

@parpy.jit
def parpy_convert_bool(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.Bool)

@parpy.jit
def parpy_convert_i8(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.I8)

@parpy.jit
def parpy_convert_i16(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.I16)

@parpy.jit
def parpy_convert_i32(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.I32)

@parpy.jit
def parpy_convert_i64(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.I64)

@parpy.jit
def parpy_convert_u8(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.U8)

@parpy.jit
def parpy_convert_u16(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.U16)

@parpy.jit
def parpy_convert_u32(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.U32)

@parpy.jit
def parpy_convert_u64(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.U64)

@parpy.jit
def parpy_convert_f16(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.F16)

@parpy.jit
def parpy_convert_f32(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.F32)

@parpy.jit
def parpy_convert_f64(dst, x):
    with parpy.gpu:
        dst[0] = parpy.builtin.convert(x[0], parpy.types.F64)

convert_tests = [
    (parpy_convert_bool, parpy.types.Bool),
    (parpy_convert_i8, parpy.types.I8),
    (parpy_convert_i16, parpy.types.I16),
    (parpy_convert_i32, parpy.types.I32),
    (parpy_convert_i64, parpy.types.I64),
    (parpy_convert_u8, parpy.types.U8),
    (parpy_convert_u16, parpy.types.U16),
    (parpy_convert_u32, parpy.types.U32),
    (parpy_convert_u64, parpy.types.U64),
    (parpy_convert_f16, parpy.types.F16),
    (parpy_convert_f32, parpy.types.F32),
    (parpy_convert_f64, parpy.types.F64),
]

def conversion_should_fail(backend, from_dtype, to_dtype):
    if backend == parpy.CompileBackend.Cuda:
        return False
    elif backend == parpy.CompileBackend.Metal:
        return parpy.types.F64 in [from_dtype, to_dtype]

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', convert_tests)
@pytest.mark.parametrize('src_dtype', data_types)
def test_run_convert(backend, test_data, src_dtype):
    def helper():
        convert_fn, dtype = test_data
        x = np.ndarray((1,), dtype=src_dtype.to_numpy())
        x[0] = 1
        dst = np.zeros((1,), dtype=dtype.to_numpy())
        opts = par_opts(backend, {})
        if conversion_should_fail(backend, src_dtype, dtype):
            with pytest.raises(TypeError):
                convert_fn(dst, x, opts=opts)
        else:
            convert_fn(dst, x, opts=opts)
            y = x.astype(dtype.to_numpy())
            assert np.allclose(dst, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', convert_tests)
@pytest.mark.parametrize('src_dtype', data_types)
def test_compile_convert(backend, test_data, src_dtype):
    convert_fn, dtype = test_data
    x = np.ndarray((1,), dtype=src_dtype.to_numpy())
    x[0] = 1
    dst = np.zeros((1,), dtype=dtype.to_numpy())
    opts = par_opts(backend, {})
    if conversion_should_fail(backend, src_dtype, dtype):
        with pytest.raises(TypeError):
            parpy.print_compiled(convert_fn, [dst, x], opts)
    else:
        code = parpy.print_compiled(convert_fn, [dst, x], opts)
        assert len(code) != 0
