import numpy as np
import parpy
import pytest

from common import *

@pytest.mark.parametrize('backend', compiler_backends)
def test_annotate_int_param(backend):
    def helper():
        @parpy.jit
        def annot_int_param(x: parpy.types.I32, y):
            with parpy.gpu:
                y[:] += x
        x = 1
        y = np.ndarray((10,), dtype=np.int32)
        annot_int_param(x, y, opts=par_opts(backend, {}))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_scalar_int_argument_expected_float(backend):
    def helper():
        @parpy.jit
        def annot_float_param(x: parpy.types.F32, y):
            with parpy.gpu:
                y[:] += x
        x = 1
        y = np.ndarray((10,), dtype=np.int32)
        with pytest.raises(TypeError) as e_info:
            annot_float_param(x, y, opts=par_opts(backend, {}))
        assert e_info.match("Parameter x was annotated with type parpy.types.F32 which is incompatible with argument type")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_argument_expected_scalar(backend):
    def helper():
        @parpy.jit
        def annot_float_param_2(x: parpy.types.F32, y):
            with parpy.gpu:
                y[:] += x
        x = np.ndarray((10,), dtype=np.float32)
        y = np.ndarray((10,), dtype=np.float32)
        with pytest.raises(TypeError) as e_info:
            annot_float_param_2(x, y, opts=par_opts(backend, {}))
        assert e_info.match("Parameter x was annotated with type parpy.types.F32 which is incompatible with argument type")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_argument_wrong_element_type(backend):
    def helper():
        @parpy.jit
        def annot_float_param_3(x: parpy.types.F32, y):
            with parpy.gpu:
                y[:] += x
        x = np.ndarray((10,), dtype=np.int32)
        y = np.ndarray((10,), dtype=np.float32)
        with pytest.raises(TypeError) as e_info:
            annot_float_param_3(x, y, opts=par_opts(backend, {}))
        assert e_info.match("Parameter x was annotated with type parpy.types.F32 which is incompatible with argument type")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_parameter_annotation(backend):
    def helper():
        N = parpy.types.symbol('N')
        @parpy.jit
        def annot_buffer_param(x: parpy.types.buffer(parpy.types.F32, (N,))):
            with parpy.gpu:
                x[:] += 1.0
        x = np.ndarray((10,), dtype=np.float32)
        annot_buffer_param(x, opts=par_opts(backend, {}))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_annotation_shape_equality(backend):
    def helper():
        N = parpy.types.symbol('N')
        @parpy.jit
        def annot_add_elemwise_inplace(
            x: parpy.types.buffer(parpy.types.F32, [N]),
            y: parpy.types.buffer(parpy.types.F32, [N])
        ):
            parpy.label('N')
            x[:] += y[:]
        x = np.ndarray((10,), dtype=np.float32)
        y = np.ones_like(x)
        z = np.ndarray((20,), dtype=np.float32)
        opts = par_opts(backend, {'N': parpy.threads(128)})
        annot_add_elemwise_inplace(x, y, opts=opts)
        with pytest.raises(TypeError) as e_info:
            annot_add_elemwise_inplace(x, z, opts=opts)
        assert e_info.match("Parameter y was annotated with type.*[N].*which is incompatible with argument type.*[20].*")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_called_function_annotation_shape_constraint(backend):
    def helper():
        N = parpy.types.symbol('N')
        @parpy.jit
        def annot_add_elemwise_inplace_helper(
            x: parpy.types.buffer(parpy.types.F32, [N]),
            y: parpy.types.buffer(parpy.types.F32, [N])
        ) -> parpy.types.I32:
            parpy.label('N')
            x[:] += y[:]
            return 0
        @parpy.jit
        def annot_add_outer(a, b, N):
            parpy.label('N')
            for i in range(N):
                n = annot_add_elemwise_inplace_helper(a[i], b[i])
        x = np.ndarray((10,10), dtype=np.float32)
        y = np.ndarray((10,20), dtype=np.float32)
        opts = par_opts(backend, {'N': parpy.threads(128)})
        with pytest.raises(TypeError) as e_info:
            annot_add_outer(x, y, 10, opts=opts)
        assert e_info.match("Parameter y was annotated with type.*incompatible with.*")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_called_function_scalar_coercion(backend):
    def helper():
        @parpy.jit
        def base_func_add_scalars(x: parpy.types.I32, y: parpy.types.I32) -> parpy.types.I32:
            return x + y
        @parpy.jit
        def calling_func_add_scalars(x, y):
            parpy.label('N')
            x[:] = base_func_add_scalars(x[:], y[:])
        x = np.ndarray((10,), dtype=np.int64)
        y = np.ndarray((10,), dtype=np.int64)
        opts = par_opts(backend, {'N': parpy.threads(128)})
        calling_func_add_scalars(x, y, opts=opts)
    run_if_backend_is_enabled(backend, helper)
