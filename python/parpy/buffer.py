import pathlib
from .parpy import CompileBackend, DataType, ElemSize
from .runtime import _compile_runtime_lib

def _check_errors(lib, rescode):
    if rescode != 0:
        msg = lib.parpy_get_error_message().decode('ascii')
        raise RuntimeError(f"Runtime library error: {msg} (code={rescode})")

def _check_not_nullptr(lib, resptr):
    if resptr == 0:
        raise RuntimeError(f"Runtime library error: {lib.parpy_get_error_message()}")
    return resptr

def _check_array_interface(intf):
    shape = intf["shape"]
    dtype = DataType(intf["typestr"])

    # We require the data pointer to be provided as part of the interface.
    if "data" in intf:
        data, ro = intf["data"]
        if ro == True:
            raise ValueError(f"Cannot construct buffer from read-only memory")
    else:
        raise ValueError(f"Buffer protocol not supported")

    # We require data to be laid out contiguously in memory
    if "strides" in intf and intf["strides"] is not None:
        raise ValueError(f"Buffers must only operate on contiguous memory")

    return data, shape, dtype

def _to_array_interface(ptr, shape, dtype):
    return {
        'data': (ptr, False),
        'strides': None,
        'typestr': str(dtype),
        'shape': shape,
        'version': 3
    }

def _extract_array_interface(a, allow_cuda=False):
    if allow_cuda and hasattr(a, "__cuda_array_interface__"):
        return _check_array_interface(a.__cuda_array_interface__)
    elif hasattr(a, "__array_interface__"):
        return _check_array_interface(a.__array_interface__)
    elif hasattr(a, "__array__"):
        return _check_array_interface(a.__array__().__array_interface__)
    else:
        raise ValueError("Failed to extract array interface")

def _resolve_dtype(dtype):
    """
    Resolves the provided dtype - provided to allow users to construct buffers
    using the more easily accessible types defined in the 'parpy.types' module.
    """
    if isinstance(dtype, ElemSize):
        return DataType.from_elem_size(dtype)
    else:
        return dtype

def _size(shape, dtype):
    sz = dtype.size()
    for dim in shape:
        sz *= dim
    return sz

def _alloc_data(shape, dtype, lib):
    nbytes = _size(shape, dtype)
    return _check_not_nullptr(lib, lib.parpy_alloc_buffer(nbytes))

def sync(backend):
    """
    Synchronizes the CPU and the target device by waiting until all running
    kernels complete.
    """
    lib = _compile_runtime_lib(backend)
    _check_errors(lib, lib.parpy_sync())

def empty(shape, dtype, backend):
    dtype = _resolve_dtype(dtype)
    if backend == CompileBackend.Cuda:
        import torch
        t = torch.empty(*shape, dtype=dtype.to_torch(), device='cuda')
        return CudaBuffer.from_array(t)
    elif backend == CompileBackend.Metal:
        lib = _compile_runtime_lib(backend)
        nbytes = _size(shape, dtype)
        buf = _check_not_nullptr(lib, lib.parpy_alloc_buffer(nbytes))
        base_buf = MetalBaseBuffer(buf, nbytes, None)
        return MetalBuffer(base_buf, shape, dtype)
    else:
        raise ValueError(f"Cannot construct buffer of type {type(b)}")

def empty_like(b):
    if isinstance(b, CudaBuffer):
        return empty(b.shape, b.dtype, CompileBackend.Cuda)
    elif isinstance(b, MetalBuffer):
        return empty(b.shape, b.dtype, CompileBackend.Metal)
    else:
        raise ValueError(f"Cannot construct buffer of type {type(b)}")

def zeros(shape, dtype, backend):
    dtype = _resolve_dtype(dtype)
    if backend == CompileBackend.Cuda:
        import torch
        t = torch.zeros(*shape, dtype=dtype.to_torch(), device='cuda')
        return CudaBuffer.from_array(t)
    elif backend == CompileBackend.Metal:
        b = empty(shape, dtype, backend)
        lib = _compile_runtime_lib(backend)
        _check_errors(lib, lib.parpy_memset(b.buf.buf, b.size(), 0))
        return b
    else:
        raise ValueError(f"Cannot construct buffer of type {type(b)}")

def zeros_like(b):
    if isinstance(b, CudaBuffer):
        return zeros(b.shape, b.dtype, CompileBackend.Cuda)
    elif isinstance(b, MetalBuffer):
        return zeros(b.shape, b.dtype, CompileBackend.Metal)
    else:
        raise ValueError(f"Cannot construct buffer of type {type(b)}")

def from_array(t, backend):
    if backend is None:
        return Buffer.from_array(t)
    elif backend == CompileBackend.Cuda:
        return CudaBuffer.from_array(t)
    elif backend == CompileBackend.Metal:
        return MetalBuffer.from_array(t)
    else:
        raise ValueError(f"Cannot convert to buffer of unknown backend {backend}")

class BaseBuffer:
    def __init__(self, buf, nbytes, src):
        if type(self) is BaseBuffer:
            raise RuntimeError(f"Cannot construct instance of BaseBuffer class")
        self.buf = buf
        self.nbytes = nbytes
        self.src = src

    def __del__(self):
        if self.src is not None:
            src_ptr, _, _ = _extract_array_interface(self.src)
        else:
            src_ptr = None
        self._deconstruct(src_ptr)

    def _deconstruct(self, src_ptr):
        pass

class CudaBaseBuffer(BaseBuffer):
    def __init__(self, buf, nbytes, src=None):
        super().__init__(buf, nbytes, src)

    def _deconstruct(self, src_ptr):
        if src_ptr is not None:
            buf_ptr, _, _ = _extract_array_interface(self.buf, allow_cuda=True)
            lib = self._get_runtime_lib()
            _check_errors(lib, lib.parpy_memcpy(src_ptr, buf_ptr, self.nbytes, 2))
            self.sync()

    def _get_runtime_lib(self):
        return _compile_runtime_lib(CompileBackend.Cuda)

    def sync(self):
        sync(CompileBackend.Cuda)

    def from_array(t):
        import torch
        try:
            data_ptr, shape, dtype = _extract_array_interface(t, allow_cuda=True)
            nbytes = _size(shape, dtype)
            # If this attribute is defined, we use the existing CUDA memory
            # allocation as-is, without any copying.
            if hasattr(t, "__cuda_array_interface__"):
                return CudaBaseBuffer(torch.as_tensor(t), nbytes)
        except Exception as e:
            raise ValueError(f"Cannot convert argument {t} to a CUDA buffer")

        if len(shape) == 0:
            buf = torch.empty((), dtype=dtype.to_torch(), device='cuda')
        else:
            buf = torch.empty(*shape, dtype=dtype.to_torch(), device='cuda')
        ptr, _, _ = _extract_array_interface(buf, allow_cuda=True)
        lib = _compile_runtime_lib(CompileBackend.Cuda)
        _check_errors(lib, lib.parpy_memcpy(ptr, data_ptr, nbytes, 1))
        return CudaBaseBuffer(buf, nbytes, src=t)

    def from_raw(ptr, shape, dtype):
        import torch
        class dummy(object):
            def __init__(self):
                self.__cuda_array_interface__ = _to_array_interface(ptr, shape, dtype)
        buf = torch.as_tensor(dummy(), device='cuda')
        nbytes = _size(shape, dtype)
        return CudaBaseBuffer(buf, nbytes)

    def copy(self):
        return CudaBaseBuffer(self.buf.detach().clone(), self.nbytes)

class MetalBaseBuffer(BaseBuffer):
    def __init__(self, buf, nbytes, src=None):
        super().__init__(buf, nbytes, src)

    def _deconstruct(self, src_ptr):
        self.sync()
        lib = self._get_runtime_lib()
        if src_ptr is not None:
            _check_errors(lib, lib.parpy_memcpy(src_ptr, self.buf, self.nbytes, 2))
        _check_errors(lib, lib.parpy_free_buffer(self.buf))

    def _get_runtime_lib(self):
        return _compile_runtime_lib(CompileBackend.Metal)

    def sync(self):
        sync(CompileBackend.Metal)

    def from_array(t):
        try:
            data_ptr, shape, dtype = _extract_array_interface(t)
        except:
            raise ValueError(f"Cannot convert argument {t} to a Metal buffer")
        data_ptr, shape, dtype = _extract_array_interface(t)
        lib = _compile_runtime_lib(CompileBackend.Metal)
        nbytes = _size(shape, dtype)
        buf = _check_not_nullptr(lib, lib.parpy_alloc_buffer(nbytes))
        _check_errors(lib, lib.parpy_memcpy(buf, data_ptr, nbytes, 1))
        return MetalBaseBuffer(buf, nbytes, src=t)

    def from_raw(ptr, shape, dtype):
        # This assumes the provided raw pointer points to a MTL::Buffer
        nbytes = _size(shape, dtype)
        return MetalBaseBuffer(ptr, nbytes)

class Buffer:
    def __init__(self, buf, shape, dtype, buf_offset):
        self.buf = buf
        self.shape = shape
        self.dtype = _resolve_dtype(dtype)
        self.buf_offset = buf_offset

    def __float__(self):
        if len(self.shape) == 0:
            return float(self.numpy().item())
        else:
            raise ValueError(f"Cannot convert buffer of shape {self.shape} to float")

    def __int__(self):
        if len(self.shape) == 0:
            return int(self.numpy().item())
        else:
            raise ValueError(f"Cannot convert buffer of shape {self.shape} to int")

    def __bool__(self):
        if len(self.shape) == 0:
            return bool(self.numpy().item())
        else:
            raise ValueError(f"Cannot convert buffer of shape {self.shape} to bool")

    def __index__(self):
        if len(self.shape) == 0:
            if self.dtype.is_integer():
                return self.__int__()
        else:
            raise ValueError(f"Cannot use buffer of shape {self.shape} and type {self.dtype} as index")

    def __getitem__(self, idx):
        import math
        idx = [idx] if isinstance(idx, int) else idx
        if len(idx) > len(self.shape):
            raise ValueError(f"Cannot use lookup index {idx} on buffer of shape {self.shape}")
        ofs = self.buf_offset
        for i, (j, k) in enumerate(zip(idx, self.shape)):
            if j < k:
                ofs += j * math.prod(self.shape[i+1:])
            else:
                raise IndexError(f"Index {j} out of bounds for dimension {k} of buffer")
        new_shape = self.shape[len(idx):]
        buf = type(self)(self.buf, new_shape, self.dtype, ofs)
        if len(new_shape) == 0:
            return buf.numpy()
        else:
            return buf

    def _get_indexed(self, buf):
        import math
        nelems = self.buf.nbytes // self.dtype.size()
        a = buf.reshape(nelems)
        size = math.prod(self.shape)
        return a[self.buf_offset:self.buf_offset+size].reshape(self.shape)

    def from_array(t):
        try:
            ptr, shape, dtype = _extract_array_interface(t, allow_cuda=True)
        except ValueError:
            raise ValueError(f"Cannot convert argument {t} to dummy buffer")
        buf = Buffer(ptr, shape, dtype, 0)
        buf.__array_interface__ = _to_array_interface(ptr, shape, dtype)
        return buf

    def size(self):
        return _size(self.shape, self.dtype)

    def numpy(self):
        import numpy as np
        class dummy(object):
            def __init__(self, buf, shape, dtype):
                self.__array_interface__ = _to_array_interface(buf, shape, dtype)
        return np.asarray(dummy(self.buf, self.shape, self.dtype))

    def torch(self):
        import torch
        return self._get_indexed(torch.as_tensor(self.numpy()))

    def copy(self):
        return type(self)(self.buf.copy(), self.shape, self.dtype, self.buf_offset)

    def reshape(self, *dims):
        import math
        curr_sz = math.prod(self.shape)
        new_shape = tuple(dims)
        new_sz = math.prod(new_shape)
        if curr_sz == new_sz:
            return type(self)(self.buf, new_shape, self.dtype, self.buf_offset)
        else:
            raise ValueError(f"Cannot reshape buffer of shape {self.shape} to {new_shape}")

    def with_type(self, new_dtype):
        raise RuntimeError(f"Cannot instantiate base Buffer class")

class CudaBuffer(Buffer):
    def __init__(self, buf, shape, dtype, buf_offset=0):
        super().__init__(buf, shape, dtype, buf_offset)
        self.__cuda_array_interface__ = self.buf.buf.__cuda_array_interface__

    def _get_ptr(self):
        ptr, _, _, = _check_array_interface(self.__cuda_array_interface__)
        return ptr + self.buf_offset * self.dtype.size()

    def from_array(t):
        _, shape, dtype = _extract_array_interface(t, allow_cuda=True)
        buf = CudaBaseBuffer.from_array(t)
        return CudaBuffer(buf, shape, dtype)

    def from_raw(ptr, shape, dtype):
        buf = CudaBaseBuffer.from_raw(ptr, shape, dtype)
        return CudaBuffer(buf, shape, dtype)

    def numpy(self):
        import numpy as np
        return self._get_indexed(np.asarray(self.buf.buf.cpu()))

    def torch(self):
        return self._get_indexed(self.buf.buf)

    def with_type(self, new_dtype):
        new_dtype = _resolve_dtype(new_dtype)
        if isinstance(new_dtype, DataType):
            if self.dtype.size() == new_dtype.size():
                return CudaBuffer(self.buf, self.shape, new_dtype, self.buf_offset)
            else:
                t = self.buf.buf.detach().clone().to(new_dtype.to_torch())
                nbytes = _size(self.shape, new_dtype)
                buf = CudaBaseBuffer(t, nbytes)
                return CudaBuffer(buf, self.shape, new_dtype, self.buf_offset)

class MetalBuffer(Buffer):
    def __init__(self, buf, shape, dtype, buf_offset=0):
        super().__init__(buf, shape, dtype, buf_offset)
        lib = self.buf._get_runtime_lib()
        ptr = lib.parpy_ptr_buffer(self.buf.buf)
        self.__array_interface__ = _to_array_interface(ptr, self.shape, self.dtype)

    def _get_ptr(self):
        lib = self.buf._get_runtime_lib()
        lib.parpy_buffer_set_offset(self.buf.buf, self.buf_offset * self.dtype.size())
        return self.buf.buf

    def _copy_to_numpy(self):
        import numpy as np
        nelems = self.buf.nbytes // self.dtype.size()
        a = np.ndarray((nelems,), dtype=self.dtype.to_numpy())
        ptr, _, _ = _check_array_interface(a.__array_interface__)
        self.buf.sync()
        lib = self.buf._get_runtime_lib()
        _check_errors(lib, lib.parpy_memcpy(ptr, self.buf.buf, self.buf.nbytes, 2))
        return a

    def from_array(t):
        _, shape, dtype = _extract_array_interface(t)
        buf = MetalBaseBuffer.from_array(t)
        return MetalBuffer(buf, shape, dtype)

    def from_raw(ptr, shape, dtype):
        buf = MetalBaseBuffer.from_raw(ptr, shape, dtype)
        return MetalBuffer(buf, shape, dtype)

    def numpy(self):
        return self._get_indexed(self._copy_to_numpy())

    def torch(self):
        import torch
        return self._get_indexed(torch.as_tensor(self._copy_to_numpy()))

    def copy(self):
        b = empty(self.shape, self.dtype, CompileBackend.Metal)
        self.buf.sync()
        lib = self.buf._get_runtime_lib()
        _check_errors(lib, lib.parpy_memcpy(b.buf.buf, self.buf.buf, self.size(), 3))
        return b

    def with_type(self, new_dtype):
        new_dtype = _resolve_dtype(new_dtype)
        if isinstance(new_dtype, DataType):
            if self.dtype.size() == new_dtype.size():
                return MetalBuffer(self.buf, self.shape, new_dtype, self.buf_offset)
            else:
                import numpy as np
                t = np.asarray(self).astype(dtype=new_dtype.to_numpy())
                return MetalBuffer.from_array(t)
