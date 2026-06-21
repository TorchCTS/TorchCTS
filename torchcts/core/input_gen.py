import torch
import random
import numpy as np

# Session-scoped shared CPU and device-resident tensors
_SHARED_CPU_TENSORS = {}
_SHARED_DEVICE_TENSORS = {}

def init_shared_data():
    global _SHARED_CPU_TENSORS
    # Create standard reference tensors on CPU
    # Seed it for reproducibility
    g = torch.Generator()
    g.manual_seed(42)
    
    _SHARED_CPU_TENSORS = {
        "float_1d": torch.randn(1024, dtype=torch.float32, generator=g),
        "float_2d": torch.randn(128, 128, dtype=torch.float32, generator=g),
        "float_pos_2d": torch.rand(128, 128, dtype=torch.float32, generator=g) + 0.1,
        "int_1d": torch.randint(-100, 100, (1024,), dtype=torch.int64, generator=g),
        "int_2d": torch.randint(-100, 100, (128, 128), dtype=torch.int64, generator=g),
        "bool_1d": torch.rand(1024, generator=g) > 0.5,
        "bool_2d": torch.rand(128, 128, generator=g) > 0.5,
    }

def refresh_shared_data(device):
    global _SHARED_CPU_TENSORS, _SHARED_DEVICE_TENSORS
    if not _SHARED_CPU_TENSORS:
        init_shared_data()
    _SHARED_DEVICE_TENSORS.clear()
    for name, cpu_tensor in _SHARED_CPU_TENSORS.items():
        _SHARED_DEVICE_TENSORS[name] = cpu_tensor.clone().to(device)

def get_shared_tensor(name, device):
    global _SHARED_DEVICE_TENSORS
    if not _SHARED_DEVICE_TENSORS:
        refresh_shared_data(device)
    return _SHARED_DEVICE_TENSORS.get(name)

def make_tensor(shape, dtype, device, layout="contiguous", low=None, high=None, positive_only=False):
    """
    Generate a tensor on `device` with specified `shape`, `dtype`, and memory `layout`.
    """
    # 1. Determine value range based on dtype
    if low is None or high is None:
        if dtype.is_floating_point:
            if positive_only:
                low = 0.1
                high = 2.0
            else:
                low = -2.0
                high = 2.0
        elif dtype.is_complex:
            low = -2.0
            high = 2.0
        elif dtype == torch.bool:
            low = 0
            high = 2
        else:  # Integer types
            if positive_only:
                low = 1
                high = 10
            else:
                low = -10
                high = 10

    # For overlapping or sliced layouts, we may create a larger base tensor
    base_shape = list(shape)
    if layout == "transpose":
        if len(shape) < 2:
            layout = "contiguous"
        else:
            # Swap last two dimensions for the base shape
            base_shape[-1], base_shape[-2] = base_shape[-2], base_shape[-1]
    elif layout == "sliced":
        # Create a tensor double the size along all dims
        base_shape = [s * 2 for s in shape]
    elif layout == "broadcast":
        # Create a tensor with 1s in some dimensions
        # and expand it later
        base_shape = [1 if i % 2 == 0 else s for i, s in enumerate(shape)]
        if all(s == 1 for s in base_shape) and len(shape) > 0:
            # Ensure at least one dimension is not 1 to allow actual broadcast if possible
            base_shape[-1] = shape[-1]
    elif layout == "overlapping":
        # Overlapping layout is created via as_strided, so we make a flat contiguous base
        # that is large enough to hold the strides
        total_elements = int(np.prod(shape)) * 2
        base_shape = [total_elements]

    # Create the base tensor as contiguous CPU/device
    # Note: FP8 dtypes are not natively supported by random/randint on CPU in older PyTorch,
    # so we generate on float32 and cast.
    is_fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)
    gen_dtype = torch.float32 if (is_fp8 or dtype.is_complex) else dtype
    
    if gen_dtype == torch.bool:
        base = torch.randint(0, 2, base_shape, dtype=torch.bool, device=device)
    elif gen_dtype.is_floating_point:
        base = torch.empty(base_shape, dtype=gen_dtype, device=device).uniform_(low, high)
    else:  # Integer types
        base = torch.randint(int(low), int(high), base_shape, dtype=gen_dtype, device=device)

    if dtype.is_complex:
        imag_base = torch.empty(base_shape, dtype=gen_dtype, device=device).uniform_(low, high)
        if dtype == torch.complex32:
            # Note: complex32 support is limited in standard PyTorch, but we can do our best
            base = torch.complex(base.half(), imag_base.half())
        elif dtype == torch.complex64:
            base = torch.complex(base, imag_base)
        else:  # complex128
            base = torch.complex(base.double(), imag_base.double())

    # Cast to final FP8 if needed
    if is_fp8:
        base = base.to(dtype)

    # Apply layout transformation
    if layout == "contiguous":
        return base
    elif layout == "transpose":
        # Transpose last two dimensions
        return base.transpose(-1, -2)
    elif layout == "sliced":
        # Slice back to original shape: t[::2, ::2, ...]
        slices = tuple(slice(0, s * 2, 2) for s in shape)
        return base[slices]
    elif layout == "broadcast":
        return base.expand(shape)
    elif layout == "overlapping":
        # Map overlapping strides: e.g. stride = 0 or 1 for all dimensions
        # Use as_strided
        strides = [1] * len(shape)
        return torch.as_strided(base, shape, strides)
    elif layout == "channels_last":
        if len(shape) == 4:
            return base.to(memory_format=torch.channels_last)
        else:
            # Fallback to contiguous if shape is not 4D
            return base
    elif layout == "channels_last_3d":
        if len(shape) == 5:
            return base.to(memory_format=torch.channels_last_3d)
        else:
            # Fallback to contiguous if shape is not 5D
            return base
    else:
        return base
