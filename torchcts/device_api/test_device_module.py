# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import torch
from torchcts.core.device import get_device_module

@pytest.mark.gate
@pytest.mark.smoke
@pytest.mark.requires("device_api")
@pytest.mark.parametrize("index", [0])
def test_device_construction(index, device):
    dev1 = torch.device(device)
    assert dev1.type == device
    
    dev2 = torch.device(f"{device}:{index}")
    assert dev2.type == device
    assert dev2.index == index

@pytest.mark.smoke
@pytest.mark.requires("device_api")
@pytest.mark.parametrize("mode", ["standard"])
def test_device_module_methods(mode, device, manifest):
    # CPU and missing-device-module checks handled at collection time in conftest.
    mod = get_device_module(device)
        
    assert hasattr(mod, "is_available"), f"torch.{device} is missing is_available()"
    assert mod.is_available() is True, f"torch.{device}.is_available() is False"
    
    assert hasattr(mod, "device_count"), f"torch.{device} is missing device_count()"
    count = mod.device_count()
    assert isinstance(count, int) and count >= 1
    assert count == manifest.get("effective_device_count", manifest.get("device_count", 1))
    
    if hasattr(mod, "current_device"):
        curr = mod.current_device()
        assert isinstance(curr, int) and curr >= 0
    
    assert hasattr(mod, "synchronize"), f"torch.{device} is missing synchronize()"
    mod.synchronize()  # should run without raising errors
    
    assert hasattr(mod, "empty_cache"), f"torch.{device} is missing empty_cache()"
    mod.empty_cache()  # should run without raising errors

@pytest.mark.smoke
@pytest.mark.requires("device_api")
@pytest.mark.parametrize("mode", ["standard"])
def test_device_memory_query(mode, device):
    # CPU and missing-device-module checks handled at collection time in conftest.
    mod = get_device_module(device)
        
    # memory_allocated is standard, but memory_reserved/get_device_properties are optional
    if hasattr(mod, "memory_allocated"):
        mem = mod.memory_allocated()
        assert isinstance(mem, int) and mem >= 0
        
    if hasattr(mod, "memory_reserved"):
        mem_res = mod.memory_reserved()
        assert isinstance(mem_res, int) and mem_res >= 0
        
    if hasattr(mod, "get_device_properties"):
        props = mod.get_device_properties(0)
        assert props is not None

@pytest.mark.gate
@pytest.mark.smoke
@pytest.mark.requires("device_api")
@pytest.mark.parametrize("shape", [(10, 10), (5, 5)])
def test_tensor_infrastructure(shape, device):
    # Create tensor on device
    t = torch.randn(*shape, device=device)
    assert t.device.type == device
    
    # Check data pointer
    ptr = t.data_ptr()
    assert isinstance(ptr, int) and ptr > 0
    
    # Check storage offset
    assert isinstance(t.storage_offset(), int) and t.storage_offset() == 0
    
    # Check storage
    try:
        store = t.untyped_storage()
        assert store is not None
        num_elements = 1
        for s in shape:
            num_elements *= s
        assert store.size() >= num_elements * 4 # float32 elements = 4 bytes
    except AttributeError:
        # Fallback for older PyTorch versions
        try:
            store = t.storage()
            assert store is not None
        except Exception:
            pass

    # Check is_<device> property if registered on Tensor class
    prop_name = f"is_{device}"
    if hasattr(t, prop_name):
        assert getattr(t, prop_name) is True

@pytest.mark.gate
@pytest.mark.smoke
@pytest.mark.requires("device_api")
def test_matmul_on_device(device, compare):
    """1024x1024 matmul that proves the backend is actually dispatching, not
    silently falling back to CPU."""
    N = 1024

    # Create inputs on CPU, copy to device
    a_cpu = torch.randn(N, N, dtype=torch.float32)
    b_cpu = torch.randn(N, N, dtype=torch.float32)
    a_dev = a_cpu.to(device)
    b_dev = b_cpu.to(device)

    # Verify inputs are on the right device
    assert a_dev.device.type == device, (
        f"Input A landed on {a_dev.device}, expected {device}")
    assert b_dev.device.type == device, (
        f"Input B landed on {b_dev.device}, expected {device}")
    assert not a_dev.is_cpu, "Input A is secretly on CPU"
    assert not b_dev.is_cpu, "Input B is secretly on CPU"

    # Run matmul on device
    c_dev = torch.mm(a_dev, b_dev)

    # Verify output is on the right device
    assert c_dev.device.type == device, (
        f"Output landed on {c_dev.device}, expected {device}")
    assert not c_dev.is_cpu, "Output is secretly on CPU"

    # Verify data pointers differ from CPU (not shared memory aliasing)
    c_cpu = torch.mm(a_cpu, b_cpu)
    assert c_dev.data_ptr() != c_cpu.data_ptr(), (
        "Device and CPU tensors share the same data_ptr — backend is fake")

    # Verify numerical correctness against CPU reference
    compare(c_dev, c_cpu, category="matmul", dtype=torch.float32)
