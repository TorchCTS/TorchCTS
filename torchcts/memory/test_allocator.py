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
from torchcts.core.device import (
    synchronize,
    empty_cache,
    memory_allocated,
    memory_reserved,
    get_device_module
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.stress
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", [1024, 2048])
def test_allocator_tracking_and_cache(size, dtype, device, manifest):
    mod = get_device_module(device)
    if mod is None or not hasattr(mod, "memory_allocated"):
        pytest.skip(f"torch.{device} does not expose allocator telemetry.")
    
    # Reset allocator state
    synchronize(device)
    empty_cache(device)
    
    initial_alloc = memory_allocated(device)
    
    # 1. Allocate tensor
    x = torch.randn(size, size, dtype=dtype, device=device) # 4MB / 16MB
    synchronize(device)
    
    alloc_after_x = memory_allocated(device)
    assert alloc_after_x > initial_alloc, "Allocated memory did not increase after allocation."
    
    # 2. Cache reuse
    del x
    synchronize(device)
    
    # Peak reserved memory
    reserved_after_del = memory_reserved(device)
    
    # Reallocate same size
    y = torch.randn(size, size, dtype=dtype, device=device)
    synchronize(device)
    
    alloc_after_y = memory_allocated(device)
    reserved_after_y = memory_reserved(device)
    
    # If caching is working, reserved memory shouldn't have doubled
    if reserved_after_del > 0:
        assert reserved_after_y <= reserved_after_del, "Caching allocator did not reuse cached buffer."
        
    del y
    synchronize(device)
    empty_cache(device)
    
    final_alloc = memory_allocated(device)
    # Allow some small residual overhead
    assert final_alloc - initial_alloc < 1024 * 1024, "Memory leak detected or empty_cache did not release memory."

@pytest.mark.stress
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("oom_factor", [1.0, 1.2])
def test_oom_recovery(oom_factor, dtype, device, manifest):
    # OOM recoverability is verified at collection time via conftest.
        
    # Get available device memory
    hw_config = manifest.get("hardware", {})
    dev_mem_list = hw_config.get("device_memory_gb", [2])
    # Assume 1st device
    dev_mem = dev_mem_list[0] * (1024 ** 3)
    
    # Try to allocate a tensor larger than available memory
    huge_shape = (int(dev_mem // 4 * oom_factor) + 1000000, 4) # slightly larger than total memory
    
    try:
        # This should raise OOM exception
        huge_tensor = torch.empty(huge_shape, dtype=dtype, device=device)
        # Force kernel launch to execute allocation if lazy
        huge_tensor.fill_(1.0)
        synchronize(device)
        del huge_tensor
    except (RuntimeError, MemoryError) as e:
        # Verify it was indeed an OOM error
        err_msg = str(e).lower()
        # Common OOM messages: "out of memory", "alloc", "limit", "mps"
        assert "memory" in err_msg or "alloc" in err_msg or "oom" in err_msg or "limit" in err_msg or "buffer" in err_msg or "size" in err_msg
        
        # Clean up
        empty_cache(device)
        synchronize(device)
        
        # Verify we can allocate a small tensor successfully after recovery
        try:
            x = torch.randn(10, 10, device=device)
            assert x.shape == (10, 10)
            del x
        except Exception as recovery_err:
            pytest.fail(f"Failed to allocate tensor after OOM recovery: {recovery_err}")
    else:
        pytest.fail("Expected an out-of-memory error, but the oversized allocation succeeded.")
