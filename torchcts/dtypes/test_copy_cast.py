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
from torchcts.core.device import synchronize

CAST_DTYPES = [
    torch.float32, torch.float16, torch.bfloat16,
    torch.int64, torch.int32, torch.int8, torch.bool
]

@pytest.mark.medium
@pytest.mark.covers("aten::_to_copy")
@pytest.mark.parametrize("src_dtype", CAST_DTYPES)
@pytest.mark.parametrize("dst_dtype", CAST_DTYPES)
def test_copy_cast_grid(src_dtype, dst_dtype, device, manifest, compare, input_gen):
    # src_dtype/dst_dtype support is verified at collection time via conftest.
        
    # Generate source tensor
    x_dev = input_gen((16, 16), src_dtype, device)
    x_cpu = x_dev.cpu()
    
    # 1. Device-to-Device Copy & Cast
    y_dev = x_dev.to(dst_dtype)
    y_cpu = x_cpu.to(dst_dtype)
    synchronize(device)
    # Use 'copy' category or loose comparison depending on lossiness
    category = "exact" if not (src_dtype.is_floating_point and not dst_dtype.is_floating_point) else "elementwise"
    compare(y_dev, y_cpu, category=category, dtype=dst_dtype)
    
    # 2. Host-to-Device Copy & Cast
    y_h2d = x_cpu.to(device=device, dtype=dst_dtype)
    synchronize(device)
    compare(y_h2d, y_cpu, category=category, dtype=dst_dtype)
    
    # 3. Device-to-Host Copy & Cast
    y_d2h = x_dev.to(device="cpu", dtype=dst_dtype)
    compare(y_d2h, y_cpu, category=category, dtype=dst_dtype)
