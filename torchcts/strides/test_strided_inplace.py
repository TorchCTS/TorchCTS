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

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.medium
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["transpose", "slice"])
def test_strided_inplace_updates(case, dtype, device, manifest, compare, input_gen):
    if case == "transpose":
        # 1. In-place fill on transpose view
        x_dev = input_gen((16, 16), dtype, device)
        x_trans_dev = x_dev.T
        
        x_cpu = x_dev.cpu()
        x_trans_cpu = x_cpu.T
        
        x_trans_dev.fill_(42.0)
        x_trans_cpu.fill_(42.0)
        synchronize(device)
        compare(x_dev, x_cpu, category="exact", dtype=dtype)
        
    elif case == "slice":
        # 2. In-place clamp on sliced view
        y_base_dev = input_gen((32,), dtype, device)
        y_base_cpu = y_base_dev.cpu()
        
        y_slice_dev = y_base_dev[::2]
        y_slice_cpu = y_base_cpu[::2]
        
        y_slice_dev.clamp_(min=-0.5, max=0.5)
        y_slice_cpu.clamp_(min=-0.5, max=0.5)
        synchronize(device)
        compare(y_base_dev, y_base_cpu, category="exact", dtype=dtype)
