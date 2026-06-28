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
@pytest.mark.covers("aten::gt.Scalar")
@pytest.mark.covers("aten::index.Tensor")
@pytest.mark.covers("aten::select.int")
@pytest.mark.covers("aten::slice.Tensor")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["mixed", "boolean"])
def test_advanced_indexing_mixed(case, dtype, device, manifest, compare, input_gen):
    x_dev = input_gen((8, 8, 8), dtype, device)
    x_cpu = x_dev.cpu()
    
    if case == "mixed":
        # 1. Mixed slice and index tensor
        idx_cpu = torch.tensor([1, 3, 5], dtype=torch.int64)
        idx_dev = idx_cpu.to(device)
        
        expected = x_cpu[:, idx_cpu, 2:6]
        actual = x_dev[:, idx_dev, 2:6]
        synchronize(device)
        compare(actual, expected, category="exact", dtype=dtype)
        
    elif case == "boolean":
        # 2. Boolean mask indexing
        mask_cpu = x_cpu[0, :, 0] > 0.0
        mask_dev = mask_cpu.to(device)
        
        expected_mask = x_cpu[0, mask_cpu, :]
        actual_mask = x_dev[0, mask_dev, :]
        synchronize(device)
        compare(actual_mask, expected_mask, category="exact", dtype=dtype)
