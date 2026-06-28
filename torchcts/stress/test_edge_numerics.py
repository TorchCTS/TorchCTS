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
import math

pytestmark = pytest.mark.covers_category("stress")

@pytest.mark.stress
@pytest.mark.parametrize("offset", [1.0, 2.0])
def test_nan_inf_propagation(offset, device):
    # NaN
    nan_tensor = torch.tensor([float('nan'), 1.0, 2.0], device=device)
    out_nan = nan_tensor + offset
    assert torch.isnan(out_nan[0])
    assert out_nan[1].item() == 1.0 + offset
    
    # Inf
    inf_tensor = torch.tensor([float('inf'), -float('inf'), 3.0], device=device)
    out_inf = inf_tensor + offset
    assert torch.isinf(out_inf[0])
    assert torch.isinf(out_inf[1])
    assert out_inf[2].item() == 3.0 + offset

@pytest.mark.stress
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_dtype_min_max(dtype, device, manifest):
    if dtype.is_floating_point:
        info = torch.finfo(dtype)
        min_val, max_val = info.min, info.max
    else:
        info = torch.iinfo(dtype)
        min_val, max_val = info.min, info.max
        
    t_min = torch.tensor([min_val], dtype=dtype, device=device)
    t_max = torch.tensor([max_val], dtype=dtype, device=device)
    
    assert t_min.item() == min_val
    assert t_max.item() == max_val
