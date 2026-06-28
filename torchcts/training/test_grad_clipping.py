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

pytestmark = pytest.mark.covers_category("training_workflow")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("clipping_method", ["norm", "value"])
def test_gradient_clipping(clipping_method, dtype, device, manifest, compare):
    w_cpu = torch.randn(8, 8, dtype=dtype)
    w_cpu.grad = torch.randn(8, 8, dtype=dtype) * 10.0 # large grads
    
    w_dev = w_cpu.clone().to(device)
    w_dev.grad = w_cpu.grad.clone().to(device)
    
    if clipping_method == "norm":
        # 1. Clip norm
        torch.nn.utils.clip_grad_norm_([w_cpu], max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([w_dev], max_norm=1.0)
        synchronize(device)
        compare(w_dev.grad, w_cpu.grad, category="elementwise", dtype=dtype)
        
    elif clipping_method == "value":
        # 2. Clip value
        torch.nn.utils.clip_grad_value_([w_cpu], clip_value=0.5)
        torch.nn.utils.clip_grad_value_([w_dev], clip_value=0.5)
        synchronize(device)
        compare(w_dev.grad, w_cpu.grad, category="exact", dtype=dtype)
