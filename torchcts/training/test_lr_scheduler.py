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

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scheduler_name", ["StepLR", "ExponentialLR"])
def test_lr_schedulers(scheduler_name, dtype, device):
    w = torch.randn(2, 2, dtype=dtype, device=device, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    
    if scheduler_name == "StepLR":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    elif scheduler_name == "ExponentialLR":
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
    
    # Check step
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)
