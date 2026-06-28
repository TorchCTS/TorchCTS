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
from torch.autograd import gradcheck

pytestmark = pytest.mark.covers_category("gradcheck")

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.requires("gradcheck")
@pytest.mark.parametrize("op_name", ["pow", "sin", "exp", "sigmoid"])
def test_gradcheck_ops(op_name, device):
    # float64 support is verified at collection time via conftest.
    # If we get here, the manifest declares float64 as supported.
    x = torch.randn(2, 2, dtype=torch.float64, device=device, requires_grad=True)
        
    if op_name == "pow":
        func = lambda inputs: inputs.pow(3).sum()
    elif op_name == "sin":
        func = lambda inputs: torch.sin(inputs).sum()
    elif op_name == "exp":
        func = lambda inputs: torch.exp(inputs).sum()
    elif op_name == "sigmoid":
        func = lambda inputs: torch.sigmoid(inputs).sum()
        
    assert gradcheck(func, (x,), eps=1e-6, atol=1e-4)
