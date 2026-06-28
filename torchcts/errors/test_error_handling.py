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

pytestmark = pytest.mark.covers_category("error_behavior")

@pytest.mark.smoke
@pytest.mark.parametrize("shapes", [
    ((4, 3), (5, 2)),
    ((2, 3), (4, 3))
])
def test_error_handling_shapes(shapes, device):
    shape_x, shape_y = shapes
    x = torch.randn(shape_x, device=device)
    y = torch.randn(shape_y, device=device)
    
    with pytest.raises(RuntimeError):
        torch.mm(x, y)

@pytest.mark.smoke
@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
def test_error_handling_cross_device(op_name, device):
    # CPU deselection handled at collection time in conftest.
        
    # 2. Cross-device operation must raise RuntimeError
    x_cpu = torch.randn(5)
    x_dev = torch.randn(5, device=device)
    
    op_fn = getattr(torch, op_name) if hasattr(torch, op_name) else getattr(x_cpu, f"__{op_name}__")
    
    with pytest.raises(RuntimeError):
        op_fn(x_cpu, x_dev)
