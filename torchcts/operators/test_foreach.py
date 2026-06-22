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

FOREACH_DTYPES = [torch.float32]

@pytest.mark.smoke
@pytest.mark.requires("foreach")
@pytest.mark.parametrize("dtype", FOREACH_DTYPES)
@pytest.mark.parametrize("op_name", ["add", "mul", "lerp"])
def test_foreach_op(dtype, op_name, device, compare, input_gen):
    tensors_dev = [input_gen((8, 8), dtype, device) for _ in range(3)]
    tensors_cpu = [t.cpu() for t in tensors_dev]
    
    if op_name == "add":
        expected = torch._foreach_add(tensors_cpu, 2.5)
        actual = torch._foreach_add(tensors_dev, 2.5)
    elif op_name == "mul":
        expected = torch._foreach_mul(tensors_cpu, 1.5)
        actual = torch._foreach_mul(tensors_dev, 1.5)
    elif op_name == "lerp":
        tensors_target_dev = [input_gen((8, 8), dtype, device) for _ in range(3)]
        tensors_target_cpu = [t.cpu() for t in tensors_target_dev]
        expected = torch._foreach_lerp(tensors_cpu, tensors_target_cpu, 0.5)
        actual = torch._foreach_lerp(tensors_dev, tensors_target_dev, 0.5)
        
    synchronize(device)
    for act, exp in zip(actual, expected):
        compare(act, exp, category="elementwise", dtype=dtype)
