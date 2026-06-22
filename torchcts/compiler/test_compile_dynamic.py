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

COMPILE_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

_SHAPE_SEQUENCES = {
    "small_varying": [(8, 8), (16, 16), (32, 16), (8, 8)],
    "medium_varying": [(4, 4), (8, 8), (12, 8), (4, 4)],
    "batch_varying": [(2, 16), (4, 16), (8, 16), (2, 16)],
}

_DYNAMIC_OPS = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "matmul": lambda a, b: torch.mm(a, b),
    "relu_add": lambda a, b: torch.nn.functional.relu(a + b),
    "gelu_mul": lambda a, b: torch.nn.functional.gelu(a * b),
}


@pytest.mark.smoke
@pytest.mark.requires("compile")
@pytest.mark.parametrize("op_name", list(_DYNAMIC_OPS.keys()))
@pytest.mark.parametrize("seq_name", list(_SHAPE_SEQUENCES.keys()))
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_dynamic_shapes(op_name, seq_name, dtype, device, compare, input_gen):
    op_fn = _DYNAMIC_OPS[op_name]
    shapes = _SHAPE_SEQUENCES[seq_name]

    compiled_fn = torch.compile(op_fn, dynamic=True)

    for shape in shapes:
        x = input_gen(shape, dtype, device)
        y_shape = (shape[1], shape[0]) if op_name == "matmul" else shape
        y = input_gen(y_shape, dtype, device)

        expected = op_fn(x, y)
        actual = compiled_fn(x, y)
        synchronize(device)

        compare(actual, expected, category="compile", dtype=dtype)


@pytest.mark.medium
@pytest.mark.requires("compile")
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
def test_compile_dynamic_batch_linear(dtype, device, compare, input_gen):
    """Test compiled Linear layer with varying batch sizes."""
    model = torch.nn.Linear(16, 8).to(device)
    if dtype != torch.float32:
        model = model.to(dtype)

    compiled_model = torch.compile(model, dynamic=True)

    for batch_size in [1, 4, 8, 2]:
        x = input_gen((batch_size, 16), dtype, device)

        expected = model(x)
        actual = compiled_model(x)
        synchronize(device)

        compare(actual, expected, category="compile", dtype=dtype)
