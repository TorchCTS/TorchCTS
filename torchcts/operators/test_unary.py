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

# Dtypes we want to test unary ops on
UNARY_FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
UNARY_INT_DTYPES = [torch.int64, torch.int32]

# Filter ops to only those that exist in torch at collection time
_UNARY_FLOAT_OPS = [op for op in [
    "abs", "neg", "sin", "cos", "exp", "log", "sqrt", "rsqrt", "erf",
    "sigmoid", "tanh", "log10", "log2", "floor", "ceil", "round", "reciprocal", "sign"
] if hasattr(torch, op)]

_UNARY_INT_OPS = [op for op in ["abs", "neg", "floor", "ceil", "round", "sign"] if hasattr(torch, op)]

# Helper to generate input range suitable for the op
def make_unary_input(op_name, shape, dtype, device, input_gen):
    # Some ops require positive values
    if op_name in ("log", "log2", "log10", "sqrt", "rsqrt"):
        return input_gen(shape, dtype, device, positive_only=True)
    elif op_name in ("acos", "asin", "acosh", "atanh"):
        # Range (-1.0, 1.0)
        t = input_gen(shape, dtype, device)
        # Normalize to (-0.9, 0.9)
        return t * 0.45
    elif op_name == "cosh" or op_name == "sinh" or op_name == "exp":
        # Keep input small to avoid infs/overflows
        t = input_gen(shape, dtype, device)
        return t * 0.1
    elif op_name == "reciprocal":
        # Avoid zeros
        t = input_gen(shape, dtype, device, positive_only=True)
        return t + 0.5
    return input_gen(shape, dtype, device)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::abs")
@pytest.mark.covers("aten::ceil")
@pytest.mark.covers("aten::cos")
@pytest.mark.covers("aten::erf")
@pytest.mark.covers("aten::exp")
@pytest.mark.covers("aten::floor")
@pytest.mark.covers("aten::log")
@pytest.mark.covers("aten::log10")
@pytest.mark.covers("aten::log2")
@pytest.mark.covers("aten::neg")
@pytest.mark.covers("aten::reciprocal")
@pytest.mark.covers("aten::round")
@pytest.mark.covers("aten::rsqrt")
@pytest.mark.covers("aten::sigmoid")
@pytest.mark.covers("aten::sign")
@pytest.mark.covers("aten::sin")
@pytest.mark.covers("aten::sqrt")
@pytest.mark.covers("aten::tanh")
@pytest.mark.parametrize("op_name", _UNARY_FLOAT_OPS)
@pytest.mark.parametrize("dtype", UNARY_FLOAT_DTYPES)
def test_unary_float_op(op_name, dtype, device, manifest, compare, input_gen):
    op_fn = getattr(torch, op_name)

    shape = (32, 32)
    x_dev = make_unary_input(op_name, shape, dtype, device, input_gen)
    x_cpu = x_dev.cpu()

    try:
        expected = op_fn(x_cpu)
        actual = op_fn(x_dev)
        synchronize(device)
    except Exception as e:
        raise RuntimeError(f"Unary op '{op_name}' failed on {device}: {e}") from e

    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::abs")
@pytest.mark.covers("aten::ceil")
@pytest.mark.covers("aten::floor")
@pytest.mark.covers("aten::neg")
@pytest.mark.covers("aten::round")
@pytest.mark.covers("aten::sign")
@pytest.mark.parametrize("op_name", _UNARY_INT_OPS)
@pytest.mark.parametrize("dtype", UNARY_INT_DTYPES)
def test_unary_int_op(op_name, dtype, device, manifest, compare, input_gen):
    op_fn = getattr(torch, op_name)

    shape = (32, 32)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()

    try:
        expected = op_fn(x_cpu)
        actual = op_fn(x_dev)
        synchronize(device)
    except Exception as e:
        raise RuntimeError(f"Unary integer op '{op_name}' failed on {device}: {e}") from e

    compare(actual, expected, category="exact", dtype=dtype)
