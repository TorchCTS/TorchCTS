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

MATMUL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::dot")
@pytest.mark.covers("aten::mv")
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("op_name", ["dot", "mv"])
def test_dot_mv(dtype, op_name, device, compare, input_gen):
    if op_name == "dot":
        x_dev = input_gen((128,), dtype, device)
        y_dev = input_gen((128,), dtype, device)
        compare(torch.dot(x_dev, y_dev), torch.dot(x_dev.cpu(), y_dev.cpu()), category="matmul", dtype=dtype)
    else:
        M_dev = input_gen((32, 64), dtype, device)
        v_dev = input_gen((64,), dtype, device)
        compare(torch.mv(M_dev, v_dev), torch.mv(M_dev.cpu(), v_dev.cpu()), category="matmul", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::mm")
@pytest.mark.parametrize("layout_a", ["contiguous", "transpose"])
@pytest.mark.parametrize("layout_b", ["contiguous", "transpose"])
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_mm_layouts(dtype, layout_a, layout_b, device, compare, input_gen):
    M, K, N = 32, 64, 48
    shape_a = (M, K)
    shape_b = (K, N)
    
    a_dev = input_gen(shape_a, dtype, device, layout=layout_a)
    b_dev = input_gen(shape_b, dtype, device, layout=layout_b)
    
    a_cpu, b_cpu = a_dev.cpu(), b_dev.cpu()
    
    expected = torch.mm(a_cpu, b_cpu)
    actual = torch.mm(a_dev, b_dev)
    synchronize(device)
    
    category = "noncontiguous_mm" if (layout_a == "transpose" or layout_b == "transpose") else "matmul"
    compare(actual, expected, category=category, dtype=dtype)


@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::_int_mm")
@pytest.mark.covers("aten::_int_mm.out")
def test_int_mm_dispatcher_variants(device):
    a_cpu = torch.arange(12, dtype=torch.int8).reshape(3, 4)
    b_cpu = torch.arange(20, dtype=torch.int8).reshape(4, 5)
    a_dev = a_cpu.to(device)
    b_dev = b_cpu.to(device)

    expected = torch.ops.aten._int_mm(a_cpu, b_cpu)
    actual = torch.ops.aten._int_mm(a_dev, b_dev)
    synchronize(device)
    assert actual.dtype == torch.int32
    assert torch.equal(actual.cpu(), expected)

    out = torch.empty((3, 5), dtype=torch.int32, device=device)
    returned = torch.ops.aten._int_mm.out(a_dev, b_dev, out=out)
    synchronize(device)
    assert returned is out
    assert torch.equal(out.cpu(), expected)


@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::bmm")
@pytest.mark.parametrize("layout_a", ["contiguous", "transpose"])
@pytest.mark.parametrize("layout_b", ["contiguous", "transpose"])
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_bmm_layouts(dtype, layout_a, layout_b, device, compare, input_gen):
    B, M, K, N = 4, 16, 32, 24
    shape_a = (B, M, K)
    shape_b = (B, K, N)
    
    a_dev = input_gen(shape_a, dtype, device, layout=layout_a)
    b_dev = input_gen(shape_b, dtype, device, layout=layout_b)
    
    a_cpu, b_cpu = a_dev.cpu(), b_dev.cpu()
    
    expected = torch.bmm(a_cpu, b_cpu)
    actual = torch.bmm(a_dev, b_dev)
    synchronize(device)
    
    compare(actual, expected, category="matmul", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::addmm")
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_addmm(dtype, device, compare, input_gen):
    M, K, N = 32, 64, 48
    beta, alpha = 0.5, 1.5
    mat_dev = input_gen((M, N), dtype, device)
    a_dev = input_gen((M, K), dtype, device)
    b_dev = input_gen((K, N), dtype, device)
    
    expected = torch.addmm(mat_dev.cpu(), a_dev.cpu(), b_dev.cpu(), beta=beta, alpha=alpha)
    actual = torch.addmm(mat_dev, a_dev, b_dev, beta=beta, alpha=alpha)
    synchronize(device)
    
    compare(actual, expected, category="matmul", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.benchmarkable
@pytest.mark.covers("aten::matmul")
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_matmul_general(dtype, device, compare, input_gen):
    # Broadcasting matmul: (2, 3, 4) x (4, 5) -> (2, 3, 5)
    a_dev = input_gen((2, 3, 4), dtype, device)
    b_dev = input_gen((4, 5), dtype, device)
    
    expected = torch.matmul(a_dev.cpu(), b_dev.cpu())
    actual = torch.matmul(a_dev, b_dev)
    synchronize(device)
    
    compare(actual, expected, category="matmul", dtype=dtype)
