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

POOL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.smoke
@pytest.mark.covers("aten::max_pool2d_with_indices")
@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_max_pool2d(dtype, device, compare, input_gen):
    shape = (2, 3, 16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected = torch.nn.functional.max_pool2d(x_cpu, kernel_size=2, stride=2)
    actual = torch.nn.functional.max_pool2d(x_dev, kernel_size=2, stride=2)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::avg_pool2d")
@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_avg_pool2d(dtype, device, compare, input_gen):
    shape = (2, 3, 16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected = torch.nn.functional.avg_pool2d(x_cpu, kernel_size=2, stride=2)
    actual = torch.nn.functional.avg_pool2d(x_dev, kernel_size=2, stride=2)
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::_adaptive_avg_pool2d")
@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_adaptive_avg_pool2d(dtype, device, compare, input_gen):
    shape = (2, 3, 16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected = torch.nn.functional.adaptive_avg_pool2d(x_cpu, output_size=(8, 8))
    actual = torch.nn.functional.adaptive_avg_pool2d(x_dev, output_size=(8, 8))
    synchronize(device)
    
    compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::adaptive_max_pool2d")
@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_adaptive_max_pool2d(dtype, device, compare, input_gen):
    shape = (2, 3, 16, 16)
    x_dev = input_gen(shape, dtype, device)
    x_cpu = x_dev.cpu()
    
    expected_out, expected_ind = torch.nn.functional.adaptive_max_pool2d(x_cpu, output_size=(8, 8))
    actual_out, actual_ind = torch.nn.functional.adaptive_max_pool2d(x_dev, output_size=(8, 8))
    synchronize(device)
    
    compare(actual_out, expected_out, category="elementwise", dtype=dtype)
    compare(actual_ind, expected_ind, category="exact", dtype=torch.int64)


@pytest.mark.smoke
@pytest.mark.covers("aten::max_unpool2d.out", surface="out_variant")
@pytest.mark.covers("aten::max_unpool3d.out", surface="out_variant")
def test_direct_max_unpool_out_variants(device, compare):
    input2_cpu = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    pooled2_cpu, indices2_cpu = torch.nn.functional.max_pool2d(
        input2_cpu, 2, 2, return_indices=True,
    )
    expected = torch.nn.functional.max_unpool2d(
        pooled2_cpu, indices2_cpu, 2, 2, output_size=(1, 1, 4, 4),
    )
    out = torch.empty_like(expected, device=device)
    actual = torch.ops.aten.max_unpool2d.out(
        pooled2_cpu.to(device),
        indices2_cpu.to(device),
        [4, 4],
        out=out,
    )
    assert actual.data_ptr() == out.data_ptr()
    synchronize(device)
    compare(out, expected, category="exact", dtype=torch.float32)

    input3_cpu = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
    pooled3_cpu, indices3_cpu = torch.nn.functional.max_pool3d(
        input3_cpu, 2, 2, return_indices=True,
    )
    expected = torch.nn.functional.max_unpool3d(
        pooled3_cpu, indices3_cpu, 2, 2, output_size=(1, 1, 4, 4, 4),
    )
    out = torch.empty_like(expected, device=device)
    actual = torch.ops.aten.max_unpool3d.out(
        pooled3_cpu.to(device),
        indices3_cpu.to(device),
        [4, 4, 4],
        [2, 2, 2],
        [0, 0, 0],
        out=out,
    )
    assert actual.data_ptr() == out.data_ptr()
    synchronize(device)
    compare(out, expected, category="exact", dtype=torch.float32)
