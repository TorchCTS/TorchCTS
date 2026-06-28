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

CREATION_DTYPES = [torch.float32, torch.int64, torch.bool]
NUMERIC_DTYPES = [torch.float32, torch.int64]

@pytest.mark.smoke
@pytest.mark.covers("aten::full")
@pytest.mark.covers("aten::ones")
@pytest.mark.covers("aten::zeros")
@pytest.mark.parametrize("dtype", CREATION_DTYPES)
def test_zeros_ones_full(dtype, device, compare):
    shape = (4, 4)
    # zeros
    expected = torch.zeros(shape, dtype=dtype)
    actual = torch.zeros(shape, dtype=dtype, device=device)
    compare(actual, expected, category="exact", dtype=dtype)
    
    # ones
    expected = torch.ones(shape, dtype=dtype)
    actual = torch.ones(shape, dtype=dtype, device=device)
    compare(actual, expected, category="exact", dtype=dtype)
    
    # full
    val = 5 if dtype != torch.bool else True
    expected = torch.full(shape, val, dtype=dtype)
    actual = torch.full(shape, val, dtype=dtype, device=device)
    compare(actual, expected, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::eye")
@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_eye(dtype, device, compare):
    expected = torch.eye(5, dtype=dtype)
    actual = torch.eye(5, dtype=dtype, device=device)
    compare(actual, expected, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::arange.start_step")
@pytest.mark.covers("aten::linspace")
@pytest.mark.covers("aten::logspace")
@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_arange_linspace_logspace(dtype, device, compare):
    # arange
    expected = torch.arange(0, 10, 2, dtype=dtype)
    actual = torch.arange(0, 10, 2, dtype=dtype, device=device)
    compare(actual, expected, category="exact", dtype=dtype)
    
    if dtype.is_floating_point:
        # linspace
        expected = torch.linspace(0.0, 1.0, 5, dtype=dtype)
        actual = torch.linspace(0.0, 1.0, 5, dtype=dtype, device=device)
        compare(actual, expected, category="elementwise", dtype=dtype)
        
        # logspace
        expected = torch.logspace(0.0, 2.0, 5, dtype=dtype)
        actual = torch.logspace(0.0, 2.0, 5, dtype=dtype, device=device)
        compare(actual, expected, category="elementwise", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::lift_fresh")
@pytest.mark.parametrize("dtype", CREATION_DTYPES)
def test_tensor_construction_methods(dtype, device, compare):
    # Scalar tensor
    val = 3.14 if dtype.is_floating_point else (1 if dtype != torch.bool else True)
    expected = torch.tensor(val, dtype=dtype)
    actual = torch.tensor(val, dtype=dtype, device=device)
    compare(actual, expected, category="exact", dtype=dtype)
    
    # List tensor
    lst = [1, 2, 3] if dtype != torch.bool else [True, False, True]
    expected = torch.tensor(lst, dtype=dtype)
    actual = torch.tensor(lst, dtype=dtype, device=device)
    compare(actual, expected, category="exact", dtype=dtype)
    
    # as_tensor
    cpu_t = torch.tensor(lst, dtype=dtype)
    actual = torch.as_tensor(cpu_t, device=device)
    compare(actual, cpu_t, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::empty.memory_format")
@pytest.mark.parametrize("dtype", CREATION_DTYPES)
def test_empty_factory(dtype, device):
    shape = (10, 10)
    # Empty tensor allocates storage on device without crashing
    t = torch.empty(shape, dtype=dtype, device=device)
    assert t.shape == shape
    assert t.dtype == dtype
    assert t.device.type == device

@pytest.mark.smoke
@pytest.mark.covers("aten::lift_fresh")
@pytest.mark.covers("aten::new_full")
@pytest.mark.covers("aten::new_ones")
@pytest.mark.covers("aten::new_zeros")
@pytest.mark.parametrize("dtype", CREATION_DTYPES)
def test_new_methods(dtype, device, compare):
    base = torch.ones((2, 2), dtype=dtype, device=device)
    
    # new_zeros
    actual = base.new_zeros((3, 3))
    expected = torch.zeros((3, 3), dtype=dtype)
    compare(actual, expected, category="exact", dtype=dtype)
    
    # new_ones
    actual = base.new_ones((3, 3))
    expected = torch.ones((3, 3), dtype=dtype)
    compare(actual, expected, category="exact", dtype=dtype)
    
    # new_full
    val = 7 if dtype != torch.bool else False
    actual = base.new_full((3, 3), val)
    expected = torch.full((3, 3), val, dtype=dtype)
    compare(actual, expected, category="exact", dtype=dtype)
    
    # new_tensor
    lst = [4, 5] if dtype != torch.bool else [False, False]
    actual = base.new_tensor(lst)
    expected = torch.tensor(lst, dtype=dtype)
    compare(actual, expected, category="exact", dtype=dtype)

@pytest.mark.smoke
@pytest.mark.covers("aten::_to_copy")
@pytest.mark.covers("aten::clone")
@pytest.mark.parametrize("dtype", CREATION_DTYPES)
def test_clone_to_methods(dtype, device, compare):
    val = torch.randn(5, 5).to(dtype) if dtype.is_floating_point else torch.randint(0, 2, (5, 5)).to(dtype)
    
    # clone
    dev_t = val.to(device)
    cloned = dev_t.clone()
    compare(cloned, val, category="exact", dtype=dtype)
    
    # to(device)
    to_dev = val.to(device)
    compare(to_dev, val, category="exact", dtype=dtype)
    
    # to(dtype)
    # convert to float32
    if dtype != torch.float32:
        f32_dev = to_dev.to(torch.float32)
        f32_expected = val.to(torch.float32)
        compare(f32_dev, f32_expected, category="exact", dtype=torch.float32)
