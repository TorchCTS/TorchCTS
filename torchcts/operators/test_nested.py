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

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.parametrize("dtype", DTYPES)
def test_nested_tensor_construction(dtype, device):
    components = [
        torch.randn(2, 3, dtype=dtype),
        torch.randn(4, 3, dtype=dtype),
        torch.randn(1, 3, dtype=dtype),
    ]
    nt = torch.nested.nested_tensor(components, dtype=dtype, device=device)
    synchronize(device)
    assert nt.is_nested
    assert nt.device.type == device


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.parametrize("dtype", DTYPES)
def test_nested_tensor_to_padded(dtype, device):
    components = [
        torch.randn(2, 4, dtype=dtype),
        torch.randn(5, 4, dtype=dtype),
    ]
    nt = torch.nested.nested_tensor(components, dtype=dtype, device=device)
    padded = nt.to_padded_tensor(padding=0.0)
    synchronize(device)
    assert padded.shape == (2, 5, 4)
    assert padded.device.type == device
    padded_cpu = padded.cpu()
    assert torch.all(padded_cpu[0, 2:, :] == 0.0)


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.parametrize("dtype", DTYPES)
def test_jagged_tensor_construction(dtype, device):
    components = [
        torch.randn(3, 5, dtype=dtype),
        torch.randn(7, 5, dtype=dtype),
        torch.randn(2, 5, dtype=dtype),
    ]
    nt = torch.nested.nested_tensor(
        components, dtype=dtype, device=device, layout=torch.jagged,
    )
    synchronize(device)
    assert nt.is_nested
    assert nt.device.type == device


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_nested_tensor_unary_op(dtype, device):
    components_cpu = [torch.randn(3, 4, dtype=dtype), torch.randn(5, 4, dtype=dtype)]
    nt_cpu = torch.nested.nested_tensor(components_cpu, dtype=dtype, device="cpu")
    nt_dev = torch.nested.nested_tensor(components_cpu, dtype=dtype, device=device)

    result_cpu = torch.abs(nt_cpu).to_padded_tensor(0.0)
    result_dev = torch.abs(nt_dev).to_padded_tensor(0.0).cpu()
    synchronize(device)
    assert torch.allclose(result_dev, result_cpu), "Nested tensor abs() mismatch"


@pytest.mark.smoke
@pytest.mark.requires("nested")
@pytest.mark.parametrize("dtype", [torch.float32])
def test_nested_tensor_add(dtype, device):
    c1 = [torch.randn(2, 3, dtype=dtype), torch.randn(4, 3, dtype=dtype)]
    c2 = [torch.randn(2, 3, dtype=dtype), torch.randn(4, 3, dtype=dtype)]

    nt1_cpu = torch.nested.nested_tensor(c1, dtype=dtype, device="cpu")
    nt2_cpu = torch.nested.nested_tensor(c2, dtype=dtype, device="cpu")
    nt1_dev = torch.nested.nested_tensor(c1, dtype=dtype, device=device)
    nt2_dev = torch.nested.nested_tensor(c2, dtype=dtype, device=device)

    result_cpu = torch.add(nt1_cpu, nt2_cpu).to_padded_tensor(0.0)
    result_dev = torch.add(nt1_dev, nt2_dev).to_padded_tensor(0.0).cpu()
    synchronize(device)
    assert torch.allclose(result_dev, result_cpu), "Nested tensor add mismatch"
