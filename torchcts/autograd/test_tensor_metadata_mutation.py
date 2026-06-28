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

from __future__ import annotations

import pytest
import torch

from torchcts.core.device import synchronize


@pytest.mark.smoke
@pytest.mark.covers("aten::get_device")
def test_get_device_metadata(device):
    tensor = torch.empty((2, 3), device=device)
    actual = torch.ops.aten.get_device.default(tensor)
    expected = -1 if tensor.device.type == "cpu" else (tensor.device.index or 0)
    assert actual == expected


@pytest.mark.smoke
@pytest.mark.covers("aten::detach_")
@pytest.mark.covers("aten::requires_grad_")
@pytest.mark.covers("aten::retain_grad")
def test_autograd_metadata_mutation_surfaces(device):
    leaf = torch.randn((2, 3), dtype=torch.float32, device=device)
    returned = torch.ops.aten.requires_grad_.default(leaf, True)
    assert returned is leaf
    assert leaf.requires_grad

    non_leaf = leaf * 2
    torch.ops.aten.retain_grad.default(non_leaf)
    loss = non_leaf.sum()
    loss.backward()
    synchronize(device)
    assert non_leaf.grad is not None
    torch.testing.assert_close(non_leaf.grad.detach().cpu(), torch.ones((2, 3), dtype=torch.float32))

    detached = non_leaf.clone()
    returned = torch.ops.aten.detach_.default(detached)
    assert returned is detached
    assert not detached.requires_grad
    assert detached.grad_fn is None
