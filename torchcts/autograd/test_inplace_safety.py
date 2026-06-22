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

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_name", ["add_", "sub_", "mul_", "div_"])
def test_inplace_safety_checks(op_name, dtype, device):
    # 1. In-place on a leaf tensor with requires_grad=True must raise RuntimeError
    x = torch.randn(10, requires_grad=True, device=device, dtype=dtype)
    op_fn = getattr(x, op_name)
    with pytest.raises(RuntimeError):
        op_fn(1.0)
        
    # 2. version counter increments after in-place operation
    y = torch.randn(10, device=device, dtype=dtype)
    v0 = y._version
    op_fn_y = getattr(y, op_name)
    op_fn_y(2.0)
    assert y._version > v0, "Version counter did not increment after in-place operation."

@pytest.mark.medium
@pytest.mark.requires("training")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inplace_op", ["add_", "sub_"])
def test_inplace_saved_for_backward_error(inplace_op, dtype, device):
    # Modifying a tensor saved for backward should raise a RuntimeError when backward is called.
    # Autograd saves references to tensors needed for gradient computation. If one is modified
    # in-place, the version counter should detect this and backward() must raise RuntimeError.
    x = torch.randn(10, requires_grad=True, device=device, dtype=dtype)
    y = x.pow(2)
    z = y * y

    # Capture state before in-place op
    ptr_before = y.data_ptr()
    ver_before = y._version
    storage_ptr_before = y.untyped_storage().data_ptr()

    op_fn = getattr(y, inplace_op)
    op_fn(1.0)

    # Capture state after in-place op
    ptr_after = y.data_ptr()
    ver_after = y._version
    storage_ptr_after = y.untyped_storage().data_ptr()

    # Build diagnostic info for failure message
    diag_lines = [
        f"In-place op: {inplace_op}",
        f"data_ptr   before={ptr_before:#x}  after={ptr_after:#x}  changed={ptr_before != ptr_after}",
        f"storage_ptr before={storage_ptr_before:#x}  after={storage_ptr_after:#x}  changed={storage_ptr_before != storage_ptr_after}",
        f"_version   before={ver_before}  after={ver_after}  bumped={ver_after > ver_before}",
    ]
    diag = "\n  ".join(diag_lines)

    try:
        z.sum().backward()
    except RuntimeError:
        return  # Correct behavior: autograd detected the in-place modification

    pytest.fail(
        f"backward() did not raise RuntimeError after in-place modification of a "
        f"tensor saved for backward.\n"
        f"  This means autograd did not detect that '{inplace_op}' mutated the tensor.\n"
        f"  Diagnostics:\n  {diag}\n"
        f"  If data_ptr or storage_ptr changed, the op allocated new storage instead of "
        f"mutating in-place.\n"
        f"  If _version was not bumped, the op did not increment the version counter."
    )
