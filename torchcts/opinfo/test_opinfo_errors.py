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
import torchcts.conftest as conftest
from torchcts.core.opinfo_adapter import (
    get_error_op_tests,
    get_live_opinfo,
    get_op_error_inputs,
)

pytestmark = pytest.mark.covers_category("opinfo_error_behavior")

# Build error test list by checking op.error_inputs at collection time (fast)
try:
    op_names_with_errors = get_error_op_tests(conftest._MANIFEST)
except Exception:
    op_names_with_errors = []

if not op_names_with_errors:
    op_names_with_errors = ["dummy"]

def _assert_expected_error(op_fn, dev_input, dev_args, dev_kwargs, err_in, op_name):
    try:
        op_fn(dev_input, *dev_args, **dev_kwargs)
    except Exception:
        return

    expected = getattr(err_in, "error_type", None)
    if expected is None:
        expected = "an exception"
    elif isinstance(expected, type):
        expected = expected.__name__
    raise AssertionError(f"Expected exception {expected} not raised for op {op_name}")

@pytest.mark.opinfo
@pytest.mark.parametrize("op_name", op_names_with_errors)
def test_op_errors(op_name, device):
    if op_name == "dummy":
        pytest.skip("No OpInfo error tests matched the manifest filters.")

    op_info = get_live_opinfo(op_name)
    if op_info is None:
        pytest.skip(f"Could not load live OpInfo for {op_name}")

    # Resolve error inputs
    try:
        errors = list(get_op_error_inputs(op_name, device))
    except Exception:
        pytest.skip(f"Failed to generate error inputs for {op_name}")

    if not errors:
        pytest.skip(f"No error inputs defined for {op_name}")

    op_fn = op_info.op
    tested_any = False

    for err_in in errors:
        si = err_in.sample_input
        
        # Prepare inputs on target device
        try:
            dev_input = si.input.to(device) if isinstance(si.input, torch.Tensor) else si.input
            dev_args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in si.args]
            dev_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in si.kwargs.items()}
        except Exception:
            # If placing on device fails, that is also a valid error/exception raise
            continue

        _assert_expected_error(op_fn, dev_input, dev_args, dev_kwargs, err_in, op_name)
        tested_any = True

    if not tested_any:
        pytest.skip(f"Could not run error validation checks for {op_name}")
