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
import re
import torch
import torchcts.conftest as conftest
from torchcts.core.opinfo_adapter import (
    get_error_op_tests,
    get_live_opinfo,
    get_op_error_inputs,
)

pytestmark = pytest.mark.covers_category("opinfo_error_behavior")

# Build error test list by checking op.error_inputs at collection time (fast)
op_names_with_errors = get_error_op_tests(conftest._MANIFEST)

if not op_names_with_errors:
    op_names_with_errors = ["dummy"]

def _expected_error_types(err_in):
    expected = getattr(err_in, "error_type", None)
    if expected is None:
        return (Exception,)
    if isinstance(expected, tuple):
        return expected
    if isinstance(expected, type):
        return (expected,)
    return (Exception,)


def _expected_error_regexes(err_in):
    regex = getattr(err_in, "error_regex", None)
    regexes = getattr(err_in, "error_regexes", None)
    values = []
    if regex:
        values.append(regex)
    if regexes:
        values.extend(regexes)
    return tuple(str(value) for value in values if value)


def _assert_exception_matches_expected(exc, err_in, op_name, stage):
    expected_types = _expected_error_types(err_in)
    if not isinstance(exc, expected_types):
        expected_names = ", ".join(t.__name__ for t in expected_types)
        raise AssertionError(
            f"{stage} for {op_name} raised {type(exc).__name__}, "
            f"expected {expected_names}: {exc}"
        ) from exc

    regexes = _expected_error_regexes(err_in)
    if regexes and not any(re.search(pattern, str(exc)) for pattern in regexes):
        raise AssertionError(
            f"{stage} for {op_name} raised expected type {type(exc).__name__}, "
            f"but message did not match {regexes}: {exc}"
        ) from exc


def _assert_expected_error(op_fn, dev_input, dev_args, dev_kwargs, err_in, op_name):
    try:
        op_fn(dev_input, *dev_args, **dev_kwargs)
    except Exception as exc:
        _assert_exception_matches_expected(exc, err_in, op_name, "operation")
        return

    expected = ", ".join(t.__name__ for t in _expected_error_types(err_in))
    raise AssertionError(f"Expected exception {expected} not raised for op {op_name}")


def _move_obj(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [_move_obj(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_move_obj(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: _move_obj(value, device) for key, value in obj.items()}
    return obj


def _cpu_error_for_error_input(op_fn, err_in):
    si = err_in.sample_input
    try:
        cpu_input = _move_obj(si.input, "cpu")
        cpu_args = _move_obj(si.args, "cpu")
        cpu_kwargs = _move_obj(si.kwargs, "cpu")
    except Exception as exc:
        return exc
    try:
        op_fn(cpu_input, *cpu_args, **cpu_kwargs)
    except Exception as exc:
        return exc
    return None

@pytest.mark.opinfo
@pytest.mark.parametrize("op_name", op_names_with_errors)
def test_op_errors(op_name, device):
    if op_name == "dummy":
        pytest.skip("No OpInfo error tests matched the manifest filters.")

    op_info = get_live_opinfo(op_name)
    assert op_info is not None, f"Could not load live OpInfo for {op_name}"

    # Resolve error inputs
    try:
        errors = list(get_op_error_inputs(op_name, device))
    except Exception as exc:
        pytest.fail(f"Failed to generate error inputs for {op_name}: {exc}")

    if not errors:
        pytest.fail(f"No error inputs defined for {op_name}")

    op_fn = op_info.op
    tested_any = False

    for err_in in errors:
        si = err_in.sample_input
        cpu_error = _cpu_error_for_error_input(op_fn, err_in)
        if cpu_error is None:
            continue
        try:
            _assert_exception_matches_expected(cpu_error, err_in, op_name, "CPU reference")
        except AssertionError:
            continue
        if device == "cpu":
            tested_any = True
            continue

        # Prepare inputs on target device
        try:
            dev_input = _move_obj(si.input, device)
            dev_args = _move_obj(si.args, device)
            dev_kwargs = _move_obj(si.kwargs, device)
        except Exception as exc:
            _assert_exception_matches_expected(exc, err_in, op_name, "device placement")
            tested_any = True
            continue

        _assert_expected_error(op_fn, dev_input, dev_args, dev_kwargs, err_in, op_name)
        tested_any = True

    if not tested_any:
        pytest.fail(f"Could not run error validation checks for {op_name}")
