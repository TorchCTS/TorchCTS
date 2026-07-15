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

from dataclasses import dataclass, field
import re

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


def _describe_sample_obj(obj):
    if isinstance(obj, torch.Tensor):
        return (
            f"Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, "
            f"device={obj.device}, layout={obj.layout})"
        )
    if isinstance(obj, list):
        return "[" + ", ".join(_describe_sample_obj(item) for item in obj) + "]"
    if isinstance(obj, tuple):
        return "(" + ", ".join(_describe_sample_obj(item) for item in obj) + ")"
    if isinstance(obj, dict):
        return "{" + ", ".join(
            f"{key}: {_describe_sample_obj(value)}" for key, value in obj.items()
        ) + "}"
    return type(obj).__name__


def _bounded_message(value, limit=600):
    try:
        text = str(value)
    except Exception as exc:
        text = f"<unprintable {type(value).__name__}: {type(exc).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class ErrorCandidateOutcome:
    index: int
    sample: str
    expected_types: tuple[str, ...]
    expected_regexes: tuple[str, ...]
    classification: str = "pending"
    cpu_placement: str = "not_attempted"
    cpu_execution: str = "not_attempted"
    target_placement: str = "not_attempted"
    target_execution: str = "not_attempted"
    cpu_exception_type: str | None = None
    cpu_exception_message: str | None = None
    target_exception_type: str | None = None
    target_exception_message: str | None = None
    notes: list[str] = field(default_factory=list)

    def set_cpu_exception(self, exc):
        self.cpu_exception_type = type(exc).__name__
        self.cpu_exception_message = _bounded_message(exc)

    def set_target_exception(self, exc):
        self.target_exception_type = type(exc).__name__
        self.target_exception_message = _bounded_message(exc)

    def format(self):
        expected = ",".join(self.expected_types) or "Exception"
        regexes = repr(self.expected_regexes)
        cpu = f"placement={self.cpu_placement}, execution={self.cpu_execution}"
        if self.cpu_exception_type:
            cpu += f" {self.cpu_exception_type}: {self.cpu_exception_message}"
        target = f"placement={self.target_placement}, execution={self.target_execution}"
        if self.target_exception_type:
            target += f" {self.target_exception_type}: {self.target_exception_message}"
        notes = f"; notes={' | '.join(self.notes)}" if self.notes else ""
        return (
            f"candidate {self.index}: classification={self.classification}; "
            f"expected_types={expected}; expected_regexes={regexes}; "
            f"cpu={cpu}; target={target}; sample={self.sample}{notes}"
        )


def _candidate_outcome(index, err_in):
    sample = err_in.sample_input
    descriptor = (
        f"input={_describe_sample_obj(sample.input)}, "
        f"args={_describe_sample_obj(sample.args)}, "
        f"kwargs={_describe_sample_obj(sample.kwargs)}"
    )
    return ErrorCandidateOutcome(
        index=index,
        sample=descriptor,
        expected_types=tuple(t.__name__ for t in _expected_error_types(err_in)),
        expected_regexes=_expected_error_regexes(err_in),
    )


def _exception_mismatch(exc, err_in, op_name, stage):
    try:
        _assert_exception_matches_expected(exc, err_in, op_name, stage)
    except AssertionError as mismatch:
        return _bounded_message(mismatch)
    return None


def _format_outcomes(outcomes):
    return "\n".join(outcome.format() for outcome in outcomes)


def _aggregate_error_outcomes(op_name, outcomes):
    backend_failures = [
        outcome for outcome in outcomes
        if outcome.classification == "backend_conformance_failure"
    ]
    if backend_failures:
        return (
            "backend_conformance_failure",
            f"OPINFO_BACKEND_ERROR_CONFORMANCE_FAILURE for {op_name}:\n"
            f"{_format_outcomes(backend_failures)}",
        )

    transport_failures = [
        outcome for outcome in outcomes
        if outcome.classification == "target_sample_transport_failure"
    ]
    if transport_failures:
        return (
            "target_sample_transport_failure",
            f"OPINFO_TARGET_SAMPLE_TRANSPORT_FAILURE for {op_name}:\n"
            f"{_format_outcomes(transport_failures)}",
        )

    if any(outcome.classification == "candidate_passed" for outcome in outcomes):
        return "passed", ""

    return (
        "oracle_metadata_unusable",
        f"OPINFO_ORACLE_METADATA_UNUSABLE for {op_name}:\n"
        f"{_format_outcomes(outcomes)}",
    )


def _evaluate_error_candidates(op_name, errors, device, op_fn, *, mover=_move_obj):
    outcomes = []
    for index, err_in in enumerate(errors):
        si = err_in.sample_input
        outcome = _candidate_outcome(index, err_in)
        outcomes.append(outcome)

        try:
            cpu_input = mover(si.input, "cpu")
            cpu_args = mover(si.args, "cpu")
            cpu_kwargs = mover(si.kwargs, "cpu")
            outcome.cpu_placement = "succeeded"
        except Exception as exc:
            outcome.cpu_placement = "failed"
            outcome.set_cpu_exception(exc)
            outcome.classification = "opinfo_oracle_metadata_rejection"
            continue

        try:
            op_fn(cpu_input, *cpu_args, **cpu_kwargs)
        except Exception as exc:
            outcome.cpu_execution = "raised"
            outcome.set_cpu_exception(exc)
            mismatch = _exception_mismatch(exc, err_in, op_name, "CPU reference")
            if mismatch is not None:
                outcome.notes.append(mismatch)
                outcome.classification = "opinfo_oracle_metadata_rejection"
                continue
        else:
            outcome.cpu_execution = "succeeded"
            outcome.classification = "opinfo_oracle_metadata_rejection"
            outcome.notes.append("CPU unexpectedly accepted the OpInfo error sample")
            continue

        if device == "cpu":
            outcome.target_placement = "same_as_cpu"
            outcome.target_execution = "documented_error"
            outcome.classification = "candidate_passed"
            continue

        try:
            dev_input = mover(si.input, device)
            dev_args = mover(si.args, device)
            dev_kwargs = mover(si.kwargs, device)
            outcome.target_placement = "succeeded"
        except Exception as exc:
            outcome.target_placement = "failed"
            outcome.set_target_exception(exc)
            outcome.classification = "target_sample_transport_failure"
            continue

        try:
            op_fn(dev_input, *dev_args, **dev_kwargs)
        except Exception as exc:
            outcome.target_execution = "raised"
            outcome.set_target_exception(exc)
            mismatch = _exception_mismatch(exc, err_in, op_name, "target operation")
            if mismatch is not None:
                outcome.notes.append(mismatch)
                outcome.classification = "backend_conformance_failure"
            else:
                outcome.classification = "candidate_passed"
        else:
            outcome.target_execution = "succeeded"
            outcome.classification = "backend_conformance_failure"
            outcome.notes.append("Target unexpectedly accepted the OpInfo error sample")
    return outcomes

@pytest.mark.opinfo
@pytest.mark.parametrize("op_name", op_names_with_errors)
def test_op_errors(op_name, device):
    if op_name == "dummy":
        pytest.fail("Empty OpInfo error selection placeholder was not deselected at collection time.")

    op_info = get_live_opinfo(op_name)
    assert op_info is not None, f"Could not load live OpInfo for {op_name}"

    # Resolve error inputs
    try:
        # Generate once on CPU so target transport is a separately observable
        # candidate outcome rather than an error-input construction failure.
        errors = list(get_op_error_inputs(op_name, "cpu"))
    except Exception as exc:
        pytest.fail(f"Failed to generate error inputs for {op_name}: {exc}")

    if not errors:
        pytest.fail(f"No error inputs defined for {op_name}")

    op_fn = op_info.op
    outcomes = _evaluate_error_candidates(op_name, errors, device, op_fn)

    status, message = _aggregate_error_outcomes(op_name, outcomes)
    if status == "passed":
        return
    pytest.fail(message)
