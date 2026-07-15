# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.
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

from types import SimpleNamespace

import pytest
import torch

import torchcts.conftest as conftest
import torchcts.generated.coverage_helpers as coverage_helpers
from torchcts.core.comparer import restore_metrics, snapshot_metrics
from torchcts.core.coverage import generated_entries_for
from torchcts.core.non_unique_output_compare import compare_non_unique_output
from torchcts.core.opinfo_adapter import InputCondition
from torchcts.opinfo.test_opinfo_errors import (
    ErrorCandidateOutcome,
    _aggregate_error_outcomes,
    _evaluate_error_candidates,
)
from torchcts.operators.test_sparse import (
    _assert_sparse_tensor_equal,
    _is_sparse_addmm_additive_layout_rejection,
)
from torchcts.sample_generation import (
    rng_arg_value,
    rng_generator_device,
    rng_uses_target_device_generator,
)


def _sample(input_value, *args, **kwargs):
    return SimpleNamespace(input=input_value, args=args, kwargs=kwargs)


def _compare(actual, expected, **_kwargs):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), equal_nan=True)


def test_error_outcome_aggregation_preserves_classification_and_details():
    oracle = ErrorCandidateOutcome(0, "input=Tensor(shape=(1,))", ("RuntimeError",), ("bad",))
    oracle.classification = "opinfo_oracle_metadata_rejection"
    transport = ErrorCandidateOutcome(1, "input=Tensor(shape=(2,))", ("RuntimeError",), ())
    transport.classification = "target_sample_transport_failure"
    status, message = _aggregate_error_outcomes("demo", [oracle, transport])
    assert status == "target_sample_transport_failure"
    assert "OPINFO_TARGET_SAMPLE_TRANSPORT_FAILURE" in message
    assert "ORACLE_METADATA_UNUSABLE" not in message

    status, message = _aggregate_error_outcomes("demo", [oracle])
    assert status == "oracle_metadata_unusable"
    assert "candidate 0" in message
    assert "input=Tensor(shape=(1,))" in message

    backend = ErrorCandidateOutcome(2, "input=int", ("ValueError",), ())
    backend.classification = "backend_conformance_failure"
    status, message = _aggregate_error_outcomes("demo", [oracle, backend])
    assert status == "backend_conformance_failure"
    assert "candidate 2" in message


def _error_input(label, *, error_type=ValueError, error_regex="documented"):
    return SimpleNamespace(
        sample_input=SimpleNamespace(input=label, args=(), kwargs={}),
        error_type=error_type,
        error_regex=error_regex,
    )


def test_all_unusable_error_candidates_report_every_outcome():
    errors = [
        _error_input("cpu_accepts"),
        _error_input("wrong_type"),
        _error_input("wrong_regex"),
        _error_input("cpu_placement_fails"),
    ]

    def mover(value, device):
        if value == "cpu_placement_fails" and device == "cpu":
            raise RuntimeError("cannot move CPU sample")
        return value

    def operation(value):
        if value == "cpu_accepts":
            return value
        if value == "wrong_type":
            raise RuntimeError("documented")
        raise ValueError("different message")

    outcomes = _evaluate_error_candidates("demo", errors, "cpu", operation, mover=mover)
    status, message = _aggregate_error_outcomes("demo", outcomes)
    assert status == "oracle_metadata_unusable"
    for index in range(4):
        assert f"candidate {index}:" in message
    assert "placement=succeeded" in message
    assert "execution=succeeded" in message
    assert "RuntimeError: documented" in message
    assert "ValueError: different message" in message
    assert "RuntimeError: cannot move CPU sample" in message


def test_target_placement_failure_is_not_bad_oracle():
    error = _error_input("invalid")

    def mover(value, device):
        if device == "privateuseone":
            raise RuntimeError("target transport failed")
        return value

    def operation(_value):
        raise ValueError("documented")

    outcomes = _evaluate_error_candidates(
        "demo",
        [error],
        "privateuseone",
        operation,
        mover=mover,
    )
    status, message = _aggregate_error_outcomes("demo", outcomes)
    assert status == "target_sample_transport_failure"
    assert "OPINFO_TARGET_SAMPLE_TRANSPORT_FAILURE" in message
    assert "OPINFO_ORACLE_METADATA_UNUSABLE" not in message
    assert outcomes[0].cpu_execution == "raised"
    assert outcomes[0].target_placement == "failed"


def test_target_accepting_invalid_error_sample_is_backend_failure():
    error = _error_input("invalid")

    def mover(value, device):
        return (device, value) if isinstance(value, str) else value

    def operation(value):
        if value[0] == "cpu":
            raise ValueError("documented")
        return value

    outcomes = _evaluate_error_candidates(
        "demo",
        [error],
        "privateuseone",
        operation,
        mover=mover,
    )
    status, message = _aggregate_error_outcomes("demo", outcomes)
    assert status == "backend_conformance_failure"
    assert "OPINFO_BACKEND_ERROR_CONFORMANCE_FAILURE" in message
    assert outcomes[0].target_execution == "succeeded"


def test_rng_generator_ownership_always_follows_target_device(monkeypatch):
    expected_names = {
        "aten::_sample_dirichlet",
        "aten::_sample_dirichlet.out",
        "aten::binomial",
        "aten::binomial.out",
        "aten::poisson.out",
    }
    entries = {}
    for surface in ("functional_data", "out_variant"):
        for entry in generated_entries_for(surface):
            if entry["name"] in expected_names:
                entries[entry["name"]] = entry
    assert set(entries) == expected_names
    assert "aten::poisson" not in entries

    created_devices = []

    class FakeGenerator:
        def __init__(self, *, device):
            self.device = device
            created_devices.append(device)

        def manual_seed(self, _seed):
            return self

    monkeypatch.setattr(torch, "Generator", FakeGenerator)
    monkeypatch.setattr(
        conftest,
        "_MANIFEST",
        {"capabilities": {"rng": True, "device_generator": False}},
    )
    for entry in entries.values():
        generator_arg = next(arg for arg in entry["args"] if arg.get("name") == "generator")
        assert rng_uses_target_device_generator(entry)
        assert rng_generator_device(entry, "privateuseone") == "privateuseone"
        cpu_generator = rng_arg_value(entry, generator_arg, torch.float32, "cpu", 67)
        target_generator = rng_arg_value(
            entry,
            generator_arg,
            torch.float32,
            "privateuseone",
            67,
        )
        assert cpu_generator is not target_generator
        assert cpu_generator.device == "cpu"
        assert target_generator.device == "privateuseone"
        item = SimpleNamespace(callspec=SimpleNamespace(params={"entry": entry}))
        skip = conftest._generated_rng_capability_skip_for_item(item)
        assert skip is not None and skip[2] == {"capability": "device_generator"}
    assert created_devices == [value for _ in entries for value in ("cpu", "privateuseone")]


def test_sparse_values_use_reduction_tolerance_while_indices_remain_exact(compare):
    indices = torch.tensor([[0, 1], [1, 0]])
    expected = torch.sparse_coo_tensor(indices, torch.tensor([1.0, 2.0]), (2, 2)).coalesce()
    actual = torch.sparse_coo_tensor(indices, torch.tensor([1.00001, 2.0]), (2, 2)).coalesce()
    _assert_sparse_tensor_equal(actual, expected, compare, value_category="reduction")
    with pytest.raises(AssertionError):
        _assert_sparse_tensor_equal(actual, expected, compare)

    expected_zero = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]),
        torch.tensor([0.0]),
        (2, 2),
    ).coalesce()
    different_index_zero = torch.sparse_coo_tensor(
        torch.tensor([[1], [1]]),
        torch.tensor([0.0]),
        (2, 2),
    ).coalesce()
    with pytest.raises(AssertionError):
        _assert_sparse_tensor_equal(
            different_index_zero,
            expected_zero,
            compare,
            value_category="reduction",
        )


def test_sparse_addmm_rejection_classifier_rejects_missing_coverage_messages():
    assert _is_sparse_addmm_additive_layout_rejection(
        RuntimeError("addmm_sparse_dense: expected strided result tensor")
    )
    assert _is_sparse_addmm_additive_layout_rejection(
        RuntimeError("sparse.addmm: input tensor must be strided")
    )
    assert not _is_sparse_addmm_additive_layout_rejection(
        RuntimeError("operator is not implemented for the SparsePrivateUse1 backend")
    )


def test_pooling_nonunique_comparer_uses_sample_input_context():
    input_tensor = torch.tensor([[[2.0, 2.0]]])
    expected = (torch.tensor([[[2.0]]]), torch.tensor([[[0]]]))
    alternate = (torch.tensor([[[2.0]]]), torch.tensor([[[1]]]))
    sample = _sample(input_tensor)
    metrics = snapshot_metrics()
    try:
        compare_non_unique_output(
            "aten::max_pool1d_with_indices",
            alternate,
            expected,
            sample=sample,
            category="reduction",
            dtype=torch.float32,
            compare=_compare,
        )
        invalid = (torch.tensor([[[2.0]]]), torch.tensor([[[2]]]))
        with pytest.raises(AssertionError, match="outside input pooling plane bounds"):
            compare_non_unique_output(
                "aten::max_pool1d_with_indices",
                invalid,
                expected,
                sample=sample,
                category="reduction",
                dtype=torch.float32,
                compare=_compare,
            )
    finally:
        restore_metrics(metrics)


def test_manual_pooling_runner_passes_target_sample_context(monkeypatch):
    cpu_sample = _sample(torch.tensor([[[2.0, 2.0]]]))
    target_sample = _sample(torch.tensor([[[2.0, 2.0]]]))
    calls = []

    def fake_run(*args, **kwargs):
        device = args[4]
        output = (torch.tensor([[[2.0]]]), torch.tensor([[[0]]]))
        return output, cpu_sample if device == "cpu" else target_sample

    def fake_compare(*args, **kwargs):
        calls.append(kwargs.get("sample"))

    monkeypatch.setattr(coverage_helpers, "_run_pooling_once", fake_run)
    monkeypatch.setattr(coverage_helpers, "_compare_multi_output_results", fake_compare)
    monkeypatch.setattr(coverage_helpers, "synchronize", lambda _device: None)
    assert coverage_helpers._run_manual_pooling_case(
        {"name": "aten::max_pool1d_with_indices", "schema": "demo"},
        object(),
        torch.float32,
        InputCondition.CLEAN,
        "privateuseone",
        _compare,
        {},
    )
    assert calls == [target_sample]
