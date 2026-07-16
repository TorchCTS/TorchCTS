# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

from types import SimpleNamespace

import pytest
import torch

import torchcts.conftest as conftest
import torchcts.generated.coverage_helpers as coverage_helpers
from torchcts.core.opinfo_adapter import InputCondition
from torchcts.opinfo.test_opinfo_errors import (
    _aggregate_error_outcomes,
    _evaluate_error_candidates,
)
from torchcts.sample_generation import (
    rng_arg_value,
    rng_uses_target_device_generator,
)


pytestmark = pytest.mark.covers_category("selftest")


def _sample(input_value, *args, **kwargs):
    return SimpleNamespace(input=input_value, args=args, kwargs=kwargs)


def _compare(actual, expected, **_kwargs):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), equal_nan=True)


def _error_input(label):
    return SimpleNamespace(
        sample_input=SimpleNamespace(input=label, args=(), kwargs={}),
        error_type=ValueError,
        error_regex="documented",
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
    generator_arg = {"name": "generator", "kwarg_only": True}
    entry = {
        "name": "aten::example_rng",
        "base_name": "example_rng",
        "args": [generator_arg],
        "generated": {"strategy": {"strategy": "manual_rng"}},
    }
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

    cpu_generator = rng_arg_value(entry, generator_arg, torch.float32, "cpu", 67)
    target_generator = rng_arg_value(
        entry,
        generator_arg,
        torch.float32,
        "privateuseone",
        67,
    )
    item = SimpleNamespace(callspec=SimpleNamespace(params={"entry": entry}))
    skip = conftest._generated_rng_capability_skip_for_item(item)

    assert rng_uses_target_device_generator(entry)
    assert cpu_generator is not target_generator
    assert cpu_generator.device == "cpu"
    assert target_generator.device == "privateuseone"
    assert created_devices == ["cpu", "privateuseone"]
    assert skip is not None and skip[2] == {"capability": "device_generator"}


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
