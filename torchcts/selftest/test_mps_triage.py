# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

import json
from types import SimpleNamespace

import pytest
import torch

import torchcts.conftest as harness
from torchcts.core import adaptive_isolation, known_segfaults, triage
from torchcts.generated import coverage_helpers


pytestmark = pytest.mark.covers_category("selftest")


def _adaptive_result_payload(results, *, completed=False):
    return {
        "metadata": {
            "device_name": "mps",
            "hardware_key": "hw",
            "pytorch_version": "2.12.1",
            "timestamp": "2026-06-28T00:00:00Z",
            "session_completed": completed,
        },
        "results": results,
        "skips": {},
    }


def _known_segfault_entry(**overrides):
    entry = {
        "id": "mps-example",
        "backend": "mps",
        "match": "nodeid",
        "nodeid": "torchcts/example.py::test_crash",
        "dispatcher": "aten::example.default",
        "evidence_scope": "exact_node",
        "classification": "confirmed_backend_crash",
        "expected_signal": "SIGSEGV",
        "repro": {"script": "repro.py", "case": "case0"},
        "reason": "standalone repro crashes",
        "owner": "torchcts",
        "pytorch_min": "2.12.0",
        "pytorch_max": None,
        "hardware": "any",
        "review_after": "2026-09-30",
    }
    for key, value in overrides.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
    return entry


def _generated_item(nodeid, entry, *, dtype=None):
    params = {"entry": entry}
    if dtype is not None:
        params["dtype"] = dtype
    return SimpleNamespace(
        nodeid=nodeid,
        fspath="torchcts/generated/test_foreach_fused.py",
        name=nodeid.rsplit("::", 1)[-1],
        callspec=SimpleNamespace(params=params),
        iter_markers=lambda name=None: iter(()),
    )


def _generated_preflight_entry():
    return {
        "name": "aten::example",
        "base_name": "example",
        "status": "covered_generated",
        "surface_kind": "functional_data",
        "returns": [{"type": "Tensor", "tensor": True}],
        "generated": {"strategy": {"strategy": "manual_elementwise", "family": "example"}},
    }


def test_opinfo_all_candidates_unusable_has_explicit_bad_oracle_triage():
    unusable = triage.classify_record({
        "error_message": "OPINFO_ORACLE_METADATA_UNUSABLE for bernoulli",
    })
    transport = triage.classify_record({
        "error_message": "OPINFO_TARGET_SAMPLE_TRANSPORT_FAILURE for bernoulli",
    })

    assert unusable["classification"] == "torchcts_bad_oracle"
    assert transport["classification"] != "torchcts_bad_oracle"


def test_generated_collection_preflight_blocks_cpu_contract(monkeypatch):
    manifest = {"supported_dtypes": {"torch.float32": True}}
    coverage_helpers._RECORDED_CONTRACT_DISPOSITION_CACHE.clear()
    monkeypatch.setattr(
        coverage_helpers,
        "contract_disposition",
        lambda name, dtype: SimpleNamespace(allowed=False, status="cpu_unsupported"),
    )

    reason, detail, extra = coverage_helpers.generated_collection_skip_for_entry(
        _generated_preflight_entry(),
        manifest,
        runner="functional_data",
        device="mps",
    )

    assert reason == "cpu_contract_unsupported"
    assert "no selected dtype is executable" in detail
    assert extra["op"] == "aten::example"


def test_adaptive_isolation_loads_crash_signal_and_timeout_candidates(tmp_path):
    latest = tmp_path / "hw_latest.json"
    latest.write_text(
        json.dumps(
            _adaptive_result_payload(
                {
                    "torchcts/a.py::test_crash": {
                        "status": "ERROR",
                        "error_type": "ProcessCrash",
                        "subprocess": {"signal": "SIGSEGV"},
                    },
                    "torchcts/b.py::test_signal": {
                        "status": "ERROR",
                        "error_type": "SubprocessFailure",
                        "subprocess": {"signal": "SIGABRT"},
                    },
                    "torchcts/c.py::test_timeout": {
                        "status": "ERROR",
                        "error_type": "TimeoutError",
                        "phase": "subprocess",
                        "failure_stage": "subprocess_timeout",
                        "subprocess": {"timed_out": True},
                    },
                }
            )
        ),
        encoding="utf-8",
    )

    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )

    assert loaded.candidates["torchcts/a.py::test_crash"].isolation_source == "adaptive_previous_crash"
    assert loaded.candidates["torchcts/b.py::test_signal"].prior_signal == "SIGABRT"
    assert loaded.candidates["torchcts/c.py::test_timeout"].isolation_source == "adaptive_previous_timeout"


def test_harness_known_segfault_no_execute_record_preserves_crash_metadata(monkeypatch):
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 8)
    entry = {
        "name": "aten::_foreach_add.Tensor_out",
        "schema": (
            "aten::_foreach_add.Tensor_out(Tensor[] self, Tensor[] other, "
            "Scalar alpha=1, *, Tensor(a!)[] out) -> ()"
        ),
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "out_variant",
        "variant_kind": "out",
        "semantic_level": 4,
        "generated": {"strategy": {"strategy": "manual_foreach", "family": "binary"}},
    }
    item = _generated_item(
        "torchcts/generated/test_foreach_fused.py::"
        "test_generated_foreach_or_fused[_foreach_add.Tensor_out[L4]-torch.float16]",
        entry,
        dtype=torch.float16,
    )
    match = known_segfaults.annotate_match(
        _known_segfault_entry(
            id="mps-generated-foreach-add-tensor-out-float16-pytorch-2-12",
            match="coverage_id",
            nodeid=item.nodeid,
            dispatcher="aten::_foreach_add.Tensor_out",
            constraints={
                "suite": ["generated"],
                "coverage_kind": ["generated"],
                "surface_kind": ["out_variant"],
                "variant_kind": ["out"],
                "strategy": ["manual_foreach"],
                "strategy_family": ["binary"],
                "semantic_level": [4],
                "dtype": ["torch.float16"],
                "coverage_id_glob": ["aten::_foreach_add.Tensor_out"],
            },
        ),
        item.nodeid,
        metadata=harness._extract_result_metadata(item),
    )

    record = harness._known_segfault_no_execute_record(item, match)

    assert harness._known_segfault_record_without_execution(match)
    assert record["status"] == "ERROR"
    assert record["phase"] == "known_segfault"
    assert record["failure_stage"] == "known_backend_crash"
    assert record["error_type"] == "KnownSegfaultNotExecuted"
    assert record["isolation_source"] == "known_segfault"
    assert record["known_segfault_classification"] == "confirmed_backend_crash"
    assert record["dispatcher_name"] == "aten::_foreach_add.Tensor_out"
    assert record["dtype"] == "torch.float16"


def test_harness_child_command_disables_adaptive_isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "_DEVICE_NAME", "mps")
    monkeypatch.setattr(harness, "_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 8)
    options = {
        "--level-exact": None,
        "--level-range": None,
        "--dtype": [],
        "--path-shape-family": ["matmul"],
        "--path-shape-category": [],
        "--path-shape-case": [],
        "--path-shape-runner": [],
        "--path-shape-resource-tier": ["heavy"],
        "--path-shape-cost-class": ["large"],
        "--path-shape-model-role": [],
        "--path-shape-dtype-group": ["float"],
        "--memory-mode": "balanced",
        "--max-device-memory": None,
        "--max-tensor-size": None,
        "--validation": True,
    }
    config = SimpleNamespace(getoption=lambda name: options.get(name))
    item = SimpleNamespace(nodeid="torchcts/example.py::test_crash", config=config)

    cmd = harness._subprocess_child_command(item)

    assert cmd[cmd.index("--adaptive-isolation") + 1] == "off"
    assert cmd[cmd.index("--known-segfault-policy") + 1] == "off"
