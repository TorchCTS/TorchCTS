# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

from __future__ import annotations

import json
from pathlib import Path

import pytest

import torchcts.conftest as harness
from torchcts.core import coverage
from torchcts.core import runtime_evidence
from torchcts.core.result_artifacts import (
    ResultArtifactError,
    dumps_result_artifact,
    load_result_artifact,
    loads_result_artifact,
    write_result_artifact,
    write_result_reference,
)
from torchcts.core.result_sanitization import (
    absolute_paths_in_text,
    sanitize_result_payload,
    sanitize_text,
    structural_absolute_paths,
)
from torchcts.core.run_log import RollingRunLog
from torchcts.core.triage import freeze_input_snapshot, parse_runlog_crash_candidates


pytestmark = pytest.mark.covers_category("selftest")


def test_result_sanitization_preserves_diagnostics_and_normalizes_commands():
    payload = {
        "command_args": ["/home/alice/torchcts/.venv/bin/python", "-m", "pytest"],
        "error_message": (
            '  File "/home/alice/torchcts/torchcts/core/report.py", line 10\n'
            "registered at /Users/runner/work/pytorch/pytorch/aten/src/ATen/foo.cpp:42\n"
            "installed at C:\\Users\\alice\\env\\Lib\\site-packages\\torch\\foo.py:7\n"
            "input file:///home/alice/private/input.json"
        ),
        "documentation": "https://docs.nvidia.com/cuda/cuda-runtime-api/index.html",
    }

    sanitized = sanitize_result_payload(payload)
    assert sanitized["command_args"][0] == "python"
    assert sanitized["error_message"] == payload["error_message"]
    assert sanitized["documentation"] == payload["documentation"]
    assert absolute_paths_in_text(sanitized["error_message"])
    assert structural_absolute_paths(sanitized) == []


def test_atomic_result_writer_sanitizes_recursively(tmp_path: Path):
    destination = tmp_path / "latest.json"
    harness._atomic_result_dump(
        destination,
        {
            "metadata": {"source_path": "/home/alice/project/results/input.json"},
            "results": {
                "/home/alice/project/test_backend.py::test_example": {
                    "error_message": 'File "/home/alice/project/test_backend.py", line 4',
                }
            },
        },
    )

    payload = load_result_artifact(destination)
    assert payload["metadata"]["source_path"] == "external/input.json"
    assert payload["results"]["external/test_backend.py::test_example"]["error_message"] == (
        'File "/home/alice/project/test_backend.py", line 4'
    )
    assert structural_absolute_paths(payload) == []


def test_runtime_probe_evidence_sanitizes_command_and_traceback(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TORCHCTS_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("TORCHCTS_HARDWARE_KEY", "test-hardware")

    class ProbeFailure:
        def to_dict(self):
            return {
                "error_type": "RuntimeError",
                "error_message": 'File "/home/alice/project/probe.py", line 2',
                "stderr_tail": "/home/alice/project/.venv/lib/python3.12/site-packages/torch/foo.py:3",
                "command_args": ["/home/alice/project/.venv/bin/python", "-c", "probe"],
            }

    record = runtime_evidence.record_harness_probe_failure(
        "capability",
        "example",
        ProbeFailure(),
        stage="declared_capability_probe",
    )

    assert record is not None
    assert record["command_args"][0] == "python"
    assert record["error_message"] == 'File "/home/alice/project/probe.py", line 2'
    assert record["stderr_tail"] == (
        "/home/alice/project/.venv/lib/python3.12/site-packages/torch/foo.py:3"
    )
    artifact = next(tmp_path.glob("*_harness_probe_failures_*.jsonl"))
    artifact_record = json.loads(artifact.read_text(encoding="utf-8"))
    assert structural_absolute_paths(artifact_record) == []


def test_coverage_result_writer_sanitizes_payload(tmp_path: Path):
    destination = tmp_path / "audit.json"

    coverage._write_result_json(destination, {"_source_path": "/home/alice/project/test_ops.py"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "_source_path": "external/test_ops.py"
    }


def test_triage_snapshot_sanitizes_legacy_result_and_report(tmp_path: Path):
    result_path = tmp_path / "hw_latest.json"
    result_path.write_text(
        json.dumps(
            {
                "metadata": {"hardware_key": "hw"},
                "results": {"case": {"error_message": "/home/alice/project/test_ops.py:4"}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "hw_report.md").write_text(
        "Failure at /home/alice/project/test_ops.py:4\n",
        encoding="utf-8",
    )

    copied = freeze_input_snapshot(result_path, json.loads(result_path.read_text()), tmp_path / "triage")

    assert len(copied) == 2
    copied_result = load_result_artifact(copied[0])
    assert copied_result["results"]["case"]["error_message"] == (
        "/home/alice/project/test_ops.py:4"
    )
    assert structural_absolute_paths(copied_result) == []
    assert Path(copied[1]).read_text(encoding="utf-8") == (
        "Failure at /home/alice/project/test_ops.py:4\n"
    )


def test_rolling_run_log_keeps_only_recent_sanitized_context(tmp_path: Path):
    path = tmp_path / "runlog.txt"
    runlog = RollingRunLog(path, window_size=4)
    runlog.open()
    for index in range(20):
        runlog.record_start(index + 0.5, f"/home/alice/project/test_ops.py::test_case[{index}]")
    runlog.close()

    text = path.read_text(encoding="utf-8")
    node_lines = [line for line in text.splitlines() if not line.startswith("#")]
    assert len(node_lines) == 4
    assert "tests started: 20; retained: 4" in text
    assert "test_case[16]" in text
    assert "test_case[19]" in text
    assert not absolute_paths_in_text(text)
    assert parse_runlog_crash_candidates(path, limit=2) == [
        "external/test_ops.py::test_case[19]",
        "external/test_ops.py::test_case[18]",
    ]


def test_sanitize_text_leaves_urls_and_relative_paths_unchanged():
    value = r"torchcts/core/report.py, \\S+, and https://example.com/a/b?q=/relative"
    assert sanitize_text(value) == value


def test_sanitize_text_recognizes_unc_paths_without_treating_regexes_as_paths():
    value = r"from \\build-server\artifacts\run\results.json; pattern \\S+"
    sanitized = sanitize_text(value)

    assert "external/results.json" in sanitized
    assert r"\\S+" in sanitized
    assert not absolute_paths_in_text(sanitized)


def test_compact_result_format_round_trips_reserved_keys_and_diagnostics():
    diagnostic = "failure at /Users/alice/project/backend.cpp:42"
    payload = {
        "metadata": {"hardware_key": "hw", "source_path": "/Users/alice/project/result.json"},
        "results": {
            "torchcts/test_ops.py::test_one": {
                "status": "FAIL",
                "error_message": diagnostic,
                "0": "literal token-like key",
                "$": "literal reference-like key",
                "~key": "literal escaped key",
            },
            "torchcts/test_ops.py::test_two": {
                "status": "FAIL",
                "error_message": diagnostic,
            },
        },
    }

    text = dumps_result_artifact(payload)
    restored = loads_result_artifact(text)

    assert restored["metadata"]["source_path"] == "external/result.json"
    assert restored["results"]["torchcts/test_ops.py::test_one"]["error_message"] == diagnostic
    assert restored["results"]["torchcts/test_ops.py::test_one"]["0"] == "literal token-like key"
    assert restored["results"]["torchcts/test_ops.py::test_one"]["$"] == "literal reference-like key"
    assert restored["results"]["torchcts/test_ops.py::test_one"]["~key"] == "literal escaped key"


def test_compact_result_format_reduces_repeated_schema_and_values():
    payload = {
        "metadata": {"hardware_key": "hw"},
        "results": {
            f"torchcts/test_ops.py::test_case[{index}]": {
                "status": "FAIL",
                "semantic_level": 4,
                "coverage_status": "covered_generated",
                "error_message": "the same long diagnostic remains exact",
            }
            for index in range(100)
        },
    }

    text = dumps_result_artifact(payload)
    plain = json.dumps(payload, separators=(",", ":")) + "\n"

    assert loads_result_artifact(text) == payload
    assert len(text) < len(plain) * 0.6


def test_partial_result_format_avoids_table_rebuild_and_remains_readable():
    payload = {
        "metadata": {"session_completed": False},
        "results": {
            "torchcts/test_ops.py::test_case": {
                "status": "ERROR",
                "error_message": "failure at /Users/alice/project/backend.cpp:42",
            }
        },
    }

    text = dumps_result_artifact(payload, optimize_tables=False)

    assert json.loads(text).get("format") is None
    assert loads_result_artifact(text) == payload


def test_compact_result_serialization_is_deterministic_across_mapping_order():
    first = {
        "metadata": {"hardware_key": "hw", "pytorch_version": "2.12.1"},
        "results": {
            "test_b": {"status": "PASS", "suite": "operators"},
            "test_a": {"status": "PASS", "suite": "operators"},
        },
    }
    second = {
        "results": {
            "test_a": {"suite": "operators", "status": "PASS"},
            "test_b": {"suite": "operators", "status": "PASS"},
        },
        "metadata": {"pytorch_version": "2.12.1", "hardware_key": "hw"},
    }

    assert dumps_result_artifact(first) == dumps_result_artifact(second)


def test_result_reference_resolves_one_canonical_artifact(tmp_path: Path):
    history = tmp_path / "hw_history" / "run.json"
    latest = tmp_path / "hw_latest.json"
    payload = {"metadata": {"hardware_key": "hw"}, "results": {}}
    write_result_artifact(history, payload)
    write_result_reference(latest, history)

    assert load_result_artifact(latest) == payload
    reference = json.loads(latest.read_text(encoding="utf-8"))
    assert reference["artifact"] == "hw_history/run.json"
    assert latest.stat().st_size < 160


def test_result_reference_rejects_parent_directory_escape(tmp_path: Path):
    latest = tmp_path / "hw_latest.json"
    latest.write_text(
        json.dumps(
            {
                "format": "torchcts_result_reference",
                "format_version": 1,
                "artifact": "../outside.json",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ResultArtifactError, match="escapes"):
        load_result_artifact(latest)
