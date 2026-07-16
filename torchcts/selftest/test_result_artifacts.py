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

from torchcts.core.result_artifacts import (
    ResultArtifactError,
    dumps_result_artifact,
    load_result_artifact,
    loads_result_artifact,
)
from torchcts.core.result_sanitization import (
    absolute_paths_in_text,
    sanitize_result_payload,
    structural_absolute_paths,
)


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


def test_compact_result_format_round_trips_reserved_keys_and_diagnostics():
    diagnostic = "failure at /Users/alice/project/backend.cpp:42"
    payload = {
        "metadata": {
            "hardware_key": "hw",
            "source_path": "/Users/alice/project/result.json",
        },
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
