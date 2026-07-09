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

import json
import os

import pytest

from torchcts.core.report import build_report, generate_report_cli


pytestmark = pytest.mark.covers_category("selftest")


def _result(nodeid, *, status="PASS", op="add", dtype="torch.float32"):
    return {
        "status": status,
        "phase": "call",
        "suite": "opinfo",
        "test_kind": "opinfo",
        "capability": "inference",
        "is_plumbing": False,
        "op": op,
        "dtype": dtype,
        "semantic_level": 1,
        "requested_level": 8,
        "semantic_level_selection": {
            "mode": "cumulative",
            "min_level": 1,
            "max_level": 8,
            "label": "requested <= 8",
        },
        "error_message": "" if status == "PASS" else "backend mismatch",
        "duration_ms": 1.0,
        "nodeid": nodeid,
    }


def _manifest_decline(nodeid, *, op="unsupported", dtype="torch.float64"):
    return {
        "status": "SKIP",
        "phase": "collection",
        "suite": "opinfo",
        "test_kind": "opinfo",
        "capability": "inference",
        "op": op,
        "dtype": dtype,
        "semantic_level": 1,
        "requested_level": 8,
        "semantic_level_selection": {
            "mode": "cumulative",
            "min_level": 1,
            "max_level": 8,
            "label": "requested <= 8",
        },
        "skip_reason": "dtype_not_supported",
        "detail": "dtype not declared by manifest",
    }


def _payload(
    *,
    session_completed=True,
    total_runnable=19000,
    result_status="PASS",
    timestamp="2026-07-09T12:00:00Z",
):
    results = {
        "torchcts/operators/test_opinfo.py::test_op[add]": _result(
            "torchcts/operators/test_opinfo.py::test_op[add]",
            status=result_status,
        )
    }
    skips = {
        f"torchcts/operators/test_opinfo.py::test_op[declined_{i}]": _manifest_decline(
            f"torchcts/operators/test_opinfo.py::test_op[declined_{i}]",
            op=f"declined_{i}",
        )
        for i in range(max(total_runnable - len(results), 0))
    }
    asserted = len(results)
    declined = len(skips)
    total = asserted + declined
    return {
        "metadata": {
            "device_name": "mps",
            "hardware_key": "Apple_Test_64gb",
            "pytorch_version": "2.12.1",
            "timestamp": timestamp,
            "elapsed_sec": 1.0,
            "collect_only": False,
            "semantic_level": 8,
            "semantic_level_selection": {
                "mode": "cumulative",
                "min_level": 1,
                "max_level": 8,
                "label": "requested <= 8",
            },
            "skip_count": len(skips),
            "session_completed": session_completed,
            "backend_manifest_assertion": {
                "schema_version": 1,
                "basis": "collection_selected_tests_plus_backend_manifest_declines",
                "asserted_test_count": asserted,
                "declined_test_count": declined,
                "total_runnable_test_count": total,
                "asserted_fraction": asserted / total if total else 0.0,
                "asserted_percent": (asserted / total * 100.0) if total else 0.0,
                "progress_bar_width": 37,
                "declined_skip_reasons": {"dtype_not_supported": declined},
            },
        },
        "results": results,
        "skips": skips,
    }


def test_report_suppresses_manifest_scorecard_for_incomplete_full_run():
    scorecard, markdown = build_report(_payload(session_completed=False, total_runnable=19000))

    assert "Backend manifest support scorecard unavailable for this run." in scorecard
    assert "This run did not complete" in scorecard
    assert "No backend support percentage is shown." in scorecard
    assert "Backend asserts via manifest.py" not in scorecard
    assert "Backend manifest support scorecard unavailable for this run." in markdown


def test_report_from_file_writes_next_to_source_json(tmp_path, monkeypatch):
    source_dir = tmp_path / "custom-results"
    source_dir.mkdir()
    source_json = source_dir / "Apple_Test_64gb_latest.json"
    source_json.write_text(json.dumps(_payload(total_runnable=2)), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert generate_report_cli(str(source_json)) == 0

    report_path = source_dir / "Apple_Test_64gb_report.md"
    assert report_path.exists()
    assert not (tmp_path / "results" / "Apple_Test_64gb_report.md").exists()


def test_report_from_file_skips_current_history_copy_for_baseline(tmp_path, monkeypatch):
    source_dir = tmp_path / "custom-results"
    history_dir = source_dir / "Apple_Test_64gb_history"
    history_dir.mkdir(parents=True)
    source_json = source_dir / "Apple_Test_64gb_latest.json"
    current = _payload(
        total_runnable=2,
        result_status="FAIL",
        timestamp="2026-07-09T12:00:00Z",
    )
    source_json.write_text(json.dumps(current), encoding="utf-8")
    previous_history = history_dir / "previous.json"
    previous_history.write_text(
        json.dumps(
            _payload(
                total_runnable=2,
                result_status="PASS",
                timestamp="2026-07-08T12:00:00Z",
            )
        ),
        encoding="utf-8",
    )
    current_history = history_dir / "current-copy.json"
    current_history.write_text(json.dumps(current), encoding="utf-8")
    os.utime(previous_history, (1, 1))
    os.utime(current_history, (2, 2))
    monkeypatch.chdir(tmp_path)

    assert generate_report_cli(str(source_json)) == 0

    report_text = (source_dir / "Apple_Test_64gb_report.md").read_text(encoding="utf-8")
    assert "REGRESSIONS SINCE LAST RUN (2026-07-08T12:00:00Z)" in report_text
    assert "1 new failures" in report_text
