# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

import pytest

from torchcts.core.report import build_report


pytestmark = pytest.mark.covers_category("selftest")


def test_report_suppresses_manifest_scorecard_for_incomplete_full_run():
    nodeid = "torchcts/operators/test_opinfo.py::test_op[add]"
    payload = {
        "metadata": {
            "device_name": "mps",
            "hardware_key": "test-hardware",
            "pytorch_version": "2.12.1",
            "timestamp": "2026-07-09T12:00:00Z",
            "elapsed_sec": 1.0,
            "collect_only": False,
            "semantic_level": 8,
            "semantic_level_selection": {
                "mode": "cumulative",
                "min_level": 1,
                "max_level": 8,
                "label": "requested <= 8",
            },
            "skip_count": 0,
            "session_completed": False,
            "backend_manifest_assertion": {
                "schema_version": 1,
                "basis": "collection_selected_tests_plus_backend_manifest_declines",
                "asserted_test_count": 1,
                "declined_test_count": 0,
                "total_runnable_test_count": 1,
                "asserted_fraction": 1.0,
                "asserted_percent": 100.0,
                "progress_bar_width": 37,
                "declined_skip_reasons": {},
            },
        },
        "results": {
            nodeid: {
                "status": "PASS",
                "phase": "call",
                "suite": "opinfo",
                "test_kind": "opinfo",
                "capability": "inference",
                "is_plumbing": False,
                "op": "add",
                "dtype": "torch.float32",
                "semantic_level": 1,
                "requested_level": 8,
                "semantic_level_selection": {
                    "mode": "cumulative",
                    "min_level": 1,
                    "max_level": 8,
                    "label": "requested <= 8",
                },
                "error_message": "",
                "duration_ms": 1.0,
                "nodeid": nodeid,
            }
        },
        "skips": {},
    }

    scorecard, markdown = build_report(payload)

    assert "Backend manifest support scorecard unavailable for this run." in scorecard
    assert "This run did not complete" in scorecard
    assert "No backend support percentage is shown." in scorecard
    assert "Backend asserts via manifest.py" not in scorecard
    assert "Backend manifest support scorecard unavailable for this run." in markdown
