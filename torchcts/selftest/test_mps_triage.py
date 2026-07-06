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
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import torchcts.conftest as harness
from torchcts.core import adaptive_isolation
from torchcts.core import known_segfaults
from torchcts.core import triage
from torchcts.core.device import stream_context
from torchcts.core.report import get_hardware_key
from torchcts.generated import coverage_helpers


pytestmark = pytest.mark.covers_category("selftest")


def _adaptive_result_payload(results, *, device="mps", hardware="hw", version="2.12.1", completed=False):
    return {
        "metadata": {
            "device_name": device,
            "hardware_key": hardware,
            "pytorch_version": version,
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


def _generated_item(nodeid, entry, *, fspath="torchcts/generated/test_out_variants.py", dtype=None):
    params = {"entry": entry}
    if dtype is not None:
        params["dtype"] = dtype
    return SimpleNamespace(
        nodeid=nodeid,
        fspath=fspath,
        name=nodeid.rsplit("::", 1)[-1],
        callspec=SimpleNamespace(params=params),
        iter_markers=lambda name=None: iter(()),
    )


def _path_shape_item(nodeid, path_shape_case):
    return SimpleNamespace(
        nodeid=nodeid,
        fspath=nodeid.split("::", 1)[0],
        name=nodeid.rsplit("::", 1)[-1],
        callspec=SimpleNamespace(params={"path_shape_case": path_shape_case}),
        iter_markers=lambda name=None: iter(()),
    )


def _reflection_pad3d_out_entry():
    return {
        "name": "aten::reflection_pad3d.out",
        "schema": "aten::reflection_pad3d.out(Tensor self, SymInt[6] padding, *, Tensor(a!) out) -> Tensor(a!)",
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "out_variant",
        "variant_kind": "out",
        "semantic_level": 3,
        "generated": {
            "strategy": {
                "strategy": "manual_padding",
                "family": "reflection_pad3d",
            }
        },
    }


def _unfold_view_alias_entry():
    return {
        "name": "aten::unfold",
        "schema": "aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)",
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "view_or_alias",
        "variant_kind": "view",
        "semantic_level": 3,
        "generated": {
            "strategy": {
                "strategy": "opinfo_view_alias",
                "opinfo_name": "unfold",
            }
        },
    }


def _hamming_window_periodic_factory_entry():
    return {
        "name": "aten::hamming_window.periodic",
        "schema": (
            "aten::hamming_window.periodic(int window_length, bool periodic, *, "
            "ScalarType? dtype=None, Layout? layout=None, Device? device=None, "
            "bool? pin_memory=None) -> Tensor"
        ),
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "factory",
        "variant_kind": "factory",
        "semantic_level": 2,
        "generated": {
            "strategy": {
                "strategy": "manual_factory",
                "family": "window",
            }
        },
    }


def _col2im_functional_entry():
    return {
        "name": "aten::col2im",
        "schema": (
            "aten::col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, "
            "int[2] dilation, int[2] padding, int[2] stride) -> Tensor"
        ),
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "semantic_level": 4,
        "generated": {
            "strategy": {
                "strategy": "manual_convolution",
                "family": "col2im",
            }
        },
    }


def _foreach_add_list_functional_entry():
    return {
        "name": "aten::_foreach_add.List",
        "schema": "aten::_foreach_add.List(Tensor[] self, Tensor[] other, *, Scalar alpha=1) -> Tensor[]",
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "semantic_level": 4,
        "generated": {
            "strategy": {
                "strategy": "manual_foreach",
                "family": "binary",
                "foreach_name": "add",
                "overload": "List",
            }
        },
    }


def _foreach_add_scalarlist_functional_entry():
    return {
        "name": "aten::_foreach_add.ScalarList",
        "schema": "aten::_foreach_add.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]",
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "semantic_level": 4,
        "generated": {
            "strategy": {
                "strategy": "manual_foreach",
                "family": "binary",
                "foreach_name": "add",
                "overload": "ScalarList",
            }
        },
    }


def _foreach_addcdiv_scalar_functional_entry():
    return {
        "name": "aten::_foreach_addcdiv.Scalar",
        "schema": (
            "aten::_foreach_addcdiv.Scalar(Tensor[] self, Tensor[] tensor1, "
            "Tensor[] tensor2, Scalar value=1) -> Tensor[]"
        ),
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "semantic_level": 4,
        "generated": {
            "strategy": {
                "strategy": "manual_foreach",
                "family": "ternary",
                "foreach_name": "addcdiv",
                "overload": "Scalar",
            }
        },
    }


def _native_batch_norm_backward_entry():
    return {
        "name": "aten::native_batch_norm_backward",
        "schema": (
            "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, "
            "Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, "
            "bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"
        ),
        "status": "covered_generated",
        "coverage_kind": "generated",
        "surface_kind": "autograd_backward",
        "variant_kind": "functional",
        "semantic_level": 3,
        "generated": {
            "strategy": {
                "strategy": "manual_multi_output_reduction",
                "family": "native_batch_norm_backward",
            }
        },
    }


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


def test_adaptive_isolation_ignores_mismatched_and_malformed_artifacts(tmp_path):
    (tmp_path / "hw_latest.json").write_text("{not json", encoding="utf-8")
    history = tmp_path / "hw_history"
    history.mkdir()
    (history / "2026-06-27T00-00-00Z.json").write_text(
        json.dumps(
            _adaptive_result_payload(
                {
                    "torchcts/a.py::test_crash": {
                        "status": "ERROR",
                        "error_type": "ProcessCrash",
                    }
                },
                device="cpu",
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

    assert not loaded.candidates
    assert loaded.warnings
    assert loaded.rejected[0]["reason"] == "metadata_mismatch"


def test_adaptive_isolation_requires_matching_latest_for_runlog_hang(tmp_path):
    (tmp_path / "hw_latest.json").write_text(
        json.dumps(_adaptive_result_payload({}, device="cpu", completed=False)),
        encoding="utf-8",
    )
    history = tmp_path / "hw_history"
    history.mkdir()
    (history / "2026-06-27T00-00-00Z.json").write_text(
        json.dumps(_adaptive_result_payload({}, completed=False)),
        encoding="utf-8",
    )
    (tmp_path / "hw_runlog.txt").write_text(
        "     1.0s  torchcts/hangs.py::test_hang\n",
        encoding="utf-8",
    )

    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )

    assert not loaded.candidates


def test_adaptive_isolation_scans_only_five_history_artifacts(tmp_path):
    history = tmp_path / "hw_history"
    history.mkdir()
    for index in range(6):
        (history / f"2026-06-27T00-00-0{index}Z.json").write_text(
            json.dumps(_adaptive_result_payload({}, completed=True)),
            encoding="utf-8",
        )

    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )

    assert len(loaded.artifacts_considered) == 5


def test_adaptive_isolation_uses_runlog_for_high_confidence_hang(tmp_path):
    (tmp_path / "hw_latest.json").write_text(
        json.dumps(
            _adaptive_result_payload(
                {
                    "torchcts/a.py::test_done": {"status": "PASS"},
                },
                completed=False,
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "hw_runlog.txt").write_text(
        "     1.0s  torchcts/a.py::test_done\n"
        "     2.0s  torchcts/hangs.py::test_hang\n",
        encoding="utf-8",
    )

    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )

    candidate = loaded.candidates["torchcts/hangs.py::test_hang"]
    assert candidate.isolation_source == "adaptive_suspected_hang"


def test_adaptive_isolation_does_not_infer_hang_from_result_or_completed_session(tmp_path):
    runlog = tmp_path / "hw_runlog.txt"
    runlog.write_text("     1.0s  torchcts/a.py::test_done\n", encoding="utf-8")
    latest = tmp_path / "hw_latest.json"
    latest.write_text(
        json.dumps(
            _adaptive_result_payload(
                {"torchcts/a.py::test_done": {"status": "PASS"}},
                completed=False,
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
    assert not loaded.candidates

    latest.write_text(
        json.dumps(_adaptive_result_payload({}, completed=True)),
        encoding="utf-8",
    )
    loaded = adaptive_isolation.load_adaptive_isolation(
        tmp_path,
        hardware_key="hw",
        device_name="mps",
        torch_version="2.12.1",
    )
    assert not loaded.candidates
    assert any(item["reason"] == "latest_json_reports_completed_session" for item in loaded.rejected)


def test_adaptive_isolation_newer_pass_resolves_older_crash(tmp_path):
    (tmp_path / "hw_latest.json").write_text(
        json.dumps(
            _adaptive_result_payload(
                {"torchcts/a.py::test_flaky": {"status": "PASS"}},
                completed=True,
            )
        ),
        encoding="utf-8",
    )
    history = tmp_path / "hw_history"
    history.mkdir()
    (history / "2026-06-27T00-00-00Z.json").write_text(
        json.dumps(
            _adaptive_result_payload(
                {
                    "torchcts/a.py::test_flaky": {
                        "status": "ERROR",
                        "error_type": "ProcessCrash",
                    }
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

    assert not loaded.candidates
    assert loaded.rejected[0]["reason"] == "newer_nonisolating_record_resolved_candidate"


def test_adaptive_isolation_filters_stale_nodeids_from_collection(tmp_path):
    loaded = adaptive_isolation.AdaptiveIsolationLoadResult(
        candidates={
            "torchcts/a.py::test_keep": adaptive_isolation.AdaptiveIsolationCandidate(
                nodeid="torchcts/a.py::test_keep",
                canonical_nodeid="torchcts/a.py::test_keep",
                isolation_source="adaptive_previous_crash",
                reason="crashed",
                evidence_path="results/hw_latest.json",
            ),
            "torchcts/b.py::test_drop": adaptive_isolation.AdaptiveIsolationCandidate(
                nodeid="torchcts/b.py::test_drop",
                canonical_nodeid="torchcts/b.py::test_drop",
                isolation_source="adaptive_previous_crash",
                reason="crashed",
                evidence_path="results/hw_latest.json",
            ),
        },
        rejected=[],
        warnings=[],
        artifacts_considered=[],
    )

    accepted, rejected = adaptive_isolation.filter_candidates_for_collection(
        loaded,
        ["torchcts/a.py::test_keep"],
    )

    assert sorted(accepted) == ["torchcts/a.py::test_keep"]
    assert rejected == [
        {
            "nodeid": "torchcts/b.py::test_drop",
            "canonical_nodeid": "torchcts/b.py::test_drop",
            "path": "results/hw_latest.json",
            "reason": "not_collected_in_current_run",
        }
    ]



def test_known_segfault_schema_accepts_and_matches_entry():
    payload = {
        "version": 1,
        "known_segfaults": [_known_segfault_entry()],
    }

    entries = known_segfaults.validate_known_segfaults([payload])
    active = known_segfaults.active_known_segfaults(
        entries,
        backend="mps",
        torch_version="2.12.1+cpu",
        hardware_key="Apple_M3_Max_128gb",
    )
    item = SimpleNamespace(nodeid="torchcts/example.py::test_crash")

    assert known_segfaults.match_known_segfault(item, active)["id"] == "mps-example"


def test_known_segfault_matching_canonicalizes_installed_package_nodeids():
    payload = {
        "version": 1,
        "known_segfaults": [
            _known_segfault_entry(
                id="mps-installed-nodeid",
                nodeid="torchcts/generated/test_out_variants.py::test_generated_out_variant[range.out_[L2]]",
                dispatcher="aten::range.out_",
                repro={"script": "repro.py", "case": "range_out_"},
            )
        ],
    }

    entries = known_segfaults.validate_known_segfaults([payload])
    active = known_segfaults.active_known_segfaults(
        entries,
        backend="mps",
        torch_version="2.12.1",
        hardware_key="Apple_M3_Max_128gb",
    )
    item = SimpleNamespace(
        nodeid=(
            ".venv/lib/python3.14/site-packages/torchcts/generated/"
            "test_out_variants.py::test_generated_out_variant[range.out_[L2]]"
        )
    )

    assert known_segfaults.canonicalize_nodeid(item.nodeid) == payload["known_segfaults"][0]["nodeid"]
    assert known_segfaults.match_known_segfault(item, active)["id"] == "mps-installed-nodeid"


def test_known_segfault_schema_accepts_dispatcher_and_coverage_id_entries():
    payload = {
        "version": 1,
        "known_segfaults": [
            _known_segfault_entry(
                id="mps-dispatcher",
                match="dispatcher",
                nodeid=None,
                dispatcher="aten::dispatcher.default",
                evidence_scope="dispatcher_surface",
            ),
            _known_segfault_entry(
                id="mps-coverage",
                match="coverage_id",
                nodeid=None,
                dispatcher="aten::coverage.default",
                coverage_id="aten::coverage.default",
                evidence_scope="constrained_metadata",
                constraints={"suite": ["generated"]},
            ),
        ],
    }

    entries = known_segfaults.validate_known_segfaults([payload])

    assert [entry["match"] for entry in entries] == ["dispatcher", "coverage_id"]


def test_known_segfault_schema_rejects_nonsensical_metadata_rules():
    bad_dispatcher = _known_segfault_entry(
        match="dispatcher",
        nodeid=None,
        evidence_scope="constrained_metadata",
    )
    bad_scope = _known_segfault_entry(
        match="dispatcher",
        nodeid=None,
        evidence_scope="exact_node",
    )

    with pytest.raises(known_segfaults.KnownSegfaultError, match="non-empty constraints"):
        known_segfaults.validate_known_segfaults([
            {"version": 1, "known_segfaults": [bad_dispatcher]}
        ])
    with pytest.raises(known_segfaults.KnownSegfaultError, match="requires match=nodeid"):
        known_segfaults.validate_known_segfaults([
            {"version": 1, "known_segfaults": [bad_scope]}
        ])


def test_known_segfault_schema_rejects_bad_constraints():
    cases = [
        (_known_segfault_entry(constraints={"nope": ["generated"]}), "unknown"),
        (_known_segfault_entry(constraints={"suite": []}), "must not be empty"),
        (_known_segfault_entry(constraints={"semantic_level": [9]}), "semantic levels"),
        (_known_segfault_entry(constraints={"dtype": []}), "must not be empty"),
        (_known_segfault_entry(constraints={"dtype": [1]}), "non-empty strings"),
    ]

    for entry, pattern in cases:
        with pytest.raises(known_segfaults.KnownSegfaultError, match=pattern):
            known_segfaults.validate_known_segfaults([
                {"version": 1, "known_segfaults": [entry]}
            ])


def test_known_segfault_dtype_constraint_uses_metadata():
    entry = known_segfaults.validate_known_segfaults([
        {
            "version": 1,
            "known_segfaults": [
                _known_segfault_entry(
                    match="dispatcher",
                    nodeid=None,
                    dispatcher="aten::example.default",
                    evidence_scope="constrained_metadata",
                    constraints={
                        "suite": ["generated"],
                        "dtype": ["torch.float64"],
                    },
                )
            ],
        }
    ])[0]
    nodeid = "torchcts/generated/test_foreach_fused.py::test_generated_foreach_or_fused[example-torch.float64]"

    assert known_segfaults.entry_matches(
        entry,
        nodeid,
        {"dispatcher_name": "aten::example.default", "suite": "generated", "dtype": "torch.float64"},
    )
    assert not known_segfaults.entry_matches(
        entry,
        nodeid,
        {"dispatcher_name": "aten::example.default", "suite": "generated", "dtype": "torch.float32"},
    )


def test_known_segfault_nodeid_wins_over_dispatcher():
    exact = _known_segfault_entry(id="exact")
    dispatcher = _known_segfault_entry(
        id="dispatcher",
        match="dispatcher",
        nodeid=None,
        dispatcher="aten::example.default",
        evidence_scope="dispatcher_surface",
    )
    entries = known_segfaults.validate_known_segfaults([
        {"version": 1, "known_segfaults": [dispatcher, exact]}
    ])

    match = known_segfaults.match_known_segfault(
        SimpleNamespace(
            nodeid="torchcts/example.py::test_crash",
            metadata={"dispatcher_name": "aten::example.default"},
        ),
        entries,
    )

    assert match["id"] == "exact"


def test_known_segfault_schema_rejects_duplicate_ids():
    entry = _known_segfault_entry(id="dup")

    with pytest.raises(known_segfaults.KnownSegfaultError, match="duplicate"):
        known_segfaults.validate_known_segfaults([
            {"version": 1, "known_segfaults": [entry, dict(entry)]}
        ])


def test_harness_known_segfault_fields_preserve_failure_semantics():
    match = {
        "id": "mps-example",
        "match": "dispatcher",
        "matched_by": "dispatcher",
        "dispatcher": "aten::example.default",
        "classification": "confirmed_backend_crash",
        "evidence_scope": "dispatcher_surface",
        "constraints": {"suite": ["generated"]},
        "matched_nodeid": "torchcts/example.py::test_crash",
        "matched_metadata": {"dispatcher_name": "aten::example.default", "suite": "generated"},
        "reason": "standalone repro crashes",
        "expected_signal": "SIGSEGV",
        "repro": {"script": "repro.py", "case": "case0"},
    }

    fields = harness._known_segfault_result_fields(match, actual_signal="SIGABRT")

    assert fields["known_segfault_id"] == "mps-example"
    assert fields["known_segfault_classification"] == "confirmed_backend_crash"
    assert fields["known_segfault_expected_signal"] == "SIGSEGV"
    assert fields["known_segfault_unexpected_signal"] == "SIGABRT"
    assert fields["known_segfault_match"] == "dispatcher"
    assert fields["known_segfault_evidence_scope"] == "dispatcher_surface"
    assert fields["known_segfault_constraints"] == {"suite": ["generated"]}
    assert fields["known_segfault_matched_nodeid"] == "torchcts/example.py::test_crash"
    assert fields["known_segfault_matched_metadata"]["dispatcher_name"] == "aten::example.default"
    assert "status" not in fields


def test_harness_known_segfault_process_classification_is_backend_generic():
    crash = _known_segfault_entry()
    wrong_value = _known_segfault_entry(
        dispatcher="aten::_grid_sampler_2d_cpu_fallback_backward",
    )

    assert harness._known_segfault_process_classification(crash) == "confirmed_backend_crash"
    assert (
        harness._known_segfault_process_classification(
            wrong_value,
            stdout="Tensor-likes are not close! Mismatched elements: 4 / 4",
        )
        == "confirmed_backend_wrong_value"
    )


def test_harness_known_segfault_audit_prints_rule_counts(capsys):
    entry = _known_segfault_entry(
        match="dispatcher",
        nodeid=None,
        dispatcher="aten::reflection_pad3d.out",
        evidence_scope="constrained_metadata",
        constraints={"suite": ["generated"], "strategy_family": ["reflection_pad3d"]},
    )
    item = _generated_item(
        "torchcts/generated/test_out_variants.py::test_generated_out_variant[reflection_pad3d.out[L3]]",
        _reflection_pad3d_out_entry(),
    )
    descriptor = harness._known_segfault_descriptor_for_item(item)
    config = SimpleNamespace(option=SimpleNamespace(verbose=1))

    harness._print_known_segfault_audit(config, [entry], [descriptor])

    out = capsys.readouterr().out
    assert "Known segfault audit: 1 active rule(s)" in out
    assert "matched=1" in out
    assert "reflection_pad3d.out" in out


def test_harness_adaptive_match_canonicalizes_nodeids(monkeypatch):
    candidate = {
        "nodeid": "torchcts/generated/test_out_variants.py::test_generated_out_variant[range.out_[L2]]",
        "canonical_nodeid": "torchcts/generated/test_out_variants.py::test_generated_out_variant[range.out_[L2]]",
        "isolation_source": "adaptive_previous_crash",
        "reason": "previous run recorded a process crash",
        "evidence_path": "results/hw_latest.json",
        "prior_status": "ERROR",
        "prior_signal": "SIGSEGV",
    }
    monkeypatch.setattr(
        harness,
        "_ADAPTIVE_ISOLATION_ACTIVE",
        {candidate["canonical_nodeid"]: candidate},
    )
    item = SimpleNamespace(
        nodeid=(
            ".venv/lib/python3.14/site-packages/torchcts/generated/"
            "test_out_variants.py::test_generated_out_variant[range.out_[L2]]"
        )
    )

    assert harness._adaptive_isolation_match_for_item(item) is candidate


def test_harness_adaptive_fields_preserve_known_segfault_precedence():
    adaptive = {
        "isolation_source": "adaptive_previous_crash",
        "reason": "previous run recorded a process crash (SIGSEGV)",
        "evidence_path": "results/hw_latest.json",
        "prior_status": "ERROR",
        "prior_signal": "SIGSEGV",
        "prior_error_type": "ProcessCrash",
        "prior_timestamp": "2026-06-28T00:00:00Z",
    }

    fields = harness._adaptive_isolation_result_fields(
        adaptive,
        known_segfault_match={"id": "known"},
        resolved=True,
    )

    assert fields["isolation_source"] == "known_segfault"
    assert fields["adaptive_isolation_source"] == "adaptive_previous_crash"
    assert fields["adaptive_isolation_prior_signal"] == "SIGSEGV"
    assert fields["adaptive_isolation_resolved"] is True
    assert "status" not in fields


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
    assert cmd[cmd.index("--path-shape-family") + 1] == "matmul"
    assert cmd[cmd.index("--path-shape-resource-tier") + 1] == "heavy"
    assert cmd[cmd.index("--path-shape-cost-class") + 1] == "large"
    assert cmd[cmd.index("--path-shape-dtype-group") + 1] == "float"


def test_harness_finalizes_adaptive_candidates_for_collected_items(monkeypatch, tmp_path, capsys):
    loaded = adaptive_isolation.AdaptiveIsolationLoadResult(
        candidates={
            "torchcts/a.py::test_keep": adaptive_isolation.AdaptiveIsolationCandidate(
                nodeid="torchcts/a.py::test_keep",
                canonical_nodeid="torchcts/a.py::test_keep",
                isolation_source="adaptive_previous_timeout",
                reason="previous subprocess isolation timed out",
                evidence_path="results/hw_latest.json",
                prior_status="ERROR",
                prior_error_type="TimeoutError",
            ),
            "torchcts/b.py::test_drop": adaptive_isolation.AdaptiveIsolationCandidate(
                nodeid="torchcts/b.py::test_drop",
                canonical_nodeid="torchcts/b.py::test_drop",
                isolation_source="adaptive_previous_crash",
                reason="previous run recorded a process crash",
                evidence_path="results/hw_latest.json",
                prior_status="ERROR",
            ),
        },
        rejected=[],
        warnings=[],
        artifacts_considered=["results/hw_latest.json"],
    )
    monkeypatch.setattr(harness, "_ADAPTIVE_ISOLATION_MODE", "auto")
    monkeypatch.setattr(harness, "_ADAPTIVE_ISOLATION_LOAD", loaded)
    monkeypatch.setattr(harness, "_ADAPTIVE_ISOLATION_WARNINGS", [])
    monkeypatch.setattr(harness, "_COLLECT_ONLY", False)
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_ARTIFACT_WRITES_ENABLED", True)
    monkeypatch.setattr(harness, "_IS_XDIST_WORKER", False)
    monkeypatch.setattr(harness, "_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(harness, "_HARDWARE_KEY", "hw")
    monkeypatch.setattr(harness, "_DEVICE_NAME", "mps")
    monkeypatch.setattr(harness, "_is_child_process", lambda: False)

    config = SimpleNamespace(option=SimpleNamespace(verbose=1))
    item = SimpleNamespace(nodeid="torchcts/a.py::test_keep")

    harness._finalize_adaptive_isolation_for_collection(config, [item])

    assert sorted(harness._ADAPTIVE_ISOLATION_ACTIVE) == ["torchcts/a.py::test_keep"]
    out = capsys.readouterr().out
    assert "Adaptive isolation: 1 node(s)" in out
    artifact = json.loads((tmp_path / "hw_adaptive_isolation.json").read_text(encoding="utf-8"))
    assert sorted(artifact["accepted"]) == ["torchcts/a.py::test_keep"]
    assert artifact["rejected"][0]["reason"] == "not_collected_in_current_run"


def test_harness_subprocess_error_record_preserves_crash_evidence():
    item = SimpleNamespace(
        nodeid="torchcts/example.py::test_crash",
        fspath="torchcts/example.py",
        name="test_crash",
        iter_markers=lambda name=None: iter(()),
    )

    record = harness._subprocess_error_record(
        item,
        "ERROR",
        "ProcessCrash",
        "crashed",
        1234.5,
        ["python", "-m", "pytest", item.nodeid],
        returncode=-11,
        stdout="stdout before crash",
        stderr=b"Fatal Python error: Segmentation fault",
    )

    assert record["nodeid"] == item.nodeid
    assert record["phase"] == "subprocess"
    assert record["failure_stage"] == "process"
    assert record["subprocess"]["command_args"] == ["python", "-m", "pytest", item.nodeid]
    assert record["subprocess"]["signal"] == "SIGSEGV"
    assert record["subprocess"]["duration_seconds"] == pytest.approx(1.2345)
    assert "Segmentation fault" in record["subprocess"]["stderr_tail"]
