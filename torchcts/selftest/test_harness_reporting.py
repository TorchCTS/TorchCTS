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
import runpy
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import torchcts.conftest as harness
import torchcts.cli as cli_module
import torchcts.core.comparer as comparer_module
import torchcts.core.coverage as coverage_module
import torchcts.core.device as device_module
import torchcts.core.dtype_contracts as dtype_contracts
import torchcts.core.opinfo_adapter as opinfo_adapter_module
import torchcts.core.reference_oracles as reference_oracles
import torchcts.core.runtime_evidence as runtime_evidence
import torchcts.core.version_rules as version_rules
import torchcts.generated.coverage_helpers as generated_helpers
import torchcts.op_metadata as op_metadata_module
import torchcts.sample_generation as sample_generation
import torchcts.rng.test_generator as rng_tests
from torchcts.core.comparer import (
    clear_metrics,
    compare_inf_propagation,
    compare_nan_propagation,
    compare_tensors,
    get_metrics,
)
from torchcts.core.manifest_schema import KNOWN_CAPABILITIES, validate_manifest
from torchcts.core.opinfo_adapter import (
    InputCondition,
    _NO_GENERIC_BACKWARD_ORACLE_OPS,
    _ieee754_enabled_for_op,
    classify_sample,
    get_op_sample_inputs,
    prepare_sample,
)
from torchcts.core.quantized_decoders import load_custom_container_decoder
from torchcts.core.report import build_report
from torchcts.core.semantic_levels import SemanticLevelError, SemanticLevelSelection, normalize_level_selection
from torchcts.core.tolerances import get_tolerance
from torchcts.dtypes.test_quantized import _run_custom_decoder_case
from torchcts.opinfo.test_opinfo_forward import _compare_special_tier as _opinfo_compare_special_tier
from torchcts.opinfo.test_opinfo_forward import _move_sample_obj
from torchcts.opinfo.test_opinfo_errors import _assert_expected_error
from torchcts.opinfo.test_opinfo_errors import _assert_exception_matches_expected

pytestmark = pytest.mark.covers_category("selftest")
_SOURCE_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_CHECKOUT_ONLY = pytest.mark.skipif(
    not (_SOURCE_REPO_ROOT / "pyproject.toml").exists(),
    reason="source checkout test requires repository files",
)


def test_build_report_counts_opinfo_and_ignores_plumbing():
    current_data = {
        "metadata": {
            "device_name": "cpu",
            "hardware_key": "cpu_test_1gb",
            "pytorch_version": torch.__version__,
            "timestamp": "2026-06-16T00:00:00Z",
            "elapsed_sec": 1,
        },
        "results": {
            "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[abs-torch.float32]": {
                "suite": "opinfo",
                "test_kind": "opinfo",
                "capability": "inference",
                "is_plumbing": False,
                "status": "PASS",
                "op": "abs",
                "dtype": "torch.float32",
            },
            "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[sin-torch.float32]": {
                "suite": "opinfo",
                "test_kind": "opinfo",
                "capability": "inference",
                "is_plumbing": False,
                "status": "FAIL",
                "op": "sin",
                "dtype": "torch.float32",
            },
            "torchcts/operators/test_unary.py::test_unary_float_op[exp-torch.float32]": {
                "suite": "operators",
                "test_kind": "handwritten",
                "capability": "inference",
                "is_plumbing": False,
                "status": "PASS",
                "op": "exp",
                "dtype": "torch.float32",
            },
            "torchcts/dtypes/test_quantized.py::test_quantized_plumbing[int4]": {
                "suite": "dtypes",
                "test_kind": "handwritten",
                "capability": "quantized_container_plumbing",
                "is_plumbing": True,
                "status": "PASS",
                "op": "int4_pack",
                "dtype": "torch.uint8",
            },
        },
        "skips": {
            "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[cos-torch.float32]": {
                "suite": "opinfo",
                "test_kind": "opinfo",
                "capability": "inference",
                "skip_reason": "op_excluded",
                "op": "cos",
                "dtype": "torch.float32",
            },
            "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[tan-torch.float32]": {
                "suite": "opinfo",
                "test_kind": "opinfo",
                "capability": "inference",
                "skip_reason": "dtype_not_listed",
                "op": "tan",
                "dtype": "torch.float32",
            },
        },
    }

    scorecard, _ = build_report(current_data, include_skips=True)

    assert "OpInfo ops discovered:     4" in scorecard
    assert "Ops tested (PASS):         1" in scorecard
    assert "Ops tested (FAIL):         1" in scorecard
    assert "Ops not run (manifest): 2" in scorecard
    assert "Ops not run (selection): 0" in scorecard
    assert "Ops not run (coverage): 0" in scorecard
    assert "Ops not run (runtime): 0" in scorecard
    assert "Ops skipped (unsupported)" not in scorecard
    assert "inference" in scorecard
    assert "2/3 passed" in scorecard


def test_build_report_counts_cpu_contract_not_run_bucket():
    current_data = {
        "metadata": {
            "device_name": "cpu",
            "hardware_key": "cpu_test_1gb",
            "pytorch_version": torch.__version__,
            "timestamp": "2026-06-16T00:00:00Z",
            "elapsed_sec": 1,
        },
        "results": {},
        "skips": {
            "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[fake-torch.complex32-clean]": {
                "suite": "opinfo",
                "test_kind": "opinfo",
                "capability": "inference",
                "skip_reason": "cpu_contract_unsupported",
                "op": "fake",
                "dtype": "torch.complex32",
            },
        },
    }

    scorecard, _ = build_report(current_data, include_skips=True)

    assert "Ops not run (CPU contract): 1" in scorecard


def test_flush_results_to_disk_is_disabled_for_dry_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_ARTIFACT_WRITES_ENABLED", False)
    monkeypatch.setattr(harness, "_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(harness, "_HARDWARE_KEY", "cpu_test_1gb")
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_START_TIME", 0.0)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_REPORT_SKIPS", True)
    monkeypatch.setattr(harness, "_SESSION_RESULTS", {"dummy": {"status": "PASS"}})
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {"dummy": {"skip_reason": "test"}})

    harness.flush_results_to_disk()

    assert not list(Path(tmp_path).glob("*.json"))


def test_flush_results_to_disk_writes_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_ARTIFACT_WRITES_ENABLED", True)
    monkeypatch.setattr(harness, "_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(harness, "_HARDWARE_KEY", "cpu_test_1gb")
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_START_TIME", 0.0)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", False)
    monkeypatch.setattr(harness, "_REPORT_SKIPS", False)
    monkeypatch.setattr(harness, "_SESSION_RESULTS", {"dummy": {"status": "PASS"}})
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})

    harness.flush_results_to_disk()

    latest_path = Path(tmp_path) / "cpu_test_1gb_latest.json"
    assert latest_path.exists()
    data = json.loads(latest_path.read_text())
    assert data["metadata"]["collect_only"] is False
    assert data["metadata"]["skip_count"] == 0
    assert data["results"]["dummy"]["status"] == "PASS"


def test_flush_results_to_disk_persists_skips_by_default(tmp_path, monkeypatch):
    skip = {"skip_reason": "capability_not_declared", "detail": "requires device_api"}
    monkeypatch.setattr(harness, "_ARTIFACT_WRITES_ENABLED", True)
    monkeypatch.setattr(harness, "_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(harness, "_HARDWARE_KEY", "cpu_test_1gb")
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_START_TIME", 0.0)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", False)
    monkeypatch.setattr(harness, "_REPORT_SKIPS", False)
    monkeypatch.setattr(harness, "_SESSION_RESULTS", {})
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {"test_node": skip})

    harness.flush_results_to_disk()

    latest_path = Path(tmp_path) / "cpu_test_1gb_latest.json"
    data = json.loads(latest_path.read_text())
    assert data["metadata"]["skip_count"] == 1
    assert data["skips"]["test_node"] == skip


def test_compile_probe_context_hides_stale_conda_env_for_regular_venv(monkeypatch):
    monkeypatch.setattr(harness.sys, "prefix", "/tmp/project/.venv")
    monkeypatch.setattr(harness.sys, "base_prefix", "/opt/homebrew/python")
    monkeypatch.setenv("CONDA_PREFIX", "/opt/anaconda3")
    monkeypatch.setenv("CONDA_EXE", "/opt/anaconda3/bin/conda")
    monkeypatch.setenv("CONDA_SHLVL", "1")

    with harness._without_stale_conda_env_for_venv():
        assert "CONDA_PREFIX" not in os.environ
        assert "CONDA_EXE" not in os.environ
        assert "CONDA_SHLVL" not in os.environ

    assert os.environ["CONDA_PREFIX"] == "/opt/anaconda3"
    assert os.environ["CONDA_EXE"] == "/opt/anaconda3/bin/conda"
    assert os.environ["CONDA_SHLVL"] == "1"


def test_compile_probe_context_preserves_active_conda_env(monkeypatch):
    monkeypatch.setattr(harness.sys, "prefix", "/opt/anaconda3/envs/project")
    monkeypatch.setattr(harness.sys, "base_prefix", "/opt/anaconda3")
    monkeypatch.setenv("CONDA_PREFIX", "/opt/anaconda3/envs/project")

    with harness._without_stale_conda_env_for_venv():
        assert os.environ["CONDA_PREFIX"] == "/opt/anaconda3/envs/project"


def test_cli_default_test_paths_exclude_selftests(tmp_path):
    for suite in cli_module.DEFAULT_TEST_SUITES:
        (tmp_path / suite).mkdir()
    (tmp_path / "selftest").mkdir()

    paths = cli_module._default_test_paths(str(tmp_path))

    assert str(tmp_path / "opinfo") in paths
    assert str(tmp_path / "generated") in paths
    assert str(tmp_path / "selftest") not in paths


def test_cli_default_test_paths_skip_missing_suite_dirs(tmp_path):
    (tmp_path / "opinfo").mkdir()

    paths = cli_module._default_test_paths(str(tmp_path))

    assert paths == [str(tmp_path / "opinfo")]


def test_probe_capability_supports_nested_named_tensor_and_fp8(monkeypatch):
    scripts = []

    def fake_run(cmd, capture_output, text, timeout):
        scripts.append(cmd[-1])
        return SimpleNamespace(returncode=0, stdout="SUCCESS\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert device_module.probe_capability("mps", "nested")
    assert device_module.probe_capability("mps", "named_tensor")
    assert device_module.probe_capability("mps", "fp8")
    assert "torch.nested.nested_tensor" in scripts[0]
    assert "names=('rows', 'cols')" in scripts[1]
    assert "torch.float8_e4m3fn" in scripts[2]


def test_probe_capability_imports_backend_before_device_probe(monkeypatch):
    scripts = []

    def fake_run(cmd, capture_output, text, timeout):
        scripts.append(cmd[-1])
        return SimpleNamespace(returncode=0, stdout="SUCCESS\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert device_module.probe_capability(
        "privateuseone",
        "sparse",
        backend_import="privateuseone_backend",
    )
    assert "_torchcts_backend_import = 'privateuseone_backend'" in scripts[0]
    assert "importlib.import_module(_torchcts_backend_import)" in scripts[0]
    assert "torch.sparse_coo_tensor" in scripts[0]


def test_probe_capability_result_preserves_failure_evidence(monkeypatch):
    def fake_run(cmd, capture_output, text, timeout):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="RuntimeError: backend refused named tensors\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = device_module.probe_capability_result("privateuseone", "named_tensor")

    assert not result.supported
    assert result.error_type == "CapabilityProbeFailed"
    assert "backend refused named tensors" in result.stderr
    assert not device_module.probe_capability("privateuseone", "named_tensor")


def test_declared_capability_probe_failure_is_diagnostic(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TORCHCTS_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("TORCHCTS_HARDWARE_KEY", "unit_hw")
    monkeypatch.setenv("TORCHCTS_DEVICE_NAME", "privateuseone")
    monkeypatch.setenv("TORCHCTS_PYTORCH_VERSION", "9.9.9")
    monkeypatch.setattr(harness, "_SESSION_PROBE_RESULTS", [])
    monkeypatch.setattr(harness, "_SESSION_PROBE_FAILURES", [])
    monkeypatch.setattr(harness, "_SESSION_PROBE_FAILURE_KEYS", set())
    monkeypatch.setattr(harness, "_COLLECT_ONLY", False)
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_KNOWN_SEGFAULT_AUDIT", False)
    caps = {"named_tensor": True}

    def fake_probe(device_name, capability):
        return device_module.CapabilityProbeResult(
            device_name=device_name,
            capability=capability,
            supported=False,
            returncode=1,
            error_type="CapabilityProbeFailed",
            error_message="NYI: named tensors only support CPU",
            stderr="NYI: named tensors only support CPU\n",
        )

    records = harness._apply_declared_capability_probes(
        caps,
        "privateuseone",
        probe_func=fake_probe,
    )

    assert caps["named_tensor"] is True
    assert len(records) == 1
    assert records[0]["probe_kind"] == "capability"
    assert records[0]["name"] == "named_tensor"
    assert records[0]["stage"] == "declared_capability_probe"
    assert records[0]["declared"] is True
    assert records[0]["outcome"] == "failed"
    assert records[0]["returncode"] == 1
    assert "NYI: named tensors" in records[0]["stderr_tail"]
    assert "tests will still run" not in capsys.readouterr().err
    path = next(tmp_path.glob("unit_hw_harness_probe_failures_*.jsonl"))
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["name"] == "named_tensor"


def test_declared_capability_probe_success_returns_pass_row(monkeypatch):
    monkeypatch.setattr(harness, "_SESSION_PROBE_FAILURES", [])
    monkeypatch.setattr(harness, "_SESSION_PROBE_FAILURE_KEYS", set())
    caps = {"named_tensor": True}

    def fake_probe(device_name, capability):
        return device_module.CapabilityProbeResult(
            device_name=device_name,
            capability=capability,
            supported=True,
            returncode=0,
            stdout="SUCCESS\n",
        )

    records = harness._apply_declared_capability_probes(
        caps,
        "privateuseone",
        probe_func=fake_probe,
    )

    assert records == [{
        "probe_kind": "capability",
        "name": "named_tensor",
        "stage": "declared_capability_probe",
        "declared": True,
        "outcome": "passed",
        "error_type": "",
        "error_message": "",
        "returncode": None,
        "timed_out": False,
        "stdout_tail": "",
        "stderr_tail": "",
        "command_args": [],
    }]
    assert harness._SESSION_PROBE_FAILURES == []


def test_declared_dtype_probe_failure_is_diagnostic(tmp_path, monkeypatch, capsys):
    def fake_zeros(*args, **kwargs):
        raise RuntimeError("value cannot be converted to type float without overflow")

    monkeypatch.setenv("TORCHCTS_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("TORCHCTS_HARDWARE_KEY", "unit_hw")
    monkeypatch.setenv("TORCHCTS_DEVICE_NAME", "privateuseone")
    monkeypatch.setenv("TORCHCTS_PYTORCH_VERSION", "9.9.9")
    monkeypatch.setattr(harness, "_SESSION_PROBE_RESULTS", [])
    monkeypatch.setattr(harness, "_SESSION_PROBE_FAILURES", [])
    monkeypatch.setattr(harness, "_SESSION_PROBE_FAILURE_KEYS", set())
    monkeypatch.setattr(harness, "_COLLECT_ONLY", False)
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_KNOWN_SEGFAULT_AUDIT", False)
    monkeypatch.setattr(harness.torch, "zeros", fake_zeros)
    supported_dtypes = {torch.float32: True}

    records = harness._apply_declared_dtype_probes(supported_dtypes, "privateuseone")

    assert supported_dtypes == {torch.float32: True}
    assert len(records) == 1
    assert records[0]["probe_kind"] == "dtype"
    assert records[0]["name"] == "torch.float32"
    assert records[0]["stage"] == "declared_dtype_probe"
    assert records[0]["declared"] is True
    assert records[0]["outcome"] == "failed"
    assert "value cannot be converted" in records[0]["error_message"]
    assert "tests will still run" not in capsys.readouterr().err
    path = next(tmp_path.glob("unit_hw_harness_probe_failures_*.jsonl"))
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["name"] == "torch.float32"


def test_declared_dtype_probe_success_returns_pass_row(monkeypatch):
    calls = []

    def fake_zeros(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(harness, "_SESSION_PROBE_FAILURES", [])
    monkeypatch.setattr(harness, "_SESSION_PROBE_FAILURE_KEYS", set())
    monkeypatch.setattr(harness.torch, "zeros", fake_zeros)

    records = harness._apply_declared_dtype_probes({torch.float32: True}, "privateuseone")

    assert len(calls) == 1
    assert records == [{
        "probe_kind": "dtype",
        "name": "torch.float32",
        "stage": "declared_dtype_probe",
        "declared": True,
        "outcome": "passed",
        "error_type": "",
        "error_message": "",
        "returncode": None,
        "timed_out": False,
        "stdout_tail": "",
        "stderr_tail": "",
        "command_args": [],
    }]
    assert harness._SESSION_PROBE_FAILURES == []


def test_probe_results_table_renders_mixed_pass_fail_and_artifact(monkeypatch):
    monkeypatch.setattr(harness, "_RESULTS_DIR", "./results")
    monkeypatch.setattr(harness, "_HARDWARE_KEY", "unit_hw")
    records = [
        harness._probe_result_record("dtype", "torch.float32", "declared_dtype_probe", "passed"),
        harness._probe_result_record(
            "capability",
            "pinned_memory",
            "declared_capability_probe",
            "failed",
            error_type="CapabilityProbeFailed",
            error_message=(
                "CUDA runtime error\n"
                "CUDA kernel errors might be asynchronously reported at some other API call\n"
                "For debugging consider passing CUDA_LAUNCH_BLOCKING=1"
            ),
        ),
    ]

    text = harness._format_probe_results_table(records, "cuda")

    assert "TorchCTS diagnostic probes" in text
    assert "Device: cuda" in text
    assert "dtype" in text and "torch.float32" in text and "PASS" in text
    assert "capability" in text and "pinned_memory" in text and "FAIL" in text
    assert "CapabilityProbeFailed: CUDA runtime error" in text
    assert "CUDA_LAUNCH_BLOCKING" not in text
    assert "Summary: 1 passed, 1 failed" in text
    assert "Full failure evidence: ./results/unit_hw_harness_probe_failures_" in text


def test_probe_results_table_renders_all_pass_without_artifact(monkeypatch):
    monkeypatch.setattr(harness, "_RESULTS_DIR", "./results")
    monkeypatch.setattr(harness, "_HARDWARE_KEY", "unit_hw")

    text = harness._format_probe_results_table(
        [harness._probe_result_record("dtype", "torch.float32", "declared_dtype_probe", "passed")],
        "cuda",
    )

    assert "torch.float32" in text
    assert "PASS" in text
    assert "Summary: 1 passed, 0 failed" in text
    assert "Full failure evidence" not in text


def test_probe_result_note_truncates_long_cuda_like_error():
    record = harness._probe_result_record(
        "dtype",
        "torch.float32",
        "declared_dtype_probe",
        "failed",
        error_type="AcceleratorError",
        error_message=(
            "AcceleratorError: " + ("x" * 200) + "\n"
            "CUDA kernel errors might be asynchronously reported at some other API call"
        ),
    )

    note = harness._probe_result_note(record)

    assert len(note) <= 120
    assert note.startswith("AcceleratorError: ")
    assert "asynchronously reported" not in note


def test_run_declared_diagnostic_probes_prints_one_table(monkeypatch, capsys):
    dtype_row = harness._probe_result_record("dtype", "torch.float32", "declared_dtype_probe", "passed")
    capability_row = harness._probe_result_record(
        "capability",
        "pinned_memory",
        "declared_capability_probe",
        "failed",
        error_type="CapabilityProbeFailed",
        error_message="pin_memory failed",
    )
    monkeypatch.setattr(harness, "_SESSION_PROBE_RESULTS", [])
    monkeypatch.setattr(harness, "_IS_XDIST_WORKER", False)
    monkeypatch.setattr(harness, "_apply_declared_dtype_probes", lambda *_args, **_kwargs: [dtype_row])
    monkeypatch.setattr(harness, "_apply_declared_capability_probes", lambda *_args, **_kwargs: [capability_row])

    records = harness._run_declared_diagnostic_probes({}, {}, "cuda", emit_table=True)

    assert records == [dtype_row, capability_row]
    assert harness._SESSION_PROBE_RESULTS == [dtype_row, capability_row]
    err = capsys.readouterr().err
    assert err.count("TorchCTS diagnostic probes") == 1
    assert "Summary: 1 passed, 1 failed" in err


def test_run_declared_diagnostic_probes_suppresses_table_in_xdist_worker(monkeypatch, capsys):
    dtype_row = harness._probe_result_record("dtype", "torch.float32", "declared_dtype_probe", "passed")
    monkeypatch.setattr(harness, "_SESSION_PROBE_RESULTS", [])
    monkeypatch.setattr(harness, "_IS_XDIST_WORKER", True)
    monkeypatch.setattr(harness, "_apply_declared_dtype_probes", lambda *_args, **_kwargs: [dtype_row])

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("capability probes should not run in xdist workers")

    monkeypatch.setattr(harness, "_apply_declared_capability_probes", fail_if_called)

    records = harness._run_declared_diagnostic_probes({}, {"named_tensor": True}, "cuda", emit_table=True)

    assert records == [dtype_row]
    assert capsys.readouterr().err == ""


def test_declared_diagnostic_probe_enablement_suppresses_special_modes(monkeypatch):
    monkeypatch.setattr(harness, "_COLLECT_ONLY", False)
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_KNOWN_SEGFAULT_AUDIT", False)
    assert harness._declared_diagnostic_probes_enabled(False)
    assert not harness._declared_diagnostic_probes_enabled(True)

    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    assert not harness._declared_diagnostic_probes_enabled(False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", False)
    monkeypatch.setattr(harness, "_SHOW_SKIPS", True)
    assert not harness._declared_diagnostic_probes_enabled(False)
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_KNOWN_SEGFAULT_AUDIT", True)
    assert not harness._declared_diagnostic_probes_enabled(False)


def test_op_metadata_loader_falls_back_to_module_adjacent_resource(tmp_path, monkeypatch):
    source_pkg = tmp_path / "source_pkg"
    source_pkg.mkdir()
    module_file = source_pkg / "op_metadata.py"
    module_file.write_text("# synthetic module path\n", encoding="utf-8")
    (source_pkg / "op_metadata.json").write_text(
        json.dumps({"version": 2, "metadata": {"loaded": "module-adjacent"}, "ops": {}}),
        encoding="utf-8",
    )

    missing_resource_root = tmp_path / "stale_site_packages" / "torchcts"
    monkeypatch.setattr(op_metadata_module, "__file__", str(module_file))
    monkeypatch.setattr(op_metadata_module.resources, "files", lambda _package: missing_resource_root)
    op_metadata_module.load_op_metadata.cache_clear()
    try:
        metadata = op_metadata_module.load_op_metadata()
    finally:
        op_metadata_module.load_op_metadata.cache_clear()

    assert metadata["metadata"]["loaded"] == "module-adjacent"


def test_dtype_contract_loader_falls_back_to_module_adjacent_package_resource(tmp_path, monkeypatch):
    source_pkg = tmp_path / "source_pkg"
    core_dir = source_pkg / "core"
    core_dir.mkdir(parents=True)
    module_file = core_dir / "dtype_contracts.py"
    module_file.write_text("# synthetic module path\n", encoding="utf-8")
    (source_pkg / "op_dtype_contracts.json").write_text(
        json.dumps({
            "version": 2,
            "format": "runtime_profile_ranges",
            "metadata": {"loaded": "module-adjacent"},
            "profiles": {},
            "contracts": {},
        }),
        encoding="utf-8",
    )

    missing_resource_root = tmp_path / "stale_site_packages" / "torchcts"
    monkeypatch.setattr(dtype_contracts, "__file__", str(module_file))
    monkeypatch.setattr(dtype_contracts.resources, "files", lambda _package: missing_resource_root)
    dtype_contracts.load_dtype_contracts.cache_clear()
    try:
        contracts = dtype_contracts.load_dtype_contracts()
    finally:
        dtype_contracts.load_dtype_contracts.cache_clear()

    assert contracts["metadata"]["loaded"] == "module-adjacent"


def test_atomic_json_dump_preserves_existing_file_on_failure(tmp_path):
    latest_path = tmp_path / "cpu_test_1gb_latest.json"
    latest_path.write_text('{"previous": true}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        harness._atomic_json_dump(latest_path, {"next": object()})

    assert json.loads(latest_path.read_text(encoding="utf-8")) == {"previous": True}
    assert not any(path.name.endswith(".tmp") for path in tmp_path.iterdir())


def test_runtime_evidence_writes_opinfo_oracle_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHCTS_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("TORCHCTS_HARDWARE_KEY", "unit_hw")
    monkeypatch.setenv("TORCHCTS_DEVICE_NAME", "privateuseone")
    monkeypatch.setenv("TORCHCTS_PYTORCH_VERSION", "9.9.9")

    runtime_evidence.record_opinfo_oracle_failure(
        "forward",
        "fake.op",
        "torch.float32",
        "cpu_reference",
        RuntimeError("reference unavailable"),
        input_condition="clean",
        sample_index=2,
        nodeid="node::id",
    )

    path = next(tmp_path.glob("unit_hw_opinfo_oracle_failures_*.jsonl"))
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["device_name"] == "privateuseone"
    assert record["hardware_key"] == "unit_hw"
    assert record["pytorch_version"] == "9.9.9"
    assert record["phase"] == "forward"
    assert record["op_name"] == "fake.op"
    assert record["sample_index"] == 2
    assert record["error_type"] == "RuntimeError"
    assert record["error_message"] == "reference unavailable"


def test_runtime_evidence_falls_back_and_safely_formats_errors(tmp_path, monkeypatch):
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("broken __str__")

    for key in (
        "TORCHCTS_RESULTS_DIR",
        "TORCHCTS_HARDWARE_KEY",
        "TORCHCTS_DEVICE_NAME",
        "TORCHCTS_PYTORCH_VERSION",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    runtime_evidence.record_opinfo_oracle_failure(
        "forward",
        "fake.op",
        "torch.float32",
        "cpu_reference",
        RuntimeError("x" * 5000),
    )
    runtime_evidence.record_opinfo_oracle_failure(
        "forward",
        "fake.op",
        "torch.float32",
        "cpu_reference",
        BadStr(),
    )

    path = next((tmp_path / "results").glob("unknown_opinfo_oracle_failures_*.jsonl"))
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["hardware_key"] == "unknown"
    assert len(records[0]["error_message"]) == 4000
    assert records[0]["error_message"].endswith("...")
    assert records[1]["error_message"].startswith("<unprintable BadStr")


def test_runtime_evidence_writes_harness_probe_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHCTS_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("TORCHCTS_HARDWARE_KEY", "unit_hw")
    monkeypatch.setenv("TORCHCTS_DEVICE_NAME", "privateuseone")
    monkeypatch.setenv("TORCHCTS_PYTORCH_VERSION", "9.9.9")

    record = runtime_evidence.record_harness_probe_failure(
        "dtype",
        "torch.float64",
        RuntimeError("x" * 5000),
        stage="declared_dtype_probe",
    )

    path = next(tmp_path.glob("unit_hw_harness_probe_failures_*.jsonl"))
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload == record
    assert payload["device_name"] == "privateuseone"
    assert payload["hardware_key"] == "unit_hw"
    assert payload["pytorch_version"] == "9.9.9"
    assert payload["probe_kind"] == "dtype"
    assert payload["name"] == "torch.float64"
    assert payload["stage"] == "declared_dtype_probe"
    assert payload["declared"] is True
    assert payload["outcome"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert len(payload["error_message"]) == 4000
    assert payload["error_message"].endswith("...")


def test_flush_results_includes_harness_probe_failures(tmp_path, monkeypatch):
    record = {
        "created_at": "2026-06-29T00:00:00Z",
        "device_name": "privateuseone",
        "hardware_key": "unit_hw",
        "pytorch_version": "9.9.9",
        "probe_kind": "dtype",
        "name": "torch.float64",
        "stage": "declared_dtype_probe",
        "declared": True,
        "outcome": "failed",
        "error_type": "RuntimeError",
        "error_message": "probe failed",
        "returncode": None,
        "timed_out": False,
        "stdout_tail": "",
        "stderr_tail": "",
        "command_args": [],
    }
    monkeypatch.setattr(harness, "_ARTIFACT_WRITES_ENABLED", True)
    monkeypatch.setattr(harness, "_IS_XDIST_WORKER", False)
    monkeypatch.setattr(harness, "_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(harness, "_HARDWARE_KEY", "unit_hw")
    monkeypatch.setattr(harness, "_DEVICE_NAME", "privateuseone")
    monkeypatch.setattr(harness, "_START_TIME", 0)
    monkeypatch.setattr(harness, "_SESSION_RESULTS", {})
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SESSION_PROBE_RESULTS", [
        harness._probe_result_record("dtype", "torch.float32", "declared_dtype_probe", "passed"),
        harness._probe_result_record(
            "dtype",
            "torch.float64",
            "declared_dtype_probe",
            "failed",
            error_type="RuntimeError",
            error_message="probe failed",
        ),
    ])
    monkeypatch.setattr(harness, "_SESSION_PROBE_FAILURES", [record])
    monkeypatch.setattr(harness, "_SESSION_COMPLETED", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", False)
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 8)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 8))

    harness.flush_results_to_disk()

    payload = json.loads((tmp_path / "unit_hw_latest.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["harness_probe_failure_count"] == 1
    assert payload["metadata"]["harness_probe_pass_count"] == 1
    assert payload["metadata"]["harness_probe_total_count"] == 2
    assert payload["metadata"]["harness_probe_failure_artifact"].endswith(
        "unit_hw_harness_probe_failures_{}.jsonl".format(os.getpid())
    )
    assert payload["harness_probe_failures"] == [record]


def test_merge_xdist_worker_files_dedupes_harness_probe_failures(tmp_path):
    base_record = {
        "probe_kind": "dtype",
        "name": "torch.float64",
        "stage": "declared_dtype_probe",
        "error_type": "RuntimeError",
        "error_message": "probe failed",
        "returncode": None,
    }
    other_record = {
        "probe_kind": "capability",
        "name": "named_tensor",
        "stage": "declared_capability_probe",
        "error_type": "CapabilityProbeFailed",
        "error_message": "probe failed",
        "returncode": 1,
    }
    latest = tmp_path / "unit_hw_latest.json"
    latest.write_text(
        json.dumps({
            "metadata": {
                "elapsed_sec": 1.0,
                "harness_probe_pass_count": 3,
                "harness_probe_total_count": 4,
            },
            "results": {},
            "skips": {},
            "harness_probe_failures": [base_record],
        }),
        encoding="utf-8",
    )
    worker = tmp_path / "unit_hw_latest.gw0.json"
    worker.write_text(
        json.dumps({
            "metadata": {
                "elapsed_sec": 2.0,
                "harness_probe_pass_count": 4,
                "harness_probe_total_count": 6,
            },
            "results": {"node": {"status": "PASS"}},
            "skips": {},
            "harness_probe_failures": [base_record, other_record],
        }),
        encoding="utf-8",
    )

    harness._merge_xdist_worker_files(str(tmp_path), "unit_hw")

    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["metadata"]["harness_probe_failure_count"] == 2
    assert payload["metadata"]["harness_probe_pass_count"] == 4
    assert payload["metadata"]["harness_probe_total_count"] == 6
    assert payload["harness_probe_failures"] == [base_record, other_record]
    assert payload["results"] == {"node": {"status": "PASS"}}


def test_opinfo_known_failure_package_policy_is_removed():
    assert not hasattr(opinfo_adapter_module, "record_known_failure")
    assert not hasattr(opinfo_adapter_module, "load_known_failures")
    assert not hasattr(opinfo_adapter_module, "_KNOWN_FAILURES_PATH")


def test_prepare_sample_preserves_special_polynomial_order_argument():
    from torch.testing._internal.opinfo.core import SampleInput

    raw = SampleInput(
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        args=(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),),
    )

    prepared = prepare_sample(
        raw,
        InputCondition.HAS_INF,
        ieee754_seed=67,
        sample_index=0,
        op_name="special.legendre_polynomial_p",
    )

    assert torch.isinf(prepared.input).any()
    assert torch.equal(prepared.args[0], raw.args[0])
    assert not torch.isinf(prepared.args[0]).any()


def test_ieee754_collection_excludes_ops_without_propagation_contract():
    assert _ieee754_enabled_for_op("empty_like", True) is False
    assert _ieee754_enabled_for_op("bernoulli", True) is False
    assert _ieee754_enabled_for_op("nn.functional.dropout2d", True) is False
    assert _ieee754_enabled_for_op("add", True) is True


def test_matrix_exp_special_value_tiers_are_never_enabled(monkeypatch):
    monkeypatch.setattr(opinfo_adapter_module, "_IEEE754_UNDEFINED_OPS", frozenset())

    for cap in (True, "matrix_exp", ["matrix_exp"]):
        assert _ieee754_enabled_for_op("matrix_exp", cap) is False
        assert _ieee754_enabled_for_op("linalg.matrix_exp", cap) is False


def test_versioned_rules_carry_forward_until_explicit_remove():
    rules = {
        "2.12": ["all_212", "fixed_in_213"],
        "2.12.1": {"add": ["patch_2121"]},
        "2.12.1+cpu": {"add": ["build_only"]},
        "2.12.2": {"add": ["patch_2122"]},
        "2.13": {
            "add": ["all_213"],
            "remove": ["fixed_in_213"],
        },
    }

    assert version_rules.cumulative_versioned_set(rules, "2.12.1+cpu") == frozenset({
        "all_212",
        "fixed_in_213",
        "patch_2121",
        "build_only",
    })
    assert version_rules.cumulative_versioned_set(rules, "2.12.9") == frozenset({
        "all_212",
        "fixed_in_213",
        "patch_2121",
        "patch_2122",
    })
    assert version_rules.cumulative_versioned_set(rules, "2.13.0") == frozenset({
        "all_212",
        "patch_2121",
        "patch_2122",
        "all_213",
    })
    assert version_rules.cumulative_versioned_set(rules, "2.13.0.dev20260628") == frozenset({
        "all_212",
        "patch_2121",
        "patch_2122",
        "all_213",
    })


def test_ieee754_undefined_loader_cascades_until_explicit_remove(tmp_path, monkeypatch):
    cache = tmp_path / "ieee754_undefined.json"
    cache.write_text(
        json.dumps({
            "2.12": ["all_212", "fixed_in_213"],
            "2.12.1": ["patch_2121"],
            "2.12.1+cpu": ["build_only"],
            "2.12.2": ["patch_2122"],
            "2.13": {
                "add": ["all_213"],
                "remove": ["fixed_in_213"],
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(opinfo_adapter_module, "_IEEE754_UNDEFINED_OPS_PATH", str(cache))

    monkeypatch.setattr(opinfo_adapter_module.torch, "__version__", "2.12.1+cpu")
    loaded = opinfo_adapter_module._load_ieee754_undefined()

    assert "all_212" in loaded
    assert "fixed_in_213" in loaded
    assert "patch_2121" in loaded
    assert "build_only" in loaded
    assert "patch_2122" not in loaded
    assert "all_213" not in loaded

    monkeypatch.setattr(opinfo_adapter_module.torch, "__version__", "2.12.0")
    loaded = opinfo_adapter_module._load_ieee754_undefined()

    assert "all_212" in loaded
    assert "fixed_in_213" in loaded
    assert "patch_2121" not in loaded
    assert "build_only" not in loaded
    assert "patch_2122" not in loaded
    assert "all_213" not in loaded

    monkeypatch.setattr(opinfo_adapter_module.torch, "__version__", "2.12.9")
    loaded = opinfo_adapter_module._load_ieee754_undefined()

    assert "all_212" in loaded
    assert "fixed_in_213" in loaded
    assert "patch_2121" in loaded
    assert "build_only" not in loaded
    assert "patch_2122" in loaded
    assert "all_213" not in loaded

    monkeypatch.setattr(opinfo_adapter_module.torch, "__version__", "2.13.0")
    loaded = opinfo_adapter_module._load_ieee754_undefined()

    assert "all_212" in loaded
    assert "fixed_in_213" not in loaded
    assert "patch_2121" in loaded
    assert "build_only" not in loaded
    assert "patch_2122" in loaded
    assert "all_213" in loaded


def test_matrix_exp_forward_collection_is_clean_only_for_regex_ieee754():
    manifest = {
        "capabilities": {"ieee754": "matrix_exp"},
        "supported_dtypes": {
            "torch.bfloat16": True,
            "torch.complex128": True,
            "torch.complex64": True,
            "torch.float16": True,
            "torch.float32": True,
            "torch.float64": True,
        },
    }

    matrix_exp_tests = [
        (op_name, dtype_name, condition)
        for op_name, dtype_name, condition in opinfo_adapter_module.get_forward_op_tests(manifest)
        if op_name == "matrix_exp"
    ]

    assert matrix_exp_tests
    assert {condition for _, _, condition in matrix_exp_tests} == {InputCondition.CLEAN}


def test_opinfo_forward_dtype_false_creates_structured_skip(monkeypatch):
    import torch.testing._internal.common_methods_invocations as cmi

    class FakeOp:
        name = "fake_manifest_dtype_op"
        dtypes = (torch.float32,)
        backward_dtypes = (torch.float32,)
        supports_autograd = True
        supports_sparse = False
        supports_sparse_csr = False

    opinfo_adapter_module.clear_pending_manifest_skips()
    monkeypatch.setattr(cmi, "op_db", [FakeOp()])

    tests = opinfo_adapter_module.get_forward_op_tests({
        "capabilities": {"ieee754": False},
        "supported_dtypes": {torch.float32: False},
    })
    skips = opinfo_adapter_module.consume_pending_manifest_skips()

    assert tests == []
    assert len(skips) == 1
    record = next(iter(skips.values()))
    assert record["suite"] == "opinfo"
    assert record["test_kind"] == "opinfo"
    assert record["op"] == "fake_manifest_dtype_op"
    assert record["dtype"] == "torch.float32"
    assert record["input_condition"] == InputCondition.CLEAN
    assert record["skip_reason"] == "dtype_not_supported"


def test_opinfo_regex_filtered_dtype_creates_structured_skip(monkeypatch):
    import torch.testing._internal.common_methods_invocations as cmi

    class FakeOp:
        name = "fake_manifest_regex_op"
        dtypes = (torch.complex64,)
        backward_dtypes = (torch.complex64,)
        supports_autograd = True
        supports_sparse = False
        supports_sparse_csr = False

    opinfo_adapter_module.clear_pending_manifest_skips()
    monkeypatch.setattr(cmi, "op_db", [FakeOp()])

    tests = opinfo_adapter_module.get_forward_op_tests({
        "capabilities": {"ieee754": False},
        "supported_dtypes": {torch.complex64: r"^fft"},
    })
    skips = opinfo_adapter_module.consume_pending_manifest_skips()

    assert tests == []
    assert len(skips) == 1
    record = next(iter(skips.values()))
    assert record["skip_reason"] == "dtype_regex_filtered"
    assert record["dtype"] == "torch.complex64"


def test_opinfo_missing_dtype_is_omitted_without_structured_skip(monkeypatch):
    import torch.testing._internal.common_methods_invocations as cmi

    class FakeOp:
        name = "fake_manifest_missing_op"
        dtypes = (torch.float32,)
        backward_dtypes = (torch.float32,)
        supports_autograd = True
        supports_sparse = False
        supports_sparse_csr = False

    opinfo_adapter_module.clear_pending_manifest_skips()
    monkeypatch.setattr(cmi, "op_db", [FakeOp()])

    tests = opinfo_adapter_module.get_forward_op_tests({
        "capabilities": {"ieee754": False},
        "supported_dtypes": {},
    })
    skips = opinfo_adapter_module.consume_pending_manifest_skips()

    assert tests == []
    assert skips == {}


def test_opinfo_backward_dtype_false_creates_structured_skip(monkeypatch):
    import torch.testing._internal.common_methods_invocations as cmi

    class FakeOp:
        name = "fake_manifest_backward_op"
        dtypes = (torch.float32,)
        backward_dtypes = (torch.float32,)
        supports_autograd = True
        supports_sparse = False
        supports_sparse_csr = False

    opinfo_adapter_module.clear_pending_manifest_skips()
    monkeypatch.setattr(cmi, "op_db", [FakeOp()])

    tests = opinfo_adapter_module.get_backward_op_tests({
        "capabilities": {},
        "supported_dtypes": {torch.float32: False},
    })
    skips = opinfo_adapter_module.consume_pending_manifest_skips()

    assert tests == []
    assert len(skips) == 1
    record = next(iter(skips.values()))
    assert record["capability"] == "training"
    assert record["op"] == "fake_manifest_backward_op"
    assert record["skip_reason"] == "dtype_not_supported"


def test_opinfo_allowed_dtype_still_creates_executable_test(monkeypatch):
    import torch.testing._internal.common_methods_invocations as cmi
    from torch.testing._internal.opinfo.core import SampleInput

    class FakeOp:
        name = "fake_manifest_allowed_op"
        dtypes = (torch.float32,)
        backward_dtypes = (torch.float32,)
        supports_autograd = True
        supports_sparse = False
        supports_sparse_csr = False

        def sample_inputs(self, device, dtype, requires_grad=False):
            yield SampleInput(torch.ones(2, dtype=dtype, device=device))

        def op(self, input):
            return input + 1

    opinfo_adapter_module.clear_pending_manifest_skips()
    opinfo_adapter_module._OPINFO_CPU_CONTRACT_CACHE.clear()
    monkeypatch.setattr(cmi, "op_db", [FakeOp()])

    tests = opinfo_adapter_module.get_forward_op_tests({
        "capabilities": {"ieee754": False},
        "supported_dtypes": {torch.float32: True},
    })
    skips = opinfo_adapter_module.consume_pending_manifest_skips()

    assert tests == [("fake_manifest_allowed_op", "torch.float32", InputCondition.CLEAN)]
    assert skips == {}


def test_opinfo_cpu_probe_records_source_expected_unsupported_mismatch():
    from torch.testing._internal.opinfo.core import SampleInput

    class FakeOp:
        name = "fake_cpu_contract_op"
        dtypes = (torch.complex32,)
        backward_dtypes = ()
        supports_autograd = False
        supports_sparse = False
        supports_sparse_csr = False

        def sample_inputs(self, device, dtype, requires_grad=False):
            yield SampleInput(torch.ones(2, dtype=dtype, device=device))

        def op(self, input):
            raise RuntimeError("not implemented for 'ComplexHalf'")

    opinfo_adapter_module._OPINFO_CPU_CONTRACT_CACHE.clear()

    disposition = opinfo_adapter_module.probe_opinfo_cpu_contract(FakeOp(), torch.complex32, phase="forward")

    assert not disposition.allowed
    assert disposition.status == dtype_contracts.CPU_UNSUPPORTED
    assert disposition.source_expected == ("torch.complex32",)
    assert disposition.mismatches == (
        dtype_contracts.SOURCE_EXPECTED_BUT_CPU_UNSUPPORTED,
    )


def test_opinfo_explicit_cpu_contract_skip_is_structured(monkeypatch):
    import torch.testing._internal.common_methods_invocations as cmi

    class FakeOp:
        name = "fake_explicit_cpu_contract_op"
        dtypes = (torch.complex32,)
        backward_dtypes = ()
        supports_autograd = False
        supports_sparse = False
        supports_sparse_csr = False

    def fake_contract_disposition(*args, **kwargs):
        return dtype_contracts.ContractDisposition(
            False,
            dtype_contracts.CPU_UNSUPPORTED,
            "cpu_contract_unsupported",
            "explicit unsupported",
            source_expected=("torch.complex32",),
            mismatches=(dtype_contracts.SOURCE_EXPECTED_BUT_CPU_UNSUPPORTED,),
        )

    opinfo_adapter_module.clear_pending_manifest_skips()
    monkeypatch.setattr(cmi, "op_db", [FakeOp()])
    monkeypatch.setattr(opinfo_adapter_module, "contract_disposition", fake_contract_disposition)

    tests = opinfo_adapter_module.get_forward_op_tests({
        "capabilities": {"ieee754": False},
        "supported_dtypes": {torch.complex32: True},
    })
    skips = opinfo_adapter_module.consume_pending_manifest_skips()

    assert tests == []
    assert len(skips) == 1
    record = next(iter(skips.values()))
    assert record["skip_reason"] == "cpu_contract_unsupported"
    assert record["cpu_contract_status"] == dtype_contracts.CPU_UNSUPPORTED
    assert record["source_expected"] == ["torch.complex32"]
    assert record["source_probe_mismatches"] == [
        dtype_contracts.SOURCE_EXPECTED_BUT_CPU_UNSUPPORTED,
    ]


def test_cpu_supported_but_missing_from_source_is_recorded():
    disposition = dtype_contracts.disposition_from_cpu_probe(
        "aten::fake_missing_source",
        torch.float32,
        supported=True,
    )

    assert disposition.allowed
    assert disposition.status == dtype_contracts.CPU_SUPPORTED
    assert disposition.mismatches == (dtype_contracts.CPU_SUPPORTED_BUT_MISSING_FROM_SOURCE,)


def test_dtype_contract_disposition_uses_generic_runtime_detail_without_probe_blob(monkeypatch):
    monkeypatch.setattr(dtype_contracts.torch, "__version__", "2.12.1")
    monkeypatch.setattr(
        dtype_contracts,
        "load_dtype_contracts",
        lambda: {
            "version": 2,
            "format": "runtime_profile_ranges",
            "metadata": {"contract_authority": "versioned_cpu_probe", "collected_versions": ["2.12.1"]},
            "profiles": {
                "p000001": {
                    "cpu_supported": {},
                    "cpu_unsupported": {"forward:clean": ["torch.complex32"]},
                    "cpu_unknown": {},
                    "cpu_pending": {},
                    "oracle_supported": {},
                    "source_expected": {},
                },
            },
            "contracts": {"aten::fake_contract": [["2.12.1", "2.12.1", "p000001"]]},
        },
    )

    disposition = dtype_contracts.contract_disposition("aten::fake_contract", torch.complex32)

    assert not disposition.allowed
    assert disposition.status == dtype_contracts.CPU_UNSUPPORTED
    assert disposition.detail == "torch.complex32 is not supported by the PyTorch CPU contract for aten::fake_contract"


def test_dtype_contract_replacement_version_entry_removes_prior_bucket(monkeypatch):
    monkeypatch.setattr(dtype_contracts.torch, "__version__", "2.8.0")
    monkeypatch.setattr(
        dtype_contracts,
        "load_dtype_contracts",
        lambda: {
            "version": 2,
            "format": "runtime_profile_ranges",
            "metadata": {"contract_authority": "versioned_cpu_probe", "collected_versions": ["2.7.0", "2.8.0"]},
            "profiles": {
                "p000001": {
                    "cpu_supported": {"forward:clean": ["torch.float32"]},
                    "cpu_unsupported": {},
                    "cpu_unknown": {},
                    "cpu_pending": {},
                    "oracle_supported": {},
                    "source_expected": {},
                },
                "p000002": {
                    "cpu_supported": {},
                    "cpu_unsupported": {"forward:clean": ["torch.float32"]},
                    "cpu_unknown": {},
                    "cpu_pending": {},
                    "oracle_supported": {},
                    "source_expected": {},
                },
            },
            "contracts": {"aten::fake_contract": [["2.7.0", "2.7.0", "p000001"], ["2.8.0", "2.8.0", "p000002"]]},
        },
    )

    disposition = dtype_contracts.contract_disposition("aten::fake_contract", torch.float32)

    assert not disposition.allowed
    assert disposition.status == dtype_contracts.CPU_UNSUPPORTED
    assert disposition.detail == "torch.float32 is not supported by the PyTorch CPU contract for aten::fake_contract"


def test_source_expected_dtypes_prefers_versioned_contract_over_global_metadata(monkeypatch):
    monkeypatch.setattr(dtype_contracts.torch, "__version__", "2.7.0")
    monkeypatch.setattr(
        dtype_contracts,
        "load_dtype_contracts",
        lambda: {
            "version": 2,
            "format": "runtime_profile_ranges",
            "metadata": {"contract_authority": "versioned_cpu_probe", "collected_versions": ["2.7.0"]},
            "profiles": {
                "p000001": {
                    "cpu_supported": {},
                    "cpu_unsupported": {},
                    "cpu_unknown": {},
                    "cpu_pending": {},
                    "oracle_supported": {},
                    "source_expected": {"*": ["torch.float32"]},
                },
            },
            "contracts": {"aten::fake_contract": [["2.7.0", "2.7.0", "p000001"]]},
        },
    )
    monkeypatch.setattr(
        dtype_contracts,
        "get_op_metadata",
        lambda _dispatcher_name: {"pytorch_dtypes": ["f64"]},
    )

    assert dtype_contracts.source_expected_dtypes("aten::fake_contract") == ("torch.float32",)


def test_generated_cpu_contract_probe_honors_explicit_out_args_on_misbucketed_out_schema():
    from torchcts.generated.generated_cases import GENERATED_CASES

    entry = next(
        candidate
        for entries in GENERATED_CASES["cases_by_surface"].values()
        for candidate in entries
        if isinstance(candidate, dict) and candidate.get("name") == "aten::_aminmax.dim_out"
    )

    assert entry["surface_kind"] == "mutating_or_inplace"
    assert any(arg.get("is_out") for arg in entry["args"])

    result = generated_helpers.probe_generated_clean_cpu_contract(
        entry,
        torch.float32,
        {},
        enforce_recorded_contract=False,
    )

    assert result["status"] == "supported"


def test_generic_backward_excludes_nondeterministic_oracles():
    assert "nn.functional.dropout" in _NO_GENERIC_BACKWARD_ORACLE_OPS
    assert "nn.functional.fractional_max_pool2d" in _NO_GENERIC_BACKWARD_ORACLE_OPS
    assert "empty_like" in _NO_GENERIC_BACKWARD_ORACLE_OPS


def test_factory_out_tensor_scalars_use_integer_literals_for_integer_dtypes():
    arg = {"name": "end", "tensor": True}

    value = sample_generation.factory_out_arg_value(arg, torch.uint64, "cpu")

    assert value.dtype == torch.uint64
    assert value.item() == 6


def test_opinfo_sample_classification_handles_unresolved_conjugates():
    from torch.testing._internal.opinfo.core import SampleInput

    sample = SampleInput(
        torch.tensor([1.0 + 2.0j], dtype=torch.complex64),
        args=(torch.tensor([3.0 + 4.0j], dtype=torch.complex64).conj(),),
        kwargs={},
    )

    assert classify_sample(sample) == InputCondition.CLEAN
    transformed = prepare_sample(sample, InputCondition.HAS_INF, op_name="dot")
    assert classify_sample(transformed) == InputCondition.HAS_INF


def test_opinfo_samples_are_reference_generated_before_backend_transfer(monkeypatch):
    from torch.testing._internal.opinfo.core import SampleInput

    calls = []

    class FakeOpInfo:
        def sample_inputs(self, device, dtype, requires_grad=False):
            calls.append((device, dtype, requires_grad))
            yield SampleInput(
                torch.ones(1, dtype=dtype, device=device, requires_grad=requires_grad)
            )

    monkeypatch.setitem(opinfo_adapter_module._live_opinfo_cache, "fake.reference", FakeOpInfo())

    samples = list(get_op_sample_inputs("fake.reference", "mps", torch.float32, requires_grad=True))

    assert calls == [("cpu", torch.float32, True)]
    assert samples[0].input.device.type == "cpu"
    assert samples[0].input.requires_grad is True


def test_opinfo_sample_generation_errors_propagate(monkeypatch):
    class FakeOpInfo:
        def sample_inputs(self, device, dtype, requires_grad=False):
            raise ValueError("broken samples")

    monkeypatch.setitem(opinfo_adapter_module._live_opinfo_cache, "fake.bad_samples", FakeOpInfo())

    with pytest.raises(RuntimeError, match=r"sample_inputs failed for fake\.bad_samples on cpu with torch.float32"):
        list(get_op_sample_inputs("fake.bad_samples", "cpu", torch.float32))


def test_opinfo_error_input_generation_errors_propagate(monkeypatch):
    class FakeOpInfo:
        def error_inputs(self, device):
            raise ValueError("broken errors")

    monkeypatch.setitem(opinfo_adapter_module._live_opinfo_cache, "fake.bad_errors", FakeOpInfo())

    with pytest.raises(RuntimeError, match=r"error_inputs failed for fake\.bad_errors on cpu"):
        list(opinfo_adapter_module.get_op_error_inputs("fake.bad_errors", "cpu"))


def test_opinfo_error_metadata_type_and_regex_are_enforced():
    err_in = SimpleNamespace(error_type=ValueError, error_regex="expected fragment")

    with pytest.raises(AssertionError, match="raised RuntimeError, expected ValueError"):
        _assert_expected_error(
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("expected fragment")),
            None,
            [],
            {},
            err_in,
            "fake.error",
        )

    with pytest.raises(AssertionError, match="message did not match"):
        _assert_expected_error(
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("different")),
            None,
            [],
            {},
            err_in,
            "fake.error",
        )

    with pytest.raises(AssertionError, match="Expected exception ValueError not raised"):
        _assert_expected_error(lambda *args, **kwargs: None, None, [], {}, err_in, "fake.error")


def test_opinfo_error_device_placement_must_match_expected_metadata():
    err_in = SimpleNamespace(error_type=RuntimeError, error_regex="placement rejected")

    _assert_exception_matches_expected(
        RuntimeError("placement rejected by backend"),
        err_in,
        "fake.error",
        "device placement",
    )
    with pytest.raises(AssertionError, match="message did not match"):
        _assert_exception_matches_expected(
            RuntimeError("unsupported dtype"),
            err_in,
            "fake.error",
            "device placement",
        )


def test_opinfo_forward_sample_move_is_recursive():
    sample = {
        "items": [torch.ones(1), (torch.zeros(1),)],
        "plain": 3,
    }

    moved = _move_sample_obj(sample, "cpu")

    assert moved["items"][0].device.type == "cpu"
    assert moved["items"][1][0].device.type == "cpu"
    assert moved["plain"] == 3


def test_opinfo_sample_classification_scans_sparse_values():
    from torch.testing._internal.opinfo.core import SampleInput

    float_sample = SampleInput(
        torch.sparse_csr_tensor(
            torch.tensor([0, 1]),
            torch.tensor([0]),
            torch.tensor([float("nan")]),
            size=(1, 1),
        )
    )
    complex_sample = SampleInput(
        torch.sparse_csr_tensor(
            torch.tensor([0, 1]),
            torch.tensor([0]),
            torch.tensor([complex(float("inf"), 0.0)], dtype=torch.complex64),
            size=(1, 1),
        )
    )

    assert classify_sample(float_sample) == InputCondition.HAS_NAN
    assert classify_sample(complex_sample) == InputCondition.HAS_INF


def test_opinfo_sample_transform_mutates_sparse_values_only():
    from torch.testing._internal.opinfo.core import SampleInput

    raw = SampleInput(
        torch.sparse_csr_tensor(
            torch.tensor([0, 2]),
            torch.tensor([0, 1]),
            torch.tensor([1.0, 2.0]),
            size=(1, 2),
        )
    )

    has_inf = prepare_sample(raw, InputCondition.HAS_INF, op_name="sparse.mm")
    clean = prepare_sample(has_inf, InputCondition.CLEAN, op_name="sparse.mm")

    assert has_inf.input.layout == torch.sparse_csr
    assert has_inf.input.crow_indices().equal(raw.input.crow_indices())
    assert has_inf.input.col_indices().equal(raw.input.col_indices())
    assert classify_sample(has_inf) == InputCondition.HAS_INF
    assert clean.input.layout == torch.sparse_csr
    assert classify_sample(clean) == InputCondition.CLEAN


def test_compare_tensors_reports_shape_after_complex_normalization():
    clear_metrics()

    with pytest.raises(AssertionError, match="Shape mismatch after comparison normalization"):
        compare_tensors(
            torch.tensor(1.0 + 2.0j, dtype=torch.complex64),
            torch.tensor(1.0, dtype=torch.float32),
            category="elementwise",
            dtype=torch.complex64,
        )


def test_build_report_marks_collection_only_sessions():
    current_data = {
        "metadata": {
            "device_name": "cpu",
            "hardware_key": "cpu_test_1gb",
            "pytorch_version": torch.__version__,
            "timestamp": "2026-06-16T00:00:00Z",
            "elapsed_sec": 1,
            "collect_only": True,
        },
        "results": {},
        "skips": {},
    }

    scorecard, _ = build_report(current_data)
    assert "Collection-only session: no tests executed." in scorecard


def test_build_report_ignores_runtime_skips_in_score_totals():
    current_data = {
        "metadata": {
            "device_name": "cpu",
            "hardware_key": "cpu_test_1gb",
            "pytorch_version": torch.__version__,
            "timestamp": "2026-06-16T00:00:00Z",
            "elapsed_sec": 1,
        },
        "results": {
            "torchcts/training/test_grad_clipping.py::test_gradient_clipping[norm-dtype0]": {
                "suite": "training",
                "test_kind": "handwritten",
                "capability": "training",
                "is_plumbing": False,
                "status": "SKIP",
                "dtype": "torch.float32",
            }
        },
        "skips": {
            "torchcts/opinfo/test_opinfo_forward.py::test_opinfo_forward[future]": {
                "suite": "opinfo",
                "test_kind": "opinfo",
                "status": "SKIP",
                "op": "future",
                "skip_reason": "unavailable_in_pytorch_runtime",
            }
        },
    }

    scorecard, _ = build_report(current_data)

    assert "training        0/0 passed" in scorecard
    assert "training        0/1 passed" not in scorecard
    assert "Ops not run (runtime): 1" in scorecard
    assert "float32" not in scorecard


def test_build_report_surfaces_structured_dtype_skips():
    current_data = {
        "metadata": {
            "device_name": "cpu",
            "hardware_key": "cpu_test_1gb",
            "pytorch_version": torch.__version__,
            "timestamp": "2026-06-16T00:00:00Z",
            "elapsed_sec": 1,
        },
        "results": {},
        "skips": {
            "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[manifest-skip-clean-abs-torch.float64]": {
                "suite": "opinfo",
                "test_kind": "opinfo",
                "capability": "inference",
                "skip_reason": "dtype_not_supported",
                "op": "abs",
                "dtype": "torch.float64",
                "detail": "torch.float64 is declared unsupported in supported_dtypes",
            },
        },
    }

    scorecard, markdown = build_report(current_data, include_skips=True)

    assert "Ops not run (manifest): 1" in scorecard
    assert "Ops not run (selection): 0" in scorecard
    assert "Ops not run (coverage): 0" in scorecard
    assert "Ops not run (runtime): 0" in scorecard
    assert "Ops skipped (unsupported)" not in scorecard
    assert "float64    0/0 ⬚ skip=1" in scorecard
    assert "**dtype_not_supported**: 1 skips" in markdown
    assert "**float64**: 1 skips" in markdown


def test_build_report_separates_selection_and_other_opinfo_not_run_reasons():
    current_data = {
        "metadata": {
            "device_name": "cpu",
            "hardware_key": "cpu_test_1gb",
            "pytorch_version": torch.__version__,
            "timestamp": "2026-06-16T00:00:00Z",
            "elapsed_sec": 1,
        },
        "results": {},
        "skips": {
            "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[abs-torch.float32]": {
                "suite": "opinfo",
                "test_kind": "opinfo",
                "capability": "inference",
                "skip_reason": "semantic_level_gt_requested",
                "op": "abs",
                "dtype": "torch.float32",
            },
            "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[sin-torch.float32]": {
                "suite": "opinfo",
                "test_kind": "opinfo",
                "capability": "inference",
                "skip_reason": "unexpected_future_reason",
                "op": "sin",
                "dtype": "torch.float32",
            },
            "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[cos-torch.float32]": {
                "suite": "opinfo",
                "test_kind": "opinfo",
                "capability": "inference",
                "op": "cos",
                "dtype": "torch.float32",
            },
        },
    }

    scorecard, _ = build_report(current_data, include_skips=True)

    assert "OpInfo ops discovered:     3" in scorecard
    assert "Ops not run (manifest): 0" in scorecard
    assert "Ops not run (selection): 1" in scorecard
    assert "Ops not run (coverage): 0" in scorecard
    assert "Ops not run (runtime): 0" in scorecard
    assert "Ops not run (other):    2" in scorecard
    assert "Ops skipped (unsupported)" not in scorecard


def test_build_report_tracks_split_rng_capabilities():
    current_data = {
        "metadata": {
            "device_name": "cpu",
            "hardware_key": "cpu_test_1gb",
            "pytorch_version": torch.__version__,
            "timestamp": "2026-06-16T00:00:00Z",
            "elapsed_sec": 1,
        },
        "results": {
            "torchcts/rng/test_generator.py::test_rng_reproducibility[42]": {
                "suite": "rng",
                "test_kind": "handwritten",
                "capability": "rng",
                "is_plumbing": False,
                "status": "PASS",
            },
            "torchcts/rng/test_generator.py::test_rng_generator_seeding[123]": {
                "suite": "rng",
                "test_kind": "handwritten",
                "capability": "device_generator",
                "is_plumbing": False,
                "status": "PASS",
            },
        },
        "skips": {
            "torchcts/rng/test_generator.py::test_uniform_distribution_properties": {
                "suite": "rng",
                "test_kind": "handwritten",
                "capability": "rng_distributions",
                "is_plumbing": False,
                "skip_reason": "capability_not_declared",
            }
        },
    }

    scorecard, _ = build_report(current_data)

    assert "rng             1/1 passed" in scorecard
    assert "device_generator 1/1 passed" in scorecard
    assert "rng_distributions SKIPPED" in scorecard


def test_build_report_summarizes_semantic_levels():
    current_data = {
        "metadata": {
            "device_name": "cpu",
            "hardware_key": "cpu_test_1gb",
            "pytorch_version": torch.__version__,
            "timestamp": "2026-06-16T00:00:00Z",
            "elapsed_sec": 1,
            "semantic_level": 3,
        },
        "results": {
            "torchcts/operators/test_binary.py::test_add": {
                "suite": "operators",
                "test_kind": "handwritten",
                "capability": "inference",
                "is_plumbing": False,
                "status": "PASS",
                "semantic_level": 1,
                "requested_level": 3,
            },
            "torchcts/generated/test_out_variants.py::test_add_out": {
                "suite": "generated",
                "test_kind": "handwritten",
                "capability": "inference",
                "is_plumbing": False,
                "status": "FAIL",
                "semantic_level": 3,
                "requested_level": 3,
            },
        },
        "skips": {
            "torchcts/workloads/test_transformer.py::test_transformer": {
                "suite": "workloads",
                "test_kind": "handwritten",
                "capability": "inference",
                "is_plumbing": False,
                "status": "SKIP",
                "semantic_level": 7,
                "requested_level": 3,
                "skip_reason": "semantic_level_gt_requested",
            },
        },
    }

    scorecard, _ = build_report(current_data, include_skips=True)

    assert "SEMANTIC LEVELS" in scorecard
    assert "requested <= 3" in scorecard
    assert "L1  pass=1" in scorecard
    assert "L3  pass=0    fail=1" in scorecard
    assert "L7  pass=0    fail=0    skip=1" in scorecard


def test_build_report_summarizes_exact_semantic_level_selection():
    current_data = {
        "metadata": {
            "device_name": "cpu",
            "hardware_key": "cpu_test_1gb",
            "pytorch_version": torch.__version__,
            "timestamp": "2026-06-16T00:00:00Z",
            "elapsed_sec": 1,
            "semantic_level": 3,
            "semantic_level_selection": {
                "mode": "exact",
                "min_level": 3,
                "max_level": 3,
                "label": "requested == 3",
            },
        },
        "results": {
            "torchcts/generated/test_out_variants.py::test_add_out": {
                "suite": "generated",
                "test_kind": "handwritten",
                "capability": "inference",
                "is_plumbing": False,
                "status": "PASS",
                "semantic_level": 3,
                "requested_level": 3,
            },
        },
        "skips": {},
    }

    scorecard, _ = build_report(current_data, include_skips=True)

    assert "requested == 3" in scorecard
    assert "L3  pass=1" in scorecard


def test_runtime_skip_reason_keeps_backend_unavailable_structured():
    item = SimpleNamespace(nodeid="torchcts/generated/test_oracle_surfaces.py::test_oracle_surface", name="test_oracle_surface")

    skip_reason = harness._runtime_skip_reason(
        "backend_not_available: aten::_philox_uniform requires MPS",
        None,
        item,
    )

    assert skip_reason == "backend_not_available"


def test_runtime_unsupported_pattern_is_classification_only():
    pattern = harness._runtime_unsupported_pattern_match(
        "value cannot be converted to type int64_t without overflow"
    )

    assert pattern == r"value cannot be converted to type .* without overflow"


def test_runtime_unsupported_error_is_not_converted_to_skip():
    class Outcome:
        def __init__(self):
            self.excinfo = (
                RuntimeError,
                RuntimeError("value cannot be converted to type int64_t without overflow"),
                None,
            )
            self.forced = False

        def force_result(self, value):
            self.forced = True

    item = SimpleNamespace()
    outcome = Outcome()
    hook = harness.pytest_runtest_call(item)
    next(hook)

    with pytest.raises(StopIteration):
        hook.send(outcome)

    assert not outcome.forced
    assert item._runtime_unsupported_error["matched_pattern"] == (
        r"value cannot be converted to type .* without overflow"
    )
    assert "value cannot be converted" in item._runtime_unsupported_error["message"]


def test_adversarial_backend_unsupported_error_is_failure():
    from torchcts.stress.test_adversarial import run_adversarial_op

    calls = {"count": 0}

    def op():
        calls["count"] += 1
        if calls["count"] == 1:
            return torch.ones(1)
        raise RuntimeError("not implemented for backend")

    with pytest.raises(pytest.fail.Exception, match="CPU succeeded, but backend raised RuntimeError"):
        run_adversarial_op(op, device="cpu")


def test_adversarial_backend_unsupported_after_cpu_error_is_failure():
    from torchcts.stress.test_adversarial import run_adversarial_op

    calls = {"count": 0}

    def op():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("CPU domain error")
        raise RuntimeError("not supported on backend")

    with pytest.raises(pytest.fail.Exception, match="backend reported unsupported"):
        run_adversarial_op(op, device="cpu")


def test_sparse_backend_unsupported_error_is_failure():
    from torchcts.operators.test_sparse import check_sparse_op

    calls = {"count": 0}

    def op(x):
        calls["count"] += 1
        if calls["count"] == 1:
            return x
        raise RuntimeError("not implemented for backend")

    with pytest.raises(pytest.fail.Exception, match="CPU succeeded, but backend raised RuntimeError"):
        check_sparse_op(op, "cpu", torch.ones(1))


def test_autocast_declared_but_failing_path_is_not_skipped(monkeypatch):
    import torchcts.training.test_mixed_precision as mixed_precision_tests

    def broken_autocast(*args, **kwargs):
        raise RuntimeError("autocast broken")

    monkeypatch.setattr(mixed_precision_tests.torch, "autocast", broken_autocast)

    with pytest.raises(RuntimeError, match="autocast broken"):
        mixed_precision_tests.test_autocast_precisions(torch.float16, "cpu", {})


def _load_release_hygiene_module():
    return runpy.run_path(str(_SOURCE_REPO_ROOT / "scripts" / "check_release_hygiene.py"))


@_SOURCE_CHECKOUT_ONLY
def test_release_hygiene_rejects_package_known_failure_cache(tmp_path):
    hygiene = _load_release_hygiene_module()
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cache_path = tmp_path / "torchcts" / "opinfo_cache" / "known_failures.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("{}\n", encoding="utf-8")

    errors = hygiene["_check_git_paths"](tmp_path)

    assert any("known_failures.json" in error for error in errors)


@_SOURCE_CHECKOUT_ONLY
def test_release_hygiene_rejects_tracked_backend_specific_text(tmp_path):
    hygiene = _load_release_hygiene_module()
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    bad = tmp_path / "bad.txt"
    bad.write_text("backend name: " + ("metal" "core") + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "bad.txt"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    errors = hygiene["_check_forbidden_text"](tmp_path)

    assert any("forbidden text" in error for error in errors)


@_SOURCE_CHECKOUT_ONLY
def test_pyproject_does_not_suppress_backend_fallback_or_pluggy_teardown_warnings():
    pyproject = (_SOURCE_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "The operator .* is not currently supported on the MPS backend" not in pyproject
    assert "PluggyTeardownRaisedWarning" not in pyproject


def _minimal_valid_manifest():
    return {
        "manifest_version": 1,
        "device_name": "auto",
        "capabilities": {
            "inference": True,
        },
    }


@pytest.mark.parametrize("template_name", ["complete", "inference", "minimal", "smoke", "training"])
def test_shipped_manifest_templates_validate(template_name):
    assert cli_module.check_manifest(cli_module.get_template_path(template_name)) == 0


@pytest.mark.parametrize("template_name", ["complete", "inference", "minimal", "smoke", "training"])
def test_shipped_manifest_templates_list_every_known_capability(template_name):
    template = runpy.run_path(cli_module.get_template_path(template_name))["manifest"]

    assert set(template["capabilities"]) == KNOWN_CAPABILITIES


def test_manifest_schema_rejects_stale_generator_capability():
    manifest = _minimal_valid_manifest()
    manifest["capabilities"]["generator"] = True

    result = validate_manifest(manifest)

    assert not result.ok
    assert "Unknown capabilities key 'generator'" in result.errors


def test_manifest_schema_rejects_stale_quantized_capability():
    manifest = _minimal_valid_manifest()
    manifest["capabilities"]["quantized"] = True

    result = validate_manifest(manifest)

    assert not result.ok
    assert "Unknown capabilities key 'quantized'" in result.errors


def test_manifest_schema_validates_semantic_level():
    manifest = _minimal_valid_manifest()
    manifest["semantic_level"] = 8

    result = validate_manifest(manifest)

    assert result.ok

    manifest["semantic_level"] = 0
    result = validate_manifest(manifest)

    assert not result.ok
    assert "semantic_level must be from 1 to 8" in result.errors


def test_manifest_schema_rejects_reference_device_surface():
    manifest = _minimal_valid_manifest()
    manifest["reference_device"] = "cpu"

    result = validate_manifest(manifest)

    assert not result.ok
    assert "Unknown top-level manifest key 'reference_device'" in result.errors


def test_manifest_schema_rejects_invalid_dtype_key():
    manifest = _minimal_valid_manifest()
    manifest["supported_dtypes"] = {"float128": True}

    result = validate_manifest(manifest)

    assert not result.ok
    assert "Invalid supported_dtypes key 'float128'" in result.errors


def test_manifest_schema_rejects_invalid_tolerance_category():
    manifest = _minimal_valid_manifest()
    manifest["tolerance_overrides"] = {
        "not_a_category:float32": {"rtol": 1e-3, "atol": 1e-3},
    }

    result = validate_manifest(manifest)

    assert not result.ok
    assert "Invalid tolerance_overrides category 'not_a_category'" in result.errors


def test_manifest_schema_rejects_invalid_decoder_path():
    manifest = _minimal_valid_manifest()
    manifest["supported_container_formats"] = {"uint8": True}
    manifest["custom_container_decoders"] = {"uint8": "not valid"}

    result = validate_manifest(manifest)

    assert not result.ok
    assert any("Invalid custom_container_decoders.uint8" in error for error in result.errors)


def test_manifest_schema_requires_decoder_when_custom_decode_enabled():
    manifest = _minimal_valid_manifest()
    manifest["capabilities"]["custom_quantized_decode"] = True

    result = validate_manifest(manifest)

    assert not result.ok
    assert (
        "'capabilities.custom_quantized_decode' requires at least one custom_container_decoders entry"
        in result.errors
    )


def test_custom_container_decoder_loader_imports_callable(tmp_path, monkeypatch):
    module_path = tmp_path / "decoder_mod.py"
    module_path.write_text(
        "def decode(packed, scale, zero_point, shape, dtype, device):\n"
        "    return packed.float()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    decoder = load_custom_container_decoder("decoder_mod:decode")

    assert decoder(torch.tensor([1], dtype=torch.uint8), None, None, (1,), torch.float32, "cpu").item() == 1.0


def test_custom_quantized_decoder_case_compares_against_reference_codec():
    def decode_uint8(packed, scale, zero_point, shape, dtype, device):
        return packed.float()

    def compare(actual, expected, category, dtype, **kwargs):
        return compare_tensors(actual, expected, category, dtype, **kwargs)

    _run_custom_decoder_case("uint8", decode_uint8, "cpu", compare)


def test_compare_tensors_checks_cpu_shape_mismatch():
    with pytest.raises(AssertionError, match="Shape mismatch"):
        compare_tensors(torch.ones(2), torch.ones(3), "exact", torch.float32)


def test_compare_tensors_checks_cpu_value_mismatch():
    with pytest.raises(AssertionError):
        compare_tensors(torch.ones(2), torch.zeros(2), "exact", torch.float32)


def test_compare_tensors_reports_non_tensor_values():
    clear_metrics()

    with pytest.raises(AssertionError, match="actual NoneType vs expected Tensor"):
        compare_tensors(None, torch.ones(2), "exact", torch.float32)

    assert get_metrics()["error_msg"] == (
        "Tensor comparison requires tensor values: "
        "actual NoneType vs expected Tensor(shape=(2,), dtype=torch.float32, device=cpu)"
    )
    clear_metrics()


def test_compare_nan_propagation_checks_complex_components():
    actual = torch.tensor([complex(float("nan"), 0.0)], dtype=torch.complex64)
    expected = torch.tensor([complex(0.0, float("nan"))], dtype=torch.complex64)

    with pytest.raises(AssertionError, match="NaN propagation mismatch"):
        compare_nan_propagation(actual, expected)


def test_compare_inf_propagation_checks_complex_components_and_signs():
    compare_inf_propagation(
        torch.tensor([complex(float("inf"), 0.0)], dtype=torch.complex64),
        torch.tensor([complex(float("inf"), 0.0)], dtype=torch.complex64),
    )

    with pytest.raises(AssertionError, match="Inf propagation mismatch"):
        compare_inf_propagation(
            torch.tensor([complex(float("inf"), 0.0)], dtype=torch.complex64),
            torch.tensor([complex(0.0, float("inf"))], dtype=torch.complex64),
        )

    with pytest.raises(AssertionError, match="Inf sign mismatch"):
        compare_inf_propagation(
            torch.tensor([complex(float("inf"), 0.0)], dtype=torch.complex64),
            torch.tensor([complex(float("-inf"), 0.0)], dtype=torch.complex64),
        )


def test_opinfo_has_nan_tier_checks_inf_propagation_too():
    actual = torch.tensor([float("nan"), 1.0], dtype=torch.float32)
    expected = torch.tensor([float("nan"), float("inf")], dtype=torch.float32)

    with pytest.raises(AssertionError, match="Inf propagation mismatch"):
        _opinfo_compare_special_tier(actual, expected, InputCondition.HAS_NAN)


def test_compare_tensors_equal_cpu_tensors_pass():
    compare_tensors(torch.ones(2), torch.ones(2), "exact", torch.float32)


def test_compare_tensors_records_quality_warning_for_usable_pass():
    clear_metrics()

    compare_tensors(
        torch.tensor([1.00003], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        "elementwise",
        torch.float32,
    )

    metrics = get_metrics()
    assert metrics["golden_pass"] is False
    assert metrics["usable_pass"] is True
    assert metrics["quality_warning"]
    clear_metrics()


def _matching_sparse_pair(layout_name, values=None):
    if values is None:
        values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    expected_values = values.clone()

    if layout_name == "coo":
        indices = torch.tensor([[0, 1], [1, 0]])
        actual = torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()
        expected = torch.sparse_coo_tensor(indices.clone(), expected_values, (2, 2)).coalesce()
        return actual, expected
    if layout_name == "csr":
        crow = torch.tensor([0, 1, 2])
        col = torch.tensor([1, 0])
        return (
            torch.sparse_csr_tensor(crow, col, values, size=(2, 2)),
            torch.sparse_csr_tensor(crow.clone(), col.clone(), expected_values, size=(2, 2)),
        )
    if layout_name == "csc":
        ccol = torch.tensor([0, 1, 2])
        row = torch.tensor([1, 0])
        return (
            torch.sparse_csc_tensor(ccol, row, values, size=(2, 2)),
            torch.sparse_csc_tensor(ccol.clone(), row.clone(), expected_values, size=(2, 2)),
        )
    if layout_name == "bsr":
        crow = torch.tensor([0, 1, 2])
        col = torch.tensor([0, 0])
        actual_values = values.reshape(2, 1, 1)
        expected_values = expected_values.reshape(2, 1, 1)
        return (
            torch.sparse_bsr_tensor(crow, col, actual_values, size=(2, 1)),
            torch.sparse_bsr_tensor(crow.clone(), col.clone(), expected_values, size=(2, 1)),
        )
    if layout_name == "bsc":
        ccol = torch.tensor([0, 2])
        row = torch.tensor([0, 1])
        actual_values = values.reshape(2, 1, 1)
        expected_values = expected_values.reshape(2, 1, 1)
        return (
            torch.sparse_bsc_tensor(ccol, row, actual_values, size=(2, 1)),
            torch.sparse_bsc_tensor(ccol.clone(), row.clone(), expected_values, size=(2, 1)),
        )
    raise AssertionError(f"unknown sparse layout fixture: {layout_name}")


def _sparse_structure_mismatch_pair(layout_name):
    actual, expected = _matching_sparse_pair(layout_name)
    if layout_name == "coo":
        expected = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]]),
            torch.tensor([1.0, 2.0]),
            (2, 2),
        ).coalesce()
    elif layout_name == "csr":
        expected = torch.sparse_csr_tensor(
            torch.tensor([0, 2, 2]),
            torch.tensor([0, 1]),
            torch.tensor([1.0, 2.0]),
            size=(2, 2),
        )
    elif layout_name == "csc":
        expected = torch.sparse_csc_tensor(
            torch.tensor([0, 2, 2]),
            torch.tensor([0, 1]),
            torch.tensor([1.0, 2.0]),
            size=(2, 2),
        )
    elif layout_name == "bsr":
        expected = torch.sparse_bsr_tensor(
            torch.tensor([0, 2, 2]),
            torch.tensor([0, 0]),
            torch.ones(2, 1, 1),
            size=(2, 1),
        )
    elif layout_name == "bsc":
        expected = torch.sparse_bsc_tensor(
            torch.tensor([0, 1]),
            torch.tensor([1]),
            torch.ones(1, 1, 1),
            size=(2, 1),
        )
    return actual, expected


@pytest.mark.parametrize("layout_name", ["coo", "csr", "csc", "bsr", "bsc"])
def test_compare_tensors_sparse_layouts_pass(layout_name):
    clear_metrics()
    actual, expected = _matching_sparse_pair(layout_name)

    compare_tensors(actual, expected, "exact", torch.float32)

    metrics = get_metrics()
    assert metrics["passed"] is True
    assert metrics["max_abs_err"] == 0.0
    assert metrics["max_rel_err"] == 0.0
    assert metrics["cosim"] == 1.0
    clear_metrics()


@pytest.mark.parametrize("layout_name", ["coo", "csr", "csc", "bsr", "bsc"])
def test_compare_tensors_sparse_structure_mismatch_records_failure(layout_name):
    clear_metrics()
    actual, expected = _sparse_structure_mismatch_pair(layout_name)

    with pytest.raises(AssertionError, match="Sparse structure mismatch|Shape mismatch"):
        compare_tensors(actual, expected, "exact", torch.float32)

    assert get_metrics()["passed"] is False
    assert get_metrics()["error_msg"]
    clear_metrics()


def test_compare_tensors_sparse_coalescedness_mismatch_fails():
    clear_metrics()
    indices = torch.tensor([[0, 1], [1, 0]])
    actual = torch.sparse_coo_tensor(indices, torch.tensor([1.0, 2.0]), (2, 2))
    expected = actual.coalesce()

    with pytest.raises(AssertionError, match="is_coalesced"):
        compare_tensors(actual, expected, "exact", torch.float32)

    assert get_metrics()["passed"] is False
    clear_metrics()


def test_compare_tensors_sparse_values_use_exact_tolerance():
    actual, expected = _matching_sparse_pair("coo")
    actual = torch.sparse_coo_tensor(
        actual._indices(),
        torch.tensor([1.0, 3.0]),
        actual.shape,
    ).coalesce()

    with pytest.raises(AssertionError):
        compare_tensors(actual, expected, "exact", torch.float32)


def test_compare_tensors_sparse_values_record_quality_warning_for_usable_pass():
    clear_metrics()
    indices = torch.tensor([[0], [0]])
    actual = torch.sparse_coo_tensor(indices, torch.tensor([1.00003]), (1, 1)).coalesce()
    expected = torch.sparse_coo_tensor(indices.clone(), torch.tensor([1.0]), (1, 1)).coalesce()

    compare_tensors(actual, expected, "elementwise", torch.float32)

    metrics = get_metrics()
    assert metrics["golden_pass"] is False
    assert metrics["usable_pass"] is True
    assert metrics["quality_warning"]
    clear_metrics()


def test_compare_tensors_empty_sparse_values_pass_with_neutral_metrics():
    clear_metrics()
    indices = torch.empty((2, 0), dtype=torch.long)
    values = torch.empty((0,), dtype=torch.float32)
    actual = torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()
    expected = torch.sparse_coo_tensor(indices.clone(), values.clone(), (2, 2)).coalesce()

    compare_tensors(actual, expected, "exact", torch.float32)

    metrics = get_metrics()
    assert metrics["max_abs_err"] == 0.0
    assert metrics["max_rel_err"] == 0.0
    assert metrics["cosim"] == 1.0
    clear_metrics()


def test_compare_tensors_sparse_block_value_shape_mismatch_fails():
    actual = torch.sparse_bsr_tensor(
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0]),
        torch.ones(2, 1, 1),
        size=(2, 2),
    )
    expected = torch.sparse_bsr_tensor(
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0]),
        torch.ones(2, 1, 2),
        size=(2, 2),
    )

    with pytest.raises(AssertionError, match="Sparse block value shape mismatch"):
        compare_tensors(actual, expected, "exact", torch.float32)


def test_compare_tensors_sparse_bsc_block_value_shape_mismatch_fails():
    actual = torch.sparse_bsc_tensor(
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0]),
        torch.ones(2, 1, 1),
        size=(2, 2),
    )
    expected = torch.sparse_bsc_tensor(
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0]),
        torch.ones(2, 2, 1),
        size=(2, 2),
    )

    with pytest.raises(AssertionError, match="Sparse block value shape mismatch"):
        compare_tensors(actual, expected, "exact", torch.float32)


def test_compare_tensors_sparse_vs_dense_fails_with_layout_mismatch():
    actual, _expected = _matching_sparse_pair("coo")

    with pytest.raises(AssertionError, match="Sparse layout mismatch"):
        compare_tensors(actual, actual.to_dense(), "exact", torch.float32)


def test_compare_nan_propagation_sparse_values_and_coordinates():
    indices = torch.tensor([[0, 1], [1, 0]])
    actual = torch.sparse_coo_tensor(
        indices,
        torch.tensor([float("nan"), 1.0]),
        (2, 2),
    ).coalesce()
    expected = torch.sparse_coo_tensor(
        indices.clone(),
        torch.tensor([float("nan"), 1.0]),
        (2, 2),
    ).coalesce()
    compare_nan_propagation(actual, expected)

    mismatched_expected = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([float("nan"), 1.0]),
        (2, 2),
    ).coalesce()
    with pytest.raises(AssertionError, match="Sparse structure mismatch"):
        compare_nan_propagation(actual, mismatched_expected)


def test_compare_nan_propagation_compressed_sparse_values():
    actual, expected = _matching_sparse_pair(
        "csr",
        values=torch.tensor([float("nan"), 1.0]),
    )

    compare_nan_propagation(actual, expected)


def test_compare_inf_propagation_complex_sparse_values_and_signs():
    actual, expected = _matching_sparse_pair(
        "csr",
        values=torch.tensor([complex(float("inf"), 0.0), complex(1.0, 0.0)], dtype=torch.complex64),
    )
    compare_inf_propagation(actual, expected)

    _same_actual, mismatched = _matching_sparse_pair(
        "csr",
        values=torch.tensor([complex(float("-inf"), 0.0), complex(1.0, 0.0)], dtype=torch.complex64),
    )
    with pytest.raises(AssertionError, match="Inf sign mismatch"):
        compare_inf_propagation(actual, mismatched)


def test_compare_tensors_unknown_sparse_layout_warns_and_densifies(monkeypatch):
    clear_metrics()
    actual, expected = _matching_sparse_pair("coo")
    monkeypatch.delitem(comparer_module._SPARSE_LAYOUT_HANDLERS, torch.sparse_coo)

    with pytest.warns(UserWarning, match="does not know how to compare"):
        compare_tensors(actual, expected, "exact", torch.float32)

    assert "falling back to dense comparison" in get_metrics()["quality_warning"]
    clear_metrics()


def test_compare_tensors_unknown_sparse_layout_warns_before_shape_failure(monkeypatch):
    clear_metrics()
    actual, _expected = _matching_sparse_pair("coo")
    monkeypatch.delitem(comparer_module._SPARSE_LAYOUT_HANDLERS, torch.sparse_coo)

    with pytest.warns(UserWarning, match="does not know how to compare"):
        with pytest.raises(AssertionError, match="Shape mismatch"):
            compare_tensors(actual, torch.ones(3, 3), "exact", torch.float32)

    assert "falling back to dense comparison" in get_metrics()["quality_warning"]
    clear_metrics()


def test_compare_unknown_sparse_densification_failure_is_actionable(monkeypatch):
    class FailingSparse:
        layout = "torch.sparse_future"

        def detach(self):
            return self

        def cpu(self):
            return self

        def to_dense(self):
            raise RuntimeError("future sparse cannot densify")

    monkeypatch.setattr(comparer_module, "_is_sparse_like_tensor", lambda _t: True)

    with pytest.raises(AssertionError, match="Could not densify sparse layout"):
        comparer_module._densify_sparse_for_unknown_compare(FailingSparse())


def test_manifest_tolerance_override_supports_string_keys():
    tol = get_tolerance(
        "matmul",
        torch.float32,
        manifest_overrides={"matmul:float32": {"rtol": 0.25, "atol": 0.5}},
    )

    assert tol.rtol == 0.25
    assert tol.atol == 0.5


def test_compare_fixture_passes_manifest_tolerance_overrides(monkeypatch):
    monkeypatch.setattr(harness, "_MANIFEST", {
        "tolerance_overrides": {
            ("exact", torch.float32): {"rtol": 1e-3, "atol": 1e-3},
        }
    })
    compare = harness.compare.__wrapped__()

    compare(
        torch.tensor([1.0002], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        "exact",
        torch.float32,
    )


def test_expected_error_helper_fails_when_op_does_not_raise():
    err_in = SimpleNamespace(error_type=RuntimeError)

    with pytest.raises(AssertionError, match="Expected exception RuntimeError not raised"):
        _assert_expected_error(lambda x: x, torch.tensor(1), [], {}, err_in, "fake_op")


def test_expected_error_helper_passes_when_op_raises():
    err_in = SimpleNamespace(error_type=RuntimeError)

    def raise_runtime_error(x):
        raise RuntimeError("expected")

    _assert_expected_error(raise_runtime_error, torch.tensor(1), [], {}, err_in, "fake_op")


def test_init_cli_accepts_smoke_template(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SOURCE_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchcts",
            "init",
            "--template",
            "smoke",
            "--non-interactive",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    manifest_text = (tmp_path / "manifest.py").read_text()
    assert "fast broad sweep" in manifest_text
    assert (tmp_path / "results").is_dir()


def test_project_venv_reexec_is_not_implicit(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_module.os, "getcwd", lambda: str(tmp_path))
    monkeypatch.delenv("TORCHCTS_USE_PROJECT_VENV", raising=False)

    assert cli_module._maybe_reexec_project_venv() is None


def test_project_venv_reexec_opt_in_requires_project_venv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module.os, "getcwd", lambda: str(tmp_path))
    monkeypatch.setenv("TORCHCTS_USE_PROJECT_VENV", "1")
    monkeypatch.delenv("_TORCHCTS_VENV_ACTIVE", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        cli_module._maybe_reexec_project_venv()

    assert exc_info.value.code == 1
    assert "TORCHCTS_USE_PROJECT_VENV=1 was set" in capsys.readouterr().err


def test_runtime_device_count_uses_backend_device_count(monkeypatch):
    class FakeDeviceModule:
        @staticmethod
        def device_count():
            return 3

    monkeypatch.setattr(device_module, "get_device_module", lambda name: FakeDeviceModule)

    assert harness._get_runtime_device_count("privateuseone") == 3


def test_runtime_device_count_clamps_backend_zero_to_one(monkeypatch):
    class FakeDeviceModule:
        @staticmethod
        def device_count():
            return 0

    monkeypatch.setattr(device_module, "get_device_module", lambda name: FakeDeviceModule)

    assert harness._get_runtime_device_count("privateuseone") == 1


def test_detect_backends_prints_probe_status(monkeypatch, capsys):
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)
    if hasattr(device_module.torch, "backends") and hasattr(device_module.torch.backends, "mps"):
        monkeypatch.setattr(device_module.torch.backends.mps, "is_available", lambda: False)
    if hasattr(device_module.torch, "mps") and hasattr(device_module.torch.mps, "is_available"):
        monkeypatch.setattr(device_module.torch.mps, "is_available", lambda: False)
    if hasattr(device_module.torch, "xpu") and hasattr(device_module.torch.xpu, "is_available"):
        monkeypatch.setattr(device_module.torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(device_module, "_scan_for_privateuse1_backends", lambda: [])

    assert device_module.detect_backends() == []

    captured = capsys.readouterr()
    assert "Probing backends...\n" == captured.out


class _CollectionConfig:
    def __init__(self, suite=None, dtype=None, level=None, level_exact=None, level_range=None):
        self.deselected = []
        self.suite = suite
        self.dtype = dtype
        self.level = level
        self.level_exact = level_exact
        self.level_range = level_range

    def getoption(self, option):
        return {
            "--suite": self.suite,
            "--validation": False,
            "--dtype": self.dtype,
            "--level": self.level,
            "--level-exact": self.level_exact,
            "--level-range": self.level_range,
        }.get(option)

    @property
    def hook(self):
        return self

    def pytest_deselected(self, items):
        self.deselected.extend(items)


class _Marker:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _CollectionItem:
    def __init__(self, name, fspath, markers=None, params=None):
        self.name = name
        self.fspath = fspath
        self.nodeid = f"{fspath}::{name}"
        self._markers = markers or {}
        self.added_markers = []
        if params is not None:
            self.callspec = SimpleNamespace(params=params)

    def get_closest_marker(self, name):
        return self._markers.get(name)

    def iter_markers(self, name=None):
        for marker_name, marker in self._markers.items():
            if name is None or marker_name == name:
                yield marker

    def add_marker(self, marker):
        self.added_markers.append(marker)


def test_collection_does_not_call_torch_compile_preflight(monkeypatch):
    def fail_compile(*args, **kwargs):
        raise AssertionError("torch.compile must not run as a collection preflight")

    monkeypatch.setattr(harness, "_COLLECT_ONLY", False)
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "privateuseone")
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"compile": True},
        "supported_dtypes": {},
        "skip_ops": [],
        "device_count": 1,
    })
    monkeypatch.setattr(harness.torch, "compile", fail_compile)

    harness.pytest_collection_modifyitems(None, _CollectionConfig(), [])


def test_dtype_manifest_disposition_handles_true_false_regex_and_missing():
    supported = {
        torch.float32: True,
        torch.float64: False,
        torch.complex64: r"^fft",
    }

    assert opinfo_adapter_module.dtype_manifest_disposition(
        torch.float32, "torch.float32", supported, "add"
    ).allowed

    unsupported = opinfo_adapter_module.dtype_manifest_disposition(
        torch.float64, "torch.float64", supported, "add"
    )
    assert not unsupported.allowed
    assert unsupported.skip_reason == "dtype_not_supported"

    regex_allowed = opinfo_adapter_module.dtype_manifest_disposition(
        torch.complex64, "torch.complex64", supported, "fft.fft"
    )
    assert regex_allowed.allowed

    regex_filtered = opinfo_adapter_module.dtype_manifest_disposition(
        torch.complex64, "torch.complex64", supported, "add"
    )
    assert not regex_filtered.allowed
    assert regex_filtered.skip_reason == "dtype_regex_filtered"

    missing = opinfo_adapter_module.dtype_manifest_disposition(
        torch.bfloat16, "torch.bfloat16", supported, "add"
    )
    assert not missing.allowed
    assert missing.skip_reason == "dtype_not_listed"


def test_collection_dtype_false_is_structured_deselection(monkeypatch):
    opinfo_adapter_module.clear_pending_manifest_skips()
    item = _CollectionItem(
        "test_unary_float_op",
        "torchcts/operators/test_fake.py",
        params={"op_name": "abs", "dtype": torch.float64},
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"inference": True},
        "supported_dtypes": {torch.float64: False},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 6)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 6))

    items = [item]
    config = _CollectionConfig()
    harness.pytest_collection_modifyitems(None, config, items)

    assert items == []
    assert config.deselected == [item]
    assert item.added_markers == []
    assert harness._SESSION_SKIPS[item.nodeid]["skip_reason"] == "dtype_not_supported"


def test_collection_dtype_true_remains_executable(monkeypatch):
    opinfo_adapter_module.clear_pending_manifest_skips()
    item = _CollectionItem(
        "test_unary_float_op",
        "torchcts/operators/test_fake.py",
        params={"op_name": "abs", "dtype": torch.float64},
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"inference": True},
        "supported_dtypes": {torch.float64: True},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 6)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 6))

    items = [item]
    config = _CollectionConfig()
    harness.pytest_collection_modifyitems(None, config, items)

    assert items == [item]
    assert config.deselected == []
    assert harness._SESSION_SKIPS == {}


def test_collection_src_dst_and_gradcheck_dtype_false_are_structured_deselections(monkeypatch):
    opinfo_adapter_module.clear_pending_manifest_skips()
    copy_item = _CollectionItem(
        "test_copy_cast",
        "torchcts/operators/test_fake.py",
        params={"src_dtype": torch.float32, "dst_dtype": torch.float64},
    )
    gradcheck_item = _CollectionItem(
        "test_gradcheck",
        "torchcts/autograd/test_fake.py",
        params={},
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"inference": True},
        "supported_dtypes": {torch.float32: True, torch.float64: False},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 6)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 6))

    items = [copy_item, gradcheck_item]
    config = _CollectionConfig()
    harness.pytest_collection_modifyitems(None, config, items)

    assert items == []
    assert config.deselected == [copy_item, gradcheck_item]
    assert harness._SESSION_SKIPS[copy_item.nodeid]["skip_reason"] == "dtype_not_supported"
    assert harness._SESSION_SKIPS[gradcheck_item.nodeid]["skip_reason"] == "dtype_not_supported"


def test_collection_fixed_dtype_contract_marker_is_structured_deselection(monkeypatch):
    opinfo_adapter_module.clear_pending_manifest_skips()
    item = _CollectionItem(
        "test_fixed_scale_tensor",
        "torchcts/dtypes/test_fake.py",
        markers={"cpu_contract_dtype": _Marker("aten::q_per_channel_scales.out", torch.float64)},
        params={},
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"inference": True},
        "supported_dtypes": {torch.float64: False},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 6)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 6))

    items = [item]
    config = _CollectionConfig()
    harness.pytest_collection_modifyitems(None, config, items)

    assert items == []
    assert config.deselected == [item]
    record = harness._SESSION_SKIPS[item.nodeid]
    assert record["skip_reason"] == "dtype_not_supported"
    assert record["dtype"] == "torch.float64"
    assert record["cpu_contract_fixed_dtype"] is True
    assert record["cpu_contract_fixed_surface"] == "aten::q_per_channel_scales.out"


def test_collection_fixed_dtype_contract_marker_allows_unrecorded_contract_when_manifest_true(monkeypatch):
    opinfo_adapter_module.clear_pending_manifest_skips()
    item = _CollectionItem(
        "test_fixed_scale_tensor",
        "torchcts/dtypes/test_fake.py",
        markers={"cpu_contract_dtype": _Marker("aten::q_per_channel_scales.out", torch.float64)},
        params={},
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"inference": True},
        "supported_dtypes": {torch.float64: True},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 6)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 6))

    items = [item]
    config = _CollectionConfig()
    harness.pytest_collection_modifyitems(None, config, items)

    assert items == [item]
    assert config.deselected == []
    assert harness._SESSION_SKIPS == {}


def test_collection_uses_probed_contract_before_source_only_overload(monkeypatch):
    opinfo_adapter_module.clear_pending_manifest_skips()
    assert harness._candidate_contract_surfaces("aten::add.Tensor") == ("aten::add",)
    assert harness._candidate_contract_surfaces("aten::slice.Tensor") == ("aten::slice",)
    assert harness._candidate_contract_surfaces("aten::_to_copy") == ("aten::to.dtype",)

    item = _CollectionItem(
        "test_add_tensor",
        "torchcts/operators/test_fake.py",
        markers={"covers": _Marker("aten::add.Tensor")},
        params={"dtype": torch.float32},
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"inference": True},
        "supported_dtypes": {torch.float32: True},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 6)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 6))

    items = [item]
    config = _CollectionConfig()
    harness.pytest_collection_modifyitems(None, config, items)

    assert items == [item]
    assert config.deselected == []
    assert harness._SESSION_SKIPS == {}


def test_semantic_level_selection_parses_exact_and_range():
    exact = normalize_level_selection(cli_level_exact=3)
    level_range = normalize_level_selection(cli_level_range="3:5")
    hyphen_range = normalize_level_selection(cli_level_range="4-6")

    assert exact == SemanticLevelSelection("exact", 3, 3)
    assert exact.label == "requested == 3"
    assert exact.contains(3)
    assert not exact.contains(2)

    assert level_range == SemanticLevelSelection("range", 3, 5)
    assert level_range.contains(3)
    assert level_range.contains(5)
    assert not level_range.contains(6)

    assert hyphen_range == SemanticLevelSelection("range", 4, 6)


def test_semantic_level_selection_rejects_conflicting_selectors():
    with pytest.raises(SemanticLevelError, match="Use only one"):
        normalize_level_selection(cli_level=4, cli_level_exact=3)

    with pytest.raises(SemanticLevelError, match="minimum must be <= maximum"):
        normalize_level_selection(cli_level_range="5:3")


def test_collection_filters_exact_semantic_level(monkeypatch):
    level_2 = _CollectionItem(
        "test_l2",
        "torchcts/operators/test_fake.py",
        markers={"semantic_level": _Marker(2)},
    )
    level_3 = _CollectionItem(
        "test_l3",
        "torchcts/operators/test_fake.py",
        markers={"semantic_level": _Marker(3)},
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"inference": True},
        "supported_dtypes": {},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 3)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("exact", 3, 3))

    items = [level_2, level_3]
    config = _CollectionConfig()
    harness.pytest_collection_modifyitems(None, config, items)

    assert items == [level_3]
    assert config.deselected == [level_2]
    assert level_2.added_markers == []
    assert level_2.nodeid in harness._SESSION_SKIPS
    assert level_3.nodeid not in harness._SESSION_SKIPS
    assert harness._SESSION_SKIPS[level_2.nodeid]["skip_reason"] == "semantic_level_out_of_range"
    assert harness._SESSION_SKIPS[level_2.nodeid]["semantic_level_selection"]["mode"] == "exact"


def test_collection_filters_semantic_level_range(monkeypatch):
    level_2 = _CollectionItem(
        "test_l2",
        "torchcts/operators/test_fake.py",
        markers={"semantic_level": _Marker(2)},
    )
    level_4 = _CollectionItem(
        "test_l4",
        "torchcts/operators/test_fake.py",
        markers={"semantic_level": _Marker(4)},
    )
    level_5 = _CollectionItem(
        "test_l5",
        "torchcts/operators/test_fake.py",
        markers={"semantic_level": _Marker(5)},
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"inference": True},
        "supported_dtypes": {},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 4)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("range", 3, 4))

    items = [level_2, level_4, level_5]
    config = _CollectionConfig()
    harness.pytest_collection_modifyitems(None, config, items)

    assert items == [level_4]
    assert config.deselected == [level_2, level_5]
    assert level_2.added_markers == []
    assert level_5.added_markers == []
    assert level_2.nodeid in harness._SESSION_SKIPS
    assert level_4.nodeid not in harness._SESSION_SKIPS
    assert level_5.nodeid in harness._SESSION_SKIPS
    assert harness._SESSION_SKIPS[level_5.nodeid]["skip_reason"] == "semantic_level_out_of_range"
    assert harness._SESSION_SKIPS[level_5.nodeid]["semantic_level_selection"]["label"] == "requested 3-4"


def test_gate_tests_still_respect_required_capabilities(monkeypatch):
    item = _CollectionItem(
        "test_device_gate",
        "torchcts/device_api/test_fake.py",
        markers={
            "gate": _Marker(),
            "requires": _Marker("device_api"),
        },
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"device_api": False},
        "supported_dtypes": {},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)

    items = [item]
    harness.pytest_collection_modifyitems(None, _CollectionConfig(), items)

    assert items == [item]
    assert item.added_markers
    assert item.nodeid in harness._SESSION_SKIPS
    assert harness._SESSION_SKIPS[item.nodeid]["skip_reason"] == "capability_not_declared"


def test_gate_tests_bypass_suite_filter_but_not_capability_filter(monkeypatch):
    item = _CollectionItem(
        "test_registration_gate",
        "torchcts/device_api/test_fake.py",
        markers={
            "gate": _Marker(),
            "requires": _Marker("inference"),
        },
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"inference": True},
        "supported_dtypes": {},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 6)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 6))

    items = [item]
    config = _CollectionConfig(suite="opinfo")
    harness.pytest_collection_modifyitems(None, config, items)

    assert items == [item]
    assert config.deselected == []
    assert harness._SESSION_SKIPS == {}


def test_gate_failure_is_recorded_without_session_exit(monkeypatch):
    item = _CollectionItem(
        "test_registration_gate",
        "torchcts/device_api/test_fake.py",
        markers={"gate": _Marker()},
    )
    monkeypatch.setattr(harness, "_SESSION_RESULTS", {})
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 6)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 6))
    monkeypatch.setattr(harness, "_DEVICE_NAME", "privateuseone")

    class _ExcInfo:
        typename = "AssertionError"
        value = AssertionError("gate failed")
        tb = None

    call = SimpleNamespace(when="call", excinfo=_ExcInfo(), duration=0.01)

    harness.pytest_runtest_makereport(item, call)

    record = harness._SESSION_RESULTS[item.nodeid]
    assert record["status"] == "FAIL"
    assert record["error_type"] == "AssertionError"
    assert "gate failed" in record["error_message"]


def test_runtime_skip_result_record_includes_structured_skip_reason(monkeypatch):
    item = _CollectionItem(
        "test_op_forward[abs-torch.float32]",
        "torchcts/opinfo/test_opinfo_forward.py",
        markers={"semantic_level": _Marker(2)},
    )
    monkeypatch.setattr(harness, "_SESSION_RESULTS", {})
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {
        item.nodeid: {
            "skip_reason": "semantic_level_gt_requested",
            "detail": "semantic_level=2 exceeds requested level 1",
        }
    })
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 1)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 1))
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_ARTIFACT_WRITES_ENABLED", False)

    class _ExcInfo:
        typename = "Skipped"
        value = pytest.skip.Exception("semantic_level=2 exceeds requested level 1")
        tb = None

    call = SimpleNamespace(when="setup", excinfo=_ExcInfo(), duration=0.01)

    harness.pytest_runtest_makereport(item, call)

    record = harness._SESSION_RESULTS[item.nodeid]
    assert record["status"] == "SKIP"
    assert record["skip_reason"] == "semantic_level_gt_requested"
    assert record["skip_detail"] == "semantic_level=2 exceeds requested level 1"
    assert record["semantic_skip_reason"] == "semantic_level_gt_requested"
    assert harness._SESSION_SKIPS[item.nodeid]["skip_reason"] == "semantic_level_gt_requested"


def test_rng_and_device_generator_capabilities_filter_independently(monkeypatch):
    rng_item = _CollectionItem(
        "test_rng_reproducibility",
        "torchcts/rng/test_generator.py",
        markers={"requires": _Marker("rng")},
    )
    device_generator_item = _CollectionItem(
        "test_rng_generator_seeding",
        "torchcts/rng/test_generator.py",
        markers={"requires": _Marker("device_generator")},
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {
            "rng": True,
            "device_generator": False,
            "rng_distributions": False,
        },
        "supported_dtypes": {},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 8)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 8))

    items = [rng_item, device_generator_item]
    harness.pytest_collection_modifyitems(None, _CollectionConfig(), items)

    assert items == [rng_item, device_generator_item]
    assert rng_item.nodeid not in harness._SESSION_SKIPS
    assert device_generator_item.nodeid in harness._SESSION_SKIPS
    assert harness._SESSION_SKIPS[device_generator_item.nodeid]["skip_reason"] == "capability_not_declared"


def test_rng_distribution_capability_filters_independently(monkeypatch):
    item = _CollectionItem(
        "test_uniform_distribution_properties",
        "torchcts/rng/test_generator.py",
        markers={"requires": _Marker("rng_distributions")},
    )
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {
            "rng": True,
            "device_generator": True,
            "rng_distributions": False,
        },
        "supported_dtypes": {},
        "skip_ops": [],
        "device_count": 1,
        "effective_device_count": 1,
    })
    monkeypatch.setattr(harness, "_SESSION_SKIPS", {})
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "_REQUESTED_SEMANTIC_LEVEL", 8)
    monkeypatch.setattr(harness, "_SEMANTIC_LEVEL_SELECTION", SemanticLevelSelection("cumulative", 1, 8))

    items = [item]
    harness.pytest_collection_modifyitems(None, _CollectionConfig(), items)

    assert items == [item]
    assert item.nodeid in harness._SESSION_SKIPS
    assert harness._SESSION_SKIPS[item.nodeid]["skip_reason"] == "capability_not_declared"


def test_device_generator_unsupported_error_is_not_swallowed(monkeypatch):
    def unsupported_generator(*args, **kwargs):
        raise TypeError("device generator unsupported")

    monkeypatch.setattr(rng_tests.torch, "Generator", unsupported_generator)

    with pytest.raises(TypeError, match="device generator unsupported"):
        rng_tests.test_rng_generator_seeding(123, "cpu", {})


def test_resource_limit_cli_overrides_mutate_manifest_limits():
    manifest = {
        "resource_limits": {
            "max_device_memory_mb": 4096,
            "max_tensor_size_mb": 2048,
        }
    }

    limits = harness._apply_resource_limit_overrides(
        manifest,
        cli_max_mem=1024,
        cli_max_tensor=16,
    )

    assert limits["max_device_memory_mb"] == 1024
    assert limits["max_tensor_size_mb"] == 16
    assert manifest["resource_limits"] is limits


def test_check_hardware_alignment_macos_warning(monkeypatch, capsys):
    monkeypatch.setattr(device_module.sys, "platform", "darwin")
    # Mock MPS to be unavailable
    if hasattr(device_module.torch, "backends") and hasattr(device_module.torch.backends, "mps"):
        monkeypatch.setattr(device_module.torch.backends.mps, "is_available", lambda: False)
    if hasattr(device_module.torch, "mps") and hasattr(device_module.torch.mps, "is_available"):
        monkeypatch.setattr(device_module.torch.mps, "is_available", lambda: False)

    assert device_module._check_hardware_alignment() is True
    captured = capsys.readouterr()
    assert "WARNING: Running on macOS, but the installed PyTorch does not have MPS" in captured.err


def test_check_hardware_alignment_windows_cuda_error(monkeypatch, capsys):
    monkeypatch.setattr(device_module.sys, "platform", "win32")
    monkeypatch.setattr(device_module.shutil, "which", lambda cmd: "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else None)
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)

    assert device_module._check_hardware_alignment() is False
    captured = capsys.readouterr()
    assert "ERROR: NVIDIA GPU hardware detected via nvidia-smi" in captured.err


def test_check_hardware_alignment_windows_rocm_error(monkeypatch, capsys):
    monkeypatch.setattr(device_module.sys, "platform", "win32")
    monkeypatch.setattr(device_module.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(device_module, "_detect_amd_gpu_windows", lambda: True)
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)

    assert device_module._check_hardware_alignment() is False
    captured = capsys.readouterr()
    assert "ERROR: AMD GPU hardware detected" in captured.err


def test_check_hardware_alignment_windows_intel_error(monkeypatch, capsys):
    monkeypatch.setattr(device_module.sys, "platform", "win32")
    monkeypatch.setattr(device_module.shutil, "which", lambda cmd: "/usr/bin/xpu-smi" if cmd == "xpu-smi" else None)
    if hasattr(device_module.torch, "xpu"):
        monkeypatch.setattr(device_module.torch.xpu, "is_available", lambda: False)
    else:
        # Mock class/module if xpu isn't present
        class FakeXPU:
            @staticmethod
            def is_available():
                return False
        monkeypatch.setattr(device_module.torch, "xpu", FakeXPU, raising=False)

    assert device_module._check_hardware_alignment() is False
    captured = capsys.readouterr()
    assert "ERROR: Intel GPU hardware detected" in captured.err


def test_coverage_classifier_handles_schema_kinds():
    assert coverage_module.classify_surface("aten::add.Tensor")[0] == "functional_data"
    assert coverage_module.classify_surface("aten::add.out")[0] == "out_variant"
    assert coverage_module.classify_surface("aten::add_.Tensor")[0] == "mutating_or_inplace"
    assert coverage_module.classify_surface("aten::view")[0] == "view_or_alias"
    assert coverage_module.classify_surface("aten::empty.memory_format")[0] == "factory"
    assert coverage_module.classify_surface("aten::result_type.Scalar_Scalar")[0] == "metadata_device"


def test_coverage_marker_parser_reads_exact_and_category(tmp_path):
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        "\n".join([
            "import pytest",
            "",
            "@pytest.mark.covers('aten::add.Tensor')",
            "def test_exact():",
            "    pass",
            "",
            "@pytest.mark.semantic_level(5, reason='advanced layout case')",
            "@pytest.mark.covers('aten::add.out', surface='out_variant')",
            "@pytest.mark.covers_category('layout_storage')",
            "def test_surface_and_category():",
            "    pass",
            "",
            "def test_unmapped():",
            "    pass",
            "",
        ]),
        encoding="utf-8",
    )

    parsed = coverage_module.collect_coverage_markers(tmp_path)

    covered = {item["name"]: item for item in parsed["markers"]}
    assert covered["test_exact"]["covers"] == ["aten::add.Tensor"]
    assert covered["test_surface_and_category"]["covers"] == ["aten::add.out"]
    assert covered["test_surface_and_category"]["categories"] == ["layout_storage"]
    assert covered["test_surface_and_category"]["surfaces"] == {"aten::add.out": "out_variant"}
    assert covered["test_surface_and_category"]["semantic_level"] == 5
    assert covered["test_surface_and_category"]["level_reason"] == "advanced layout case"
    assert parsed["unmapped_tests"] == [
        {
            "nodeid": f"{tmp_path.name}/test_sample.py::test_unmapped",
            "reason": "missing @pytest.mark.covers marker",
            "semantic_level": 4,
            "level_reason": "Custom tests default to broad production coverage.",
            "level_source": "suite_default",
        }
    ]
    assert parsed["errors"] == []


def test_coverage_marker_parser_reads_module_pytestmark(tmp_path):
    test_file = tmp_path / "test_module_markers.py"
    test_file.write_text(
        "\n".join([
            "import pytest",
            "",
            "pytestmark = [",
            "    pytest.mark.covers_category('training_workflow'),",
            "    pytest.mark.requires('training'),",
            "    pytest.mark.semantic_level(level=4, reason='training workflow'),",
            "]",
            "",
            "def test_module_marked():",
            "    pass",
            "",
        ]),
        encoding="utf-8",
    )

    parsed = coverage_module.collect_coverage_markers(tmp_path)

    assert parsed["unmapped_tests"] == []
    assert parsed["markers"][0]["categories"] == ["training_workflow"]
    assert parsed["markers"][0]["capabilities"] == ["training"]
    assert parsed["markers"][0]["semantic_level"] == 4
    assert parsed["markers"][0]["level_reason"] == "training workflow"
    assert parsed["markers"][0]["source"] == "marker"


def test_coverage_marker_parser_reports_invalid_semantic_level(tmp_path):
    test_file = tmp_path / "test_bad_level.py"
    test_file.write_text(
        "\n".join([
            "import pytest",
            "",
            "@pytest.mark.semantic_level(9)",
            "@pytest.mark.covers('aten::add.Tensor')",
            "def test_bad_level():",
            "    pass",
            "",
        ]),
        encoding="utf-8",
    )

    parsed = coverage_module.collect_coverage_markers(tmp_path)

    assert parsed["markers"][0]["covers"] == ["aten::add.Tensor"]
    assert parsed["errors"]
    assert "semantic_level marker must be from 1 to 8" in parsed["errors"][0]


def test_coverage_path_rules_map_builtin_category_suites(tmp_path):
    package_root = tmp_path / "torchcts"
    test_dir = package_root / "strides"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "test_noncontiguous.py"
    test_file.write_text(
        "\n".join([
            "def test_layout_behavior():",
            "    pass",
            "",
        ]),
        encoding="utf-8",
    )

    parsed = coverage_module.collect_coverage_markers(package_root)

    assert parsed["unmapped_tests"] == []
    assert parsed["markers"][0]["categories"] == ["layout_storage", "stride_behavior"]
    assert parsed["markers"][0]["source"] == "path_rule"


def test_coverage_audit_rejects_invalid_exact_marker(monkeypatch):
    inventory = {
        "entries": [
            {
                "name": "aten::add.Tensor",
                "base_name": "add",
                "overload": "Tensor",
                "surface_kind": "functional_data",
                "variant_kind": "functional",
            }
        ]
    }
    marker_data = {
        "markers": [
            {
                "nodeid": "test_sample.py::test_bad_marker",
                "path": "test_sample.py",
                "covers": ["aten::definitely_missing"],
                "categories": [],
            }
        ],
        "unmapped_tests": [],
    }

    monkeypatch.setattr(coverage_module, "build_dispatcher_inventory", lambda: inventory)
    monkeypatch.setattr(coverage_module, "build_opinfo_map", lambda: {"bases": {}, "exact": {}})
    monkeypatch.setattr(coverage_module, "collect_coverage_markers", lambda root=None: marker_data)
    monkeypatch.setattr(
        coverage_module,
        "load_exclusions",
        lambda inventory: {"exclusions": [], "errors": [], "warnings": []},
    )

    audit = coverage_module.build_audit()

    assert audit["errors"] == [
        "test_sample.py::test_bad_marker marks unknown dispatcher surface 'aten::definitely_missing'"
    ]


def test_opinfo_mapping_does_not_cover_alias_view_semantics():
    entry = {
        "name": "aten::view",
        "base_name": "view",
        "overload": "",
        "surface_kind": "view_or_alias",
        "variant_kind": "view",
    }
    opinfo_map = {
        "bases": {"view": ["view"]},
        "exact": {"view": ["view"]},
    }

    covered, matches = coverage_module._opinfo_covers(entry, opinfo_map)

    assert covered is False
    assert matches == ["view"]


def test_opinfo_mapping_keeps_inplace_functional_candidates_without_covering():
    entry = {
        "name": "aten::add_.Tensor",
        "base_name": "add_",
        "overload": "Tensor",
        "surface_kind": "mutating_or_inplace",
        "variant_kind": "inplace",
    }
    opinfo_map = {
        "bases": {"add": ["add"]},
        "exact": {"add": ["add"]},
    }

    covered, matches = coverage_module._opinfo_covers(entry, opinfo_map)

    assert covered is False
    assert matches == ["add"]


def test_opinfo_mapping_covers_curated_dunder_bitwise_aliases():
    class FakeOp:
        name = "bitwise_and"

    assert coverage_module._alias_override_names_for_opinfo(FakeOp()) == {"__and__"}

    entry = {
        "name": "aten::__and__.Tensor",
        "base_name": "__and__",
        "overload": "Tensor",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
    }
    opinfo_map = {
        "bases": {"__and__": ["bitwise_and"]},
        "exact": {},
    }

    covered, matches = coverage_module._opinfo_covers(entry, opinfo_map)

    assert covered is True
    assert matches == ["bitwise_and"]


def test_generated_out_strategy_is_narrow_and_declarative():
    opinfo_map = {
        "bases": {
            "abs": ["abs"],
            "absolute": ["abs"],
            "add": ["add"],
        },
        "exact": {},
        "supports_out": {
            "abs": True,
            "add": True,
        },
    }
    abs_out = {
        "name": "aten::abs.out",
        "base_name": "abs",
        "overload": "out",
        "surface_kind": "out_variant",
        "variant_kind": "out",
        "args": [{"name": "self"}, {"name": "out"}],
        "tensor_returns": [{"name": ""}],
    }
    absolute_alias = {
        **abs_out,
        "name": "aten::absolute.out",
        "base_name": "absolute",
    }
    scalar_overload = {
        **abs_out,
        "name": "aten::add.Scalar_out",
        "base_name": "add",
        "overload": "Scalar_out",
    }

    assert coverage_module._generated_strategy_for_entry(abs_out, opinfo_map) == {
        "strategy": "opinfo_out",
        "opinfo_name": "abs",
    }
    assert coverage_module._generated_strategy_for_entry(absolute_alias, opinfo_map) is None
    assert coverage_module._generated_strategy_for_entry(scalar_overload, opinfo_map) is None


def test_generated_inplace_strategy_is_unary_and_declarative():
    opinfo_map = {
        "bases": {
            "abs": ["abs"],
            "add": ["add"],
        },
        "exact": {},
        "supports_out": {},
    }
    abs_inplace = {
        "name": "aten::abs_",
        "base_name": "abs_",
        "overload": "",
        "surface_kind": "mutating_or_inplace",
        "variant_kind": "inplace",
        "args": [{"name": "self", "tensor": True}],
        "tensor_returns": [{"name": ""}],
    }
    add_tensor_inplace = {
        **abs_inplace,
        "name": "aten::add_.Tensor",
        "base_name": "add_",
        "overload": "Tensor",
        "args": [{"name": "self", "tensor": True}, {"name": "other", "tensor": True}],
    }
    internal_inplace = {
        **abs_inplace,
        "name": "aten::_foreach_abs_",
        "base_name": "_foreach_abs_",
    }

    assert coverage_module._generated_strategy_for_entry(abs_inplace, opinfo_map) == {
        "strategy": "opinfo_inplace_unary",
        "opinfo_name": "abs",
    }
    assert coverage_module._generated_strategy_for_entry(add_tensor_inplace, opinfo_map) == {
        "strategy": "manual_elementwise",
        "family": "add",
    }
    assert coverage_module._generated_strategy_for_entry(internal_inplace, opinfo_map) == {
        "strategy": "manual_foreach",
        "family": "unary",
        "foreach_name": "abs",
    }


def test_generated_view_strategy_requires_exact_alias_surface():
    opinfo_map = {
        "bases": {
            "view": ["view"],
            "view_copy": ["view_copy"],
        },
        "exact": {},
        "supports_out": {},
    }
    view_entry = {
        "name": "aten::view",
        "base_name": "view",
        "overload": "",
        "surface_kind": "view_or_alias",
        "variant_kind": "view",
        "returns": [{"alias": {"is_write": False}}],
        "tensor_returns": [{"name": ""}],
    }
    copy_entry = {
        **view_entry,
        "name": "aten::view_copy",
        "base_name": "view_copy",
        "returns": [{"alias": None}],
    }
    list_return_entry = {
        **view_entry,
        "name": "aten::chunk",
        "base_name": "chunk",
        "tensor_returns": [{"name": ""}, {"name": ""}],
    }

    assert coverage_module._generated_strategy_for_entry(view_entry, opinfo_map) == {
        "strategy": "opinfo_view_alias",
        "opinfo_name": "view",
    }
    assert coverage_module._generated_strategy_for_entry(copy_entry, opinfo_map) == {
        "strategy": "manual_shape",
        "family": "view_copy",
    }
    assert coverage_module._generated_strategy_for_entry(list_return_entry, opinfo_map) == {
        "strategy": "manual_shape",
        "family": "chunk",
    }


def test_generated_factory_strategy_is_explicit_and_safe():
    opinfo_map = {
        "bases": {},
        "exact": {},
        "supports_out": {},
    }
    hann_window = {
        "name": "aten::hann_window",
        "base_name": "hann_window",
        "overload": "",
        "surface_kind": "factory",
        "variant_kind": "factory",
        "tensor_args": [],
        "tensor_returns": [{"name": ""}],
    }
    internal_factory = {
        **hann_window,
        "name": "aten::_efficientzerotensor",
        "base_name": "_efficientzerotensor",
    }
    tensor_arg_factory = {
        **hann_window,
        "name": "aten::hann_window",
        "tensor_args": [{"name": "self"}],
    }

    assert coverage_module._generated_strategy_for_entry(hann_window, opinfo_map) == {
        "strategy": "manual_factory",
        "family": "window",
    }
    assert coverage_module._generated_strategy_for_entry(internal_factory, opinfo_map) == {
        "strategy": "manual_factory",
        "family": "zero_tensor",
    }
    assert coverage_module._generated_strategy_for_entry(tensor_arg_factory, opinfo_map) is None


def test_generated_foreach_strategy_is_safe_functional_subset():
    opinfo_map = {
        "bases": {},
        "exact": {},
        "supports_out": {},
    }
    foreach_abs = {
        "name": "aten::_foreach_abs",
        "base_name": "_foreach_abs",
        "overload": "",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "tensor_returns": [{"name": ""}],
    }
    foreach_add_scalar = {
        **foreach_abs,
        "name": "aten::_foreach_add.Scalar",
        "base_name": "_foreach_add",
        "overload": "Scalar",
    }
    foreach_copy = {
        **foreach_abs,
        "name": "aten::_foreach_copy",
        "base_name": "_foreach_copy",
    }

    assert coverage_module._generated_strategy_for_entry(foreach_abs, opinfo_map) == {
        "strategy": "manual_foreach",
        "family": "unary",
        "foreach_name": "abs",
    }
    assert coverage_module._generated_strategy_for_entry(foreach_add_scalar, opinfo_map) == {
        "strategy": "manual_foreach",
        "family": "binary",
        "foreach_name": "add",
        "overload": "Scalar",
    }
    assert coverage_module._generated_strategy_for_entry(foreach_copy, opinfo_map) == {
        "strategy": "manual_foreach",
        "family": "copy",
        "foreach_name": "copy",
    }


def test_generated_bitwise_strategy_covers_out_and_inplace_variants():
    opinfo_map = {
        "bases": {},
        "exact": {},
        "supports_out": {},
    }
    bitwise_out = {
        "name": "aten::bitwise_and.Tensor_out",
        "base_name": "bitwise_and",
        "overload": "Tensor_out",
        "surface_kind": "out_variant",
        "variant_kind": "out",
        "args": [{"name": "self"}, {"name": "other"}, {"name": "out"}],
        "tensor_returns": [{"name": ""}],
    }
    dunder_shift_out = {
        **bitwise_out,
        "name": "aten::__lshift__.Scalar_out",
        "base_name": "__lshift__",
        "overload": "Scalar_out",
    }
    bitwise_inplace = {
        "name": "aten::bitwise_xor_.Tensor",
        "base_name": "bitwise_xor_",
        "overload": "Tensor",
        "surface_kind": "mutating_or_inplace",
        "variant_kind": "inplace",
        "tensor_returns": [{"name": ""}],
    }

    assert coverage_module._generated_strategy_for_entry(bitwise_out, opinfo_map) == {
        "strategy": "manual_bitwise",
        "family": "bitwise_and",
    }
    assert coverage_module._generated_strategy_for_entry(dunder_shift_out, opinfo_map) == {
        "strategy": "manual_bitwise",
        "family": "bitwise_left_shift",
    }
    assert coverage_module._generated_strategy_for_entry(bitwise_inplace, opinfo_map) == {
        "strategy": "manual_bitwise",
        "family": "bitwise_xor",
    }


def test_generated_special_math_strategy_covers_functional_out_and_inplace_variants():
    opinfo_map = {"bases": {}, "exact": {}, "supports_out": {}}
    functional = {
        "name": "aten::special_airy_ai",
        "base_name": "special_airy_ai",
        "overload": "",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "tensor_returns": [{"name": ""}],
        "args": [{"name": "x", "tensor": True}],
    }
    out = {
        **functional,
        "name": "aten::special_airy_ai.out",
        "overload": "out",
        "surface_kind": "out_variant",
        "variant_kind": "out",
        "args": [{"name": "x", "tensor": True}, {"name": "out", "tensor": True}],
    }
    inplace = {
        **functional,
        "name": "aten::xlogy_.Tensor",
        "base_name": "xlogy_",
        "overload": "Tensor",
        "surface_kind": "mutating_or_inplace",
        "variant_kind": "inplace",
        "args": [{"name": "self", "tensor": True}, {"name": "other", "tensor": True}],
    }

    assert coverage_module._generated_strategy_for_entry(functional, opinfo_map) == {
        "strategy": "manual_special_math",
        "family": "special_airy_ai",
    }
    assert coverage_module._generated_strategy_for_entry(out, opinfo_map) == {
        "strategy": "manual_special_math",
        "family": "special_airy_ai",
    }
    assert coverage_module._generated_strategy_for_entry(inplace, opinfo_map) == {
        "strategy": "manual_special_math",
        "family": "xlogy",
    }


def test_generated_elementwise_strategy_covers_functional_out_and_inplace_variants():
    opinfo_map = {"bases": {}, "exact": {}, "supports_out": {}}
    functional = {
        "name": "aten::_add_relu.Tensor",
        "base_name": "_add_relu",
        "overload": "Tensor",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "tensor_returns": [{"name": ""}],
        "args": [{"name": "self", "type": "Tensor", "tensor": True}, {"name": "other", "type": "Tensor", "tensor": True}],
    }
    out = {
        "name": "aten::div.out_mode",
        "base_name": "div",
        "overload": "out_mode",
        "surface_kind": "out_variant",
        "variant_kind": "out",
        "tensor_returns": [{"name": ""}],
        "args": [
            {"name": "self", "type": "Tensor", "tensor": True},
            {"name": "other", "type": "Tensor", "tensor": True},
            {"name": "rounding_mode", "type": "str?", "tensor": False},
            {"name": "out", "type": "Tensor", "tensor": True},
        ],
    }
    inplace = {
        "name": "aten::add_.Tensor",
        "base_name": "add_",
        "overload": "Tensor",
        "surface_kind": "mutating_or_inplace",
        "variant_kind": "inplace",
        "tensor_returns": [{"name": ""}],
        "args": [{"name": "self", "type": "Tensor", "tensor": True}, {"name": "other", "type": "Tensor", "tensor": True}],
    }

    assert coverage_module._generated_strategy_for_entry(functional, opinfo_map) == {
        "strategy": "manual_elementwise",
        "family": "_add_relu",
    }
    assert coverage_module._generated_strategy_for_entry(out, opinfo_map) == {
        "strategy": "manual_elementwise",
        "family": "div",
    }
    assert coverage_module._generated_strategy_for_entry(inplace, opinfo_map) == {
        "strategy": "manual_elementwise",
        "family": "add",
    }


def test_generated_reduction_strategy_covers_numeric_dim_out_only():
    opinfo_map = {"bases": {}, "exact": {}, "supports_out": {}}
    numeric_dim = {
        "name": "aten::sum.IntList_out",
        "base_name": "sum",
        "overload": "IntList_out",
        "surface_kind": "out_variant",
        "variant_kind": "out",
        "tensor_returns": [{"name": ""}],
        "args": [
            {"name": "self", "type": "Tensor", "tensor": True},
            {"name": "dim", "type": "Optional[List[int]]", "tensor": False},
            {"name": "keepdim", "type": "bool", "tensor": False},
            {"name": "dtype", "type": "Optional[int]", "tensor": False},
            {"name": "out", "type": "Tensor", "tensor": True},
        ],
    }
    dimname = {
        **numeric_dim,
        "name": "aten::sum.DimnameList_out",
        "overload": "DimnameList_out",
        "args": [
            {"name": "self", "type": "Tensor", "tensor": True},
            {"name": "dim", "type": "List[str]", "tensor": False},
            {"name": "keepdim", "type": "bool", "tensor": False},
            {"name": "dtype", "type": "Optional[int]", "tensor": False},
            {"name": "out", "type": "Tensor", "tensor": True},
        ],
    }

    assert coverage_module._generated_strategy_for_entry(numeric_dim, opinfo_map) == {
        "strategy": "manual_reduction",
        "family": "sum",
    }
    assert coverage_module._generated_strategy_for_entry(dimname, opinfo_map) is None


def test_generated_factory_out_strategy_covers_plain_and_names_none_out():
    opinfo_map = {"bases": {}, "exact": {}, "supports_out": {}}
    plain = {
        "name": "aten::linspace.out",
        "base_name": "linspace",
        "overload": "out",
        "surface_kind": "out_variant",
        "variant_kind": "out",
        "tensor_returns": [{"name": ""}],
        "args": [
            {"name": "start", "type": "number", "tensor": False},
            {"name": "end", "type": "number", "tensor": False},
            {"name": "steps", "type": "int", "tensor": False},
            {"name": "out", "type": "Tensor", "tensor": True},
        ],
    }
    named = {
        **plain,
        "name": "aten::ones.names_out",
        "base_name": "ones",
        "overload": "names_out",
        "args": [
            {"name": "size", "type": "List[int]", "tensor": False},
            {"name": "names", "type": "Optional[List[str]]", "tensor": False},
            {"name": "out", "type": "Tensor", "tensor": True},
        ],
    }

    assert coverage_module._generated_strategy_for_entry(plain, opinfo_map) == {
        "strategy": "manual_factory_out",
        "family": "linspace",
    }
    assert coverage_module._generated_strategy_for_entry(named, opinfo_map) == {
        "strategy": "manual_factory_out",
        "family": "ones",
    }


def test_generated_strategy_covers_current_unknown_families():
    opinfo_map = {"bases": {}, "exact": {}, "supports_out": {}}
    grid_fallback = {
        "name": "aten::_grid_sampler_2d_cpu_fallback",
        "base_name": "_grid_sampler_2d_cpu_fallback",
        "overload": "",
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "tensor_returns": [{"name": ""}],
        "args": [
            {"name": "input", "type": "Tensor", "tensor": True},
            {"name": "grid", "type": "Tensor", "tensor": True},
            {"name": "interpolation_mode", "type": "int", "tensor": False},
            {"name": "padding_mode", "type": "int", "tensor": False},
            {"name": "align_corners", "type": "bool", "tensor": False},
        ],
    }
    grid_backward = {
        "name": "aten::grid_sampler_2d_backward",
        "base_name": "grid_sampler_2d_backward",
        "overload": "",
        "surface_kind": "autograd_backward",
        "variant_kind": "functional",
        "tensor_returns": [{"name": ""}, {"name": ""}],
        "args": [
            {"name": "grad_output", "type": "Tensor", "tensor": True},
            {"name": "input", "type": "Tensor", "tensor": True},
            {"name": "grid", "type": "Tensor", "tensor": True},
            {"name": "interpolation_mode", "type": "int", "tensor": False},
            {"name": "padding_mode", "type": "int", "tensor": False},
            {"name": "align_corners", "type": "bool", "tensor": False},
            {"name": "output_mask", "type": "List[bool]", "tensor": False},
        ],
    }
    segment_backward = {
        "name": "aten::_segment_reduce_backward",
        "base_name": "_segment_reduce_backward",
        "overload": "",
        "surface_kind": "autograd_backward",
        "variant_kind": "functional",
        "tensor_returns": [{"name": ""}],
        "args": [
            {"name": "grad", "type": "Tensor", "tensor": True},
            {"name": "output", "type": "Tensor", "tensor": True},
            {"name": "data", "type": "Tensor", "tensor": True},
            {"name": "reduce", "type": "str", "tensor": False},
            {"name": "lengths", "type": "Optional[Tensor]", "tensor": True, "kwarg_only": True},
            {"name": "offsets", "type": "Optional[Tensor]", "tensor": True, "kwarg_only": True},
            {"name": "axis", "type": "int", "tensor": False, "kwarg_only": True},
            {"name": "initial", "type": "Optional[number]", "tensor": False, "kwarg_only": True},
        ],
    }

    assert coverage_module._generated_strategy_for_entry(grid_fallback, opinfo_map) == {
        "strategy": "manual_grid",
        "family": "_grid_sampler_2d_cpu_fallback",
    }
    assert coverage_module._generated_strategy_for_entry(grid_backward, opinfo_map) == {
        "strategy": "manual_grid_backward",
        "family": "grid_sampler_2d_backward",
    }
    assert coverage_module._generated_strategy_for_entry(segment_backward, opinfo_map) == {
        "strategy": "manual_reduction",
        "family": "_segment_reduce_backward",
    }


def test_linalg_matrix_exp_generated_ieee754_tiers_are_clean_only():
    manifest = {"capabilities": {"ieee754": True}, "supported_dtypes": {"float32": True}}
    functional_entry = {"name": "aten::linalg_matrix_exp", "base_name": "linalg_matrix_exp"}
    out_entry = {"name": "aten::linalg_matrix_exp.out", "base_name": "linalg_matrix_exp"}

    assert generated_helpers._manual_linalg_input_conditions(manifest, functional_entry, torch.float32) == [
        InputCondition.CLEAN
    ]
    assert generated_helpers._manual_linalg_input_conditions(manifest, out_entry, torch.float32) == [
        InputCondition.CLEAN
    ]


def test_coverage_exclusion_validation_rejects_unknown_names(tmp_path):
    inventory = {
        "entries": [
            {
                "name": "aten::add.Tensor",
                "base_name": "add",
            }
        ]
    }
    exclusions = tmp_path / "coverage_exclusions.json"
    exclusions.write_text(
        json.dumps({
            "version": 1,
            "exclusions": [
                {
                    "name": "aten::definitely_missing",
                    "match": "exact",
                    "surface": "functional_data",
                    "category": "manual_future_scope",
                    "reason": "Synthetic selftest exclusion.",
                    "owner": "torchcts",
                    "review_after": "2099-01-01",
                }
            ],
        }),
        encoding="utf-8",
    )

    result = coverage_module.load_exclusions(
        inventory,
        package_path=exclusions,
        project_path=tmp_path / "missing.json",
    )

    assert any("not a known dispatcher overload" in error for error in result["errors"])


def test_coverage_exclusion_validation_accepts_runtime_unavailable_names(tmp_path, monkeypatch):
    inventory = {
        "entries": [
            {
                "name": "aten::add.Tensor",
                "base_name": "add",
            }
        ]
    }
    monkeypatch.setattr(
        coverage_module,
        "runtime_unavailable_op_entries",
        lambda **_kwargs: [
            {
                "name": "aten::future",
                "base_name": "future",
            }
        ],
    )
    exclusions = tmp_path / "coverage_exclusions.json"
    exclusions.write_text(
        json.dumps({
            "version": 1,
            "exclusions": [
                {
                    "name": "aten::future",
                    "match": "exact",
                    "surface": "functional_data",
                    "category": "manual_future_scope",
                    "reason": "Synthetic selftest exclusion for a version-known future op.",
                    "owner": "torchcts",
                    "review_after": "2099-01-01",
                }
            ],
        }),
        encoding="utf-8",
    )

    result = coverage_module.load_exclusions(
        inventory,
        package_path=exclusions,
        project_path=tmp_path / "missing.json",
    )

    assert result["errors"] == []


def test_coverage_default_commands_write_default_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert coverage_module.run_inventory_command() == 0
    assert (tmp_path / coverage_module.DEFAULT_INVENTORY_PATH).exists()

    assert coverage_module.run_audit_command() == 0
    assert (tmp_path / coverage_module.DEFAULT_AUDIT_PATH).exists()
    assert (tmp_path / coverage_module.DEFAULT_UNKNOWNS_PATH).exists()
    assert (tmp_path / coverage_module.DEFAULT_UNMAPPED_TESTS_PATH).exists()
    assert (tmp_path / coverage_module.DEFAULT_SUMMARY_PATH).exists()
    assert (tmp_path / coverage_module.DEFAULT_SEMANTIC_LEVELS_PATH).exists()
    assert (tmp_path / coverage_module.DEFAULT_GENERATED_CASES_PATH).exists()

    captured = capsys.readouterr()
    assert "Wrote coverage audit" in captured.out


def test_coverage_evidence_pack_writes_portable_archive(tmp_path, monkeypatch):
    from torchcts.core import evidence_pack as evidence_pack_module

    audit = {
        "metadata": {
            "pytorch_version": "test",
            "generated_at": "2026-07-01T00:00:00Z",
            "total_aten_overloads": 1,
            "surface_counts": {},
            "unknown_count": 0,
        },
        "entries": [
            {
                "name": "aten::_fused_dropout",
                "schema": "aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)",
                "status": "covered_backend_pack",
                "coverage_kind": "backend_pack",
                "oracle": {
                    "oracle_id": "fused_dropout_backend_pack",
                    "backend_gate": "cuda",
                },
                "pending_review": None,
            },
            {
                "name": "aten::cudnn_convolution",
                "schema": (
                    "aten::cudnn_convolution(Tensor self, Tensor weight, SymInt[] padding, "
                    "SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, "
                    "bool deterministic, bool allow_tf32) -> Tensor"
                ),
                "status": "pending_backend_pack",
                "coverage_kind": "backend_pack",
                "surface_kind": "functional_data",
                "variant_kind": "functional",
                "oracle": None,
                "pending_review": {
                    "blocker_type": "needs_backend_pack",
                    "backend_gate": "cuda",
                    "required_closure": "implement_backend_gated_runner",
                },
                "exclusion": {
                    "name": "^aten::.*cudnn.*$",
                    "match": "regex",
                    "category": "backend_specific_internal",
                },
            }
        ],
        "warnings": [],
        "errors": [],
    }

    monkeypatch.setattr(evidence_pack_module, "_utc_stamp", lambda: "20260701T000000Z")
    monkeypatch.setattr(evidence_pack_module.socket, "gethostname", lambda: "unit-host")
    monkeypatch.setattr(evidence_pack_module.coverage, "build_audit", lambda: audit)
    monkeypatch.setattr(
        evidence_pack_module.coverage,
        "build_pending_review_artifact",
        lambda _audit: {"metadata": {"record_count": 0}, "records": []},
    )
    monkeypatch.setattr(evidence_pack_module.coverage, "render_summary_markdown", lambda _audit: "# Summary\n")
    monkeypatch.setattr(evidence_pack_module.coverage, "render_pending_review_markdown", lambda _audit: "# Pending\n")

    result = evidence_pack_module.build_evidence_pack(
        device="cuda",
        output_dir=tmp_path,
        run_oracles=False,
    )

    archive_path = Path(result["archive"])
    assert archive_path.exists()
    assert Path(result["staging_dir"]).exists()

    prefix = "torchcts-evidence-unit-host-cuda-20260701T000000Z"
    with tarfile.open(archive_path) as archive:
        names = set(archive.getnames())
        assert f"{prefix}/environment.json" in names
        assert f"{prefix}/coverage/audit.json" in names
        assert f"{prefix}/coverage/pending_review.json" in names
        assert f"{prefix}/oracles/backend_pack_evidence.json" in names
        evidence_file = archive.extractfile(f"{prefix}/oracles/backend_pack_evidence.json")
        assert evidence_file is not None
        evidence = json.loads(evidence_file.read().decode("utf-8"))

    assert evidence["metadata"]["record_count"] == 2
    by_surface = {record["surface"]: record for record in evidence["records"]}
    assert by_surface["aten::_fused_dropout"]["oracle_result"]["skipped"] is True
    assert by_surface["aten::cudnn_convolution"]["oracle"] is None
    assert by_surface["aten::cudnn_convolution"]["backend_gate"] == "cuda"
    assert by_surface["aten::cudnn_convolution"]["oracle_result"]["reason"] == "no oracle spec registered"


def test_coverage_evidence_pack_selects_explicit_backend_gates():
    from torchcts.core import evidence_pack as evidence_pack_module

    audit = {
        "entries": [
            {
                "name": "aten::cuda_only",
                "coverage_kind": "backend_pack",
                "pending_review": {"backend_gate": "cuda"},
            },
            {
                "name": "aten::rocm_only",
                "coverage_kind": "backend_pack",
                "pending_review": {"backend_gate": "rocm"},
            },
            {
                "name": "aten::fbgemm_only",
                "coverage_kind": "backend_pack",
                "pending_review": {"backend_gate": "fbgemm"},
            },
            {
                "name": "aten::cpu_build_only",
                "coverage_kind": "backend_pack",
                "pending_review": {"backend_gate": "cpu_build"},
            },
            {
                "name": "aten::any_backend",
                "coverage_kind": "backend_pack",
                "pending_review": {"backend_gate": "any"},
            },
            {
                "name": "aten::generated",
                "coverage_kind": "generated",
                "pending_review": {"backend_gate": "cuda"},
            },
        ]
    }

    default_cuda = evidence_pack_module._select_targets(audit, "cuda")
    assert [target["surface"] for target in default_cuda] == [
        "aten::any_backend",
        "aten::cuda_only",
    ]

    explicit = evidence_pack_module._select_targets(
        audit,
        "cuda",
        backend_gates=["rocm,fbgemm+cpu_build"],
    )
    assert [target["surface"] for target in explicit] == [
        "aten::any_backend",
        "aten::cpu_build_only",
        "aten::fbgemm_only",
        "aten::rocm_only",
    ]

    all_gates = evidence_pack_module._select_targets(audit, "cpu", backend_gates=["all"])
    assert [target["surface"] for target in all_gates] == [
        "aten::any_backend",
        "aten::cpu_build_only",
        "aten::cuda_only",
        "aten::fbgemm_only",
        "aten::rocm_only",
    ]


def test_cli_routes_coverage_evidence_pack(monkeypatch):
    from torchcts.core import evidence_pack as evidence_pack_module

    calls = []

    def fake_run_evidence_pack_command(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(evidence_pack_module, "run_evidence_pack_command", fake_run_evidence_pack_command)
    args = SimpleNamespace(
        coverage_command="evidence-pack",
        device="cuda",
        output_dir="out",
        surface=["aten::_fused_dropout"],
        backend_gate=["cuda+rocm"],
        no_run_oracles=True,
        include_all_backend_packs=True,
        strict_unknowns=False,
    )

    assert cli_module.run_coverage_command(args) == 0
    assert calls == [{
        "device": "cuda",
        "output_dir": "out",
        "surfaces": ["aten::_fused_dropout"],
        "backend_gates": ["cuda+rocm"],
        "run_oracles": False,
        "include_all_backend_packs": True,
    }]


def test_coverage_materializes_generated_cases(tmp_path, monkeypatch):
    generated_json = tmp_path / "results" / "coverage" / "generated_cases.json"
    generated_module = tmp_path / "torchcts" / "generated" / "generated_cases.py"
    monkeypatch.setattr(coverage_module, "DEFAULT_GENERATED_CASES_PATH", generated_json)
    monkeypatch.setattr(coverage_module, "DEFAULT_GENERATED_CASES_MODULE_PATH", generated_module)

    audit = {
        "metadata": {
            "pytorch_version": "test",
            "generated_at": "2026-06-24T00:00:00Z",
        },
        "entries": [
            {
                "name": "aten::add.out",
                "base_name": "add",
                "overload": "out",
                "schema": "aten::add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
                "surface_kind": "out_variant",
                "variant_kind": "out",
                "status": "covered_generated",
                "semantic_level": 3,
                "semantic_levels": [3],
                "min_semantic_level": 3,
                "max_semantic_level": 3,
                "level_reason": "out= variants include additional mutation/identity semantics.",
                "level_source": "surface_default",
                "generated": {"strategy": {"strategy": "opinfo_out", "opinfo_name": "add"}},
            },
            {
                "name": "aten::add.Tensor",
                "base_name": "add",
                "overload": "Tensor",
                "schema": "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
                "surface_kind": "functional_data",
                "variant_kind": "functional",
                "status": "covered_handwritten",
                "generated": {"strategy": None},
            },
        ],
    }

    manifest = coverage_module.write_generated_cases_artifacts(audit, write_module=True)
    loaded = coverage_module._load_generated_cases_manifest()

    assert generated_json.exists()
    assert generated_module.exists()
    assert manifest["metadata"]["case_count"] == 1
    assert loaded["cases_by_surface"]["out_variant"][0]["name"] == "aten::add.out"
    assert loaded["cases_by_surface"]["out_variant"][0]["semantic_level"] == 3
    assert loaded["cases_by_surface"]["out_variant"][0]["semantic_levels"] == [3]
    assert coverage_module.generated_entries_for("out_variant") == loaded["cases_by_surface"]["out_variant"]


def test_coverage_check_rebuilds_and_ignores_stale_audit(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    audit_path = tmp_path / coverage_module.DEFAULT_AUDIT_PATH
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "unknown_count": 1,
                    "generated_case_depth": {},
                    "semantic_level_counts": {"2": 1},
                },
                "entries": [
                    {
                        "name": "aten::synthetic_unknown",
                        "surface_kind": "functional_data",
                        "status": "unknown",
                        "has_tensor_args": True,
                        "has_tensor_returns": False,
                        "semantic_level": 2,
                        "semantic_levels": [2],
                        "min_semantic_level": 2,
                        "max_semantic_level": 2,
                        "level_reason": "Functional data surface default.",
                        "level_source": "surface_default",
                    }
                ],
            }
        )
    )
    fresh_audit = {
        "metadata": {
            "pytorch_version": torch.__version__,
            "generated_at": "2026-06-28T00:00:00Z",
            "total_aten_overloads": 0,
            "surface_counts": {},
            "status_counts": {},
            "semantic_level_counts": {},
            "semantic_level_descriptions": {},
            "semantic_level_status_counts": {},
            "semantic_level_surface_counts": {},
            "generated_case_depth": {},
            "unknown_count": 0,
        },
        "entries": [],
        "coverage_markers": [],
        "unmapped_tests": [],
        "errors": [],
        "warnings": [],
    }
    monkeypatch.setattr(coverage_module, "build_audit", lambda root=None: fresh_audit)

    assert coverage_module.run_check_command() == 0

    captured = capsys.readouterr()
    assert "unknown tensor-touching ATen surfaces" not in captured.out
    assert "Coverage audit is internally consistent." in captured.out

    assert coverage_module.run_check_command(strict_unknowns=True) == 0
    captured = capsys.readouterr()
    assert "strict_unknowns enabled" not in captured.out


def test_generated_entries_and_skip_reasons(capsys):
    audit = {
        "entries": [
            {"name": "aten::add.out", "surface_kind": "out_variant", "status": "unknown", "semantic_level": 3},
            {"name": "aten::empty.out", "surface_kind": "out_variant", "status": "excluded"},
            {"name": "aten::view", "surface_kind": "view_or_alias", "status": "covered_handwritten"},
        ]
    }

    entries = coverage_module.generated_entries_for("out_variant", audit=audit)

    assert [entry["name"] for entry in entries] == ["aten::add.out", "aten::empty.out"]
    assert generated_helpers.generated_case_id(entries[0]) == "add.out[L3]"

    from torchcts.generated.coverage_helpers import skip_until_strategy_exists

    with pytest.raises(pytest.skip.Exception, match="coverage_unknown"):
        skip_until_strategy_exists(entries[0], "out_variant")
    with pytest.raises(pytest.skip.Exception, match="coverage_excluded"):
        skip_until_strategy_exists(entries[1], "out_variant")


def test_coverage_audit_synthesizes_runtime_unavailable_metadata_entries(monkeypatch):
    live_entry = {
        "name": "aten::sample.Tensor",
        "base_name": "sample",
        "overload": "Tensor",
        "schema": "aten::sample.Tensor(Tensor self) -> Tensor",
        "args": [{"name": "self", "type": "Tensor", "tensor": True}],
        "returns": [{"name": "", "type": "Tensor", "tensor": True}],
        "tensor_args": [{"name": "self", "type": "Tensor", "tensor": True}],
        "tensor_returns": [{"name": "", "type": "Tensor", "tensor": True}],
        "has_tensor_args": True,
        "has_tensor_returns": True,
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "dispatch": {"CPU": True},
    }
    unavailable_schema = {
        "min": "2.9.0",
        "max": None,
        "schema_hash": "test",
        "schema": "aten::future(Tensor self) -> Tensor",
        "args": [{"name": "self", "type": "Tensor", "tensor": True}],
        "returns": [{"name": "", "type": "Tensor", "tensor": True}],
        "surface_kind": "functional_data",
        "variant_kind": "functional",
        "base_name": "future",
        "overload": "",
    }
    metadata = {
        "version": 2,
        "metadata": {"collected_versions": ["2.8.0", "2.9.0"]},
        "ops": {
            "aten::sample.Tensor": {
                "introduced": "2.7.0",
                "removed": None,
                "versions_seen": ["2.8.0"],
                "versions_missing": ["2.9.0"],
                "schema_ranges": [
                    {
                        **unavailable_schema,
                        "schema": "aten::sample.Tensor(Tensor self) -> Tensor",
                        "base_name": "sample",
                        "overload": "Tensor",
                        "min": "2.7.0",
                    }
                ],
            },
            "aten::future": {
                "introduced": "2.9.0",
                "removed": None,
                "versions_seen": ["2.9.0"],
                "versions_missing": ["2.8.0"],
                "schema_ranges": [unavailable_schema],
            },
        },
    }
    monkeypatch.setattr(coverage_module.torch, "__version__", "2.8.0")
    monkeypatch.setattr(
        coverage_module,
        "build_dispatcher_inventory",
        lambda: {
            "metadata": {"pytorch_version": "2.8.0", "total_aten_overloads": 1},
            "entries": [live_entry],
        },
    )
    monkeypatch.setattr(coverage_module, "build_opinfo_map", lambda: {"bases": {}, "exact": {}})
    monkeypatch.setattr(coverage_module, "load_exclusions", lambda inventory: {"errors": [], "warnings": [], "exclusions": []})
    monkeypatch.setattr(
        coverage_module,
        "collect_coverage_markers",
        lambda root=None: {
            "markers": [
                {
                    "nodeid": "torchcts/generated/test_future.py::test_future",
                    "path": "torchcts/generated/test_future.py",
                    "covers": ["aten::future"],
                    "categories": [],
                    "generated": True,
                }
            ],
            "errors": [],
            "warnings": [],
            "unmapped_tests": [],
        },
    )
    monkeypatch.setattr(op_metadata_module, "load_op_metadata", lambda: metadata)
    monkeypatch.setattr(coverage_module, "runtime_unavailable_op_entries", op_metadata_module.runtime_unavailable_op_entries)

    audit = coverage_module.build_audit()

    by_name = {entry["name"]: entry for entry in audit["entries"]}
    assert by_name["aten::future"]["status"] == "unavailable_in_pytorch_runtime"
    assert by_name["aten::future"]["coverage_kind"] == "runtime_unavailable"
    assert audit["metadata"]["status_counts"]["unavailable_in_pytorch_runtime"] == 1
    assert audit["metadata"]["unknown_count"] == 1
    assert audit["errors"] == []

    summary = coverage_module.render_summary_markdown(audit)
    assert "Runtime-unavailable overloads: 1" in summary
    assert "`unavailable_in_pytorch_runtime`: 1" in summary

    generated = coverage_module.build_generated_cases_manifest(audit)
    assert generated["cases_by_surface"]["functional_data"][0]["status"] == "unavailable_in_pytorch_runtime"

    with pytest.raises(pytest.skip.Exception, match="unavailable_in_pytorch_runtime"):
        generated_helpers.skip_until_strategy_exists(by_name["aten::future"], "functional_data")


def test_cli_dtype_filter_normalizes_and_deduplicates():
    effective, labels = harness._normalize_cli_dtype_filter([
        "float32",
        "torch.float32",
        "complex128",
    ])

    assert list(effective) == [torch.float32, torch.complex128]
    assert all(value is True for value in effective.values())
    assert labels == ["torch.float32", "torch.complex128"]


def test_cli_dtype_filter_rejects_unknown_dtype():
    with pytest.raises(pytest.UsageError, match="Unknown --dtype value"):
        harness._normalize_cli_dtype_filter(["definitely_not_a_dtype"])


def test_cli_dtype_filter_replaces_effective_manifest_dtypes():
    manifest = {
        "supported_dtypes": {
            torch.float32: True,
            torch.float64: False,
            "torch.complex128": "_foreach",
        }
    }

    labels = harness._apply_cli_dtype_filter(manifest, ["float64", "torch.complex128"])

    assert labels == ["torch.float64", "torch.complex128"]
    assert manifest["supported_dtypes"] == {
        torch.float64: True,
        torch.complex128: True,
    }
    assert manifest["dtype_filter"] == ["torch.float64", "torch.complex128"]
    assert manifest["_declared_supported_dtypes"] == [
        {"dtype": "torch.float32", "value": True},
        {"dtype": "torch.float64", "value": False},
        {"dtype": "torch.complex128", "value": "_foreach"},
    ]


def test_generated_foreach_collection_is_dtype_parametrized(monkeypatch):
    monkeypatch.setattr(
        generated_helpers,
        "_generated_clean_cpu_contract_allows",
        lambda entry, dtype, manifest: True,
    )
    monkeypatch.setattr(
        generated_helpers,
        "contract_disposition",
        lambda op_name, dtype: dtype_contracts.ContractDisposition(True, "cpu_supported"),
    )
    entry = {
        "name": "aten::_foreach_add.List",
        "base_name": "_foreach_add",
        "overload": "List",
        "status": "covered_generated",
        "surface_kind": "functional_data",
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
    manifest = {
        "supported_dtypes": {
            torch.float32: True,
            torch.float64: True,
            torch.float16: False,
            torch.complex128: "_foreach_add",
            torch.bfloat16: "definitely_not_foreach",
        }
    }

    cases = generated_helpers.generated_foreach_dtype_cases([entry], manifest)
    ids = [generated_helpers.generated_foreach_case_id(case) for case in cases]

    assert [dtype for _entry, dtype in cases] == [torch.complex128, torch.float32, torch.float64]
    assert ids == [
        "_foreach_add.List[L4]-torch.complex128",
        "_foreach_add.List[L4]-torch.float32",
        "_foreach_add.List[L4]-torch.float64",
    ]


def test_generated_foreach_collection_honors_cli_dtype_override(monkeypatch):
    monkeypatch.setattr(
        generated_helpers,
        "_generated_clean_cpu_contract_allows",
        lambda entry, dtype, manifest: True,
    )
    monkeypatch.setattr(
        generated_helpers,
        "contract_disposition",
        lambda op_name, dtype: dtype_contracts.ContractDisposition(True, "cpu_supported"),
    )
    entry = {
        "name": "aten::_foreach_add.List",
        "base_name": "_foreach_add",
        "overload": "List",
        "status": "covered_generated",
        "surface_kind": "functional_data",
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

    manifest = {"supported_dtypes": {torch.float32: True}}
    harness._apply_cli_dtype_filter(manifest, ["float64", "torch.complex128"])

    cases = generated_helpers.generated_foreach_dtype_cases([entry], manifest)

    assert [dtype for _entry, dtype in cases] == [torch.complex128, torch.float64]


def test_generated_manual_shape_loop_uses_effective_manifest_dtypes(monkeypatch):
    entry = {
        "name": "aten::squeeze",
        "base_name": "squeeze",
        "status": "covered_generated",
        "surface_kind": "view_or_alias",
        "semantic_level": 3,
        "generated": {"strategy": {"strategy": "manual_shape", "family": "squeeze"}},
    }
    manifest = {
        "supported_dtypes": {
            torch.float32: True,
            torch.complex128: True,
        }
    }
    harness._apply_cli_dtype_filter(manifest, ["torch.float32"])
    seen = []

    monkeypatch.setattr(generated_helpers, "_dispatcher_callable", lambda _entry: object())
    monkeypatch.setattr(
        generated_helpers,
        "_manual_shape_input_conditions",
        lambda _manifest, _entry, _dtype: [InputCondition.CLEAN],
    )
    monkeypatch.setattr(
        generated_helpers,
        "_run_manual_shape_case",
        lambda _entry, _callable, dtype, _condition, _device, _compare, _manifest: seen.append(dtype) or True,
    )

    generated_helpers.run_manual_shape_strategy(entry, "cpu", compare=None, manifest=manifest)

    assert seen == [torch.float32]


def test_generated_functional_variants_do_not_collect_manual_foreach(monkeypatch):
    import torchcts.generated.test_functional_variants as functional_variants

    foreach_entry = {
        "name": "aten::_foreach_add.List",
        "status": "covered_generated",
        "generated": {"strategy": {"strategy": "manual_foreach"}},
    }
    elementwise_entry = {
        "name": "aten::add.Tensor",
        "status": "covered_generated",
        "generated": {"strategy": {"strategy": "manual_elementwise"}},
    }
    monkeypatch.setattr(
        functional_variants,
        "generated_cases",
        lambda surface_kind: [foreach_entry, elementwise_entry],
    )

    assert functional_variants._functional_cases() == [elementwise_entry]


def test_coverage_audit_uses_oracle_status_and_metadata():
    audit = coverage_module.build_audit()
    by_name = {entry["name"]: entry for entry in audit["entries"]}

    sobol = by_name["aten::_sobol_engine_draw"]
    quantized_legacy = by_name["aten::quantized_lstm.input_legacy"]
    empty_quantized = by_name["aten::_empty_affine_quantized"]
    dynamic_int4 = by_name["aten::_dyn_quant_matmul_4bit"]

    assert sobol["status"] == "covered_oracle"
    assert sobol["coverage_kind"] == "oracle"
    assert sobol["oracle"]["oracle_id"] == "sobol_engine_state"

    assert empty_quantized["status"] == "covered_oracle"
    assert empty_quantized["oracle"]["oracle_id"] == "quantized_affine_allocation"

    assert quantized_legacy["status"] == "excluded_deprecated_or_removed"
    assert dynamic_int4["status"] == "covered_oracle"
    assert dynamic_int4["oracle"]["oracle_id"] == "dynamic_int4_pack_matmul_value_oracle"
    assert audit["metadata"]["status_counts"]["covered_oracle"] >= 13
    assert audit["metadata"]["status_counts"].get("pending_oracle", 0) == 0


def test_dynamic_int4_reference_covers_nibbles_groups_bias_and_shape_errors():
    from torchcts.core.reference_oracles import (
        dynamic_int4_matmul_reference,
        unpack_dynamic_int4_weight_bytes,
    )

    weights = torch.tensor([[0x89, 0x8F], [0x98, 0x88]], dtype=torch.uint8)
    unpacked = unpack_dynamic_int4_weight_bytes(weights, in_features=3, out_features=2)
    assert unpacked.tolist() == [[9.0, 8.0, 15.0], [8.0, 9.0, 8.0]]

    grouped_weights = torch.tensor([[0x99, 0x99, 0xFF, 0xFF]], dtype=torch.uint8)
    scales = torch.tensor([[2.0, 0.5]], dtype=torch.float32)
    bias = torch.tensor([1.25], dtype=torch.float32)
    input_tensor = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    actual = dynamic_int4_matmul_reference(
        input_tensor,
        grouped_weights,
        scales,
        bias,
        block_size=4,
        in_features=8,
        out_features=1,
    )
    logical_weights = torch.tensor([[2.0, 2.0, 2.0, 2.0, 3.5, 3.5, 3.5, 3.5]], dtype=torch.float32)
    expected = input_tensor @ logical_weights.T + bias
    assert torch.equal(actual, expected)

    with pytest.raises(ValueError, match="exactly"):
        unpack_dynamic_int4_weight_bytes(torch.tensor([0x88], dtype=torch.uint8), in_features=4, out_features=1)
    with pytest.raises(ValueError, match="must divide"):
        dynamic_int4_matmul_reference(
            input_tensor,
            grouped_weights,
            scales,
            None,
            block_size=3,
            in_features=8,
            out_features=1,
        )


def test_coverage_audit_has_no_current_unknown_closure_set():
    formerly_unknown = {
        "aten::_grid_sampler_2d_cpu_fallback",
        "aten::_grid_sampler_2d_cpu_fallback.out",
        "aten::_grid_sampler_2d_cpu_fallback_backward",
        "aten::_segment_reduce_backward",
        "aten::_segment_reduce_backward.out",
        "aten::empty.names_out",
        "aten::full.names_out",
        "aten::grid_sampler_2d_backward",
        "aten::grid_sampler_2d_backward.out",
        "aten::grid_sampler_3d_backward",
        "aten::grid_sampler_3d_backward.out",
        "aten::ones.names_out",
        "aten::zeros.names_out",
    }

    audit = coverage_module.build_audit()
    by_name = {entry["name"]: entry for entry in audit["entries"]}

    assert sorted(formerly_unknown - set(by_name)) == []
    assert sorted(name for name in formerly_unknown if by_name[name]["status"] == "unknown") == []


def test_coverage_audit_publishes_pending_review_metadata():
    audit = coverage_module.build_audit()
    by_name = {entry["name"]: entry for entry in audit["entries"]}
    review = coverage_module.build_pending_review_artifact(audit)

    flash = by_name["aten::_scaled_dot_product_flash_attention"]
    raw_quantized_flash = by_name["aten::_flash_attention_forward.quantized"]
    quantized_flash = by_name["aten::_scaled_dot_product_flash_attention.quantized"]
    pin_memory = by_name["aten::_pin_memory"]
    fused_dropout = by_name["aten::_fused_dropout"]
    nested_softmax = by_name["aten::_nested_tensor_softmax_with_shape"]

    assert flash["status"] == "covered_property"
    assert flash["oracle"]["oracle_id"] == "privateuse1_attention_public_sdpa"

    if raw_quantized_flash["status"] == "unavailable_in_pytorch_runtime":
        assert raw_quantized_flash["coverage_kind"] == "runtime_unavailable"
    else:
        assert raw_quantized_flash["status"] == "covered_property"
        assert raw_quantized_flash["oracle"]["oracle_id"] == "quantized_flash_attention_public_sdpa"
        assert raw_quantized_flash["oracle"]["backend_gate"] == "any"

    if quantized_flash["status"] == "unavailable_in_pytorch_runtime":
        assert quantized_flash["coverage_kind"] == "runtime_unavailable"
    else:
        assert quantized_flash["status"] == "covered_property"
        assert quantized_flash["oracle"]["oracle_id"] == "quantized_flash_attention_public_sdpa"
        assert quantized_flash["oracle"]["backend_gate"] == "any"

    assert pin_memory["status"] == "covered_property"
    assert pin_memory["oracle"]["oracle_id"] == "privateuse1_pin_memory_noop"

    assert fused_dropout["status"] == "pending_backend_pack"
    assert fused_dropout["pending_review"]["blocker_type"] == "needs_backend_pack"
    assert fused_dropout["pending_review"]["backend_gate"] == "cuda"

    assert nested_softmax["status"] == "pending_property"
    assert nested_softmax["pending_review"]["blocker_type"] == "needs_valid_internal_inputs"

    assert review["metadata"]["pending_or_excluded_count"] > 0
    assert review["metadata"]["blocker_counts"]["needs_public_proxy_proof"] >= 1
    assert coverage_module._validate_audit_consistency(audit) == []


def test_oracle_runner_executes_cpu_oracle_surfaces():
    from torchcts.core.oracles import run_oracle_for_surface

    run_oracle_for_surface("aten::_sobol_engine_initialize_state_", "cpu")
    run_oracle_for_surface("aten::_empty_affine_quantized", "cpu")
    run_oracle_for_surface("aten::_weight_int4pack_mm_for_cpu", "cpu")
    run_oracle_for_surface("aten::_native_batch_norm_legit.no_stats_out", "cpu")
    run_oracle_for_surface("aten::_fw_primal_copy", "cpu")
    run_oracle_for_surface("aten::_fw_primal_copy.out", "cpu")
    run_oracle_for_surface("aten::_make_dual_copy", "cpu")
    run_oracle_for_surface("aten::_make_dual_copy.out", "cpu")
    run_oracle_for_surface("aten::_nested_select_backward", "cpu")


def test_privateuse1_oracle_surfaces_require_privateuse1_device():
    from torchcts.core.oracles import OracleUnavailable, run_oracle_for_surface

    with pytest.raises(OracleUnavailable, match="requires a PrivateUse1 backend"):
        run_oracle_for_surface("aten::_scaled_dot_product_flash_attention", "cpu")

    with pytest.raises(OracleUnavailable, match="requires a PrivateUse1 backend"):
        run_oracle_for_surface("aten::_pin_memory", "cpu")


def test_mps_int4_oracle_dimension_guard():
    from torchcts.core.oracles import _validate_tinygemm_int4_dimensions

    _validate_tinygemm_int4_dimensions(out_features=16, in_features=128, group_size=32, inner_k_tiles=8)

    with pytest.raises(ValueError, match="inner_k_tiles \\* 16"):
        _validate_tinygemm_int4_dimensions(out_features=16, in_features=64, group_size=32, inner_k_tiles=8)


def test_oracle_strategy_skips_pending_backend_pack():
    from torchcts.generated.coverage_helpers import run_oracle_strategy

    entry = {
        "name": "aten::_philox_uniform",
        "status": "pending_backend_pack",
        "surface_kind": "rng",
        "semantic_level": 5,
    }

    with pytest.raises(pytest.skip.Exception, match="pending_backend_pack"):
        run_oracle_strategy(entry, "cpu")


def test_generated_strategy_dispatch_reads_strategy_dict(monkeypatch):
    calls = []

    def fake_manual_bitwise(entry, device, compare, manifest):
        calls.append((entry["name"], device))

    monkeypatch.setattr(generated_helpers, "run_manual_bitwise_strategy", fake_manual_bitwise)

    entry = {
        "name": "aten::bitwise_and.Tensor_out",
        "generated": {"strategy": {"strategy": "manual_bitwise", "family": "bitwise_and"}},
    }
    generated_helpers.run_generated_out_strategy(entry, "cpu", None, {})

    assert calls == [("aten::bitwise_and.Tensor_out", "cpu")]


def test_indexing_sample_builds_uint16_isin_without_uint16_arange():
    entry = {
        "name": "aten::isin.Tensor_Scalar",
        "base_name": "isin",
        "args": [
            {"name": "elements", "tensor": True},
            {"name": "test_element", "tensor": False},
        ],
        "generated": {"strategy": {"strategy": "manual_indexing", "family": "isin"}},
    }

    sample = sample_generation.indexing_sample(entry, torch.uint16)

    assert sample.input.dtype == torch.uint16
    assert tuple(sample.input.shape) == (3, 4)


def test_rng_sample_uses_cpu_generator_for_cpu_generator_dispatchers():
    entry = {
        "name": "aten::poisson",
        "base_name": "poisson",
        "args": [
            {"name": "self", "tensor": True},
            {"name": "generator", "type": "Generator?"},
        ],
    }

    args, _kwargs = sample_generation.rng_call_parts(entry, torch.float32, "cpu", seed=123)

    assert args[1].device.type == "cpu"
    assert not sample_generation.rng_uses_target_device_generator(entry)


def test_rng_sample_keeps_randint_like_tensor_high_on_cpu():
    entry = {
        "name": "aten::randint_like.Tensor_out",
        "base_name": "randint_like",
        "args": [
            {"name": "self", "tensor": True},
            {"name": "high", "tensor": True},
            {"name": "out", "tensor": True},
        ],
    }

    args, _kwargs = sample_generation.rng_call_parts(entry, torch.float32, "meta", seed=123)

    assert args[0].device.type == "meta"
    assert args[1].device.type == "cpu"


def test_rng_tensor_equal_handles_complex32_cpu_tensors():
    complex32 = getattr(torch, "complex32", None)
    if complex32 is None:
        pytest.skip("complex32 unavailable")
    value = torch.rand((2, 2), dtype=complex32)

    assert generated_helpers._rng_tensor_equal(value, value.clone())


def test_make_tensor_values_uint16_nonzero_is_cpu_safe():
    values = sample_generation.make_tensor_values(torch.uint16, domain="nonzero")

    assert values.dtype == torch.uint16
    assert bool((values != 0).all())


def test_foreach_tensor_scalar_args_keep_packed_scalars_on_cpu():
    entry = {
        "name": "aten::_foreach_addcdiv.Tensor",
        "base_name": "_foreach_addcdiv",
        "surface_kind": "functional_data",
        "generated": {
            "strategy": {
                "strategy": "manual_foreach",
                "family": "ternary",
                "foreach_name": "addcdiv",
                "overload": "Tensor",
            }
        },
    }
    sample = sample_generation.foreach_sample(entry, torch.float32)

    moved_args = generated_helpers._move_foreach_args_to_device(entry, sample.args, "meta")

    assert moved_args[0][0].device.type == "meta"
    assert moved_args[1][0].device.type == "meta"
    assert moved_args[2].device.type == "cpu"


def test_public_sample_generation_structured_inputs_and_expected():
    op_inputs = sample_generation.get_inputs_for_op(
        "linear",
        dtype=torch.float32,
        generate_results=True,
        batch=2,
        in_features=4,
        out_features=3,
    )

    assert op_inputs.op_name == "aten::linear"
    assert op_inputs.dispatcher_name == "aten::linear"
    assert op_inputs.signature_id == "aten::linear:functional"
    assert op_inputs.case_id == "with_bias"
    assert [param.name for param in op_inputs.params] == ["input", "weight", "bias"]
    assert [param.purpose for param in op_inputs.params] == ["activation", "weight", "bias"]
    assert op_inputs.metadata["category"] == "matmul"
    assert op_inputs.metadata["surface_kind"] == "functional_data"
    assert op_inputs.metadata["variant_kind"] == "functional"
    assert op_inputs.metadata["source"] == "torchcts"
    assert op_inputs.expected is not None
    assert op_inputs.expected.ok
    assert tuple(op_inputs.expected.value.shape) == (2, 3)


def test_matmul_family_reference_handles_complex32_without_native_cpu_kernel():
    complex32 = getattr(torch, "complex32", None)
    if complex32 is None:
        pytest.skip("complex32 unavailable")
    lhs = torch.tensor([[1 + 2j, 3 - 4j]], dtype=torch.complex64).to(complex32)
    rhs = torch.tensor([[2 - 1j], [-3 + 0.5j]], dtype=torch.complex64).to(complex32)

    expected = (lhs.to(torch.complex64) @ rhs.to(torch.complex64)).to(complex32)
    actual = reference_oracles.matmul_family_reference("aten::matmul.out", (lhs, rhs), {})

    assert actual.dtype == complex32
    assert torch.allclose(actual, expected)


def test_public_sample_generation_expected_uses_matmul_reference_for_complex32():
    complex32 = getattr(torch, "complex32", None)
    if complex32 is None:
        pytest.skip("complex32 unavailable")

    op_inputs = sample_generation.get_inputs_for_op(
        "matmul",
        dtype=complex32,
        generate_results=True,
        lhs_shape=(2, 3),
        rhs_shape=(3, 2),
    )

    assert op_inputs.expected is not None
    assert op_inputs.expected.ok
    assert op_inputs.expected.metadata["reference"] == "torchcts.core.reference_oracles.matmul_family_reference"
    assert op_inputs.expected.value.dtype == complex32
    assert tuple(op_inputs.expected.value.shape) == (2, 2)


def test_generated_matmul_expected_falls_back_when_native_cpu_kernel_is_missing():
    complex32 = getattr(torch, "complex32", None)
    if complex32 is None:
        pytest.skip("complex32 unavailable")
    entry = {
        "name": "aten::addmm.out",
        "base_name": "addmm",
        "surface_kind": "out_variant",
    }
    input_value = torch.ones((2, 2), dtype=complex32)
    mat1 = torch.tensor([[1 + 1j, 2 - 1j], [3 + 0j, 4 + 0.5j]], dtype=torch.complex64).to(complex32)
    mat2 = torch.tensor([[1 - 2j, 2 + 0j], [0.5 + 1j, -1 + 2j]], dtype=torch.complex64).to(complex32)

    expected, used_native = generated_helpers._native_or_reference_matmul_expected(
        entry,
        torch.ops.aten.addmm.out,
        (input_value, mat1, mat2),
        {"beta": 1, "alpha": 1},
    )

    manual = reference_oracles.matmul_family_reference(
        "aten::addmm.out",
        (input_value, mat1, mat2),
        {"beta": 1, "alpha": 1},
    )
    assert not used_native
    assert expected.dtype == complex32
    assert torch.allclose(expected, manual)


def test_public_sample_generation_iterates_planned_cases_and_conditions():
    cases = list(sample_generation.iter_inputs_for_op(
        "matmul",
        dtypes=(torch.float32,),
        input_conditions=(sample_generation.InputCondition.CLEAN,),
    ))

    assert len(cases) >= 4
    assert {case.metadata["case_id"] for case in cases} >= {
        "matrix_matrix",
        "vector_matrix",
        "matrix_vector",
        "broadcast_batch_matrix",
    }
    assert all(case.metadata["dispatcher_name"] == "aten::matmul" for case in cases)
    assert all(case.metadata["signature_id"] == "aten::matmul:functional" for case in cases)
    assert all(case.metadata["input_condition"] == sample_generation.InputCondition.CLEAN for case in cases)


def test_public_sample_generation_get_all_inputs_freezes_iterator():
    cases = sample_generation.get_all_inputs_for_op(
        "matmul",
        dtypes=(torch.float32,),
        input_conditions=(sample_generation.InputCondition.CLEAN,),
        generate_results=True,
    )

    assert isinstance(cases, tuple)
    assert len(cases) >= 4
    assert {case.case_id for case in cases} >= {"matrix_matrix", "vector_matrix", "matrix_vector"}
    assert all(case.expected and case.expected.ok for case in cases)


def test_public_sample_generation_exact_variant_and_binary_cases():
    op_inputs = sample_generation.get_inputs_for_op(
        "add",
        variant="Tensor",
        dtype=torch.float32,
        generate_results=True,
    )

    assert op_inputs.op_name == "aten::add.Tensor"
    assert op_inputs.metadata["dispatcher_name"] == "aten::add.Tensor"
    assert op_inputs.metadata["variant_kind"] == "functional"
    assert op_inputs.expected is not None
    assert op_inputs.expected.ok
    assert op_inputs.kwargs()["alpha"] == 1

    cases = list(sample_generation.iter_inputs_for_op(
        "add.Tensor",
        dtypes=(torch.float32,),
        input_conditions=(sample_generation.InputCondition.CLEAN,),
        generate_results=True,
    ))
    assert {case.metadata["case_id"] for case in cases} == {"same_shape", "broadcast_rhs"}
    assert all(case.expected and case.expected.ok for case in cases)


def test_public_sample_generation_case_specs_are_first_class():
    specs = sample_generation.sample_case_specs_for_op("matmul")

    assert {spec.case_id for spec in specs} >= {
        "matrix_matrix",
        "vector_matrix",
        "matrix_vector",
        "broadcast_batch_matrix",
    }
    assert all(spec.required for spec in specs)
    assert all(spec.purpose for spec in specs)
    assert any("rank_polymorphic" in spec.tags for spec in specs)

    inplace_addmm_specs = sample_generation.sample_case_specs_for_op("aten::addmm_")
    assert {spec.case_id for spec in inplace_addmm_specs} == {"full_bias"}


def test_public_sample_generation_exact_dispatcher_iterates_strategy_cases():
    entry = sample_generation.dispatcher_entry("aten::matmul.out")
    entry = {
        **entry,
        "generated": {"strategy": {"strategy": "manual_matmul", "family": "matmul"}},
        "status": "covered_generated",
    }

    cases = list(sample_generation.iter_inputs_for_op(
        "aten::matmul.out",
        audit={"entries": [entry]},
        dtypes=(torch.float32,),
        input_conditions=(sample_generation.InputCondition.CLEAN,),
    ))

    assert {case.metadata["case_id"] for case in cases} >= {
        "matrix_matrix",
        "vector_matrix",
        "matrix_vector",
        "broadcast_batch_matrix",
    }
    assert all(case.op_name == "aten::matmul.out" for case in cases)


def test_public_sample_generation_manual_shape_sample():
    entry = sample_generation.dispatcher_entry("aten::squeeze_copy.out")
    entry = {
        **entry,
        "generated": {"strategy": {"strategy": "manual_shape", "family": "squeeze_copy"}},
    }

    op_inputs = sample_generation.get_inputs_for_op(
        "aten::squeeze_copy.out",
        audit={"entries": [entry]},
        dtype=torch.float32,
    )

    assert op_inputs.strategy_name == "manual_shape"
    assert op_inputs.metadata["case_id"] == "shape_default"
    assert tuple(op_inputs.positional_args()[0].shape) == (2, 1, 3, 1)


def test_manual_shape_tensor_split_control_tensor_stays_on_cpu():
    input_value, args, _kwargs, case_id = sample_generation.shape_args_for_entry(
        "aten::tensor_split.tensor_indices_or_sections",
        torch.float32,
        device="meta",
    )

    assert case_id == "shape_dim_list"
    assert input_value.device.type == "meta"
    assert args[0].device.type == "cpu"


def test_coverage_generated_case_depth_reports_semantic_cases():
    entry = sample_generation.dispatcher_entry("aten::matmul.out")
    strategy = {"strategy": "manual_matmul", "family": "matmul"}

    depth = coverage_module._generated_case_depth_for_entry(entry, strategy)

    assert depth["planned_count"] >= 4
    assert depth["required_count"] >= 4
    assert "matrix_vector" in depth["case_ids"]
    assert "matmul" in depth["tags"]
    assert depth["semantic_levels"] == [4]
    assert depth["min_semantic_level"] == 4
    assert all(case["semantic_level"] == 4 for case in depth["cases"])


def test_sample_generation_publishes_semantic_level_for_external_callers():
    matmul = sample_generation.get_inputs_for_op("matmul", sample_index=0)
    binary = sample_generation.get_inputs_for_op("binary", sample_index=0)
    matmul_specs = sample_generation.sample_case_specs_for_op("matmul")

    assert matmul.metadata["semantic_level"] == 4
    assert matmul.metadata["level_source"] == "generated_case"
    assert binary.metadata["semantic_level"] == 1
    assert matmul_specs
    assert all(spec.semantic_level == 4 for spec in matmul_specs)


def test_generated_shape_strategy_is_exact_and_declarative():
    squeeze = sample_generation.dispatcher_entry("aten::squeeze")
    cat_out = sample_generation.dispatcher_entry("aten::cat.out")
    squeeze_inplace = sample_generation.dispatcher_entry("aten::squeeze_")
    chunk = sample_generation.dispatcher_entry("aten::chunk")
    named_squeeze = sample_generation.dispatcher_entry("aten::squeeze.dimname")

    assert coverage_module._generated_strategy_for_entry(squeeze, {}) == {
        "strategy": "manual_shape",
        "family": "squeeze",
    }
    assert coverage_module._generated_strategy_for_entry(cat_out, {}) == {
        "strategy": "manual_shape",
        "family": "cat",
    }
    assert coverage_module._generated_strategy_for_entry(squeeze_inplace, {}) == {
        "strategy": "manual_shape",
        "family": "squeeze",
    }
    assert coverage_module._generated_strategy_for_entry(chunk, {}) == {
        "strategy": "manual_shape",
        "family": "chunk",
    }
    assert coverage_module._generated_strategy_for_entry(named_squeeze, {}) is None


def test_public_sample_generation_realistic_weight_distribution():
    weight = sample_generation.make_weight_tensor(
        torch.float32,
        shape=(16, 8),
        distribution="kaiming_normal",
        seed=123,
    )

    assert tuple(weight.shape) == (16, 8)
    assert torch.isfinite(weight).all()
    assert abs(float(weight.mean())) < 0.2
    assert 0.1 < float(weight.std()) < 1.0
