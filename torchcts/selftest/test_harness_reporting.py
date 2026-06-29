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
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import torchcts.conftest as harness
import torchcts.cli as cli_module
import torchcts.core.coverage as coverage_module
import torchcts.core.device as device_module
import torchcts.core.opinfo_adapter as opinfo_adapter_module
import torchcts.core.reference_oracles as reference_oracles
import torchcts.core.runtime_evidence as runtime_evidence
import torchcts.core.version_rules as version_rules
import torchcts.generated.coverage_helpers as generated_helpers
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
    assert "Ops skipped (manifest):    1" in scorecard
    assert "Ops skipped (unsupported): 1" in scorecard
    assert "inference" in scorecard
    assert "2/3 passed" in scorecard


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


def test_declared_capability_probe_failure_is_manifest_overclaim():
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

    failures = harness._apply_declared_capability_probes(
        caps,
        "privateuseone",
        probe_func=fake_probe,
    )

    assert caps["named_tensor"] is True
    assert len(failures) == 1
    assert "Manifest overclaim: capability 'named_tensor'" in failures[0]
    assert "NYI: named tensors" in failures[0]


def test_declared_dtype_probe_failure_is_manifest_overclaim(monkeypatch):
    def fake_zeros(*args, **kwargs):
        raise RuntimeError("value cannot be converted to type float without overflow")

    monkeypatch.setattr(harness.torch, "zeros", fake_zeros)
    supported_dtypes = {torch.float32: True}

    failures = harness._apply_declared_dtype_probes(supported_dtypes, "privateuseone")

    assert supported_dtypes == {torch.float32: True}
    assert len(failures) == 1
    assert "Manifest overclaim: dtype 'torch.float32'" in failures[0]
    assert "value cannot be converted" in failures[0]


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
        "skips": {},
    }

    scorecard, _ = build_report(current_data)

    assert "training        0/0 passed" in scorecard
    assert "training        0/1 passed" not in scorecard
    assert "float32" not in scorecard


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
    pattern = harness._hardware_unsupported_pattern_match(
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
    repo_root = Path(__file__).resolve().parents[2]
    return runpy.run_path(str(repo_root / "scripts" / "check_release_hygiene.py"))


def test_release_hygiene_rejects_package_known_failure_cache(tmp_path):
    hygiene = _load_release_hygiene_module()
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cache_path = tmp_path / "torchcts" / "opinfo_cache" / "known_failures.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("{}\n", encoding="utf-8")

    errors = hygiene["_check_git_paths"](tmp_path)

    assert any("known_failures.json" in error for error in errors)


def test_release_hygiene_rejects_tracked_backend_specific_text(tmp_path):
    hygiene = _load_release_hygiene_module()
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    bad = tmp_path / "bad.txt"
    bad.write_text("backend name: " + ("metal" "core") + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "bad.txt"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    errors = hygiene["_check_forbidden_text"](tmp_path)

    assert any("forbidden text" in error for error in errors)


def test_pyproject_does_not_suppress_backend_fallback_or_pluggy_teardown_warnings():
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")

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
    def __init__(self, name, fspath, markers=None):
        self.name = name
        self.fspath = fspath
        self.nodeid = f"{fspath}::{name}"
        self._markers = markers or {}
        self.added_markers = []

    def get_closest_marker(self, name):
        return self._markers.get(name)

    def iter_markers(self, name=None):
        for marker_name, marker in self._markers.items():
            if name is None or marker_name == name:
                yield marker

    def add_marker(self, marker):
        self.added_markers.append(marker)


def test_collection_dry_run_does_not_call_torch_compile_before_backend_import(monkeypatch):
    def fail_compile(*args, **kwargs):
        raise AssertionError("torch.compile must not run during backend-import-free collection")

    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
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
    harness.pytest_collection_modifyitems(None, _CollectionConfig(), items)

    assert items == [level_2, level_3]
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
    harness.pytest_collection_modifyitems(None, _CollectionConfig(), items)

    assert items == [level_2, level_4, level_5]
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


class _BenchmarkConfig:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def getoption(self, option, default=None):
        if option == "--benchmark":
            return self.enabled
        return default


class _FixtureInfo:
    argnames = []


class _PyfuncItem:
    def __init__(self, marker=None):
        self.config = _BenchmarkConfig()
        self.funcargs = {}
        self._fixtureinfo = _FixtureInfo()
        self.name = "test_fake_benchmarkable"
        self._marker = marker
        self.calls = 0

    def get_closest_marker(self, name):
        if name == "benchmarkable":
            return self._marker
        return None

    def obj(self):
        self.calls += 1


def test_benchmark_mode_skips_unmarked_tests():
    item = _PyfuncItem(marker=None)

    with pytest.raises(pytest.skip.Exception, match="Benchmark mode only runs tests marked benchmarkable"):
        harness.pytest_pyfunc_call(item)

    assert item.calls == 0


def test_benchmark_mode_runs_marked_tests(monkeypatch):
    monkeypatch.setattr(harness, "_DEVICE_NAME", "cpu")
    monkeypatch.setattr(harness, "clear_metrics", lambda: None)
    monkeypatch.setattr(harness, "synchronize", lambda device_name: None)

    item = _PyfuncItem(marker=object())

    assert harness.pytest_pyfunc_call(item) is True
    assert item.calls == 110
    assert item.bench_stats["repetitions"] == 100


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
    fused_dropout = by_name["aten::_fused_dropout"]
    nested_softmax = by_name["aten::_nested_tensor_softmax_with_shape"]

    assert flash["status"] == "pending_property"
    assert flash["pending_review"]["blocker_type"] == "needs_public_proxy_proof"
    assert flash["pending_review"]["required_closure"] == "prove_public_proxy_or_add_direct_runner"

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
