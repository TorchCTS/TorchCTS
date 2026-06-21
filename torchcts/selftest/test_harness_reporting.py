import json
from pathlib import Path

import pytest
import torch

import torchcts.conftest as harness
import torchcts.core.device as device_module
from torchcts.core.report import build_report


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
                "capability": "quantized",
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
    assert data["results"]["dummy"]["status"] == "PASS"


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
    def __init__(self):
        self.deselected = []

    def getoption(self, option):
        return {
            "--suite": None,
            "--validation": False,
            "--dtype": None,
        }.get(option)

    @property
    def hook(self):
        return self

    def pytest_deselected(self, items):
        self.deselected.extend(items)


def test_collection_dry_run_does_not_call_torch_compile_before_backend_import(monkeypatch):
    def fail_compile(*args, **kwargs):
        raise AssertionError("torch.compile must not run during backend-import-free collection")

    monkeypatch.setattr(harness, "_COLLECT_ONLY", True)
    monkeypatch.setattr(harness, "_SHOW_SKIPS", False)
    monkeypatch.setattr(harness, "_DEVICE_NAME", "metalcore")
    monkeypatch.setattr(harness, "_MANIFEST", {
        "capabilities": {"compile": True},
        "supported_dtypes": {},
        "skip_ops": [],
        "device_count": 1,
    })
    monkeypatch.setattr(harness.torch, "compile", fail_compile)

    harness.pytest_collection_modifyitems(None, _CollectionConfig(), [])


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
