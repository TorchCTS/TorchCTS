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

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.covers_category("selftest")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_site_stats.py"
if not SCRIPT_PATH.exists() or not (REPO_ROOT / "pyproject.toml").exists():
    pytest.skip("site stats script selftests require a source checkout", allow_module_level=True)


def _load_site_stats_module():
    spec = importlib.util.spec_from_file_location("generate_site_stats", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_site_stats_no_collect_writes_markdown(tmp_path):
    module = _load_site_stats_module()
    output = tmp_path / "site-stats.md"

    assert module.main(["--no-collect", "--output", str(output)]) == 0

    text = output.read_text(encoding="utf-8")
    assert "# TorchCTS Site Stats" in text
    assert "## Headline Stats" in text
    assert "## Dispatcher Coverage Summary" in text
    assert "## Known Crash Isolation Stats" in text
    assert "Pytest collection included | no" in text
    assert "Installed TorchCTS metadata version" in text
    assert "pyproject.toml version" in text
    assert "TorchCTS import path" in text
    assert "Unknown tensor-touching surfaces | 0" in text
    assert "Ops skipped (unsupported)" not in text


def test_site_stats_script_imports_checkout_package():
    module = _load_site_stats_module()

    try:
        installed_version = importlib.metadata.version("torchcts")
    except importlib.metadata.PackageNotFoundError:
        installed_version = module.torchcts.__version__

    assert module.torchcts.__version__ == installed_version
    assert Path(module.torchcts.__file__).resolve() == (REPO_ROOT / "torchcts" / "__init__.py").resolve()


def test_site_stats_collection_parser_groups_nodes():
    module = _load_site_stats_module()
    stats = module._collection_stats([
        "torchcts/opinfo/test_opinfo_forward.py::test_op_forward[abs-torch.float32]",
        "torchcts/generated/test_out_variants.py::test_generated_out_variant[add.out[L2]]",
        "torchcts/operators/test_binary.py::test_binary_float_op[add-torch.float32]",
        "torchcts/selftest/test_harness_reporting.py::test_report",
    ])

    assert stats["total"] == 4
    assert stats["parameterized"] == 3
    assert stats["suites"]["opinfo"] == 1
    assert stats["suites"]["generated"] == 1
    assert stats["suites"]["operators"] == 1
    assert stats["suites"]["selftest"] == 1
    assert stats["kinds"]["opinfo"] == 1
    assert stats["kinds"]["generated"] == 1
    assert stats["kinds"]["handwritten"] == 1
    assert stats["kinds"]["selftest"] == 1
    assert stats["visible_dtype_tokens"]["torch.float32"] == 2
    assert stats["visible_level_tokens"]["L2"] == 1
    assert stats["decisions"]["executable"] == 4


def test_site_stats_structured_collection_decisions_and_level_eight_render():
    module = _load_site_stats_module()
    audit = {
        "metadata": {
            "generated_at": "2026-06-30T00:00:00Z",
            "total_aten_overloads": 4,
            "unknown_count": 0,
            "status_counts": {
                "covered_handwritten": 2,
                "not_backend_relevant": 1,
                "unavailable_in_pytorch_runtime": 1,
            },
            "surface_counts": {},
            "coverage_kind_counts": {},
            "semantic_level_counts": {"1": 2},
            "semantic_level_status_counts": {"1": {"covered_handwritten": 2}},
            "semantic_level_surface_counts": {"1": {"functional_data": 2}},
            "semantic_level_descriptions": {str(level): f"level {level}" for level in range(1, 9)},
            "generated_case_depth": {"by_semantic_level": {"1": 2}},
            "pending_blocker_counts": {},
            "pending_backend_gate_counts": {},
        },
        "entries": [],
        "coverage_markers": [],
        "category_markers": [],
        "unmapped_tests": [],
        "warnings": [],
        "errors": [],
    }
    records = [
        {
            "nodeid": "torchcts/stress/test_adversarial.py::test_empty_tensors[sum]",
            "file": "torchcts/stress/test_adversarial.py",
            "suite": "stress",
            "test_kind": "handwritten",
            "function": "test_empty_tensors",
            "semantic_level": 8,
            "capability": "inference",
            "dtype": "torch.float32",
            "dtype_fields": {"dtype": "torch.float32"},
            "dispatcher_name": "aten::sum",
            "coverage_id": "aten::sum",
            "coverage_kind": "handwritten",
            "surface_kind": "functional_data",
            "variant_kind": "functional",
            "strategy": None,
            "strategy_family": None,
            "decision": "executable",
            "skip_reason": None,
            "skip_detail": None,
        },
        {
            "nodeid": "torchcts/generated/test_out_variants.py::test_generated_out_variant[fake[L4]]",
            "file": "torchcts/generated/test_out_variants.py",
            "suite": "generated",
            "test_kind": "generated",
            "function": "test_generated_out_variant",
            "semantic_level": 4,
            "capability": "inference",
            "dtype": "torch.float64",
            "dtype_fields": {"dtype": "torch.float64"},
            "dispatcher_name": "aten::fake.out",
            "coverage_id": "aten::fake.out",
            "coverage_kind": "generated",
            "surface_kind": "out_variant",
            "variant_kind": "out_variant",
            "strategy": "manual_fake",
            "strategy_family": "fake",
            "decision": "structured_deselected",
            "skip_reason": "dtype_not_supported",
            "skip_detail": "not claimed",
        },
        {
            "nodeid": "torchcts/device_api/test_device_module.py::test_device_module_methods",
            "file": "torchcts/device_api/test_device_module.py",
            "suite": "device_api",
            "test_kind": "handwritten",
            "function": "test_device_module_methods",
            "semantic_level": 6,
            "capability": "device_api",
            "dtype": None,
            "dtype_fields": {},
            "dispatcher_name": None,
            "coverage_id": None,
            "coverage_kind": "category",
            "surface_kind": None,
            "variant_kind": None,
            "strategy": None,
            "strategy_family": None,
            "decision": "pytest_skip_marked",
            "skip_reason": "cpu_not_applicable",
            "skip_detail": "not applicable",
        },
    ]
    collection = {
        "command": ["python", "-m", "pytest"],
        "command_display": ["python", "-m", "pytest", "--collect-only", "-q", "torchcts", "--validation", "--level", "8"],
        "nodes": [record["nodeid"] for record in records],
        "collected_from_summary": 3,
        "structured_collection": {"records": records},
    }

    text = module.render_markdown(audit=audit, collection=collection, include_collect=True)

    assert "## Pytest Collection Decisions" in text
    assert "| executable | 1 |" in text
    assert "| structured_deselected | 1 |" in text
    assert "| pytest_skip_marked | 1 |" in text
    assert "| Backend-relevant overloads | 2 |" in text
    assert "| Runtime-unavailable overloads | 1 |" in text
    assert "| runtime_unavailable | 1 |" in text
    assert "## Semantic Level Overview" in text
    assert "| 8 | 1 | 1 | 0 | 0 | 0 | 0 | level 8 |" in text
    assert "## Pytest Nodes By Semantic Level" in text
    assert "| 8 | 1 |" in text
    assert "## Coverage Surfaces By Semantic Level" in text
    assert "| 8 | 0 |" in text
    assert "## Generated Dispatcher Cases By Semantic Level" in text
    assert "## Semantic Level Descriptions" in text
    assert "level 8" in text
