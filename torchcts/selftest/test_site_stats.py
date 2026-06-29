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

import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.covers_category("selftest")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_site_stats.py"


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
    assert "Unknown tensor-touching surfaces | 0" in text
    assert "Ops skipped (unsupported)" not in text


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
