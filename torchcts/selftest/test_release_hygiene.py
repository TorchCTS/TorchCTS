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

from importlib import util
from pathlib import Path

import pytest


pytestmark = pytest.mark.covers_category("selftest")


def _release_hygiene_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_release_hygiene.py"
    spec = util.spec_from_file_location("check_release_hygiene", script_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_hygiene_allows_tracked_sample_result_artifacts():
    hygiene = _release_hygiene_module()

    assert hygiene._is_denied_path("sample-results/mps-full-run-macos/results/Apple_M3_Max_128gb_report.md") is None
    assert hygiene._is_denied_path("results/Apple_M3_Max_128gb_report.md") == "denied component 'results'"
    assert hygiene._is_denied_path("sample-results/mps-full-run-macos/scratch/tmp.txt") == "denied component 'scratch'"
