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

import pytest

from torchcts.generated.coverage_helpers import (
    generated_case_id,
    generated_cases,
    run_generated_functional_strategy,
)


pytestmark = pytest.mark.generated


def _functional_cases():
    cases = []
    for entry in generated_cases("functional_data"):
        if entry is None:
            return [None]
        strategy = entry.get("generated", {}).get("strategy") or {}
        if entry.get("status") in {"unknown", "excluded"} or strategy.get("strategy"):
            cases.append(entry)
    return cases or [None]


@pytest.mark.covers_category("generated_functional_variants")
@pytest.mark.parametrize("entry", _functional_cases(), ids=generated_case_id)
def test_generated_functional_variant(entry, device, compare, manifest):
    run_generated_functional_strategy(entry, device, compare, manifest)
