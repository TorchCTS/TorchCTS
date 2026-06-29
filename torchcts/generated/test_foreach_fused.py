# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
    generated_foreach_case_id,
    generated_foreach_dtype_cases,
    generated_cases,
    run_manual_foreach_strategy,
)


pytestmark = pytest.mark.generated


def _foreach_or_fused_cases():
    cases = []
    for surface_kind in ("functional_data", "out_variant", "mutating_or_inplace"):
        cases.extend(
            entry
            for entry in generated_cases(surface_kind)
            if entry is not None and (entry["base_name"].startswith("_foreach") or "fused" in entry["base_name"])
        )
    return cases or [None]


def pytest_generate_tests(metafunc):
    if {"entry", "dtype"}.issubset(metafunc.fixturenames):
        import torchcts.conftest as harness

        cases = generated_foreach_dtype_cases(
            _foreach_or_fused_cases(),
            getattr(harness, "_MANIFEST", {}) or {},
        )
        metafunc.parametrize(
            ("entry", "dtype"),
            cases,
            ids=[generated_foreach_case_id(case) for case in cases],
        )


@pytest.mark.covers_category("generated_foreach_fused")
def test_generated_foreach_or_fused(entry, dtype, device, compare, manifest):
    run_manual_foreach_strategy(entry, device, compare, manifest, dtype=dtype)
