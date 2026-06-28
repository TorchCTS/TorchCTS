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
    run_oracle_strategy,
)


pytestmark = pytest.mark.generated


ORACLE_STATUSES = {
    "covered_oracle",
    "covered_backend_pack",
    "covered_property",
    "pending_oracle",
    "pending_backend_pack",
    "pending_property",
}

SURFACE_KINDS = (
    "autograd_backward",
    "factory",
    "functional_data",
    "layout_storage",
    "metadata_device",
    "mutating_or_inplace",
    "out_variant",
    "rng",
    "view_or_alias",
)


def _oracle_cases():
    cases = []
    seen = set()
    for surface_kind in SURFACE_KINDS:
        for entry in generated_cases(surface_kind):
            if entry is None:
                continue
            if entry.get("status") not in ORACLE_STATUSES:
                continue
            if entry["name"] in seen:
                continue
            seen.add(entry["name"])
            cases.append(entry)
    return cases or [None]


@pytest.mark.covers_category("oracle_surfaces")
@pytest.mark.parametrize("entry", _oracle_cases(), ids=generated_case_id)
def test_oracle_surface(entry, device):
    run_oracle_strategy(entry, device)
