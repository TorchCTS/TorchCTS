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

from pathlib import Path

import pytest


pytestmark = pytest.mark.covers_category("selftest")

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_README = REPO_ROOT / "README.md"

if not PACKAGE_README.exists() or not (REPO_ROOT / "docs").is_dir() or not (REPO_ROOT / "pyproject.toml").exists():
    pytest.skip("public docs checks require a source checkout", allow_module_level=True)

PUBLIC_DOCS = [
    PACKAGE_README,
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "harness.md",
    REPO_ROOT / "docs" / "release.md",
    REPO_ROOT / "docs" / "coverage" / "README.md",
    REPO_ROOT / "docs" / "coverage" / "oracle-authoring.md",
    REPO_ROOT / "docs" / "coverage" / "contract-evidence.md",
    REPO_ROOT / "docs" / "coverage" / "backend-packs.md",
    REPO_ROOT / "docs" / "coverage" / "exclusions.md",
]

RESEARCH_NOTE_FRAGMENTS = [
    "maybe",
    "probably",
    "appears",
    "seems",
    "not sure",
    "unclear",
    "scratch",
    "theory",
    "hypothesis",
    "partial probe",
    "we think",
    "open questions",
    "todo",
    "tbd",
]

STALE_POLICY_FRAGMENTS = [
    "known CPU reference failures",
    "known CPU failures",
    "Unsupported features are skipped",
    "Use `issues.md`",
    "Use issues.md",
]

REQUIRED_PUBLIC_CONTROLS = [
    "--dtype",
    "--adaptive-isolation",
    "--known-segfault-audit",
    "coverage check --fail-on-unknown",
    "scripts/check_release_hygiene.py",
]


def _markdown_section(text: str, title: str) -> str:
    marker = f"## {title}"
    start = text.index(marker)
    next_start = text.find("\n## ", start + len(marker))
    if next_start == -1:
        return text[start:]
    return text[start:next_start]


def test_public_coverage_docs_exist_and_are_linked():
    missing = [str(path.relative_to(REPO_ROOT)) for path in PUBLIC_DOCS if not path.exists()]
    assert missing == []

    docs_index = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    package_readme = PACKAGE_README.read_text(encoding="utf-8")

    assert "(harness.md)" in docs_index
    assert "(coverage/README.md)" in docs_index
    assert "(release.md)" in docs_index
    assert "https://github.com/TorchCTS/TorchCTS/blob/main/docs/harness.md" in package_readme
    assert "https://github.com/TorchCTS/TorchCTS/blob/main/docs/coverage/README.md" in package_readme
    assert "https://github.com/TorchCTS/TorchCTS/blob/main/docs/release.md" in package_readme


def test_package_readme_is_pypi_long_description_source():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_readme = PACKAGE_README.read_text(encoding="utf-8")
    release_doc = (REPO_ROOT / "docs" / "release.md").read_text(encoding="utf-8")

    assert 'readme = "README.md"' in pyproject
    assert "PyPI" in package_readme
    assert "long description" in release_doc
    assert "root `README.md`" in release_doc


def test_public_coverage_docs_do_not_contain_research_note_language():
    failures = []
    for path in PUBLIC_DOCS:
        text = path.read_text(encoding="utf-8").lower()
        for fragment in RESEARCH_NOTE_FRAGMENTS:
            if fragment in text:
                failures.append(f"{path.relative_to(REPO_ROOT)} contains {fragment!r}")

    assert failures == []


def test_public_docs_do_not_contain_stale_runtime_policy_language():
    failures = []
    for path in PUBLIC_DOCS:
        text = path.read_text(encoding="utf-8")
        for fragment in STALE_POLICY_FRAGMENTS:
            if fragment in text:
                failures.append(f"{path.relative_to(REPO_ROOT)} contains stale fragment {fragment!r}")

    assert failures == []


def test_public_docs_cover_current_runtime_and_release_controls():
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)

    missing = [fragment for fragment in REQUIRED_PUBLIC_CONTROLS if fragment not in combined]
    assert missing == []

    assert "diagnostic probe evidence" in combined
    assert "structured accounting" in combined
    assert "subprocess isolation never skips" in combined
    assert "PyPI README validation" in combined


def test_dynamic_int4_contract_evidence_matches_current_audit():
    from torchcts.core import coverage as coverage_module

    audit = coverage_module.build_audit()
    by_name = {entry["name"]: entry for entry in audit["entries"]}
    contract_doc = (REPO_ROOT / "docs" / "coverage" / "contract-evidence.md").read_text(encoding="utf-8")
    section = _markdown_section(contract_doc, "Dynamic 4-Bit Quantization Pack And Matmul")

    surfaces = [
        "aten::_dyn_quant_pack_4bit_weight",
        "aten::_dyn_quant_matmul_4bit",
    ]
    for surface in surfaces:
        entry = by_name[surface]
        assert entry["status"] == "covered_oracle"
        assert entry["oracle"]["oracle_id"] == "dynamic_int4_pack_matmul_value_oracle"
        assert surface in section

    assert "Status: `covered_oracle`" in section
    assert "dynamic_int4_pack_matmul_value_oracle" in section
    assert "Status: `pending_oracle`" not in section
