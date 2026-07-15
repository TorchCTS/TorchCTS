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

from copy import deepcopy
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_MANIFEST = REPO_ROOT / "evidence" / "pytorch" / "dtype-contracts" / "manifest.json"
if not (REPO_ROOT / "scripts" / "pytorch_dtype_evidence_store.py").is_file() or not TRACKED_MANIFEST.is_file():
    pytest.skip("dtype evidence selftests require a source checkout", allow_module_level=True)

from scripts import pytorch_dtype_evidence_store as evidence_store
from scripts import pytorch_dtype_evidence_taxonomy as taxonomy


pytestmark = pytest.mark.covers_category("selftest")

def _sample_entry(value: str) -> dict:
    return {
        "cpu_supported": {"forward:clean": [value]},
        "evidence": {
            "pytorch_version": "placeholder",
            "source": "selftest",
            "version_rule": "placeholder",
        },
        "replace_contract": True,
    }


def _sample_expanded() -> dict:
    universal = _sample_entry("torch.float32")
    early = _sample_entry("torch.float32")
    late = _sample_entry("torch.float64")
    return {
        "version": 2,
        "format": "expanded_evidence",
        "metadata": {
            "collected_versions": ["2.7.0", "2.8.0", "2.9.0"],
            "contract_authority": "versioned_cpu_probe",
            "generated_by": "selftest",
        },
        "contracts": {
            "aten::add.Tensor": {
                version: deepcopy(universal)
                for version in ("2.7.0", "2.8.0", "2.9.0")
            },
            "aten::aminmax": {
                "2.7.0": early,
                "2.8.0": deepcopy(early),
                "2.9.0": late,
            },
            "aten::matmul": {
                "2.8.0": _sample_entry("torch.float32"),
                "2.9.0": _sample_entry("torch.float32"),
            },
        },
        "warnings": [],
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _records_for(store: Path, op: str) -> list[dict]:
    records = []
    for path in store.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["op"] == op:
                records.append(record)
    return records


def test_taxonomy_covers_tracked_evidence_without_fallbacks():
    expanded = evidence_store.load_evidence(TRACKED_MANIFEST, verify_canonical=True)
    classifications = taxonomy.classify_operators(set(expanded["contracts"]))

    assert classifications
    assert set(classifications) == set(expanded["contracts"])
    assert all(item.category in taxonomy.CATEGORIES for item in classifications.values())
    assert all(item.rule_id for item in classifications.values())
    assert not {item.category for item in classifications.values()} & {"other", "misc", "uncategorized"}
    assert all(reason.strip() for _category, reason in taxonomy.EXACT_OVERRIDES.values())


@pytest.mark.parametrize(
    ("op", "category"),
    [
        ("aten::linalg.svd", "linear-algebra/decomposition"),
        ("aten::linalg_svd", "linear-algebra/decomposition"),
        ("aten::nn.functional.scaled_dot_product_attention", "neural-network/attention"),
        ("aten::fft.fft", "spectral/fft"),
        ("aten::special.airy_ai", "math/special"),
        ("aten::masked.sum", "tensor/masked"),
        ("aten::signal.windows.hann", "spectral/windows"),
        ("aten::__radd__", "language/operator-protocol"),
        ("aten::trace_backward", "autograd/backward"),
        ("aten::_cast_Float", "tensor/conversion"),
        ("aten::torch.ops.aten._safe_softmax.default", "neural-network/softmax"),
    ],
)
def test_taxonomy_representative_aliases_and_semantic_families(op, category):
    assert taxonomy.classify_operator(op, taxonomy.load_metadata_categories()).category == category


def test_writer_uses_universal_and_exact_scoped_records(tmp_path):
    expanded = _sample_expanded()
    manifest = evidence_store.write_evidence_store(tmp_path / "store", expanded)
    reloaded = evidence_store.load_evidence_store(manifest, verify_canonical=True)

    assert evidence_store.normalize_expanded_evidence(reloaded) == evidence_store.normalize_expanded_evidence(expanded)
    assert "versions" not in _records_for(manifest.parent, "aten::add.Tensor")[0]
    assert [record["versions"] for record in _records_for(manifest.parent, "aten::aminmax")] == [
        ["2.7.0", "2.8.0"],
        ["2.9.0"],
    ]
    assert _records_for(manifest.parent, "aten::matmul")[0]["versions"] == ["2.8.0", "2.9.0"]


def test_writer_is_byte_deterministic_for_shuffled_input(tmp_path):
    expanded = _sample_expanded()
    shuffled = deepcopy(expanded)
    shuffled["contracts"] = dict(reversed(list(shuffled["contracts"].items())))
    for op, versioned in list(shuffled["contracts"].items()):
        shuffled["contracts"][op] = dict(reversed(list(versioned.items())))

    first = evidence_store.write_evidence_store(tmp_path / "first", expanded)
    second = evidence_store.write_evidence_store(tmp_path / "second", shuffled)

    assert _tree_bytes(first.parent) == _tree_bytes(second.parent)


def test_legacy_v2_reader_matches_v3_after_structural_normalization(tmp_path):
    expanded = _sample_expanded()
    legacy_path = tmp_path / "legacy.jsonl"
    records = [
        {
            "record_kind": "metadata",
            "version": 2,
            "format": "expanded_evidence_jsonl",
            "metadata": expanded["metadata"],
        },
        *[
            {"record_kind": "op_contract_evidence", "op": op, "versions": versioned}
            for op, versioned in expanded["contracts"].items()
        ],
    ]
    legacy_path.write_text("\n".join(evidence_store.canonical_json(record) for record in records) + "\n")
    manifest = evidence_store.write_evidence_store(tmp_path / "store", expanded)

    legacy = evidence_store.load_legacy_evidence_jsonl(legacy_path)
    v3 = evidence_store.load_evidence_store(manifest, verify_canonical=True)
    assert evidence_store.normalize_expanded_evidence(legacy) == evidence_store.normalize_expanded_evidence(v3)


def test_reader_rejects_explicit_scope_covering_every_version(tmp_path):
    manifest = evidence_store.write_evidence_store(tmp_path / "store", _sample_expanded())
    record_path = next(path for path in manifest.parent.rglob("*.jsonl") if "add.Tensor" in path.read_text())
    records = [json.loads(line) for line in record_path.read_text().splitlines()]
    record = next(item for item in records if item["op"] == "aten::add.Tensor")
    record["versions"] = ["2.7.0", "2.8.0", "2.9.0"]
    record_path.write_text("\n".join(evidence_store.canonical_json(item) for item in records) + "\n")

    with pytest.raises(ValueError, match="all-version contract must omit versions"):
        evidence_store.load_evidence_store(manifest, verify_canonical=True)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record.update(versions=[]), "non-empty string list"),
        (lambda record: record.update(versions=["9.9.9"]), "unknown versions"),
        (lambda record: record["contract"].update(replace_contract=True), "repeats replace_contract"),
    ],
)
def test_reader_rejects_invalid_scopes_and_structural_duplication(tmp_path, mutate, message):
    manifest = evidence_store.write_evidence_store(tmp_path / "store", _sample_expanded())
    record_path = next(path for path in manifest.parent.rglob("*.jsonl") if "aminmax" in path.read_text())
    records = [json.loads(line) for line in record_path.read_text().splitlines()]
    record = next(item for item in records if item["op"] == "aten::aminmax")
    mutate(record)
    record_path.write_text("\n".join(evidence_store.canonical_json(item) for item in records) + "\n")

    with pytest.raises(ValueError, match=message):
        evidence_store.load_evidence_store(manifest, verify_canonical=True)
