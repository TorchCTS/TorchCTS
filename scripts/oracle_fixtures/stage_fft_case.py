#!/usr/bin/env python3
"""Print the review-ready FFT contract case."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "evidence/oracles/raw/SRC-DIRECT-DFT/fft-contracts.json"


def main():
    record = json.loads(RAW.read_text(encoding="ascii"))["records"]["CP-FFT"]
    inputs, expected = {}, {}
    for name in ("public_c2c", "public_c2r", "generated_c2c"):
        row = record[name]
        inputs[name] = {
            key: value for key, value in row.items()
            if key not in {"spec", "contributor_mask"}
        }
        expected[name] = {
            "spec": row["spec"],
            "contributor_mask": row["contributor_mask"],
        }
    inputs["comparison"] = {
        key: value for key, value in record["comparison"].items() if key != "expected"
    }
    expected["comparison"] = {"expected": record["comparison"]["expected"]}
    case = {
        "schema_version": 1,
        "case_id": "cp-fft-contracts",
        "oracle_kind": "fft_contract",
        "validation_class": "V2_ADMISSIBILITY",
        "case_pack": "CP-FFT",
        "oracle_ids": [
            "public_fft_contract_spec", "generated_c2c_fft_contract_spec",
            "fft_source_contributor_mask", "compare_fft_nonfinite_groups",
        ],
        "implementation_entry_points": [
            "torchcts.core.fft_contract.public_fft_contract_spec",
            "torchcts.core.fft_contract.generated_c2c_fft_contract_spec",
            "torchcts.core.fft_contract.fft_source_contributor_mask",
            "torchcts.core.fft_contract.compare_fft_nonfinite_groups",
        ],
        "dispatcher_surfaces": [],
        "applicability": {
            "backend": "all",
            "input_condition": "shape_and_nonfinite_contributor_contract",
            "pytorch": ">=2.7.0,<2.12.2",
        },
        "inputs": inputs,
        "expected": expected,
        "comparison": {
            "kind": "exact_shape_mask_and_group_admissibility",
            "asserts": ["value", "dtype", "shape", "contributor_set"],
            "derivation": "Hand enumeration of transform dimensions, truncation, Hermitian self-conjugate lanes, and batch groups.",
        },
        "source_ids": ["SRC-DIRECT-DFT", "SRC-MANUAL-ENUM", "SRC-PYTORCH-SOURCE"],
        "generator": "scripts/oracle_fixtures/generate_fft_contract_records.py",
        "raw_evidence": ["raw/SRC-DIRECT-DFT/fft-contracts.json"],
        "created_at": "2026-07-20",
        "review": {
            "reviewed_by": ["fixture-schema-implementation", "independent-shape-enumeration"],
            "reviewed_at": "2026-07-20",
            "conclusion": "implementation_correct",
        },
    }
    print(json.dumps(case, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
