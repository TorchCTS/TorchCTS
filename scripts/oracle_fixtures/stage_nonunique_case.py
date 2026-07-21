#!/usr/bin/env python3
"""Print the review-ready non-unique-output contract inventory case."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "evidence/oracles/raw/SRC-MANUAL-ENUM/nonunique-contracts.json"


def main():
    records = json.loads(RAW.read_text(encoding="ascii"))["records"]["CP-NON-UNIQUE"]
    case = {
        "schema_version": 1,
        "case_id": "cp-non-unique-contracts",
        "oracle_kind": "legality_contract",
        "validation_class": "V2_ADMISSIBILITY",
        "case_pack": "CP-NON-UNIQUE",
        "oracle_ids": [record["family"] for record in records],
        "implementation_entry_points": [
            "torchcts.core.non_unique_output_compare.contract_for_op_name",
            "torchcts.core.non_unique_output_compare.compare_non_unique_output_if_applicable",
            "torchcts.core.non_unique_output_compare.compare_non_unique_output",
        ],
        "dispatcher_surfaces": [record["representative_op"] for record in records],
        "applicability": {
            "backend": "all",
            "input_condition": "reviewed_ambiguous_output_family",
            "pytorch": ">=2.7.0,<2.12.2",
        },
        "inputs": {"representative_ops": [record["representative_op"] for record in records]},
        "admissible_results": {"contracts": records},
        "comparison": {
            "kind": "reviewed_legality_and_routing_contract",
            "asserts": ["legal_set", "dtype", "shape", "routing"],
            "derivation": "One explicit admissibility rule and representative routing surface for every runtime contract family.",
        },
        "source_ids": ["SRC-MANUAL-ENUM", "SRC-PYTORCH-SOURCE"],
        "generator": "scripts/oracle_fixtures/generate_nonunique_records.py",
        "raw_evidence": ["raw/SRC-MANUAL-ENUM/nonunique-contracts.json"],
        "created_at": "2026-07-20",
        "review": {
            "reviewed_by": ["fixture-schema-implementation", "admissibility-contract-review"],
            "reviewed_at": "2026-07-20",
            "conclusion": "implementation_correct",
        },
    }
    print(json.dumps(case, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
