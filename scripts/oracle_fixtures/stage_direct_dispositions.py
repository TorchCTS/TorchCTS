#!/usr/bin/env python3
"""Print the review-ready direct registry disposition case."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "evidence/oracles/raw/SRC-MANUAL-ENUM/direct-dispositions.json"


def main():
    payload = json.loads(RAW.read_text(encoding="ascii"))
    records = payload["records"]["CP-DIRECT-DISPOSITIONS"]
    case = {
        "schema_version": 1,
        "case_id": "cp-direct-dispositions",
        "oracle_kind": "direct_oracle",
        "validation_class": "V6_DISPOSITION",
        "case_pack": "CP-DIRECT-DISPOSITIONS",
        "oracle_ids": sorted({record["oracle_id"] for record in records}),
        "implementation_entry_points": [
            "torchcts.core.oracles.all_oracle_specs",
            "torchcts.core.oracles.oracle_spec_for",
        ],
        "dispatcher_surfaces": [record["surface"] for record in records],
        "applicability": {
            "backend": "registry_only",
            "input_condition": "reviewed_surface_disposition_not_backend_execution",
            "pytorch": ">=2.7.0,<2.12.2",
        },
        "inputs": {"surfaces": [record["surface"] for record in records]},
        "admissible_results": {
            "records": records,
            "counts": payload["counts"],
            "boundary": "This record validates routing and disposition only. It does not claim backend semantics without the case pack and physical evidence named by each accepted status.",
        },
        "comparison": {
            "kind": "exact_reviewed_registry_disposition",
            "asserts": ["routing", "shape", "dtype", "legal_set"],
            "derivation": "Reviewed snapshot of exact dispatcher surfaces, status, runner, backend gate, case pack, and evidence link.",
        },
        "source_ids": ["SRC-MANUAL-ENUM", "SRC-PYTORCH-SOURCE"],
        "generator": "scripts/oracle_fixtures/generate_direct_dispositions.py",
        "raw_evidence": ["raw/SRC-MANUAL-ENUM/direct-dispositions.json"],
        "created_at": "2026-07-20",
        "review": {
            "reviewed_by": ["fixture-schema-implementation", "direct-registry-disposition-review"],
            "reviewed_at": "2026-07-20",
            "conclusion": "implementation_correct",
        },
    }
    print(json.dumps(case, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
