#!/usr/bin/env python3
"""Print one review-ready backward or routing case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "evidence/oracles/raw/SRC-NUMPY-FORMULA/backward.json"

ENTRY_POINTS = {
    "CP-LINALG-BWD": [
        "vander_backward_reference",
        "complex_rpow_backward_reference",
        "condition_number_backward_reference",
    ],
    "CP-NORM-SPARSE-BWD": [
        "group_norm_backward_reference",
        "sampled_addmm_backward_reference",
        "renorm_inf_backward_reference",
    ],
    "CP-ROUTING": ["has_opinfo_backward_reference", "resolve_opinfo_backward_reference"],
}

EXPECTED_KEYS = {
    "CP-LINALG-BWD": {
        "vander_expected", "rpow_exponent_expected", "rpow_base_expected", "cond_expected",
    },
    "CP-NORM-SPARSE-BWD": {
        "group_input_expected", "group_weight_expected", "group_bias_expected",
        "sampled_input_expected", "sampled_left_expected", "sampled_right_expected",
        "renorm_expected",
    },
}


def build(pack):
    record = json.loads(RAW.read_text(encoding="ascii"))["records"][pack]
    if pack == "CP-ROUTING":
        rows = record["has_reference"]
        inputs = {"has_reference": [{"op": row["op"], "dtype": row["dtype"]} for row in rows]}
        expected = {"has_reference": [row["expected"] for row in rows]}
        comparison = {
            "kind": "exact_routing_table",
            "asserts": ["value", "dtype", "shape"],
            "derivation": "Finite reviewed truth table for every gated operation and negative control.",
        }
        validation_class = "V3_ROUTING"
    else:
        keys = EXPECTED_KEYS[pack]
        inputs = {key: value for key, value in record.items() if key not in keys}
        expected = {key: value for key, value in record.items() if key in keys}
        comparison = {
            "kind": "mixed_exact_and_ulp",
            "ulp_ceiling": 4,
            "asserts": ["value", "dtype", "shape"],
            "derivation": "Exact matrix calculus and 100-digit complex formulas with explicit float32 rounding.",
        }
        validation_class = "V1_FIXED_VALUE"
    slug = pack.removeprefix("CP-").lower().replace("-", "_")
    return {
        "schema_version": 1,
        "case_id": f"cp-{slug}-backward",
        "oracle_kind": "permanent_reference",
        "validation_class": validation_class,
        "case_pack": pack,
        "oracle_ids": ENTRY_POINTS[pack],
        "implementation_entry_points": [
            f"torchcts.core.backward_references.{name}" for name in ENTRY_POINTS[pack]
        ],
        "dispatcher_surfaces": [],
        "applicability": {
            "backend": "cpu_development_host",
            "input_condition": "fixed_backward_formula_or_routing_record",
            "pytorch": ">=2.7.0,<2.12.2",
        },
        "inputs": inputs,
        "expected": expected,
        "comparison": comparison,
        "source_ids": ["SRC-NUMPY-FORMULA", "SRC-SYMPY-EXACT", "SRC-IEEE-C", "SRC-PYTORCH-SOURCE"],
        "generator": "scripts/oracle_fixtures/generate_backward_records.py",
        "raw_evidence": ["raw/SRC-NUMPY-FORMULA/backward.json"],
        "created_at": "2026-07-20",
        "review": {
            "reviewed_by": ["fixture-schema-implementation", "independent-formula-reproduction"],
            "reviewed_at": "2026-07-20",
            "conclusion": "implementation_correct",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", choices=sorted(ENTRY_POINTS))
    args = parser.parse_args()
    print(json.dumps(build(args.pack), indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
