#!/usr/bin/env python3
"""Print one review-ready case from the independent exact-core record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "evidence/oracles/raw/SRC-SYMPY-EXACT/exact-core.json"

ENTRY_POINTS = {
    "CP-CAST": ["quantized_opmath_tensor", "cast_public_result", "saturate_weight_to_fp16_reference"],
    "CP-MATMUL": ["matmul_family_determinate_reference", "matmul_family_reference"],
    "CP-SOFTMARGIN": ["soft_margin_loss_reference", "soft_margin_loss_backward_reference"],
    "CP-SEGMENT": ["segment_reduce_prod_backward_reference", "segment_reduce_prod_reference"],
    "CP-LINEAR-BWD": ["linear_backward_reference"],
    "CP-POOL": ["max_pool2d_backward_reference"],
    "CP-INT4": ["pack_int4_values", "unpack_int4_values", "tinygemm_int4_dequantize_reference", "tinygemm_int4_matmul_reference", "unpack_dynamic_int4_weight_bytes", "dynamic_int4_dequantize_reference", "dynamic_int4_matmul_reference"],
    "CP-QUANT": ["weight_int8pack_mm_reference"],
    "CP-HISTC": ["histc_integer_count_reference"],
    "CP-IM2COL": ["im2col_reference", "col2im_reference"],
    "CP-LOGIT": ["logit_backward_reference"],
    "CP-POLYNOMIAL": ["laguerre_polynomial_reference", "shifted_chebyshev_polynomial_reference"],
}

EXPECTED_KEYS = {
    "CP-CAST": {"opmath_expected", "public_bfloat16_expected", "saturation_expected"},
    "CP-MATMUL": {"expected", "determinate_mask", "exact_rational_output"},
    "CP-SOFTMARGIN": {"loss_none", "loss_sum", "loss_mean", "backward_none", "backward_mean", "decimal_loss"},
    "CP-SEGMENT": {"expected_forward", "expected_backward", "no_zero_forward", "no_zero_backward"},
    "CP-LINEAR-BWD": {"expected_input_grad", "expected_weight_grad", "expected_bias_grad"},
    "CP-POOL": {"expected_input_grad", "winner_indices_yx"},
    "CP-INT4": {"packed_even_high", "packed_even_low", "expected_dequant", "expected_matmul", "dynamic_unpacked", "expected_dynamic_dequant", "expected_dynamic_matmul"},
    "CP-QUANT": {"expected"},
    "CP-HISTC": {"bins_1", "bins_2", "bins_4"},
    "CP-IM2COL": {"columns", "reconstructed"},
    "CP-LOGIT": {"expected_none", "expected_eps"},
    "CP-POLYNOMIAL": {"laguerre", "shifted"},
}

ULP = {"CP-SOFTMARGIN": 3, "CP-LOGIT": 2}


def build(pack):
    record = json.loads(RAW.read_text(encoding="ascii"))["records"][pack]
    expected_keys = EXPECTED_KEYS[pack]
    inputs = {key: value for key, value in record.items() if key not in expected_keys}
    expected = {key: value for key, value in record.items() if key in expected_keys}
    comparison = {
        "kind": "float32_ulp" if pack in ULP else "exact_bits_and_metadata",
        "asserts": ["value", "dtype", "shape"],
        "derivation": "Independent scalar loops and exact/high-precision formula evaluation with declared public-dtype rounding.",
    }
    if pack in ULP:
        comparison["ulp_ceiling"] = ULP[pack]
    slug = pack.removeprefix("CP-").lower().replace("-", "_")
    return {
        "schema_version": 1,
        "case_id": f"cp-{slug}-exact-core",
        "oracle_kind": "value_reference",
        "validation_class": "V1_FIXED_VALUE",
        "case_pack": pack,
        "oracle_ids": ENTRY_POINTS[pack],
        "implementation_entry_points": [
            f"torchcts.core.reference_oracles.{name}" for name in ENTRY_POINTS[pack]
        ],
        "dispatcher_surfaces": [],
        "applicability": {
            "backend": "cpu_development_host",
            "input_condition": "fixed_formula_record",
            "pytorch": ">=2.7.0,<2.12.2",
        },
        "inputs": inputs,
        "expected": expected,
        "comparison": comparison,
        "source_ids": ["SRC-SYMPY-EXACT", "SRC-IEEE-C", "SRC-PYTORCH-SOURCE"],
        "generator": "scripts/oracle_fixtures/generate_exact_core_records.py",
        "raw_evidence": ["raw/SRC-SYMPY-EXACT/exact-core.json"],
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
