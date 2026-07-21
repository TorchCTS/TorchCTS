#!/usr/bin/env python3
"""Print one review-ready case from the independent remaining-core record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "evidence/oracles/raw/SRC-NUMPY-FORMULA/remaining-core.json"

ENTRY_POINTS = {
    "CP-COMPLEX-ARITH": [
        "has_nonnegative_integral_complex_exponent",
        "complex_mul_reference",
        "foreach_complex_compound_reference",
    ],
    "CP-COMPLEX-LOSS": ["binary_cross_entropy_with_logits_reference"],
    "CP-COMPLEX-UNARY": [
        "complex_sigmoid_reference",
        "complex_rsqrt_reference",
        "complex_expm1_reference",
    ],
    "CP-LDEXP-CUMPROD": [
        "complex_ldexp_reference",
        "complex_integral_ldexp_reference",
        "complex_cumprod_reference",
        "wide_ldexp_reference",
    ],
    "CP-GRAD-COV": ["complex_gradient_reference", "complex_covariance_reference"],
    "CP-LANCZOS": ["lanczos3_coefficient", "lanczos2d_aa_reference"],
    "CP-GRID": [
        "grid_sampler_backward_f32_reference",
        "grid_sampler_3d_backward_f32_reference",
        "grid_sampler_forward_f32_reference",
    ],
    "CP-CONV": ["conv_transpose_f32_reference", "conv_transpose3d_f32_reference"],
    "CP-MATRIXEXP": ["matrix_exp_f64_reference"],
    "CP-SPECIAL": [
        "dirichlet_grad_reference",
        "standard_gamma_grad_reference",
        "i0_reference",
        "polygamma_reference",
        "regularized_gamma_reference",
        "kaiser_window_reference",
    ],
}

EXPECTED_KEYS = {
    "CP-COMPLEX-ARITH": {
        "multiplied", "foreach_add_alpha_half", "foreach_addcmul",
    },
    "CP-COMPLEX-LOSS": {"expected_none", "expected_sum", "expected_mean", "decimal"},
    "CP-COMPLEX-UNARY": {"sigmoid", "rsqrt", "expm1"},
    "CP-LDEXP-CUMPROD": {
        "complex_expected", "integral_expected", "cumprod_expected", "wide_expected",
    },
    "CP-GRAD-COV": {"gradient_expected", "covariance_expected"},
    "CP-LANCZOS": {"coefficients", "output"},
    "CP-GRID": {"forward", "grad_input", "grad_grid"},
    "CP-CONV": {"transpose1d_expected", "transpose3d_expected"},
    "CP-MATRIXEXP": {"expected", "decimal"},
    "CP-SPECIAL": {
        "dirichlet_expected", "gamma_grad_expected", "i0_expected",
        "polygamma_expected", "regularized_lower", "regularized_upper",
        "kaiser_expected",
    },
}

ULP_CEILINGS = {
    "CP-COMPLEX-LOSS": 3,
    "CP-COMPLEX-UNARY": 4,
    "CP-LDEXP-CUMPROD": 4,
    "CP-LANCZOS": 8,
    "CP-MATRIXEXP": 8,
    "CP-SPECIAL": 4,
}


def build(pack):
    record = json.loads(RAW_PATH.read_text(encoding="ascii"))["records"][pack]
    expected_keys = EXPECTED_KEYS[pack]
    inputs = {key: value for key, value in record.items() if key not in expected_keys}
    expected = {key: value for key, value in record.items() if key in expected_keys}
    comparison = {
        "kind": "ulp_with_exact_metadata" if pack in ULP_CEILINGS else "exact_bits_and_metadata",
        "asserts": ["value", "dtype", "shape"],
        "derivation": "Independent scalar loops or 100-digit formula evaluation followed by explicit public-dtype rounding.",
    }
    if pack in ULP_CEILINGS:
        comparison["ulp_ceiling"] = ULP_CEILINGS[pack]
    sources = ["SRC-NUMPY-FORMULA", "SRC-IEEE-C", "SRC-PYTORCH-SOURCE"]
    if pack in {"CP-COMPLEX-LOSS", "CP-COMPLEX-UNARY", "CP-LDEXP-CUMPROD", "CP-LANCZOS", "CP-MATRIXEXP", "CP-SPECIAL"}:
        sources.insert(1, "SRC-DLMF-MP")
    slug = pack.removeprefix("CP-").lower().replace("-", "_")
    module = "torchcts.core.high_precision_reference" if pack == "CP-SPECIAL" else "torchcts.core.reference_oracles"
    return {
        "schema_version": 1,
        "case_id": f"cp-{slug}-remaining-core",
        "oracle_kind": "value_reference",
        "validation_class": "V1_FIXED_VALUE",
        "case_pack": pack,
        "oracle_ids": ENTRY_POINTS[pack],
        "implementation_entry_points": [f"{module}.{name}" for name in ENTRY_POINTS[pack]],
        "dispatcher_surfaces": [],
        "applicability": {
            "backend": "cpu_development_host",
            "input_condition": "fixed_independent_formula_record",
            "pytorch": ">=2.7.0,<2.12.2",
        },
        "inputs": inputs,
        "expected": expected,
        "comparison": comparison,
        "source_ids": sources,
        "generator": "scripts/oracle_fixtures/generate_remaining_core_records.py",
        "raw_evidence": ["raw/SRC-NUMPY-FORMULA/remaining-core.json"],
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
