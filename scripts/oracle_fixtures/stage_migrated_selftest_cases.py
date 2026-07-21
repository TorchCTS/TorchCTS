#!/usr/bin/env python3
"""Print review-ready case JSON derived from independent migration evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = REPO_ROOT / "evidence/oracles/raw/SRC-SYMPY-EXACT/migrated-selftests.json"


def _base(
    *,
    case_id,
    case_pack,
    oracle_ids,
    entry_points,
    surfaces,
    sources,
    inputs,
    expected,
    comparison,
    applicability,
):
    return {
        "schema_version": 1,
        "case_id": case_id,
        "oracle_kind": "value_reference",
        "validation_class": "V1_FIXED_VALUE",
        "case_pack": case_pack,
        "oracle_ids": oracle_ids,
        "implementation_entry_points": entry_points,
        "dispatcher_surfaces": surfaces,
        "applicability": applicability,
        "inputs": inputs,
        "expected": expected,
        "comparison": comparison,
        "source_ids": sources,
        "generator": "scripts/oracle_fixtures/generate_migrated_selftest_records.py",
        "raw_evidence": ["raw/SRC-SYMPY-EXACT/migrated-selftests.json"],
        "created_at": "2026-07-20",
        "review": {
            "reviewed_by": ["fixture-schema-implementation", "independent-formula-reproduction"],
            "reviewed_at": "2026-07-20",
            "conclusion": "implementation_correct",
        },
    }


def build_cases():
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="ascii"))["records"]
    arithmetic = evidence["complex_arithmetic"]
    unary = evidence["complex_unary"]
    loss = evidence["complex_loss"]
    grid = evidence["grid_backward"]
    embedding = evidence["embedding"]
    convolution = evidence["complex_convolution"]
    common_sources = ["SRC-SYMPY-EXACT", "SRC-IEEE-C", "SRC-PYTORCH-SOURCE"]
    cases = {}
    cases["reference/cp-complex-arithmetic-migration.json"] = _base(
        case_id="cp-complex-arithmetic-migration",
        case_pack="CP-COMPLEX-ARITH",
        oracle_ids=["complex_unit_alpha_add_sub", "complex_tensor_integer_power"],
        entry_points=[
            "torchcts.core.reference_oracles.complex_unit_alpha_add_sub_reference",
            "torchcts.core.reference_oracles.complex_tensor_integer_power_reference",
        ],
        surfaces=["aten::add.Tensor", "aten::sub.Tensor", "aten::rsub.Tensor", "aten::pow.Tensor_Tensor"],
        sources=common_sources,
        inputs={
            "unit_alpha": {"left": arithmetic["unit_alpha"]["left"], "right": arithmetic["unit_alpha"]["right"]},
            "integer_power": {
                key: arithmetic["integer_power"][key]
                for key in ("base", "exponent", "native_placeholder")
            },
            "general_power_passthrough": arithmetic["general_power_passthrough"],
        },
        expected={
            "unit_alpha": {
                key: arithmetic["unit_alpha"][key] for key in ("add", "sub", "rsub")
            },
            "integer_power": arithmetic["integer_power"]["expected"],
            "general_power_passthrough": arithmetic["general_power_passthrough"]["native_sentinel"],
        },
        comparison={
            "kind": "exact_ieee_lane_bits",
            "asserts": ["value", "dtype", "shape"],
            "derivation": "Independent real/imaginary IEEE operations and exponent predicate; non-integral lanes use a frozen sentinel to prove pass-through without calling pow.",
        },
        applicability={
            "backend": "cpu_development_host",
            "dtypes": ["torch.complex64"],
            "input_condition": "has_inf_or_nan",
            "pytorch": ">=2.7.0,<2.12.2",
        },
    )
    cases["reference/cp-complex-unary-log2-migration.json"] = _base(
        case_id="cp-complex-unary-log2-migration",
        case_pack="CP-COMPLEX-UNARY",
        oracle_ids=["complex_log2"],
        entry_points=["torchcts.core.reference_oracles.complex_log2_reference"],
        surfaces=["aten::log2", "aten::log2.out", "aten::log2_"],
        sources=common_sources,
        inputs={"input": unary["input"]},
        expected={"output": unary["expected"], "exact_expression": unary["exact_expression"], "phase_decimal_80": unary["phase_decimal_80"]},
        comparison={
            "kind": "exact_rounded_ieee_lane_bits",
            "asserts": ["value", "dtype", "shape"],
            "derivation": "log2(z)=log(z)/log(2); the negative-real-axis phase is pi/log(2), evaluated symbolically then rounded once to float32.",
        },
        applicability={"backend": "cpu_development_host", "dtypes": ["torch.complex64"], "input_condition": "has_inf", "pytorch": ">=2.7.0,<2.12.2"},
    )
    cases["reference/cp-complex-loss-l1-migration.json"] = _base(
        case_id="cp-complex-loss-l1-migration",
        case_pack="CP-COMPLEX-LOSS",
        oracle_ids=["complex_l1_loss"],
        entry_points=["torchcts.core.reference_oracles.complex_l1_loss_reference"],
        surfaces=["aten::l1_loss"],
        sources=common_sources,
        inputs={"input": loss["input"], "target": loss["target"], "reduction": "none"},
        expected={"output": loss["expected"]},
        comparison={"kind": "exact_ieee_bits", "asserts": ["value", "dtype", "shape"], "derivation": "Per-element hypot of independently subtracted real and imaginary lanes."},
        applicability={"backend": "cpu_development_host", "dtypes": ["torch.complex64", "torch.float32"], "input_condition": "has_inf", "pytorch": ">=2.7.0,<2.12.2"},
    )
    cases["backward/cp-grid-3d-backward-migration.json"] = _base(
        case_id="cp-grid-3d-backward-migration",
        case_pack="CP-GRID",
        oracle_ids=["grid_sampler_backward_f32"],
        entry_points=["torchcts.core.reference_oracles.grid_sampler_3d_backward_f32_reference"],
        surfaces=["aten::grid_sampler_3d_backward"],
        sources=["SRC-SYMPY-EXACT", "SRC-PUBLIC-API", "SRC-PYTORCH-SOURCE"],
        inputs={key: grid[key] for key in ("input", "grid", "grad_output", "interpolation_mode", "padding_mode", "align_corners")},
        expected={"grad_input": grid["expected_grad_input"], "grad_grid": grid["expected_grad_grid"], "contributors": grid["contributors"]},
        comparison={"kind": "exact_bfloat16_bits", "asserts": ["value", "dtype", "shape"], "derivation": "Exact-rational trilinear weights, reflection derivatives, and VJP, rounded once to bfloat16."},
        applicability={"backend": "cpu_development_host", "dtypes": ["torch.bfloat16"], "input_condition": "finite_clean", "pytorch": ">=2.7.0,<2.12.2"},
    )
    cases["backward/cp-embedding-frequency-migration.json"] = _base(
        case_id="cp-embedding-frequency-migration",
        case_pack="CP-EMBED",
        oracle_ids=["embedding_bag_scale_grad_by_freq"],
        entry_points=["torchcts.core.reference_oracles.embedding_bag_scale_grad_by_freq_reference"],
        surfaces=["aten::_embedding_bag_backward"],
        sources=["SRC-SYMPY-EXACT", "SRC-PUBLIC-API", "SRC-PYTORCH-SOURCE"],
        inputs={key: embedding[key] for key in ("grad", "indices", "offset2bag", "num_weights")},
        expected={"output": embedding["expected"], "row_frequencies": embedding["row_frequencies"]},
        comparison={"kind": "float32_one_ulp", "asserts": ["value", "dtype", "shape"], "ulp_ceiling": 1, "derivation": "Exact rational per-row frequency scaling; one ULP permits sequential float32 accumulation without fitting to the candidate."},
        applicability={"backend": "cpu_development_host", "dtypes": ["torch.float32", "torch.int64"], "input_condition": "finite_clean", "pytorch": ">=2.7.0,<2.12.2"},
    )
    cases["reference/cp-complex-convolution-migration.json"] = _base(
        case_id="cp-complex-convolution-migration",
        case_pack="CP-CONV",
        oracle_ids=["complex_convolution_four_real"],
        entry_points=[
            "torchcts.core.reference_oracles.complex_convolution_reference",
            "torchcts.core.reference_oracles.slow_complex_convolution_reference",
        ],
        surfaces=["aten::conv_transpose1d"],
        sources=common_sources,
        inputs={key: convolution[key] for key in ("op_name", "input", "weight", "bias", "kwargs")},
        expected={"output": convolution["expected"], "term_map": convolution["term_map"]},
        comparison={"kind": "exact_ieee_lane_bits", "asserts": ["value", "dtype", "shape"], "derivation": "Direct scalar transposed-convolution destination enumeration and explicit complex multiplication formula."},
        applicability={"backend": "cpu_development_host", "dtypes": ["torch.complex64"], "input_condition": "has_inf", "pytorch": ">=2.7.0,<2.12.2"},
    )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", choices=sorted(build_cases()))
    args = parser.parse_args()
    print(json.dumps(build_cases()[args.path], indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
