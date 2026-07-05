"""Helpers for deterministic path-shape corpus specs."""

from __future__ import annotations

import sys
from typing import Any


FAMILY_BUDGETS = {
    "matmul": {"default_target": 140, "default_hard_max": 160, "heavy_target": 70, "total_hard_max": 250},
    "convolution": {"default_target": 90, "default_hard_max": 110, "heavy_target": 50, "total_hard_max": 180},
    "attention": {"default_target": 80, "default_hard_max": 100, "heavy_target": 70, "total_hard_max": 190},
    "reduction": {"default_target": 85, "default_hard_max": 105, "heavy_target": 35, "total_hard_max": 150},
    "indexing": {"default_target": 80, "default_hard_max": 100, "heavy_target": 30, "total_hard_max": 140},
    "sorting": {"default_target": 55, "default_hard_max": 70, "heavy_target": 25, "total_hard_max": 110},
    "fft": {"default_target": 45, "default_hard_max": 60, "heavy_target": 40, "total_hard_max": 110},
    "normalization": {"default_target": 65, "default_hard_max": 80, "heavy_target": 25, "total_hard_max": 120},
    "spatial": {"default_target": 65, "default_hard_max": 80, "heavy_target": 35, "total_hard_max": 130},
    "linear_algebra": {"default_target": 35, "default_hard_max": 50, "heavy_target": 30, "total_hard_max": 90},
    "model_patterns": {"default_target": 55, "default_hard_max": 75, "heavy_target": 40, "total_hard_max": 130},
    "broadcasting": {"default_target": 55, "default_hard_max": 70, "heavy_target": 20, "total_hard_max": 100},
}

SUITE_BUDGET = {
    "max_default_extra_ratio": 0.05,
    "max_all_tiers_extra_ratio": 0.10,
    "default_target": 850,
    "default_hard_max": 950,
    "all_tiers_hard_max": 1800,
}

DTYPE_GROUPS = {
    "float": ["torch.float32", "torch.float64", "torch.float16", "torch.bfloat16"],
    "integer": ["torch.int8", "torch.int16", "torch.int32", "torch.int64"],
    "bool": ["torch.bool"],
    "complex": ["torch.complex64"],
}

RESOURCE_TIERS = {
    "smoke": {"default": True},
    "standard": {"default": True},
    "heavy": {"default": False},
    "stress": {"default": False},
}

COVERS_BY_RUNNER = {
    "matmul.mm": ["aten::mm"],
    "matmul.bmm": ["aten::bmm"],
    "matmul.matmul": ["aten::matmul"],
    "matmul.addmm": ["aten::addmm"],
    "matmul.linear": ["aten::linear"],
    "convolution.conv1d": ["aten::convolution"],
    "convolution.conv2d": ["aten::convolution"],
    "convolution.conv3d": ["aten::convolution"],
    "convolution.conv_transpose2d": ["aten::convolution"],
    "attention.sdpa": ["aten::scaled_dot_product_attention"],
    "reduction.sum": ["aten::sum.dim_IntList"],
    "reduction.mean": ["aten::mean.dim"],
    "reduction.amax": ["aten::amax"],
    "reduction.argmax": ["aten::argmax"],
    "reduction.prod": ["aten::prod.dim_int"],
    "indexing.index_select": ["aten::index_select"],
    "indexing.gather": ["aten::gather"],
    "indexing.scatter_add": ["aten::scatter_add_"],
    "indexing.scatter_reduce": ["aten::scatter_reduce_"],
    "indexing.take": ["aten::take"],
    "indexing.masked_select": ["aten::masked_select"],
    "sorting.sort": ["aten::sort"],
    "sorting.topk": ["aten::topk"],
    "sorting.kthvalue": ["aten::kthvalue"],
    "fft.rfft": ["aten::fft_rfft"],
    "fft.irfft": ["aten::fft_irfft"],
    "fft.fft": ["aten::fft_fft"],
    "fft.fft2": ["aten::fft_fft2"],
    "normalization.layer_norm": ["aten::layer_norm"],
    "normalization.group_norm": ["aten::group_norm"],
    "normalization.batch_norm": ["aten::batch_norm"],
    "spatial.avg_pool2d": ["aten::avg_pool2d"],
    "spatial.max_pool2d": ["aten::max_pool2d_with_indices"],
    "spatial.adaptive_avg_pool2d": ["aten::adaptive_avg_pool2d"],
    "spatial.interpolate": ["aten::upsample_nearest2d", "aten::upsample_bilinear2d"],
    "linear_algebra.solve": ["aten::linalg_solve"],
    "linear_algebra.cholesky": ["aten::linalg_cholesky"],
    "linear_algebra.qr": ["aten::linalg_qr"],
    "linear_algebra.eigh": ["aten::linalg_eigh"],
    "linear_algebra.svdvals": ["aten::linalg_svdvals"],
    "model_patterns.vision_block": ["aten::convolution", "aten::relu"],
    "model_patterns.transformer_block_fragment": ["aten::scaled_dot_product_attention"],
    "model_patterns.patch_embedding": ["aten::convolution"],
    "model_patterns.depthwise_separable": ["aten::convolution"],
    "model_patterns.residual_norm": ["aten::add.Tensor", "aten::layer_norm"],
    "broadcasting.where": ["aten::where.self"],
    "broadcasting.add": ["aten::add.Tensor"],
    "broadcasting.mul": ["aten::mul.Tensor"],
    "broadcasting.masked_fill": ["aten::masked_fill.Scalar"],
}

FAMILY_CATEGORY = {
    "model_patterns": "model_pattern_path_shape",
    "linear_algebra": "linear_algebra_path_shape",
}


def dtype_suffix(dtype: str) -> str:
    return dtype.split(".", 1)[1].replace("float", "f").replace("bfloat", "bf").replace("complex", "c")


def slug(parts: list[Any] | tuple[Any, ...]) -> str:
    return "_".join(str(part).replace("-", "m").replace(".", "p").replace(" ", "_") for part in parts if part != "")


def cost_limits(cost_class: str) -> dict[str, int]:
    if cost_class == "tiny":
        return {"max_tensor_mb": 8, "max_workspace_mb": 32}
    if cost_class == "small":
        return {"max_tensor_mb": 16, "max_workspace_mb": 64}
    if cost_class == "medium":
        return {"max_tensor_mb": 64, "max_workspace_mb": 256}
    return {"max_tensor_mb": 256, "max_workspace_mb": 1024}


def case(
    *,
    runner: str,
    family: str,
    name: str,
    shape: dict[str, Any],
    tier: str = "standard",
    cost_class: str = "tiny",
    dtype: str = "torch.float32",
    suite: str = "operators",
    semantic_level: int = 5,
    level_reason: str | None = None,
    categories: list[str] | None = None,
    layout: str = "contiguous",
    stride_mode: str = "contiguous",
    model_role: str = "synthetic_branch_probe",
    branch_intent: list[str] | None = None,
    reference: str | None = None,
    source_note: str | None = None,
    device_expectation: dict[str, str] | None = None,
) -> dict[str, Any]:
    dtype_group = next(
        (group for group, values in DTYPE_GROUPS.items() if dtype in values),
        None,
    )
    if dtype_group is None:
        raise ValueError(f"unknown path-shape dtype {dtype!r}")
    categories = list(categories or [])
    family_category = FAMILY_CATEGORY.get(family, f"{family}_path_shape")
    for category in ("path_shape", "algorithmic_shape", family_category):
        if category not in categories:
            categories.append(category)
    if layout not in {"contiguous", "nchw", "bhsd"} and "layout_path_shape" not in categories:
        categories.append("layout_path_shape")
    if tier in {"heavy", "stress"} and "resource_boundary" not in categories:
        categories.append("resource_boundary")
    if tier == "stress" and "adversarial_boundary" not in categories:
        categories.append("adversarial_boundary")
    branch_intent = branch_intent or [name]
    case_id = slug([family, runner.split(".", 1)[1], name, dtype_suffix(dtype), tier])
    return {
        "case_id": case_id,
        "runner": runner,
        "family": family,
        "suite": suite,
        "semantic_level": semantic_level,
        "level_reason": level_reason or f"{family} {name} exercises {', '.join(branch_intent)}.",
        "categories": categories,
        "resource_tier": tier,
        "cost_class": cost_class,
        "dtype": dtype,
        "dtype_group": dtype_group,
        "model_role": model_role,
        "branch_intent": branch_intent,
        "device_expectation": device_expectation or {
            "cpu": "must_pass",
            "cuda": "should_pass",
            "mps": "should_pass",
            "privateuse1": "may_skip",
        },
        "shape": shape,
        "layout": layout,
        "stride_mode": stride_mode,
        "reference": reference or f"{runner}_cpu_reference",
        "source_note": source_note or f"path-shape-catalog.md#{family}-{runner.split('.', 1)[1]}",
        "limits": cost_limits(cost_class),
        "covers": COVERS_BY_RUNNER[runner],
    }


def limit(cases: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(cases) < count:
        raise ValueError(f"spec generated {len(cases)} cases but requested {count}")
    return cases[:count]


def base_corpus(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "suite_budget": SUITE_BUDGET,
        "collection_baseline": {
            "test_count": 18953,
            "command": "python -m pytest --collect-only -q torchcts --validation --level 8",
            "date_measured": "2026-07-03",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch_version": "2.12.1",
            "torchcts_version": "0.3.5",
            "notes": "Baseline before expanded path-shape corpus.",
        },
        "resource_tiers": RESOURCE_TIERS,
        "dtype_groups": DTYPE_GROUPS,
        "family_budgets": FAMILY_BUDGETS,
        "cases": cases,
        "waivers": [],
    }
