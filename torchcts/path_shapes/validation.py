"""Validation for the curated path-shape corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

from torchcts.core.semantic_levels import validate_semantic_level


PATH_SHAPE_CATEGORIES = frozenset({
    "path_shape",
    "algorithmic_shape",
    "matmul_path_shape",
    "convolution_path_shape",
    "attention_path_shape",
    "reduction_path_shape",
    "fft_path_shape",
    "indexing_path_shape",
    "spatial_path_shape",
    "normalization_path_shape",
    "model_pattern_path_shape",
    "sorting_path_shape",
    "linear_algebra_path_shape",
    "broadcasting_path_shape",
    "layout_path_shape",
    "resource_boundary",
    "adversarial_boundary",
})

PATH_SHAPE_FAMILIES = frozenset({
    "matmul",
    "convolution",
    "attention",
    "reduction",
    "indexing",
    "sorting",
    "fft",
    "normalization",
    "spatial",
    "linear_algebra",
    "model_patterns",
    "broadcasting",
})

PATH_SHAPE_FAMILY_ALIASES = {
    "conv": "convolution",
    "sdpa": "attention",
    "sort": "sorting",
    "linalg": "linear_algebra",
    "workloads": "model_patterns",
    "pooling": "spatial",
    "pooling_spatial": "spatial",
    "vision": "model_patterns",
    "pointwise_broadcast": "broadcasting",
}

PATH_SHAPE_RESOURCE_TIERS = frozenset({"smoke", "standard", "heavy", "stress"})
PATH_SHAPE_COST_CLASSES = frozenset({"tiny", "small", "medium", "large"})
PATH_SHAPE_DEVICE_EXPECTATIONS = frozenset({"must_pass", "should_pass", "may_skip", "not_applicable"})

PATH_SHAPE_RUNNERS = frozenset({
    "matmul.mm",
    "matmul.bmm",
    "matmul.matmul",
    "matmul.addmm",
    "matmul.linear",
    "convolution.conv1d",
    "convolution.conv2d",
    "convolution.conv3d",
    "convolution.conv_transpose2d",
    "attention.sdpa",
    "reduction.sum",
    "reduction.mean",
    "reduction.amax",
    "reduction.argmax",
    "reduction.prod",
    "indexing.index_select",
    "indexing.gather",
    "indexing.scatter_add",
    "indexing.scatter_reduce",
    "indexing.take",
    "indexing.masked_select",
    "sorting.sort",
    "sorting.topk",
    "sorting.kthvalue",
    "fft.rfft",
    "fft.irfft",
    "fft.fft",
    "fft.fft2",
    "normalization.layer_norm",
    "normalization.group_norm",
    "normalization.batch_norm",
    "spatial.avg_pool2d",
    "spatial.max_pool2d",
    "spatial.adaptive_avg_pool2d",
    "spatial.interpolate",
    "linear_algebra.solve",
    "linear_algebra.cholesky",
    "linear_algebra.qr",
    "linear_algebra.eigh",
    "linear_algebra.svdvals",
    "model_patterns.vision_block",
    "model_patterns.transformer_block_fragment",
    "model_patterns.patch_embedding",
    "model_patterns.depthwise_separable",
    "model_patterns.residual_norm",
    "broadcasting.where",
    "broadcasting.add",
    "broadcasting.mul",
    "broadcasting.masked_fill",
})

PATH_SHAPE_SUITES = frozenset({"operators", "workloads", "strides", "stress"})
_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_FORBIDDEN_AXIS_KEYS = frozenset({
    "dtypes",
    "layouts",
    "stride_modes",
    "shapes",
    "batch_sizes",
    "cartesian",
    "batched",
})


class PathShapeValidationError(ValueError):
    """Raised when the path-shape corpus is malformed."""


def _error(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def _require_mapping(errors: list[str], value: Any, path: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        _error(errors, path, "must be an object")
        return None
    return value


def _require_string(errors: list[str], value: Any, path: str, *, nonempty: bool = True) -> str | None:
    if not isinstance(value, str):
        _error(errors, path, "must be a string")
        return None
    if nonempty and not value:
        _error(errors, path, "must not be empty")
        return None
    return value


def _require_string_list(errors: list[str], value: Any, path: str) -> list[str] | None:
    if not isinstance(value, list):
        _error(errors, path, "must be a list")
        return None
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            _error(errors, f"{path}[{index}]", "must be a non-empty string")
        else:
            result.append(item)
    return result


def _require_positive_int(errors: list[str], value: Any, path: str) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _error(errors, path, "must be a positive integer")
        return None
    return value


def _require_nonnegative_number(errors: list[str], value: Any, path: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        _error(errors, path, "must be a non-negative number")


def _validate_shape_values(errors: list[str], value: Any, path: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        if value < 0:
            _error(errors, path, "shape integers must be non-negative")
        return
    if isinstance(value, float):
        if value < 0:
            _error(errors, path, "shape numbers must be non-negative")
        return
    if isinstance(value, str) or value is None:
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_shape_values(errors, item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_shape_values(errors, item, f"{path}.{key}")
        return
    _error(errors, path, "contains unsupported shape metadata")


def _default_resource_tiers(resource_tiers: dict[str, Any]) -> set[str]:
    return {
        tier
        for tier, metadata in resource_tiers.items()
        if isinstance(metadata, dict) and metadata.get("default") is True
    }


def _budget_value(errors: list[str], budget_map: dict[str, Any], path: str, key: str) -> int | None:
    return _require_positive_int(errors, budget_map.get(key), f"{path}.{key}")


def validate_path_shape_corpus(
    corpus: dict[str, Any],
    *,
    strict_budget: bool = False,
    enforce_targets: bool = False,
) -> dict[str, Any]:
    """Validate and return a deterministic summary for a path-shape corpus."""

    errors: list[str] = []
    budget_warnings: list[str] = []
    root = _require_mapping(errors, corpus, "corpus")
    if root is None:
        raise PathShapeValidationError("\n".join(errors))

    if root.get("schema_version") != 2:
        _error(errors, "schema_version", "must be 2")

    suite_budget = _require_mapping(errors, root.get("suite_budget"), "suite_budget") or {}
    suite_default_target = _budget_value(errors, suite_budget, "suite_budget", "default_target")
    suite_default_hard_max = _budget_value(errors, suite_budget, "suite_budget", "default_hard_max")
    suite_all_tiers_hard_max = _budget_value(errors, suite_budget, "suite_budget", "all_tiers_hard_max")
    for ratio_key in ("max_default_extra_ratio", "max_all_tiers_extra_ratio"):
        value = suite_budget.get(ratio_key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            _error(errors, f"suite_budget.{ratio_key}", "must be a positive number")

    baseline = _require_mapping(errors, root.get("collection_baseline"), "collection_baseline") or {}
    _require_positive_int(errors, baseline.get("test_count"), "collection_baseline.test_count")
    for key in ("command", "date_measured", "python_version", "torch_version", "torchcts_version"):
        _require_string(errors, baseline.get(key), f"collection_baseline.{key}")

    resource_tiers = _require_mapping(errors, root.get("resource_tiers"), "resource_tiers") or {}
    for tier, metadata in resource_tiers.items():
        if tier not in PATH_SHAPE_RESOURCE_TIERS:
            _error(errors, f"resource_tiers.{tier}", "unknown resource tier")
        if not isinstance(metadata, dict):
            _error(errors, f"resource_tiers.{tier}", "must be an object")
        elif not isinstance(metadata.get("default"), bool):
            _error(errors, f"resource_tiers.{tier}.default", "must be boolean")

    dtype_groups = _require_mapping(errors, root.get("dtype_groups"), "dtype_groups") or {}
    dtype_group_values: dict[str, set[str]] = {}
    for group, values in dtype_groups.items():
        group_values = _require_string_list(errors, values, f"dtype_groups.{group}") or []
        dtype_group_values[group] = set(group_values)
        for dtype in group_values:
            if not dtype.startswith("torch."):
                _error(errors, f"dtype_groups.{group}", f"invalid dtype {dtype!r}")

    family_budgets = _require_mapping(errors, root.get("family_budgets"), "family_budgets") or {}
    missing_budget_families = sorted(PATH_SHAPE_FAMILIES - set(family_budgets))
    for family in missing_budget_families:
        _error(errors, f"family_budgets.{family}", "is required")
    for family, budget in family_budgets.items():
        if family in PATH_SHAPE_FAMILY_ALIASES:
            _error(errors, f"family_budgets.{family}", f"use {PATH_SHAPE_FAMILY_ALIASES[family]!r}")
        elif family not in PATH_SHAPE_FAMILIES:
            _error(errors, f"family_budgets.{family}", "unknown family")
        budget_map = _require_mapping(errors, budget, f"family_budgets.{family}") or {}
        values = {
            key: _budget_value(errors, budget_map, f"family_budgets.{family}", key)
            for key in ("default_target", "default_hard_max", "heavy_target", "total_hard_max")
        }
        if values["default_target"] and values["default_hard_max"] and values["default_target"] > values["default_hard_max"]:
            _error(errors, f"family_budgets.{family}", "default_target must be <= default_hard_max")
        if values["default_hard_max"] and values["total_hard_max"] and values["default_hard_max"] > values["total_hard_max"]:
            _error(errors, f"family_budgets.{family}", "default_hard_max must be <= total_hard_max")

    cases = root.get("cases")
    if not isinstance(cases, list):
        _error(errors, "cases", "must be a list")
        cases = []

    seen_ids: set[str] = set()
    by_family = Counter()
    by_resource_tier = Counter()
    by_semantic_level = Counter()
    by_cost_class = Counter()
    by_runner = Counter()
    by_dtype = Counter()
    by_dtype_group = Counter()
    default_by_family = Counter()
    all_by_family = Counter()
    default_total = 0

    default_tiers = _default_resource_tiers(resource_tiers)

    required_fields = (
        "case_id",
        "runner",
        "family",
        "suite",
        "semantic_level",
        "level_reason",
        "categories",
        "resource_tier",
        "cost_class",
        "dtype",
        "dtype_group",
        "model_role",
        "branch_intent",
        "device_expectation",
        "shape",
        "reference",
        "source_note",
        "limits",
        "covers",
    )

    for index, case in enumerate(cases):
        path = f"cases[{index}]"
        case_map = _require_mapping(errors, case, path)
        if case_map is None:
            continue

        for key in sorted(_FORBIDDEN_AXIS_KEYS & set(case_map)):
            _error(errors, f"{path}.{key}", "cartesian axis fields are not allowed in executable corpus rows")

        for field in required_fields:
            if field not in case_map:
                _error(errors, f"{path}.{field}", "is required")

        case_id = _require_string(errors, case_map.get("case_id"), f"{path}.case_id")
        if case_id:
            if not _CASE_ID_RE.match(case_id):
                _error(errors, f"{path}.case_id", "must match ^[a-z0-9][a-z0-9_]*$")
            if case_id in seen_ids:
                _error(errors, f"{path}.case_id", f"duplicate case_id {case_id!r}")
            seen_ids.add(case_id)

        runner = _require_string(errors, case_map.get("runner"), f"{path}.runner")
        if runner and runner not in PATH_SHAPE_RUNNERS:
            _error(errors, f"{path}.runner", f"unknown runner {runner!r}")

        family = _require_string(errors, case_map.get("family"), f"{path}.family")
        if family in PATH_SHAPE_FAMILY_ALIASES:
            _error(errors, f"{path}.family", f"use {PATH_SHAPE_FAMILY_ALIASES[family]!r}")
        elif family and family not in PATH_SHAPE_FAMILIES:
            _error(errors, f"{path}.family", f"unknown family {family!r}")

        if runner and family and runner.split(".", 1)[0] != family:
            if not (family == "model_patterns" and runner.startswith("model_patterns.")):
                _error(errors, f"{path}.runner", "runner prefix must match family")

        suite = _require_string(errors, case_map.get("suite"), f"{path}.suite")
        if suite and suite not in PATH_SHAPE_SUITES:
            _error(errors, f"{path}.suite", f"unknown suite {suite!r}")

        try:
            validate_semantic_level(case_map.get("semantic_level"), field_name=f"{path}.semantic_level")
        except Exception as exc:
            _error(errors, f"{path}.semantic_level", str(exc))

        _require_string(errors, case_map.get("level_reason"), f"{path}.level_reason")
        _require_string(errors, case_map.get("model_role"), f"{path}.model_role")
        _require_string(errors, case_map.get("reference"), f"{path}.reference")
        _require_string(errors, case_map.get("source_note"), f"{path}.source_note")

        shape = _require_mapping(errors, case_map.get("shape"), f"{path}.shape")
        if shape is not None:
            _validate_shape_values(errors, shape, f"{path}.shape")

        limits = _require_mapping(errors, case_map.get("limits"), f"{path}.limits") or {}
        for key in ("max_tensor_mb", "max_workspace_mb"):
            if key in limits:
                _require_nonnegative_number(errors, limits[key], f"{path}.limits.{key}")
        if "max_tensor_mb" not in limits:
            _error(errors, f"{path}.limits.max_tensor_mb", "is required")

        categories = _require_string_list(errors, case_map.get("categories"), f"{path}.categories") or []
        if "path_shape" not in categories:
            _error(errors, f"{path}.categories", "must include 'path_shape'")
        for category in categories:
            if category not in PATH_SHAPE_CATEGORIES:
                _error(errors, f"{path}.categories", f"unknown category {category!r}")

        tier = _require_string(errors, case_map.get("resource_tier"), f"{path}.resource_tier")
        if tier and tier not in PATH_SHAPE_RESOURCE_TIERS:
            _error(errors, f"{path}.resource_tier", f"unknown resource tier {tier!r}")

        cost_class = _require_string(errors, case_map.get("cost_class"), f"{path}.cost_class")
        if cost_class and cost_class not in PATH_SHAPE_COST_CLASSES:
            _error(errors, f"{path}.cost_class", f"unknown cost class {cost_class!r}")

        dtype = _require_string(errors, case_map.get("dtype"), f"{path}.dtype")
        if dtype and not dtype.startswith("torch."):
            _error(errors, f"{path}.dtype", "must be a concrete torch dtype string")
        dtype_group = _require_string(errors, case_map.get("dtype_group"), f"{path}.dtype_group")
        if dtype_group and dtype_group not in dtype_group_values:
            _error(errors, f"{path}.dtype_group", f"unknown dtype group {dtype_group!r}")
        elif dtype and dtype_group and dtype not in dtype_group_values[dtype_group]:
            _error(errors, f"{path}.dtype", f"{dtype!r} is not in dtype group {dtype_group!r}")

        branch_intent = _require_string_list(errors, case_map.get("branch_intent"), f"{path}.branch_intent") or []
        if not branch_intent:
            _error(errors, f"{path}.branch_intent", "must not be empty")

        covers = _require_string_list(errors, case_map.get("covers"), f"{path}.covers") or []
        if not covers:
            _error(errors, f"{path}.covers", "must not be empty")

        device_expectation = _require_mapping(errors, case_map.get("device_expectation"), f"{path}.device_expectation") or {}
        for device_name, expectation in device_expectation.items():
            if device_name == "notes":
                if not isinstance(expectation, str):
                    _error(errors, f"{path}.device_expectation.notes", "must be a string")
                continue
            if expectation not in PATH_SHAPE_DEVICE_EXPECTATIONS:
                _error(errors, f"{path}.device_expectation.{device_name}", f"unknown expectation {expectation!r}")
        if "cpu" not in device_expectation:
            _error(errors, f"{path}.device_expectation.cpu", "is required")

        if family:
            by_family[family] += 1
            all_by_family[family] += 1
        if runner:
            by_runner[runner] += 1
        if tier:
            by_resource_tier[tier] += 1
        if cost_class:
            by_cost_class[cost_class] += 1
        if dtype:
            by_dtype[dtype] += 1
        if dtype_group:
            by_dtype_group[dtype_group] += 1
        try:
            by_semantic_level[str(validate_semantic_level(case_map.get("semantic_level")))] += 1
        except Exception:
            pass
        if tier in default_tiers:
            default_total += 1
            if family:
                default_by_family[family] += 1

    if suite_default_hard_max is not None and default_total > suite_default_hard_max:
        _error(errors, "suite_budget.default_hard_max", f"default-selected cases {default_total} exceed {suite_default_hard_max}")
    if suite_all_tiers_hard_max is not None and len(cases) > suite_all_tiers_hard_max:
        _error(errors, "suite_budget.all_tiers_hard_max", f"all-tier cases {len(cases)} exceed {suite_all_tiers_hard_max}")
    if suite_default_target is not None and default_total < suite_default_target:
        budget_warnings.append(f"default-selected cases {default_total} are below target {suite_default_target}")

    if strict_budget and isinstance(baseline.get("test_count"), int):
        baseline_count = baseline["test_count"]
        default_ratio = suite_budget.get("max_default_extra_ratio")
        all_tiers_ratio = suite_budget.get("max_all_tiers_extra_ratio")
        if isinstance(default_ratio, (int, float)) and not isinstance(default_ratio, bool):
            default_ratio_limit = int(baseline_count * default_ratio)
            if default_total > default_ratio_limit:
                _error(
                    errors,
                    "suite_budget.max_default_extra_ratio",
                    f"default-selected cases {default_total} exceed baseline ratio limit {default_ratio_limit}",
                )
        if isinstance(all_tiers_ratio, (int, float)) and not isinstance(all_tiers_ratio, bool):
            all_tiers_ratio_limit = int(baseline_count * all_tiers_ratio)
            if len(cases) > all_tiers_ratio_limit:
                _error(
                    errors,
                    "suite_budget.max_all_tiers_extra_ratio",
                    f"all-tier cases {len(cases)} exceed baseline ratio limit {all_tiers_ratio_limit}",
                )

    for family in sorted(PATH_SHAPE_FAMILIES):
        budget = family_budgets.get(family) or {}
        default_count = default_by_family.get(family, 0)
        total_count = all_by_family.get(family, 0)
        default_target = budget.get("default_target")
        default_hard_max = budget.get("default_hard_max")
        heavy_target = budget.get("heavy_target")
        total_hard_max = budget.get("total_hard_max")
        if isinstance(default_hard_max, int) and default_count > default_hard_max:
            _error(errors, f"family_budgets.{family}.default_hard_max", f"default-selected cases {default_count} exceed {default_hard_max}")
        if isinstance(total_hard_max, int) and total_count > total_hard_max:
            _error(errors, f"family_budgets.{family}.total_hard_max", f"all-tier cases {total_count} exceed {total_hard_max}")
        if isinstance(default_target, int) and default_count < default_target:
            budget_warnings.append(f"{family} default-selected cases {default_count} are below target {default_target}")
        heavy_count = sum(1 for case in cases if case.get("family") == family and case.get("resource_tier") == "heavy")
        if isinstance(heavy_target, int) and heavy_count < heavy_target:
            budget_warnings.append(f"{family} heavy cases {heavy_count} are below target {heavy_target}")

    if enforce_targets:
        for warning in budget_warnings:
            _error(errors, "budget_targets", warning)

    if errors:
        raise PathShapeValidationError("\n".join(errors))

    return {
        "schema_version": root.get("schema_version"),
        "case_count": len(cases),
        "default_selected_case_count": default_total,
        "by_family": dict(sorted(by_family.items())),
        "by_resource_tier": dict(sorted(by_resource_tier.items())),
        "by_semantic_level": dict(sorted(by_semantic_level.items(), key=lambda item: int(item[0]))),
        "by_cost_class": dict(sorted(by_cost_class.items())),
        "by_runner": dict(sorted(by_runner.items())),
        "by_dtype": dict(sorted(by_dtype.items())),
        "by_dtype_group": dict(sorted(by_dtype_group.items())),
        "default_selected_by_family": dict(sorted(default_by_family.items())),
        "resource_tiers": sorted(resource_tiers),
        "default_resource_tiers": sorted(default_tiers),
        "family_budgets": family_budgets,
        "suite_budget": suite_budget,
        "collection_baseline": baseline,
        "budget_warnings": budget_warnings,
        "waiver_count": len(root.get("waivers") or []),
    }


def corpus_case_index(corpus: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validate_path_shape_corpus(corpus)
    return {case["case_id"]: case for case in corpus.get("cases", [])}


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: defaultdict[str, int] = defaultdict(int)
    by_resource_tier: defaultdict[str, int] = defaultdict(int)
    by_semantic_level: defaultdict[str, int] = defaultdict(int)
    by_cost_class: defaultdict[str, int] = defaultdict(int)
    by_runner: defaultdict[str, int] = defaultdict(int)
    for case in cases:
        by_family[str(case.get("family") or "unknown")] += 1
        by_resource_tier[str(case.get("resource_tier") or "unknown")] += 1
        by_semantic_level[str(case.get("semantic_level") or "unknown")] += 1
        by_cost_class[str(case.get("cost_class") or "unknown")] += 1
        by_runner[str(case.get("runner") or "unknown")] += 1
    return {
        "case_count": len(cases),
        "by_family": dict(sorted(by_family.items())),
        "by_resource_tier": dict(sorted(by_resource_tier.items())),
        "by_semantic_level": dict(sorted(by_semantic_level.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 99)),
        "by_cost_class": dict(sorted(by_cost_class.items())),
        "by_runner": dict(sorted(by_runner.items())),
    }
