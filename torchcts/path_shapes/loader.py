"""Load, filter, and materialize the path-shape corpus for pytest."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from torchcts.path_shapes.validation import (
    PathShapeValidationError,
    summarize_cases,
    validate_path_shape_corpus,
)


CORPUS_PATH = Path(__file__).with_name("corpus.json")


def _split_selector_values(values: Iterable[str] | str | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    result: set[str] = set()
    for value in values:
        for part in str(value).replace("+", ",").split(","):
            part = part.strip()
            if part:
                result.add(part)
    return result


def _option_set(config: Any, name: str) -> set[str]:
    if config is None:
        return set()
    try:
        value = config.getoption(name)
    except Exception:
        return set()
    return _split_selector_values(value)


def load_path_shape_corpus(path: str | Path | None = None) -> dict[str, Any]:
    corpus_path = Path(path) if path is not None else CORPUS_PATH
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    validate_path_shape_corpus(payload)
    return payload


def corpus_summary(
    path: str | Path | None = None,
    *,
    strict_budget: bool = False,
    enforce_targets: bool = False,
) -> dict[str, Any]:
    return validate_path_shape_corpus(
        load_path_shape_corpus(path),
        strict_budget=strict_budget,
        enforce_targets=enforce_targets,
    )


def _default_resource_tiers(corpus: dict[str, Any]) -> set[str]:
    tiers = corpus.get("resource_tiers") or {}
    return {
        tier
        for tier, metadata in tiers.items()
        if isinstance(metadata, dict) and metadata.get("default") is True
    }


def select_path_shape_cases(
    *,
    corpus: dict[str, Any] | None = None,
    config: Any = None,
    runners: Iterable[str] | str | None = None,
    families: Iterable[str] | str | None = None,
    categories: Iterable[str] | str | None = None,
    case_ids: Iterable[str] | str | None = None,
    resource_tiers: Iterable[str] | str | None = None,
    model_roles: Iterable[str] | str | None = None,
    dtype_groups: Iterable[str] | str | None = None,
    cost_classes: Iterable[str] | str | None = None,
) -> list[dict[str, Any]]:
    corpus = corpus or load_path_shape_corpus()
    validate_path_shape_corpus(corpus)

    selected_runners = _split_selector_values(runners) | _option_set(config, "--path-shape-runner")
    selected_families = _split_selector_values(families) | _option_set(config, "--path-shape-family")
    selected_categories = _split_selector_values(categories) | _option_set(config, "--path-shape-category")
    selected_case_ids = _split_selector_values(case_ids) | _option_set(config, "--path-shape-case")
    selected_resource_tiers = _split_selector_values(resource_tiers) | _option_set(config, "--path-shape-resource-tier")
    selected_model_roles = _split_selector_values(model_roles) | _option_set(config, "--path-shape-model-role")
    selected_dtype_groups = _split_selector_values(dtype_groups) | _option_set(config, "--path-shape-dtype-group")
    selected_cost_classes = _split_selector_values(cost_classes) | _option_set(config, "--path-shape-cost-class")

    if not selected_resource_tiers:
        selected_resource_tiers = _default_resource_tiers(corpus)

    cases = []
    for case in corpus.get("cases", []):
        if selected_runners and case.get("runner") not in selected_runners:
            continue
        if selected_families and case.get("family") not in selected_families:
            continue
        if selected_categories and not (set(case.get("categories") or []) & selected_categories):
            continue
        if selected_case_ids and case.get("case_id") not in selected_case_ids:
            continue
        if selected_resource_tiers and case.get("resource_tier") not in selected_resource_tiers:
            continue
        if selected_model_roles and case.get("model_role") not in selected_model_roles:
            continue
        if selected_dtype_groups and case.get("dtype_group") not in selected_dtype_groups:
            continue
        if selected_cost_classes and case.get("cost_class") not in selected_cost_classes:
            continue
        cases.append(dict(case))
    return cases


def pytest_params_for_runner(runners: Iterable[str] | str, config: Any = None):
    import pytest

    params = []
    for case in select_path_shape_cases(config=config, runners=runners):
        marks = [
            pytest.mark.semantic_level(case["semantic_level"], reason=case["level_reason"]),
            pytest.mark.path_shape,
        ]
        for surface in case.get("covers") or []:
            marks.append(pytest.mark.covers(surface))
        for category in case.get("categories") or []:
            marks.append(pytest.mark.covers_category(category))
        if case.get("suite") == "workloads":
            marks.append(pytest.mark.workload)
        if case.get("suite") == "stress":
            marks.append(pytest.mark.stress)
        params.append(pytest.param(case, id=case["case_id"], marks=marks))
    return params


def selected_path_shape_summary(config: Any = None) -> dict[str, Any]:
    try:
        cases = select_path_shape_cases(config=config)
    except PathShapeValidationError:
        raise
    return summarize_cases(cases)
