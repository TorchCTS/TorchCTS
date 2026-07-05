# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from torchcts.cli import (
    _path_shape_pytest_args,
    _path_shape_pytest_selector_args,
    _run_path_shapes_list,
)
from torchcts.path_shapes import (
    PathShapeValidationError,
    corpus_summary,
    load_path_shape_corpus,
    pytest_params_for_runner,
    select_path_shape_cases,
    validate_path_shape_corpus,
)


pytestmark = pytest.mark.covers_category("selftest")


class _Config:
    def __init__(self, **options):
        self.options = options

    def getoption(self, name):
        return self.options.get(name)


def test_path_shape_corpus_validates_checked_in_file():
    summary = corpus_summary()

    assert summary["case_count"] == 1320
    assert summary["default_selected_case_count"] == 850
    assert "matmul" in summary["by_family"]
    assert "standard" in summary["by_resource_tier"]
    assert "heavy" in summary["by_resource_tier"]
    assert "tiny" in summary["by_cost_class"]
    assert summary["default_resource_tiers"] == ["smoke", "standard"]
    assert not summary["budget_warnings"]


def test_path_shape_selector_filters_family_and_resource_tier():
    corpus = load_path_shape_corpus()

    matmul_cases = select_path_shape_cases(corpus=corpus, families="matmul")
    assert matmul_cases
    assert {case["family"] for case in matmul_cases} == {"matmul"}

    modified = copy.deepcopy(corpus)
    heavy_case = copy.deepcopy(modified["cases"][0])
    heavy_case["case_id"] = "matmul_heavy_hidden_case_f32"
    heavy_case["resource_tier"] = "heavy"
    modified["cases"].append(heavy_case)

    default_cases = select_path_shape_cases(corpus=modified)
    assert "matmul_heavy_hidden_case_f32" not in {case["case_id"] for case in default_cases}

    heavy_cases = select_path_shape_cases(corpus=modified, case_ids="matmul_heavy_hidden_case_f32", resource_tiers="heavy")
    assert {case["case_id"] for case in heavy_cases} == {"matmul_heavy_hidden_case_f32"}


def test_path_shape_selector_filters_runner_and_cost_class():
    corpus = load_path_shape_corpus()

    cases = select_path_shape_cases(
        corpus=corpus,
        runners="attention.sdpa",
        cost_classes="small",
        resource_tiers="standard,heavy",
    )

    assert cases
    assert {case["runner"] for case in cases} == {"attention.sdpa"}
    assert {case["cost_class"] for case in cases} == {"small"}


def test_path_shape_pytest_params_are_concrete_runner_cases():
    params = pytest_params_for_runner("matmul.mm", _Config())

    assert params
    case_ids = [param.id for param in params]
    assert "matmul_mm_m1_k31_n33_nn_contiguous_f32_standard" in case_ids
    assert all(param.values[0]["runner"] == "matmul.mm" for param in params)
    assert all(isinstance(param.values[0]["dtype"], str) for param in params)


def test_path_shape_validator_rejects_cartesian_axis_fields():
    corpus = load_path_shape_corpus()
    broken = copy.deepcopy(corpus)
    broken["cases"][0]["dtypes"] = ["torch.float32", "torch.float16"]

    with pytest.raises(PathShapeValidationError, match="cartesian axis fields"):
        validate_path_shape_corpus(broken)


def test_path_shape_validator_rejects_duplicate_case_ids():
    corpus = load_path_shape_corpus()
    broken = copy.deepcopy(corpus)
    broken["cases"][1]["case_id"] = broken["cases"][0]["case_id"]

    with pytest.raises(PathShapeValidationError, match="duplicate case_id"):
        validate_path_shape_corpus(broken)


def test_path_shape_validator_enforces_budget_caps():
    corpus = load_path_shape_corpus()
    broken = copy.deepcopy(corpus)
    broken["family_budgets"]["matmul"]["default_hard_max"] = 1

    with pytest.raises(PathShapeValidationError, match="default-selected cases"):
        validate_path_shape_corpus(broken)


def test_path_shape_config_selectors_do_not_create_new_combinations():
    corpus = load_path_shape_corpus()
    config = _Config(**{"--path-shape-family": ["matmul"], "--path-shape-dtype-group": ["float"], "--path-shape-cost-class": ["small"]})

    selected = select_path_shape_cases(corpus=corpus, config=config)

    assert selected
    assert len(selected) <= len(corpus["cases"])
    assert {case["family"] for case in selected} == {"matmul"}
    assert {case["dtype_group"] for case in selected} == {"float"}
    assert {case["cost_class"] for case in selected} == {"small"}


def _cli_args(**overrides):
    defaults = {
        "family": None,
        "category": None,
        "case_id": None,
        "runner": None,
        "resource_tier": None,
        "all_resource_tiers": False,
        "model_role": None,
        "dtype_group": None,
        "cost_class": None,
        "device": None,
        "level": 8,
        "level_exact": None,
        "level_range": None,
        "validation": False,
        "subprocess_per_shape": False,
        "output_dir": None,
        "json": False,
        "require_cases": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_path_shape_cli_selector_args_map_to_pytest_filters():
    args = _cli_args(
        family=["matmul"],
        category=["layout_path_shape"],
        case_id=["matmul_mm_m1_k31_n33_nn_padded_f32_standard"],
        runner=["matmul.mm"],
        resource_tier=["standard"],
        model_role=["decode"],
        dtype_group=["float"],
        cost_class=["small"],
    )

    assert _path_shape_pytest_selector_args(args) == [
        "--path-shape-family",
        "matmul",
        "--path-shape-category",
        "layout_path_shape",
        "--path-shape-case",
        "matmul_mm_m1_k31_n33_nn_padded_f32_standard",
        "--path-shape-runner",
        "matmul.mm",
        "--path-shape-resource-tier",
        "standard",
        "--path-shape-cost-class",
        "small",
        "--path-shape-model-role",
        "decode",
        "--path-shape-dtype-group",
        "float",
    ]


def test_path_shape_cli_run_args_delegate_to_existing_pytest_modules():
    args = _cli_args(
        device="cpu",
        level=7,
        validation=True,
        subprocess_per_shape=True,
        output_dir="results/path-shapes",
        family=["matmul"],
    )

    pytest_args = _path_shape_pytest_args(args, ["-q"])

    assert pytest_args[:8] == [
        "--device",
        "cpu",
        "--level",
        "7",
        "--validation",
        "--subprocess-per-test",
        "--results-dir",
        "results/path-shapes",
    ]
    assert "--path-shape-family" in pytest_args
    assert "matmul" in pytest_args
    assert "-q" in pytest_args
    assert any(arg.endswith("torchcts/operators/test_operator_path_shapes.py") for arg in pytest_args)
    assert any(arg.endswith("torchcts/workloads/test_workload_path_shapes.py") for arg in pytest_args)


def test_path_shape_cli_list_can_require_selected_cases(capsys):
    args = _cli_args(family=["no_such_family"], require_cases=True)

    assert _run_path_shapes_list(args) == 1
    assert "Selected path-shape cases: 0" in capsys.readouterr().out
