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

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


MIN_SEMANTIC_LEVEL = 1
MAX_SEMANTIC_LEVEL = 8
DEFAULT_REQUESTED_SEMANTIC_LEVEL = 3


class SemanticLevelError(ValueError):
    """Raised when semantic level metadata is malformed."""


@dataclass(frozen=True)
class SemanticLevelInfo:
    level: int
    reason: str
    source: str


@dataclass(frozen=True)
class SemanticLevelSelection:
    mode: str
    min_level: int
    max_level: int

    def contains(self, level: int) -> bool:
        level = validate_semantic_level(level)
        return self.min_level <= level <= self.max_level

    @property
    def label(self) -> str:
        if self.mode == "cumulative":
            return f"requested <= {self.max_level}"
        if self.mode == "exact":
            return f"requested == {self.min_level}"
        if self.min_level == self.max_level:
            return f"requested == {self.min_level}"
        return f"requested {self.min_level}-{self.max_level}"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "min_level": self.min_level,
            "max_level": self.max_level,
            "label": self.label,
        }


SEMANTIC_LEVEL_DESCRIPTIONS: dict[int, str] = {
    1: "Core primitive behavior that every backend should run continuously.",
    2: "Normal correctness coverage for common tensor-producing and tensor-consuming surfaces.",
    3: "Mainstream framework semantics such as mutation, aliasing, RNG, metadata, and generated variants.",
    4: "Broad production behavior including training/autograd-adjacent and family-specialized cases.",
    5: "Advanced numeric, layout, storage, sparse, nested, and stride-sensitive behavior.",
    6: "Specialized backend integration such as compiler, device API, allocator, quantization-adjacent, and low-level implementation surfaces.",
    7: "Heavy integration and workload coverage that validates realistic model or multi-device behavior.",
    8: "Release-depth stress and adversarial coverage intended for exhaustive validation passes.",
}


SUITE_DEFAULT_LEVELS: dict[str, SemanticLevelInfo] = {
    "opinfo": SemanticLevelInfo(2, "OpInfo cases are broad normal correctness coverage.", "suite_default"),
    "operators": SemanticLevelInfo(2, "Operator tests cover normal backend correctness.", "suite_default"),
    "generated": SemanticLevelInfo(3, "Generated dispatcher cases default to mainstream framework coverage.", "suite_default"),
    "autograd": SemanticLevelInfo(3, "Autograd tests cover mainstream framework behavior.", "suite_default"),
    "dtypes": SemanticLevelInfo(4, "Dtype-specialized suites are broad production coverage.", "suite_default"),
    "strides": SemanticLevelInfo(5, "Stride and memory-format tests are advanced layout coverage.", "suite_default"),
    "rng": SemanticLevelInfo(3, "RNG behavior is mainstream framework coverage.", "suite_default"),
    "serialization": SemanticLevelInfo(3, "Serialization behavior is mainstream framework coverage.", "suite_default"),
    "errors": SemanticLevelInfo(3, "Error semantics are mainstream framework coverage.", "suite_default"),
    "selftest": SemanticLevelInfo(1, "Harness selftests are core infrastructure checks.", "suite_default"),
    "training": SemanticLevelInfo(4, "Training workflows are broad production coverage.", "suite_default"),
    "compiler": SemanticLevelInfo(6, "Compile integration is a specialized backend feature.", "suite_default"),
    "device_api": SemanticLevelInfo(6, "Device API behavior is specialized backend integration.", "suite_default"),
    "memory": SemanticLevelInfo(6, "Allocator and memory-manager behavior is specialized backend integration.", "suite_default"),
    "workloads": SemanticLevelInfo(7, "Workloads are heavy conformance and integration coverage.", "suite_default"),
    "stress": SemanticLevelInfo(8, "Stress suites are exhaustive release-depth coverage.", "suite_default"),
    "multi_device": SemanticLevelInfo(7, "Multi-device behavior is heavy backend integration coverage.", "suite_default"),
    "custom": SemanticLevelInfo(4, "Custom tests default to broad production coverage.", "suite_default"),
}


STRATEGY_DEFAULT_LEVELS: dict[str, SemanticLevelInfo] = {
    "manual_elementwise": SemanticLevelInfo(1, "Primitive elementwise dispatcher coverage.", "generated_strategy"),
    "manual_bitwise": SemanticLevelInfo(1, "Primitive bitwise dispatcher coverage.", "generated_strategy"),
    "manual_reduction": SemanticLevelInfo(2, "Common reduction dispatcher coverage.", "generated_strategy"),
    "manual_factory": SemanticLevelInfo(2, "Factory behavior is normal backend correctness coverage.", "generated_strategy"),
    "manual_factory_out": SemanticLevelInfo(2, "Factory out= behavior is normal backend correctness coverage.", "generated_strategy"),
    "manual_shape": SemanticLevelInfo(3, "Shape/view behavior is mainstream framework coverage.", "generated_strategy"),
    "manual_indexing": SemanticLevelInfo(3, "Indexing behavior is mainstream framework coverage.", "generated_strategy"),
    "manual_loss": SemanticLevelInfo(3, "Loss-function behavior is mainstream framework coverage.", "generated_strategy"),
    "manual_pooling": SemanticLevelInfo(3, "Pooling behavior is mainstream framework coverage.", "generated_strategy"),
    "manual_padding": SemanticLevelInfo(3, "Padding behavior is mainstream framework coverage.", "generated_strategy"),
    "manual_upsample": SemanticLevelInfo(3, "Upsample behavior is mainstream framework coverage.", "generated_strategy"),
    "manual_grid": SemanticLevelInfo(3, "Grid sampling behavior is mainstream framework coverage.", "generated_strategy"),
    "manual_multi_output_reduction": SemanticLevelInfo(3, "Multi-output reductions are mainstream framework coverage.", "generated_strategy"),
    "manual_matmul": SemanticLevelInfo(4, "Matmul-family generated cases are broad production coverage by default.", "generated_strategy"),
    "manual_convolution": SemanticLevelInfo(4, "Convolution-family generated cases are broad production coverage by default.", "generated_strategy"),
    "manual_foreach": SemanticLevelInfo(4, "Foreach generated cases are broad production coverage.", "generated_strategy"),
    "manual_rnn_cell": SemanticLevelInfo(4, "RNN-cell generated cases are broad production coverage.", "generated_strategy"),
    "manual_linalg": SemanticLevelInfo(5, "Linalg generated cases are advanced numeric coverage.", "generated_strategy"),
    "manual_fft": SemanticLevelInfo(5, "FFT generated cases are advanced numeric coverage.", "generated_strategy"),
    "manual_special_math": SemanticLevelInfo(5, "Special-math generated cases are advanced numeric coverage.", "generated_strategy"),
    "manual_metadata": SemanticLevelInfo(3, "Metadata behavior is mainstream framework coverage.", "generated_strategy"),
    "manual_rng": SemanticLevelInfo(3, "RNG generated cases are mainstream framework coverage.", "generated_strategy"),
    "opinfo_out": SemanticLevelInfo(3, "OpInfo-derived out= coverage is mainstream framework coverage.", "generated_strategy"),
    "opinfo_inplace_unary": SemanticLevelInfo(3, "OpInfo-derived in-place coverage is mainstream framework coverage.", "generated_strategy"),
    "opinfo_view_alias": SemanticLevelInfo(3, "OpInfo-derived view/alias coverage is mainstream framework coverage.", "generated_strategy"),
}


SURFACE_DEFAULT_LEVELS: dict[str, SemanticLevelInfo] = {
    "functional_data": SemanticLevelInfo(2, "Functional data surface default.", "surface_default"),
    "out_variant": SemanticLevelInfo(3, "out= variants include additional mutation/identity semantics.", "surface_default"),
    "mutating_or_inplace": SemanticLevelInfo(3, "Mutating variants include additional mutation/identity semantics.", "surface_default"),
    "view_or_alias": SemanticLevelInfo(3, "View and alias variants require storage/metadata semantics.", "surface_default"),
    "factory": SemanticLevelInfo(2, "Factory surfaces are normal backend correctness coverage.", "surface_default"),
    "rng": SemanticLevelInfo(3, "RNG surfaces are mainstream framework coverage.", "surface_default"),
    "layout_storage": SemanticLevelInfo(5, "Layout and storage surfaces are advanced backend coverage.", "surface_default"),
    "metadata_device": SemanticLevelInfo(3, "Metadata/device behavior is mainstream framework coverage.", "surface_default"),
    "autograd_backward": SemanticLevelInfo(4, "Backward surfaces are broad production training coverage.", "surface_default"),
}


def validate_semantic_level(value: Any, *, field_name: str = "semantic_level") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticLevelError(f"{field_name} must be an integer from {MIN_SEMANTIC_LEVEL} to {MAX_SEMANTIC_LEVEL}")
    if not MIN_SEMANTIC_LEVEL <= value <= MAX_SEMANTIC_LEVEL:
        raise SemanticLevelError(f"{field_name} must be from {MIN_SEMANTIC_LEVEL} to {MAX_SEMANTIC_LEVEL}")
    return value


def semantic_level_description(level: int) -> str:
    return SEMANTIC_LEVEL_DESCRIPTIONS[validate_semantic_level(level)]


def normalize_requested_level(manifest: dict | None = None, cli_level: int | None = None) -> int:
    if cli_level is not None:
        return validate_semantic_level(cli_level, field_name="--level")
    if manifest and "semantic_level" in manifest:
        return validate_semantic_level(manifest["semantic_level"], field_name="semantic_level")
    return DEFAULT_REQUESTED_SEMANTIC_LEVEL


def parse_semantic_level_range(value: Any, *, field_name: str = "--level-range") -> tuple[int, int]:
    if isinstance(value, str):
        text = value.strip()
        separator = ":" if ":" in text else "-" if "-" in text else None
        if separator:
            left, right = text.split(separator, 1)
            try:
                min_level = int(left.strip())
                max_level = int(right.strip())
            except ValueError as exc:
                raise SemanticLevelError(f"{field_name} must be formatted as MIN:MAX with integer levels") from exc
        else:
            try:
                min_level = max_level = int(text)
            except ValueError as exc:
                raise SemanticLevelError(f"{field_name} must be formatted as MIN:MAX with integer levels") from exc
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        min_level = validate_semantic_level(value[0], field_name=f"{field_name} minimum")
        max_level = validate_semantic_level(value[1], field_name=f"{field_name} maximum")
    else:
        raise SemanticLevelError(f"{field_name} must be formatted as MIN:MAX")

    min_level = validate_semantic_level(min_level, field_name=f"{field_name} minimum")
    max_level = validate_semantic_level(max_level, field_name=f"{field_name} maximum")
    if min_level > max_level:
        raise SemanticLevelError(f"{field_name} minimum must be <= maximum")
    return min_level, max_level


def normalize_level_selection(
    manifest: dict | None = None,
    *,
    cli_level: int | None = None,
    cli_level_exact: int | None = None,
    cli_level_range: str | None = None,
) -> SemanticLevelSelection:
    explicit = [
        name
        for name, value in (
            ("--level", cli_level),
            ("--level-exact", cli_level_exact),
            ("--level-range", cli_level_range),
        )
        if value is not None
    ]
    if len(explicit) > 1:
        raise SemanticLevelError("Use only one of --level, --level-exact, or --level-range")
    if cli_level_exact is not None:
        level = validate_semantic_level(cli_level_exact, field_name="--level-exact")
        return SemanticLevelSelection("exact", level, level)
    if cli_level_range is not None:
        min_level, max_level = parse_semantic_level_range(cli_level_range)
        mode = "exact" if min_level == max_level else "range"
        return SemanticLevelSelection(mode, min_level, max_level)
    level = normalize_requested_level(manifest, cli_level=cli_level)
    return SemanticLevelSelection("cumulative", MIN_SEMANTIC_LEVEL, level)


def marker_value_to_level(args: tuple[Any, ...], kwargs: dict[str, Any], *, marker_name: str = "semantic_level") -> int:
    if args:
        return validate_semantic_level(args[0], field_name=f"{marker_name} marker")
    if "level" in kwargs:
        return validate_semantic_level(kwargs["level"], field_name=f"{marker_name} marker")
    raise SemanticLevelError(f"{marker_name} marker requires a level argument")


def suite_for_path(path: str) -> str:
    normalized = str(PurePosixPath(path.replace("\\", "/")))
    for suite_name in SUITE_DEFAULT_LEVELS:
        if f"/{suite_name}/" in f"/{normalized}" or normalized.startswith(f"{suite_name}/"):
            return suite_name
    return "custom"


def suite_default_level(suite: str) -> SemanticLevelInfo:
    return SUITE_DEFAULT_LEVELS.get(suite, SUITE_DEFAULT_LEVELS["custom"])


def generated_level_for_entry(entry: dict) -> SemanticLevelInfo:
    generated = entry.get("generated") or {}
    strategy = generated.get("strategy") or {}
    strategy_name = strategy.get("strategy")
    if strategy_name in STRATEGY_DEFAULT_LEVELS:
        return STRATEGY_DEFAULT_LEVELS[strategy_name]
    surface_kind = entry.get("surface_kind")
    if surface_kind in SURFACE_DEFAULT_LEVELS:
        return SURFACE_DEFAULT_LEVELS[surface_kind]
    return SUITE_DEFAULT_LEVELS["generated"]


def case_level_for_entry(entry: dict, case_id: str, tags: tuple[str, ...] = ()) -> SemanticLevelInfo:
    base = generated_level_for_entry(entry)
    surface_kind = entry.get("surface_kind")
    name = entry.get("name", "")
    tags_set = set(tags)
    level = base.level
    reason = base.reason

    if surface_kind == "layout_storage":
        level = max(level, 5)
        reason = "Layout/storage case requires advanced backend coverage."
    if "broadcast" in tags_set:
        level = max(level, 2)
        reason = "Broadcasting case is normal backend correctness coverage."
    if "rank_polymorphic" in tags_set or "batched" in tags_set:
        level = max(level, 3)
        reason = "Rank-polymorphic or batched case is mainstream framework coverage."
    if "foreach" in tags_set:
        level = max(level, 4)
        reason = "Foreach case is broad production coverage."
    if "linalg" in tags_set or "fft" in tags_set or "special_math" in tags_set:
        level = max(level, 5)
        reason = "Advanced numeric family case."
    if "quant" in name or "quantized" in name or "int4" in name:
        level = max(level, 6)
        reason = "Quantization-adjacent case is specialized backend coverage."

    return SemanticLevelInfo(level, reason, "generated_case")
