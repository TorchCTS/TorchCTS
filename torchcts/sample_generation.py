# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Reusable sample generation for TorchCTS dispatcher coverage.

This module is intentionally importable outside of pytest.  Backend projects can
use it directly to request TorchCTS sample inputs for a dispatcher entry,
generated coverage case, or dispatcher name, then invoke the operator however
their own harness needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
import re
from typing import Any, Iterable, Iterator

import torch

from torchcts.core.opinfo_adapter import (
    InputCondition,
    get_op_sample_inputs,
    prepare_sample,
    str_to_dtype,
)
from torchcts.core.semantic_levels import (
    case_level_for_entry,
    validate_semantic_level,
)
from torchcts.op_metadata import get_op_metadata


DEFAULT_SAMPLE_SHAPE = (3, 4)
DEFAULT_IEEE754_SEED = 67

DEFAULT_SHORTCUT_CASES = {
    "matmul": (
        {"lhs_shape": (3, 4), "rhs_shape": (4, 5), "case_id": "matrix_matrix"},
        {"lhs_shape": (4,), "rhs_shape": (4, 5), "case_id": "vector_matrix"},
        {"lhs_shape": (3, 4), "rhs_shape": (4,), "case_id": "matrix_vector"},
        {"lhs_shape": (2, 3, 4), "rhs_shape": (4, 5), "case_id": "broadcast_batch_matrix"},
    ),
    "mm": (
        {"m": 3, "k": 4, "n": 5, "case_id": "small_rectangular"},
        {"m": 8, "k": 8, "n": 8, "case_id": "square"},
    ),
    "bmm": (
        {"batch": 2, "m": 3, "k": 4, "n": 5, "case_id": "small_batch"},
        {"batch": 5, "m": 2, "k": 8, "n": 3, "case_id": "larger_batch"},
    ),
    "addmm": (
        {"m": 3, "k": 4, "n": 5, "bias_shape": (3, 5), "case_id": "full_bias"},
        {"m": 3, "k": 4, "n": 5, "bias_shape": (5,), "case_id": "broadcast_bias"},
    ),
    "addbmm": (
        {"batch": 2, "m": 3, "k": 4, "n": 5, "input_shape": (3, 5), "case_id": "full_bias"},
        {"batch": 2, "m": 3, "k": 4, "n": 5, "input_shape": (5,), "case_id": "broadcast_bias"},
    ),
    "baddbmm": (
        {"batch": 2, "m": 3, "k": 4, "n": 5, "input_shape": (2, 3, 5), "case_id": "full_bias"},
        {"batch": 2, "m": 3, "k": 4, "n": 5, "input_shape": (1, 3, 5), "case_id": "broadcast_bias"},
    ),
    "chain_matmul": (
        {"shapes": ((3, 4), (4, 5), (5, 2)), "case_id": "matrix_chain"},
    ),
    "linear": (
        {"batch": 2, "in_features": 4, "out_features": 5, "bias": True, "case_id": "with_bias"},
        {"batch": 2, "in_features": 4, "out_features": 5, "bias": False, "case_id": "without_bias"},
    ),
    "conv2d": (
        {"height": 8, "width": 8, "kernel_size": 3, "padding": 1, "groups": 1, "case_id": "same_padding"},
        {"height": 9, "width": 7, "kernel_size": (3, 2), "padding": 0, "groups": 1, "case_id": "valid_rectangular"},
        {"in_channels": 4, "out_channels": 4, "height": 8, "width": 8, "kernel_size": 3, "padding": 1, "groups": 4, "case_id": "depthwise"},
    ),
    "binary": (
        {"shape": (3, 4), "other_shape": (3, 4), "case_id": "same_shape"},
        {"shape": (2, 3, 4), "other_shape": (4,), "case_id": "broadcast_rhs"},
    ),
}

DEFAULT_CASE_PURPOSES = {
    "default": "Default representative input case.",
    "metadata_default": "Metadata-derived representative input case.",
    "same_shape": "Same-shape tensor operands.",
    "broadcast_rhs": "Broadcasting right-hand tensor operand.",
    "matrix_matrix": "2-D matrix times 2-D matrix.",
    "vector_matrix": "1-D vector times 2-D matrix.",
    "matrix_vector": "2-D matrix times 1-D vector.",
    "broadcast_batch_matrix": "Batched matmul with broadcast batch dimensions.",
    "matrix_chain": "Compatible matrix chain multiplication.",
    "small_rectangular": "Small non-square matrix multiplication.",
    "square": "Square matrix multiplication.",
    "small_batch": "Small batched matrix multiplication.",
    "larger_batch": "Larger batched matrix multiplication.",
    "full_bias": "Bias/input already has the full output shape.",
    "broadcast_bias": "Bias/input broadcasts across the matrix output.",
    "with_bias": "Linear or convolution sample with bias.",
    "without_bias": "Linear or convolution sample without bias.",
    "same_padding": "Convolution sample with padding preserving spatial size.",
    "valid_rectangular": "Convolution sample with rectangular kernel and no padding.",
    "depthwise": "Depthwise grouped convolution sample.",
    "opinfo_sample_0": "First PyTorch OpInfo sample input.",
    "factory_default": "Default factory argument set.",
    "factory_out_default": "Default factory out= argument set.",
    "fft_forward": "Finite FFT transform with explicit size, dimension, and normalization arguments.",
    "foreach_list": "Foreach tensor-list argument case.",
    "indexing_default": "Deterministic indexing, masking, or scatter placement case.",
    "rng_default": "Random-number surface semantic case.",
    "multi_output_reduction_dim": "Reduction or ordering case returning values and indices.",
    "upsample_forward": "Forward interpolation case with explicit output size.",
    "pooling_forward": "Forward dense pooling case.",
    "convolution_forward": "Forward dense convolution case.",
    "loss_forward": "Forward dense loss-function case.",
    "linalg_forward": "Forward dense linalg case.",
    "metadata_behavior": "Backend metadata or scalar-return behavior case.",
    "padding_forward": "Forward dense padding case.",
    "grid_backward": "Grid-sampler backward case with valid gradient, input, grid, and output mask.",
    "bitwise_unary": "Unary bitwise tensor case.",
    "bitwise_tensor_tensor": "Tensor/tensor bitwise case.",
    "bitwise_tensor_scalar": "Tensor/scalar bitwise case.",
    "bitwise_scalar_tensor": "Scalar/tensor bitwise case.",
    "special_domain": "Domain-safe special-math input case.",
    "elementwise_domain": "Domain-safe elementwise input case.",
    "reduction_dim": "Reduction over an explicit dimension.",
    "shape_default": "Default shape/view transformation case.",
    "shape_dim": "Single-dimension shape/view transformation case.",
    "shape_dim_list": "Multiple-dimension shape/view transformation case.",
    "shape_permutation": "Dimension permutation shape/view transformation case.",
    "shape_tensor_sequence": "Tensor sequence shape construction case.",
    "shape_copy": "Shape-preserving copy case.",
}

SUPPORTED_SAMPLE_STRATEGIES = frozenset({
    "manual_bitwise",
    "manual_elementwise",
    "manual_factory",
    "manual_factory_out",
    "manual_fft",
    "manual_foreach",
    "manual_convolution",
    "manual_grid",
    "manual_grid_backward",
    "manual_indexing",
    "manual_loss",
    "manual_linalg",
    "manual_matmul",
    "manual_metadata",
    "manual_multi_output_reduction",
    "manual_padding",
    "manual_pooling",
    "manual_reduction",
    "manual_rng",
    "manual_rnn_cell",
    "manual_shape",
    "manual_special_math",
    "manual_upsample",
    "opinfo_inplace_unary",
    "opinfo_out",
    "opinfo_view_alias",
})


class SampleGenerationError(RuntimeError):
    """Base error for TorchCTS sample generation failures."""


class UnsupportedSampleStrategy(SampleGenerationError):
    """Raised when TorchCTS has no direct sample generator for a strategy."""


def _normalize_shape(shape: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


@dataclass(frozen=True)
class GeneratedSample:
    """A dispatcher-ready sample that does not require pytest or OpInfo types.

    For normal tensor-input operators, ``input`` is the first dispatcher
    argument and ``args`` are the remaining positional arguments.  For factories
    and other no-input surfaces, ``has_input`` is false and ``args`` is the full
    positional argument tuple.
    """

    dispatcher_name: str
    strategy_name: str
    dtype: torch.dtype
    device: str
    input_condition: str = InputCondition.CLEAN
    family: str | None = None
    input: Any = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    has_input: bool = True
    sample_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def call_args(self) -> tuple[Any, ...]:
        """Return positional arguments ready for a dispatcher call."""

        if self.has_input:
            return (self.input, *self.args)
        return self.args

    def to(self, device: str) -> "GeneratedSample":
        """Return this sample with all nested tensors moved to ``device``."""

        return replace(
            self,
            device=device,
            input=move_to_device(self.input, device),
            args=move_to_device(self.args, device),
            kwargs=move_to_device(self.kwargs, device),
        )

    def as_sample_input(self):
        """Return a PyTorch ``SampleInput`` for consumers that want that shape."""

        if not self.has_input:
            raise SampleGenerationError("Factory-style samples do not have a SampleInput input value")
        from torch.testing._internal.opinfo.core import SampleInput

        return SampleInput(self.input, args=self.args, kwargs=dict(self.kwargs))


@dataclass(frozen=True)
class GeneratedParam:
    """One generated dispatcher parameter with semantic metadata."""

    name: str
    type: str
    value: Any
    purpose: str = "argument"
    position: int | None = None
    keyword_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedExpected:
    """CPU reference result or structured CPU reference error."""

    value: Any = None
    type: str | None = None
    computed_on: str = "cpu"
    error: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class GeneratedCaseSpec:
    """One semantic input case family for an op surface.

    This is intentionally not a dtype/device/sample-value object.  A case spec
    names an equivalence class such as "broadcast_rhs" or "matrix_vector";
    dtype/device/NaN/Inf expansion happens later when the case is materialized.
    """

    case_id: str
    purpose: str
    params: dict[str, Any] = field(default_factory=dict)
    required: bool = True
    tags: tuple[str, ...] = ()
    source: str = "torchcts"
    semantic_level: int | None = None
    level_reason: str | None = None
    level_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "case_id": self.case_id,
            "purpose": self.purpose,
            "params": dict(self.params),
            "required": self.required,
            "tags": list(self.tags),
            "source": self.source,
        }
        if self.semantic_level is not None:
            data["semantic_level"] = self.semantic_level
            data["level_reason"] = self.level_reason
            data["level_source"] = self.level_source
        return data


@dataclass(frozen=True)
class GeneratedOpInputs:
    """Structured generated inputs for a dispatcher op or convenience family."""

    op_name: str
    variant: str
    dtype: torch.dtype
    device: str
    params: tuple[GeneratedParam, ...]
    input_condition: str = InputCondition.CLEAN
    strategy_name: str = "structured"
    family: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    expected: GeneratedExpected | None = None

    def positional_args(self) -> tuple[Any, ...]:
        """Return positional values ordered by dispatcher position."""

        ordered = sorted(
            (param for param in self.params if param.position is not None and not param.keyword_only),
            key=lambda param: param.position,
        )
        return tuple(param.value for param in ordered)

    def kwargs(self) -> dict[str, Any]:
        """Return keyword-only values, excluding absent ``None`` placeholders."""

        return {param.name: param.value for param in self.params if param.keyword_only and param.value is not None}

    def call_args(self) -> tuple[Any, ...]:
        return self.positional_args()

    @property
    def dispatcher_name(self) -> str:
        return str(self.metadata.get("dispatcher_name", self.op_name))

    @property
    def signature_id(self) -> str:
        return str(self.metadata.get("signature_id", self.dispatcher_name))

    @property
    def case_id(self) -> str:
        return str(self.metadata.get("case_id", f"case_{self.metadata.get('case_index', 0)}"))

    def to(self, device: str) -> "GeneratedOpInputs":
        return replace(
            self,
            device=device,
            params=tuple(
                replace(param, value=move_to_device(param.value, device))
                for param in self.params
            ),
        )

    def as_generated_sample(self) -> GeneratedSample:
        positional = self.positional_args()
        if positional:
            return GeneratedSample(
                dispatcher_name=self.op_name,
                strategy_name=self.strategy_name,
                family=self.family,
                dtype=self.dtype,
                device=self.device,
                input_condition=self.input_condition,
                input=positional[0],
                args=tuple(positional[1:]),
                kwargs=self.kwargs(),
                has_input=True,
                metadata=dict(self.metadata),
            )
        return GeneratedSample(
            dispatcher_name=self.op_name,
            strategy_name=self.strategy_name,
            family=self.family,
            dtype=self.dtype,
            device=self.device,
            input_condition=self.input_condition,
            args=(),
            kwargs=self.kwargs(),
            has_input=False,
            metadata=dict(self.metadata),
        )

    def with_expected(self, expected: GeneratedExpected) -> "GeneratedOpInputs":
        return replace(self, expected=expected)


@dataclass(frozen=True)
class DistributionSpec:
    """A deterministic tensor-value profile for backend/workload sampling."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


REAL_WORLD_DISTRIBUTIONS = frozenset({
    "activation_gelu",
    "activation_relu",
    "attention_logits",
    "embedding_weight",
    "gradient",
    "kaiming_normal",
    "kaiming_uniform",
    "layernorm_bias",
    "layernorm_weight",
    "logits",
    "normal",
    "probabilities",
    "sparse_activation",
    "uniform",
    "xavier_normal",
    "xavier_uniform",
})


def move_to_device(obj, device: str):
    """Move tensors nested inside common containers to ``device``."""

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    if isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    return obj


def manifest_dtype_items(manifest: dict) -> list[tuple[torch.dtype, str]]:
    """Return enabled manifest dtypes as ``(torch.dtype, original_key)`` pairs."""

    items = []
    for dtype_key, enabled in manifest.get("supported_dtypes", {}).items():
        if not enabled:
            continue
        if isinstance(dtype_key, torch.dtype):
            dtype = dtype_key
            dtype_str = str(dtype_key)
        else:
            dtype = str_to_dtype(str(dtype_key))
            dtype_str = str(dtype_key)
        if dtype is not None:
            items.append((dtype, dtype_str))
    return sorted(items, key=lambda item: item[1])


def ieee754_enabled(manifest: dict | None, op_name: str) -> bool:
    """Return whether NaN/Inf sample tiers should be generated for ``op_name``."""

    manifest = manifest or {}
    cap = manifest.get("capabilities", {}).get("ieee754", True)
    if cap is True:
        return True
    if cap is False or cap is None:
        return False
    if isinstance(cap, str):
        return bool(re.search(cap, op_name))
    if isinstance(cap, (list, tuple)):
        return any(re.search(pattern, op_name) for pattern in cap)
    return False


def input_conditions_for(manifest: dict | None, op_name: str, dtype: torch.dtype) -> list[str]:
    """Return clean plus optional IEEE754 input conditions for a dtype/op pair."""

    conditions = [InputCondition.CLEAN]
    if (dtype.is_floating_point or dtype.is_complex) and ieee754_enabled(manifest, op_name):
        conditions.extend([InputCondition.HAS_NAN, InputCondition.HAS_INF])
    return conditions


def make_tensor_values(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    offset: float = 0.0,
    domain: str = "mixed",
    shape: Iterable[int] = DEFAULT_SAMPLE_SHAPE,
    noncontiguous: bool = False,
) -> torch.Tensor:
    """Create deterministic tensor values for a dtype/domain/shape combination.

    Domains are semantic constraints used by operator families.  For example,
    ``positive`` avoids invalid log/sqrt inputs, ``unit`` stays inside inverse
    trig domains, and ``nonzero`` avoids division-by-zero references.
    """

    shape = tuple(int(dim) for dim in shape)
    count = max(1, math.prod(shape))
    if domain == "unit":
        base = torch.linspace(-0.75 + offset, 0.75 + offset, count, dtype=torch.float32).reshape(shape)
        base = base.clamp(-0.9, 0.9)
    elif domain in {"positive", "nonzero"}:
        base = torch.linspace(0.25 + offset, 1.75 + offset, count, dtype=torch.float32).reshape(shape)
    else:
        base = torch.linspace(-1.25 + offset, 1.25 + offset, count, dtype=torch.float32).reshape(shape)

    if dtype == torch.bool:
        tensor = base > 0
    elif dtype.is_complex:
        tensor = torch.complex(base, base / 4).to(dtype)
    elif dtype.is_floating_point:
        tensor = base.to(dtype)
    else:
        tensor = torch.round(base * 4).to(dtype)
        if domain == "nonzero":
            tensor64 = tensor.to(torch.int64)
            tensor64[tensor64 == 0] = 1
            tensor = tensor64.to(dtype)

    if noncontiguous and tensor.ndim >= 2:
        tensor = tensor.t().contiguous().t()
    return tensor.to(device)


def _cpu_generator(seed: int, salt: int = 0) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) ^ (int(salt) * 0x9E3779B1))
    return generator


def _fan_in_out(shape: Iterable[int]) -> tuple[int, int]:
    shape = _normalize_shape(shape)
    if len(shape) < 2:
        fan = shape[0] if shape else 1
        return fan, fan
    receptive_field = math.prod(shape[2:]) if len(shape) > 2 else 1
    fan_in = shape[1] * receptive_field
    fan_out = shape[0] * receptive_field
    return max(1, fan_in), max(1, fan_out)


def _coerce_distribution_spec(distribution: str | DistributionSpec | None, params: dict[str, Any]) -> DistributionSpec:
    if isinstance(distribution, DistributionSpec):
        merged = dict(distribution.params)
        merged.update(params)
        return DistributionSpec(distribution.name, merged)
    return DistributionSpec(distribution or "normal", dict(params))


def _apply_realistic_domain(base: torch.Tensor, domain: str) -> torch.Tensor:
    if domain == "unit":
        return base.clamp(-0.9, 0.9)
    if domain == "positive":
        return base.abs() + 0.05
    if domain == "nonzero":
        return torch.where(base.abs() < 0.05, base.sign().clamp(min=0) + 0.25, base)
    if domain == "probability":
        return base.sigmoid().clamp(1e-4, 1.0 - 1e-4)
    return base


def _cast_realistic_values(base: torch.Tensor, dtype: torch.dtype, device: str, *, complex_scale: float = 0.5) -> torch.Tensor:
    if dtype == torch.bool:
        tensor = base > base.mean()
    elif dtype.is_complex:
        imag = torch.roll(base, shifts=1).mul(complex_scale)
        tensor = torch.complex(base, imag).to(dtype)
    elif dtype.is_floating_point:
        tensor = base.to(dtype)
    else:
        tensor = torch.round(base).to(dtype)
    return tensor.to(device)


def make_distribution_tensor(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    shape: Iterable[int] = DEFAULT_SAMPLE_SHAPE,
    distribution: str | DistributionSpec = "normal",
    seed: int = DEFAULT_IEEE754_SEED,
    domain: str = "mixed",
    noncontiguous: bool = False,
    **params,
) -> torch.Tensor:
    """Create deterministic tensors that approximate common workload profiles.

    The goal is not to synthesize training data.  It is to hit backend-relevant
    numeric ranges that resemble real model tensors: fan-scaled weights,
    rectified or sparse activations, attention logits, probability rows, and
    heavy-tailed gradients.
    """

    shape = _normalize_shape(shape)
    count = max(1, math.prod(shape))
    spec = _coerce_distribution_spec(distribution, params)
    name = spec.name
    p = spec.params
    generator = _cpu_generator(seed, hash((name, shape)) & 0xFFFF)

    if name == "uniform":
        low = float(p.get("low", -1.0))
        high = float(p.get("high", 1.0))
        base = torch.empty(shape, dtype=torch.float32).uniform_(low, high, generator=generator)
    elif name == "xavier_uniform":
        fan_in, fan_out = _fan_in_out(shape)
        gain = float(p.get("gain", 1.0))
        bound = gain * math.sqrt(6.0 / float(fan_in + fan_out))
        base = torch.empty(shape, dtype=torch.float32).uniform_(-bound, bound, generator=generator)
    elif name == "xavier_normal":
        fan_in, fan_out = _fan_in_out(shape)
        gain = float(p.get("gain", 1.0))
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        base = torch.empty(shape, dtype=torch.float32).normal_(0.0, std, generator=generator)
    elif name == "kaiming_uniform":
        fan_in, _fan_out = _fan_in_out(shape)
        gain = float(p.get("gain", math.sqrt(2.0)))
        bound = math.sqrt(3.0) * gain / math.sqrt(float(fan_in))
        base = torch.empty(shape, dtype=torch.float32).uniform_(-bound, bound, generator=generator)
    elif name == "kaiming_normal":
        fan_in, _fan_out = _fan_in_out(shape)
        gain = float(p.get("gain", math.sqrt(2.0)))
        std = gain / math.sqrt(float(fan_in))
        base = torch.empty(shape, dtype=torch.float32).normal_(0.0, std, generator=generator)
    elif name == "embedding_weight":
        embedding_dim = int(p.get("embedding_dim", shape[-1] if shape else 1))
        std = float(p.get("std", 1.0 / math.sqrt(max(1, embedding_dim))))
        base = torch.empty(shape, dtype=torch.float32).normal_(0.0, std, generator=generator)
    elif name == "layernorm_weight":
        std = float(p.get("std", 0.02))
        base = torch.empty(shape, dtype=torch.float32).normal_(1.0, std, generator=generator)
    elif name == "layernorm_bias":
        std = float(p.get("std", 0.001))
        base = torch.empty(shape, dtype=torch.float32).normal_(0.0, std, generator=generator)
    elif name == "activation_relu":
        std = float(p.get("std", 1.0))
        zero_probability = float(p.get("zero_probability", 0.5))
        base = torch.empty(shape, dtype=torch.float32).normal_(0.0, std, generator=generator).clamp_min(0.0)
        mask = torch.rand(shape, generator=generator) < zero_probability
        base = base.masked_fill(mask, 0.0)
    elif name == "activation_gelu":
        std = float(p.get("std", 0.7))
        base = torch.empty(shape, dtype=torch.float32).normal_(0.0, std, generator=generator)
    elif name == "sparse_activation":
        density = float(p.get("density", 0.15))
        std = float(p.get("std", 1.0))
        values = torch.empty(shape, dtype=torch.float32).normal_(0.0, std, generator=generator)
        mask = torch.rand(shape, generator=generator) < density
        base = torch.where(mask, values, torch.zeros_like(values))
    elif name == "attention_logits":
        key_dim = int(p.get("key_dim", shape[-1] if shape else 64))
        std = float(p.get("std", 1.0 / math.sqrt(max(1, key_dim))))
        base = torch.empty(shape, dtype=torch.float32).normal_(0.0, std, generator=generator)
    elif name == "logits":
        std = float(p.get("std", 1.5))
        base = torch.empty(shape, dtype=torch.float32).normal_(0.0, std, generator=generator)
    elif name == "probabilities":
        dim = int(p.get("dim", -1))
        logits = torch.empty(shape, dtype=torch.float32).normal_(0.0, 1.0, generator=generator)
        base = torch.softmax(logits, dim=dim)
    elif name == "gradient":
        std = float(p.get("std", 0.03))
        outlier_probability = float(p.get("outlier_probability", 0.01))
        outlier_scale = float(p.get("outlier_scale", 10.0))
        base = torch.empty(shape, dtype=torch.float32).normal_(0.0, std, generator=generator)
        outliers = torch.empty(shape, dtype=torch.float32).normal_(0.0, std * outlier_scale, generator=generator)
        mask = torch.rand(shape, generator=generator) < outlier_probability
        base = torch.where(mask, outliers, base)
    elif name == "normal":
        mean = float(p.get("mean", 0.0))
        std = float(p.get("std", 1.0))
        base = torch.empty(shape, dtype=torch.float32).normal_(mean, std, generator=generator)
    else:
        raise UnsupportedSampleStrategy(f"Unknown tensor distribution profile {name!r}")

    if count == 0:
        base = torch.empty(shape, dtype=torch.float32)
    base = _apply_realistic_domain(base, domain)
    tensor = _cast_realistic_values(base, dtype, device)
    if noncontiguous and tensor.ndim >= 2:
        tensor = tensor.t().contiguous().t()
    return tensor


def make_weight_tensor(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    shape: Iterable[int],
    distribution: str | DistributionSpec = "kaiming_uniform",
    seed: int = DEFAULT_IEEE754_SEED,
    noncontiguous: bool = False,
    **params,
) -> torch.Tensor:
    """Create a deterministic model-weight tensor with fan-scaled values."""

    return make_distribution_tensor(
        dtype,
        device,
        shape=shape,
        distribution=distribution,
        seed=seed,
        noncontiguous=noncontiguous,
        **params,
    )


def make_activation_tensor(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    shape: Iterable[int] = DEFAULT_SAMPLE_SHAPE,
    distribution: str | DistributionSpec = "activation_gelu",
    seed: int = DEFAULT_IEEE754_SEED,
    noncontiguous: bool = False,
    **params,
) -> torch.Tensor:
    """Create a deterministic activation-like tensor for workload tests."""

    return make_distribution_tensor(
        dtype,
        device,
        shape=shape,
        distribution=distribution,
        seed=seed,
        noncontiguous=noncontiguous,
        **params,
    )


def make_gradient_tensor(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    shape: Iterable[int] = DEFAULT_SAMPLE_SHAPE,
    seed: int = DEFAULT_IEEE754_SEED,
    noncontiguous: bool = False,
    **params,
) -> torch.Tensor:
    """Create a small-magnitude gradient tensor with optional rare outliers."""

    return make_distribution_tensor(
        dtype,
        device,
        shape=shape,
        distribution="gradient",
        seed=seed,
        noncontiguous=noncontiguous,
        **params,
    )


def make_scalar_tensor(dtype: torch.dtype, value, device: str = "cpu") -> torch.Tensor:
    if dtype == torch.bool:
        value = bool(value)
    elif dtype.is_complex:
        value = complex(value, value / 4)
    elif not dtype.is_floating_point:
        value = int(value)
    return torch.tensor(value, dtype=dtype, device=device)


def make_scalar_list(dtype: torch.dtype, values) -> list:
    if dtype == torch.bool:
        return [bool(value) for value in values]
    if dtype.is_complex:
        return [complex(value, value / 4) for value in values]
    if dtype.is_floating_point:
        return [float(value) for value in values]
    return [int(value) for value in values]


def make_packed_scalars(dtype: torch.dtype, values, device: str = "cpu") -> torch.Tensor:
    if dtype == torch.bool:
        values = [bool(value) for value in values]
    elif dtype.is_complex:
        values = [complex(value, value / 4) for value in values]
    elif not dtype.is_floating_point:
        values = [int(value) for value in values]
    return torch.tensor(values, dtype=dtype, device=device)


def _sample_input(input_value, args: tuple = (), kwargs: dict | None = None):
    from torch.testing._internal.opinfo.core import SampleInput

    return SampleInput(input_value, args=args, kwargs=kwargs or {})


def _make_generated_sample(
    *,
    dispatcher_name: str,
    strategy_name: str,
    dtype: torch.dtype,
    device: str,
    input_value,
    args: tuple = (),
    kwargs: dict | None = None,
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    family: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GeneratedSample:
    sample = _sample_input(
        move_to_device(input_value, "cpu"),
        args=move_to_device(args, "cpu"),
        kwargs=move_to_device(kwargs or {}, "cpu"),
    )
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=dispatcher_name,
    )
    prepared_input = move_to_device(prepared.input, device)
    prepared_args = move_to_device(prepared.args, device)
    prepared_kwargs = move_to_device(prepared.kwargs, device)
    return GeneratedSample(
        dispatcher_name=dispatcher_name,
        strategy_name=strategy_name,
        family=family,
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        input=prepared_input,
        args=tuple(prepared_args),
        kwargs=dict(prepared_kwargs),
        has_input=True,
        sample_index=sample_index,
        metadata=dict(metadata or {}),
    )


def _wrap_prepared_sample(
    *,
    entry: dict,
    strategy_name: str,
    dtype: torch.dtype,
    device: str,
    input_condition: str,
    prepared,
    family: str | None = None,
    sample_index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> GeneratedSample:
    return GeneratedSample(
        dispatcher_name=entry["name"],
        strategy_name=strategy_name,
        family=family,
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        input=prepared.input,
        args=tuple(prepared.args),
        kwargs=dict(prepared.kwargs),
        has_input=True,
        sample_index=sample_index,
        metadata=dict(metadata or {}),
    )


def _value_type(value) -> str:
    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, torch.Tensor) for item in value):
        return "tensor_list"
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, int) for item in value):
        return "int_list"
    if value is None:
        return "none"
    return type(value).__name__


def _param(
    name: str,
    value,
    purpose: str,
    position: int,
    *,
    param_type: str | None = None,
    keyword_only: bool = False,
    metadata: dict[str, Any] | None = None,
) -> GeneratedParam:
    return GeneratedParam(
        name=name,
        type=param_type or _value_type(value),
        value=value,
        purpose=purpose,
        position=position,
        keyword_only=keyword_only,
        metadata=dict(metadata or {}),
    )


def _matmul_output_shape(lhs_shape: Iterable[int], rhs_shape: Iterable[int]) -> tuple[int, ...]:
    lhs_shape = _normalize_shape(lhs_shape)
    rhs_shape = _normalize_shape(rhs_shape)
    if not lhs_shape or not rhs_shape:
        raise ValueError("matmul inputs must be at least 1-D")

    if len(lhs_shape) == 1 and len(rhs_shape) == 1:
        if lhs_shape[0] != rhs_shape[0]:
            raise ValueError(f"1-D matmul shape mismatch: {lhs_shape} @ {rhs_shape}")
        return ()
    if len(lhs_shape) == 1:
        if lhs_shape[0] != rhs_shape[-2]:
            raise ValueError(f"vector-matrix matmul shape mismatch: {lhs_shape} @ {rhs_shape}")
        return (*rhs_shape[:-2], rhs_shape[-1])
    if len(rhs_shape) == 1:
        if lhs_shape[-1] != rhs_shape[0]:
            raise ValueError(f"matrix-vector matmul shape mismatch: {lhs_shape} @ {rhs_shape}")
        return (*lhs_shape[:-2], lhs_shape[-2])
    if lhs_shape[-1] != rhs_shape[-2]:
        raise ValueError(f"matrix matmul shape mismatch: {lhs_shape} @ {rhs_shape}")
    return (*torch.broadcast_shapes(lhs_shape[:-2], rhs_shape[:-2]), lhs_shape[-2], rhs_shape[-1])


def make_matmul_sample(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    lhs_shape: Iterable[int] = (3, 4),
    rhs_shape: Iterable[int] = (4, 5),
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    lhs_domain: str = "mixed",
    rhs_domain: str = "mixed",
    lhs_distribution: str | DistributionSpec | None = None,
    rhs_distribution: str | DistributionSpec | None = None,
    noncontiguous: bool = False,
) -> GeneratedSample:
    """Return a dispatcher-ready sample for ``torch.matmul``.

    Shapes follow PyTorch matmul broadcasting rules and are validated before
    tensors are created.  Use ``sample.call_args()`` to pass the pair directly
    to ``torch.matmul`` or ``torch.ops.aten.matmul``.
    """

    lhs_shape = _normalize_shape(lhs_shape)
    rhs_shape = _normalize_shape(rhs_shape)
    output_shape = _matmul_output_shape(lhs_shape, rhs_shape)
    if lhs_distribution is None:
        lhs = make_tensor_values(dtype, device, shape=lhs_shape, domain=lhs_domain, noncontiguous=noncontiguous)
    else:
        lhs = make_distribution_tensor(
            dtype,
            device,
            shape=lhs_shape,
            domain=lhs_domain,
            distribution=lhs_distribution,
            seed=seed,
            noncontiguous=noncontiguous,
        )
    if rhs_distribution is None:
        rhs = make_tensor_values(dtype, device, shape=rhs_shape, domain=rhs_domain, offset=0.25, noncontiguous=noncontiguous)
    else:
        rhs = make_distribution_tensor(
            dtype,
            device,
            shape=rhs_shape,
            domain=rhs_domain,
            distribution=rhs_distribution,
            seed=seed + 1,
            noncontiguous=noncontiguous,
        )
    return _make_generated_sample(
        dispatcher_name="aten::matmul",
        strategy_name="shortcut_matmul",
        family="matmul",
        dtype=dtype,
        device=device,
        input_value=lhs,
        args=(rhs,),
        input_condition=input_condition,
        seed=seed,
        sample_index=sample_index,
        metadata={"lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "output_shape": output_shape},
    )


def make_matmul_inputs(
    dtype: torch.dtype,
    device: str = "cpu",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(lhs, rhs)`` tensors suitable for ``torch.matmul``."""

    return make_matmul_sample(dtype, device, **kwargs).call_args()


def make_mm_sample(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    m: int = 3,
    k: int = 4,
    n: int = 5,
    **kwargs,
) -> GeneratedSample:
    """Return a dispatcher-ready 2-D matrix multiplication sample."""

    sample = make_matmul_sample(dtype, device, lhs_shape=(m, k), rhs_shape=(k, n), **kwargs)
    return replace(sample, dispatcher_name="aten::mm", strategy_name="shortcut_mm", family="mm")


def make_mm_inputs(dtype: torch.dtype, device: str = "cpu", **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mat1, mat2)`` tensors suitable for ``torch.mm``."""

    return make_mm_sample(dtype, device, **kwargs).call_args()


def make_bmm_sample(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    batch: int = 2,
    m: int = 3,
    k: int = 4,
    n: int = 5,
    **kwargs,
) -> GeneratedSample:
    """Return a dispatcher-ready batched matrix multiplication sample."""

    sample = make_matmul_sample(dtype, device, lhs_shape=(batch, m, k), rhs_shape=(batch, k, n), **kwargs)
    return replace(sample, dispatcher_name="aten::bmm", strategy_name="shortcut_bmm", family="bmm")


def make_bmm_inputs(dtype: torch.dtype, device: str = "cpu", **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(batch1, batch2)`` tensors suitable for ``torch.bmm``."""

    return make_bmm_sample(dtype, device, **kwargs).call_args()


def make_chain_matmul_sample(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    shapes: Iterable[Iterable[int]] = ((3, 4), (4, 5), (5, 2)),
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    distribution: str | DistributionSpec = "xavier_normal",
    noncontiguous: bool = False,
) -> GeneratedSample:
    """Return a dispatcher-ready Tensor-list sample for ``torch.chain_matmul``."""

    normalized_shapes = tuple(_normalize_shape(shape) for shape in shapes)
    if len(normalized_shapes) < 2:
        raise ValueError("chain_matmul requires at least two matrices")
    for left, right in zip(normalized_shapes, normalized_shapes[1:]):
        if len(left) != 2 or len(right) != 2 or left[1] != right[0]:
            raise ValueError(f"incompatible chain_matmul shapes: {normalized_shapes}")
    matrices = [
        make_distribution_tensor(
            dtype,
            device,
            shape=shape,
            distribution=distribution,
            seed=seed + index,
            noncontiguous=noncontiguous,
        )
        for index, shape in enumerate(normalized_shapes)
    ]
    return _make_generated_sample(
        dispatcher_name="aten::chain_matmul",
        strategy_name="shortcut_chain_matmul",
        family="chain_matmul",
        dtype=dtype,
        device=device,
        input_value=matrices,
        input_condition=input_condition,
        seed=seed,
        sample_index=sample_index,
        metadata={"shapes": normalized_shapes},
    )


def make_chain_matmul_inputs(dtype: torch.dtype, device: str = "cpu", **kwargs) -> tuple[list[torch.Tensor]]:
    """Return ``(matrices,)`` suitable for ``torch.chain_matmul``."""

    return make_chain_matmul_sample(dtype, device, **kwargs).call_args()


def make_addmm_sample(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    m: int = 3,
    k: int = 4,
    n: int = 5,
    bias_shape: Iterable[int] | None = None,
    beta=1,
    alpha=1,
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    bias_distribution: str | DistributionSpec = "layernorm_bias",
    mat1_distribution: str | DistributionSpec = "activation_gelu",
    mat2_distribution: str | DistributionSpec = "kaiming_uniform",
    noncontiguous: bool = False,
) -> GeneratedSample:
    """Return a dispatcher-ready sample for ``torch.addmm``."""

    bias_shape = _normalize_shape(bias_shape or (m, n))
    bias = make_distribution_tensor(
        dtype,
        device,
        shape=bias_shape,
        distribution=bias_distribution,
        seed=seed,
        noncontiguous=noncontiguous,
    )
    mat1 = make_distribution_tensor(
        dtype,
        device,
        shape=(m, k),
        distribution=mat1_distribution,
        seed=seed + 1,
        noncontiguous=noncontiguous,
    )
    mat2 = make_distribution_tensor(
        dtype,
        device,
        shape=(k, n),
        distribution=mat2_distribution,
        seed=seed + 2,
        noncontiguous=noncontiguous,
    )
    return _make_generated_sample(
        dispatcher_name="aten::addmm",
        strategy_name="shortcut_addmm",
        family="addmm",
        dtype=dtype,
        device=device,
        input_value=bias,
        args=(mat1, mat2),
        kwargs={"beta": beta, "alpha": alpha},
        input_condition=input_condition,
        seed=seed,
        sample_index=sample_index,
        metadata={"bias_shape": bias_shape, "mat1_shape": (m, k), "mat2_shape": (k, n)},
    )


def make_addmm_inputs(dtype: torch.dtype, device: str = "cpu", **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(input, mat1, mat2)`` tensors suitable for ``torch.addmm``."""

    return make_addmm_sample(dtype, device, **kwargs).call_args()


def make_addbmm_sample(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    batch: int = 2,
    m: int = 3,
    k: int = 4,
    n: int = 5,
    input_shape: Iterable[int] | None = None,
    beta=1,
    alpha=1,
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    input_distribution: str | DistributionSpec = "layernorm_bias",
    batch1_distribution: str | DistributionSpec = "activation_gelu",
    batch2_distribution: str | DistributionSpec = "kaiming_uniform",
    noncontiguous: bool = False,
) -> GeneratedSample:
    """Return a dispatcher-ready sample for ``torch.addbmm``."""

    input_shape = _normalize_shape(input_shape or (m, n))
    input_value = make_distribution_tensor(
        dtype,
        device,
        shape=input_shape,
        distribution=input_distribution,
        seed=seed,
        noncontiguous=noncontiguous,
    )
    batch1 = make_distribution_tensor(
        dtype,
        device,
        shape=(batch, m, k),
        distribution=batch1_distribution,
        seed=seed + 1,
        noncontiguous=noncontiguous,
    )
    batch2 = make_distribution_tensor(
        dtype,
        device,
        shape=(batch, k, n),
        distribution=batch2_distribution,
        seed=seed + 2,
        noncontiguous=noncontiguous,
    )
    return _make_generated_sample(
        dispatcher_name="aten::addbmm",
        strategy_name="shortcut_addbmm",
        family="addbmm",
        dtype=dtype,
        device=device,
        input_value=input_value,
        args=(batch1, batch2),
        kwargs={"beta": beta, "alpha": alpha},
        input_condition=input_condition,
        seed=seed,
        sample_index=sample_index,
        metadata={
            "input_shape": input_shape,
            "batch1_shape": (batch, m, k),
            "batch2_shape": (batch, k, n),
        },
    )


def make_addbmm_inputs(dtype: torch.dtype, device: str = "cpu", **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(input, batch1, batch2)`` tensors suitable for ``torch.addbmm``."""

    return make_addbmm_sample(dtype, device, **kwargs).call_args()


def make_baddbmm_sample(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    batch: int = 2,
    m: int = 3,
    k: int = 4,
    n: int = 5,
    input_shape: Iterable[int] | None = None,
    beta=1,
    alpha=1,
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    input_distribution: str | DistributionSpec = "layernorm_bias",
    batch1_distribution: str | DistributionSpec = "activation_gelu",
    batch2_distribution: str | DistributionSpec = "kaiming_uniform",
    noncontiguous: bool = False,
) -> GeneratedSample:
    """Return a dispatcher-ready sample for ``torch.baddbmm``."""

    input_shape = _normalize_shape(input_shape or (batch, m, n))
    input_value = make_distribution_tensor(
        dtype,
        device,
        shape=input_shape,
        distribution=input_distribution,
        seed=seed,
        noncontiguous=noncontiguous,
    )
    batch1 = make_distribution_tensor(
        dtype,
        device,
        shape=(batch, m, k),
        distribution=batch1_distribution,
        seed=seed + 1,
        noncontiguous=noncontiguous,
    )
    batch2 = make_distribution_tensor(
        dtype,
        device,
        shape=(batch, k, n),
        distribution=batch2_distribution,
        seed=seed + 2,
        noncontiguous=noncontiguous,
    )
    return _make_generated_sample(
        dispatcher_name="aten::baddbmm",
        strategy_name="shortcut_baddbmm",
        family="baddbmm",
        dtype=dtype,
        device=device,
        input_value=input_value,
        args=(batch1, batch2),
        kwargs={"beta": beta, "alpha": alpha},
        input_condition=input_condition,
        seed=seed,
        sample_index=sample_index,
        metadata={
            "input_shape": input_shape,
            "batch1_shape": (batch, m, k),
            "batch2_shape": (batch, k, n),
        },
    )


def make_baddbmm_inputs(dtype: torch.dtype, device: str = "cpu", **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(input, batch1, batch2)`` tensors suitable for ``torch.baddbmm``."""

    return make_baddbmm_sample(dtype, device, **kwargs).call_args()


def make_linear_sample(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    batch: int = 2,
    in_features: int = 4,
    out_features: int = 5,
    bias: bool = True,
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    activation_distribution: str | DistributionSpec = "activation_gelu",
    weight_distribution: str | DistributionSpec = "kaiming_uniform",
    bias_distribution: str | DistributionSpec = "layernorm_bias",
    noncontiguous: bool = False,
) -> GeneratedSample:
    """Return ``(input, weight, bias)`` semantics for ``torch.nn.functional.linear``."""

    activation = make_distribution_tensor(
        dtype,
        device,
        shape=(batch, in_features),
        distribution=activation_distribution,
        seed=seed,
        noncontiguous=noncontiguous,
    )
    weight = make_distribution_tensor(
        dtype,
        device,
        shape=(out_features, in_features),
        distribution=weight_distribution,
        seed=seed + 1,
        noncontiguous=noncontiguous,
    )
    bias_tensor = None
    if bias:
        bias_tensor = make_distribution_tensor(
            dtype,
            device,
            shape=(out_features,),
            distribution=bias_distribution,
            seed=seed + 2,
        )
    return _make_generated_sample(
        dispatcher_name="aten::linear",
        strategy_name="shortcut_linear",
        family="linear",
        dtype=dtype,
        device=device,
        input_value=activation,
        args=(weight, bias_tensor),
        input_condition=input_condition,
        seed=seed,
        sample_index=sample_index,
        metadata={
            "input_shape": tuple(activation.shape),
            "weight_shape": tuple(weight.shape),
            "bias": bias,
        },
    )


def make_linear_inputs(dtype: torch.dtype, device: str = "cpu", **kwargs) -> tuple:
    """Return ``(input, weight, bias)`` values suitable for ``torch.nn.functional.linear``."""

    return make_linear_sample(dtype, device, **kwargs).call_args()


def _pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    values = tuple(value)
    if len(values) != 2:
        raise ValueError(f"expected an int or pair, got {value!r}")
    return int(values[0]), int(values[1])


def make_conv2d_sample(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    batch: int = 2,
    in_channels: int = 3,
    out_channels: int = 4,
    height: int = 8,
    width: int = 8,
    kernel_size: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    bias: bool = True,
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    activation_distribution: str | DistributionSpec = "activation_relu",
    weight_distribution: str | DistributionSpec = "kaiming_normal",
    bias_distribution: str | DistributionSpec = "layernorm_bias",
    noncontiguous: bool = False,
) -> GeneratedSample:
    """Return a dispatcher-ready sample for 2-D convolution."""

    if in_channels % groups != 0:
        raise ValueError("in_channels must be divisible by groups")
    if out_channels % groups != 0:
        raise ValueError("out_channels must be divisible by groups")

    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    activation = make_distribution_tensor(
        dtype,
        device,
        shape=(batch, in_channels, height, width),
        distribution=activation_distribution,
        seed=seed,
        noncontiguous=noncontiguous,
    )
    weight = make_distribution_tensor(
        dtype,
        device,
        shape=(out_channels, in_channels // groups, *kernel_size),
        distribution=weight_distribution,
        seed=seed + 1,
        noncontiguous=noncontiguous,
    )
    bias_tensor = None
    if bias:
        bias_tensor = make_distribution_tensor(
            dtype,
            device,
            shape=(out_channels,),
            distribution=bias_distribution,
            seed=seed + 2,
        )
    return _make_generated_sample(
        dispatcher_name="aten::convolution",
        strategy_name="shortcut_conv2d",
        family="conv2d",
        dtype=dtype,
        device=device,
        input_value=activation,
        args=(weight, bias_tensor, stride, padding, dilation, False, (0, 0), groups),
        input_condition=input_condition,
        seed=seed,
        sample_index=sample_index,
        metadata={
            "input_shape": tuple(activation.shape),
            "weight_shape": tuple(weight.shape),
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
        },
    )


def make_conv2d_inputs(dtype: torch.dtype, device: str = "cpu", **kwargs) -> tuple:
    """Return values suitable for ``torch.nn.functional.conv2d``.

    The returned tuple is ``(input, weight, bias, stride, padding, dilation, groups)``.
    """

    sample = make_conv2d_sample(dtype, device, **kwargs)
    input_value, weight, bias, stride, padding, dilation, _transposed, _output_padding, groups = sample.call_args()
    return input_value, weight, bias, stride, padding, dilation, groups


def make_binary_sample(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    shape: Iterable[int] = DEFAULT_SAMPLE_SHAPE,
    other_shape: Iterable[int] | None = None,
    domain: str = "mixed",
    other_domain: str | None = None,
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    distribution: str | DistributionSpec | None = None,
    other_distribution: str | DistributionSpec | None = None,
    noncontiguous: bool = False,
) -> GeneratedSample:
    """Return a generic two-tensor elementwise/broadcast sample."""

    shape = _normalize_shape(shape)
    other_shape = _normalize_shape(other_shape or shape)
    torch.broadcast_shapes(shape, other_shape)
    if distribution is None:
        lhs = make_tensor_values(dtype, device, shape=shape, domain=domain, noncontiguous=noncontiguous)
    else:
        lhs = make_distribution_tensor(
            dtype,
            device,
            shape=shape,
            domain=domain,
            distribution=distribution,
            seed=seed,
            noncontiguous=noncontiguous,
        )
    if other_distribution is None:
        rhs = make_tensor_values(
            dtype,
            device,
            shape=other_shape,
            domain=other_domain or domain,
            offset=0.25,
            noncontiguous=noncontiguous,
        )
    else:
        rhs = make_distribution_tensor(
            dtype,
            device,
            shape=other_shape,
            domain=other_domain or domain,
            distribution=other_distribution,
            seed=seed + 1,
            noncontiguous=noncontiguous,
        )
    return _make_generated_sample(
        dispatcher_name="aten::binary",
        strategy_name="shortcut_binary",
        family="binary",
        dtype=dtype,
        device=device,
        input_value=lhs,
        args=(rhs,),
        input_condition=input_condition,
        seed=seed,
        sample_index=sample_index,
        metadata={"shape": shape, "other_shape": other_shape},
    )


def make_binary_inputs(dtype: torch.dtype, device: str = "cpu", **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(lhs, rhs)`` tensors for binary elementwise operators."""

    return make_binary_sample(dtype, device, **kwargs).call_args()


def _shape_base_tensor(
    dtype: torch.dtype,
    device: str,
    *,
    shape: Iterable[int] = (2, 1, 3, 1),
    noncontiguous: bool = False,
    domain: str = "mixed",
) -> torch.Tensor:
    return make_tensor_values(dtype, device, shape=shape, domain=domain, noncontiguous=noncontiguous)


def shape_args_for_entry(entry_name: str, dtype: torch.dtype, device: str = "cpu") -> tuple[Any, tuple, dict, str]:
    """Return ``(input, args, kwargs, case_id)`` for exact manual shape surfaces."""

    if entry_name.startswith("aten::_cast_"):
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (False,), {}, "shape_copy"
    if entry_name in {"aten::alias", "aten::detach", "aten::resolve_conj", "aten::resolve_neg"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (), {}, "shape_copy"
    if entry_name in {"aten::_conj_copy", "aten::_conj_copy.out", "aten::alias_copy", "aten::alias_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (), {}, "shape_copy"
    if entry_name in {"aten::copy", "aten::copy.out"}:
        src = _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True)
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (src, False), {}, "shape_copy"
    if entry_name in {"aten::_copy_from", "aten::_copy_from.out"}:
        dst = _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True)
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (dst, False), {}, "shape_copy"
    if entry_name in {"aten::_copy_from_and_resize", "aten::_copy_from_and_resize.out"}:
        dst = _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True)
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (dst,), {}, "shape_copy"
    if entry_name in {"aten::_neg_view", "aten::_neg_view_copy", "aten::_neg_view_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (), {}, "shape_copy"
    if entry_name in {"aten::data", "aten::lift", "aten::lift.out", "aten::zero", "aten::zero.out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (), {}, "shape_copy"
    if entry_name in {"aten::_new_zeros_with_same_feature_meta", "aten::_new_zeros_with_same_feature_meta.out"}:
        other = _shape_base_tensor(dtype, device, shape=(4, 5))
        return _shape_base_tensor(dtype, device, shape=(2, 3), noncontiguous=True), (other,), {"self_num_batch_dims": 0}, "shape_copy"
    if entry_name == "aten::type_as":
        other = _shape_base_tensor(dtype, device, shape=(1,))
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (other,), {}, "shape_copy"
    if entry_name in {"aten::lift_fresh_copy", "aten::lift_fresh_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (), {}, "shape_copy"
    if entry_name in {"aten::detach_copy", "aten::detach_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (), {}, "shape_copy"
    if entry_name in {"aten::_reshape_copy", "aten::view_copy", "aten::view_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4)), ([2, 6],), {}, "shape_default"
    if entry_name == "aten::_reshape_from_tensor":
        shape = torch.tensor([2, 6], dtype=torch.long, device=device)
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (shape,), {}, "shape_default"
    if entry_name in {"aten::view.dtype", "aten::view_copy.dtype", "aten::view_copy.dtype_out"}:
        return torch.arange(8, dtype=torch.int16, device=device), (torch.float32,), {}, "shape_copy"
    if entry_name in {"aten::_reshape_alias", "aten::_reshape_alias_copy", "aten::_reshape_alias_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4)), ([2, 6], [6, 1]), {}, "shape_copy"
    if entry_name in {"aten::diagonal_copy", "aten::diagonal_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (0, 0, 1), {}, "shape_dim_list"
    if entry_name == "aten::narrow_copy.out":
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (1, 1, 2), {}, "shape_dim"
    if entry_name == "aten::narrow.Tensor":
        start = torch.tensor(1, dtype=torch.long, device=device)
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (1, start, 2), {}, "shape_dim"
    if entry_name in {"aten::permute_copy", "aten::permute_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(2, 3, 4)), ([2, 0, 1],), {}, "shape_permutation"
    if entry_name in {"aten::select_copy.int", "aten::select_copy.int_out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (1, 2), {}, "shape_dim"
    if entry_name in {"aten::slice_copy.Tensor", "aten::slice_copy.Tensor_out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (1, 1, 3, 1), {}, "shape_dim"
    if entry_name == "aten::slice_inverse":
        src = _shape_base_tensor(dtype, device, shape=(3, 2), noncontiguous=True)
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (src, 1, 1, 3, 1), {}, "shape_dim"
    if entry_name in {"aten::transpose_copy.int", "aten::transpose_copy.int_out"}:
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (0, 1), {}, "shape_permutation"
    if entry_name in {"aten::unfold_copy", "aten::unfold_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (1, 2, 1), {}, "shape_dim"
    if entry_name in {"aten::unsqueeze_copy", "aten::unsqueeze_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (1,), {}, "shape_dim"
    if entry_name == "aten::chunk":
        return _shape_base_tensor(dtype, device, shape=(4, 4)), (2, 0), {}, "shape_dim"
    if entry_name in {"aten::split.sizes", "aten::split_with_sizes", "aten::split_with_sizes_copy", "aten::split_with_sizes_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(4, 4)), ([2, 2], 0), {}, "shape_dim_list"
    if entry_name in {"aten::split_copy.Tensor", "aten::split_copy.Tensor_out", "aten::unsafe_split.Tensor"}:
        return _shape_base_tensor(dtype, device, shape=(4, 4)), (2, 0), {}, "shape_dim"
    if entry_name == "aten::unsafe_split_with_sizes":
        return _shape_base_tensor(dtype, device, shape=(4, 4)), ([2, 2], 0), {}, "shape_dim_list"
    if entry_name == "aten::tensor_split.indices":
        return _shape_base_tensor(dtype, device, shape=(4, 4)), ([2], 0), {}, "shape_dim_list"
    if entry_name == "aten::tensor_split.sections":
        return _shape_base_tensor(dtype, device, shape=(4, 4)), (2, 0), {}, "shape_dim"
    if entry_name == "aten::tensor_split.tensor_indices_or_sections":
        return _shape_base_tensor(dtype, device, shape=(4, 4)), (torch.tensor([2], dtype=torch.long, device="cpu"), 0), {}, "shape_dim_list"
    if entry_name in {"aten::hsplit.int", "aten::vsplit.int", "aten::dsplit.int"}:
        shape = (2, 2, 4) if entry_name == "aten::dsplit.int" else (4, 4)
        return _shape_base_tensor(dtype, device, shape=shape), (2,), {}, "shape_dim"
    if entry_name in {"aten::hsplit.array", "aten::vsplit.array", "aten::dsplit.array"}:
        shape = (2, 2, 4) if entry_name == "aten::dsplit.array" else (4, 4)
        return _shape_base_tensor(dtype, device, shape=shape), ([1],), {}, "shape_dim_list"
    if entry_name == "aten::unbind_copy.int_out":
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (0,), {}, "shape_dim"
    if entry_name in {"aten::fft_fftshift", "aten::fft_ifftshift"}:
        return _shape_base_tensor(dtype, device, shape=(4, 4)), ([0, 1],), {}, "shape_dim_list"
    if entry_name == "aten::flatten_dense_tensors":
        tensors = [
            make_tensor_values(dtype, device, shape=(2, 3), domain="mixed", offset=0.0),
            make_tensor_values(dtype, device, shape=(4,), domain="mixed", offset=0.5),
        ]
        return tensors, (), {}, "shape_tensor_sequence"
    if entry_name in {"aten::channel_shuffle", "aten::channel_shuffle.out", "aten::native_channel_shuffle"}:
        return _shape_base_tensor(dtype, device, shape=(2, 4, 2, 2)), (2,), {}, "shape_permutation"
    if entry_name in {"aten::pixel_shuffle", "aten::pixel_shuffle.out"}:
        return _shape_base_tensor(dtype, device, shape=(1, 4, 2, 2)), (2,), {}, "shape_permutation"
    if entry_name in {"aten::pixel_unshuffle", "aten::pixel_unshuffle.out"}:
        return _shape_base_tensor(dtype, device, shape=(1, 1, 4, 4)), (2,), {}, "shape_permutation"
    if entry_name == "aten::unflatten.int":
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (1, [2, 2]), {}, "shape_dim_list"
    if entry_name == "aten::unflatten_dense_tensors":
        flat = make_tensor_values(dtype, device, shape=(10,), domain="mixed")
        tensors = [
            torch.empty((2, 3), dtype=dtype, device=device),
            torch.empty((4,), dtype=dtype, device=device),
        ]
        return flat, (tensors,), {}, "shape_tensor_sequence"
    if entry_name in {"aten::numpy_T"}:
        return _shape_base_tensor(dtype, device, shape=(2, 3, 4)), (), {}, "shape_permutation"
    if entry_name in {"aten::view_as_real_copy", "aten::view_as_real_copy.out"}:
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        return _shape_base_tensor(complex_dtype, device, shape=(3, 4), noncontiguous=True), (), {}, "shape_copy"
    if entry_name in {"aten::view_as_complex", "aten::view_as_complex_copy", "aten::view_as_complex_copy.out"}:
        real_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        return _shape_base_tensor(real_dtype, device, shape=(3, 4, 2)), (), {}, "shape_copy"
    if entry_name == "aten::t_copy.out":
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (), {}, "shape_permutation"
    if entry_name in {"aten::_to_copy", "aten::_to_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (), {
            "dtype": dtype,
            "layout": torch.strided,
            "device": torch.device(device),
            "pin_memory": False,
            "non_blocking": False,
            "memory_format": None,
        }, "shape_copy"
    if entry_name == "aten::to.device":
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (
            torch.device(device),
            dtype,
            False,
            True,
            None,
        ), {}, "shape_copy"
    if entry_name == "aten::to.dtype":
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (
            dtype,
            False,
            True,
            None,
        ), {}, "shape_copy"
    if entry_name == "aten::to.dtype_layout":
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (), {
            "dtype": dtype,
            "layout": torch.strided,
            "device": torch.device(device),
            "pin_memory": False,
            "non_blocking": False,
            "copy": True,
            "memory_format": None,
        }, "shape_copy"
    if entry_name == "aten::to.other":
        other = _shape_base_tensor(dtype, device, shape=(3, 4))
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (
            other,
            False,
            True,
            None,
        ), {}, "shape_copy"
    if entry_name == "aten::to_dense":
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (None,), {
            "masked_grad": None,
        }, "shape_copy"

    if entry_name in {"aten::adjoint", "aten::matrix_H"}:
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (), {}, "shape_permutation"
    if entry_name == "aten::linalg_diagonal":
        return _shape_base_tensor(dtype, device, shape=(2, 3, 4)), (), {
            "offset": 0,
            "dim1": -2,
            "dim2": -1,
        }, "shape_dim"
    if entry_name == "aten::contiguous":
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (), {
            "memory_format": torch.contiguous_format,
        }, "shape_copy"
    if entry_name in {"aten::moveaxis.int", "aten::movedim.int"}:
        return _shape_base_tensor(dtype, device, shape=(2, 3, 4)), (0, 2), {}, "shape_permutation"
    if entry_name in {"aten::moveaxis.intlist", "aten::movedim.intlist"}:
        return _shape_base_tensor(dtype, device, shape=(2, 3, 4)), ([0, 2], [2, 0]), {}, "shape_permutation"
    if entry_name in {"aten::swapaxes", "aten::swapdims", "aten::swapaxes_", "aten::swapdims_"}:
        return _shape_base_tensor(dtype, device, shape=(2, 3, 4)), (0, 2), {}, "shape_permutation"
    if entry_name == "aten::transpose_":
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (0, 1), {}, "shape_permutation"
    if entry_name == "aten::t_":
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (), {}, "shape_permutation"
    if entry_name == "aten::zero_":
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (), {}, "shape_copy"
    if entry_name == "aten::fill_diagonal_":
        fill_value = 3.25 if dtype.is_floating_point or dtype.is_complex else 3
        return _shape_base_tensor(dtype, device, shape=(4, 4)), (fill_value, False), {}, "shape_dim"
    if entry_name in {"aten::tril_", "aten::triu_"}:
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (0,), {}, "shape_dim"
    if entry_name == "aten::unsqueeze_":
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (1,), {}, "shape_dim"

    if entry_name in {"aten::squeeze", "aten::squeeze_", "aten::squeeze_copy", "aten::squeeze_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(2, 1, 3, 1)), (), {}, "shape_default"
    if entry_name in {"aten::squeeze.dim", "aten::squeeze_.dim", "aten::squeeze_copy.dim", "aten::squeeze_copy.dim_out"}:
        return _shape_base_tensor(dtype, device, shape=(2, 1, 3, 1)), (1,), {}, "shape_dim"
    if entry_name in {"aten::squeeze.dims", "aten::squeeze_.dims", "aten::squeeze_copy.dims", "aten::squeeze_copy.dims_out"}:
        return _shape_base_tensor(dtype, device, shape=(2, 1, 3, 1)), ([1, 3],), {}, "shape_dim_list"

    if entry_name in {"aten::expand_copy", "aten::expand_copy.out"}:
        return _shape_base_tensor(dtype, device, shape=(1, 3, 1)), ([2, 3, 4],), {"implicit": False}, "shape_dim_list"
    if entry_name in {"aten::set", "aten::set.out", "aten::set_"}:
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (), {}, "shape_storage_alias"
    if entry_name in {"aten::set.source_Tensor", "aten::set.source_Tensor_out", "aten::set_.source_Tensor"}:
        source = _shape_base_tensor(dtype, device, shape=(3, 4))
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (source,), {}, "shape_storage_alias"
    if entry_name == "aten::set_.source_Tensor_storage_offset":
        source = _shape_base_tensor(dtype, device, shape=(3, 4))
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (source, 1, [2, 3], [4, 1]), {}, "shape_storage_alias"
    if entry_name in {"aten::set.source_Storage", "aten::set.source_Storage_out", "aten::set_.source_Storage"}:
        source = _shape_base_tensor(dtype, device, shape=(3, 4))
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (source.untyped_storage(),), {}, "shape_storage_alias"
    if entry_name in {
        "aten::set.source_Storage_storage_offset",
        "aten::set.source_Storage_storage_offset_out",
        "aten::set_.source_Storage_storage_offset",
    }:
        source = _shape_base_tensor(dtype, device, shape=(3, 4))
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (
            source.untyped_storage(),
            1,
            [2, 3],
            [4, 1],
        ), {}, "shape_storage_alias"
    if entry_name == "aten::set_data":
        source = _shape_base_tensor(dtype, device, shape=(3, 4))
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (source,), {}, "shape_storage_alias"
    if entry_name == "aten::as_strided_copy.out":
        return _shape_base_tensor(dtype, device, shape=(3, 4)), ([2, 2], [4, 1], 0), {}, "shape_copy"
    if entry_name == "aten::as_strided_":
        return _shape_base_tensor(dtype, device, shape=(3, 4)), ([2, 2], [4, 1], 0), {}, "shape_copy"
    if entry_name == "aten::as_strided_scatter.out":
        src = _shape_base_tensor(dtype, device, shape=(2, 2), noncontiguous=True)
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (src, [2, 2], [4, 1], 0), {}, "shape_copy"
    if entry_name == "aten::clone.out":
        return _shape_base_tensor(dtype, device, shape=(3, 4), noncontiguous=True), (), {}, "shape_copy"
    if entry_name == "aten::diag.out":
        return _shape_base_tensor(dtype, device, shape=(4,)), (0,), {}, "shape_dim"
    if entry_name == "aten::diag_embed.out":
        return _shape_base_tensor(dtype, device, shape=(4,)), (0, -2, -1), {}, "shape_dim"
    if entry_name == "aten::diff.out":
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (1, 1, None, None), {}, "shape_dim"
    if entry_name == "aten::glu.out":
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (1,), {}, "shape_dim"
    if entry_name == "aten::nonzero_static.out":
        value = torch.tensor([[1, 0, 2], [0, 3, 0]], dtype=torch.int64, device=device)
        return value, (), {"size": 4, "fill_value": -1}, "shape_index_output"
    if entry_name == "aten::msort.out":
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (), {}, "shape_permutation"
    if entry_name == "aten::argsort.stable_out":
        return _shape_base_tensor(dtype, device, shape=(3, 4)), (), {"stable": True, "dim": 1, "descending": False}, "shape_index_output"
    if entry_name == "aten::flip.out":
        return _shape_base_tensor(dtype, device, shape=(2, 3)), ([1],), {}, "shape_dim_list"
    if entry_name == "aten::repeat.out":
        return _shape_base_tensor(dtype, device, shape=(2, 3)), ([2, 1],), {}, "shape_dim_list"
    if entry_name == "aten::repeat_interleave.Tensor_out":
        repeats = torch.tensor([1, 2, 1], dtype=torch.long, device=device)
        return repeats, (), {"output_size": 4}, "shape_index_output"
    if entry_name == "aten::roll.out":
        return _shape_base_tensor(dtype, device, shape=(2, 3)), ([1], [0]), {}, "shape_dim_list"
    if entry_name == "aten::rot90.out":
        return _shape_base_tensor(dtype, device, shape=(2, 3)), (1, [0, 1]), {}, "shape_permutation"
    if entry_name in {"aten::cat.out", "aten::concat.out", "aten::concatenate.out"}:
        tensors = [
            make_tensor_values(dtype, device, shape=(2, 3), domain="mixed", offset=0.0),
            make_tensor_values(dtype, device, shape=(2, 3), domain="mixed", offset=0.5),
        ]
        return tensors, (0,), {}, "shape_tensor_sequence"
    if entry_name == "aten::block_diag.out":
        tensors = [
            make_tensor_values(dtype, device, shape=(2, 2), domain="mixed", offset=0.0),
            make_tensor_values(dtype, device, shape=(1, 3), domain="mixed", offset=0.5),
        ]
        return tensors, (), {}, "shape_tensor_sequence"
    if entry_name == "aten::pad_sequence":
        sequences = [
            make_tensor_values(dtype, device, shape=(3, 2), domain="mixed", offset=0.0),
            make_tensor_values(dtype, device, shape=(2, 2), domain="mixed", offset=0.5),
        ]
        return sequences, (False, 0.0, "right"), {}, "shape_tensor_sequence"
    if entry_name == "aten::nonzero_numpy":
        value = torch.tensor([[1, 0, 2], [0, 3, 0]], dtype=torch.int64, device=device)
        return value, (), {}, "shape_index_output"
    if entry_name in {"aten::_stack", "aten::_stack.out", "aten::stack.out"}:
        tensors = [
            make_tensor_values(dtype, device, shape=(2, 3), domain="mixed", offset=0.0),
            make_tensor_values(dtype, device, shape=(2, 3), domain="mixed", offset=0.5),
        ]
        return tensors, (0,), {}, "shape_tensor_sequence"
    if entry_name in {"aten::hstack.out", "aten::vstack.out", "aten::row_stack.out"}:
        tensors = [
            make_tensor_values(dtype, device, shape=(2, 3), domain="mixed", offset=0.0),
            make_tensor_values(dtype, device, shape=(2, 3), domain="mixed", offset=0.5),
        ]
        return tensors, (), {}, "shape_tensor_sequence"
    if entry_name == "aten::_chunk_cat.out":
        tensors = [
            make_tensor_values(dtype, device, shape=(2, 4), domain="mixed", offset=0.0),
            make_tensor_values(dtype, device, shape=(2, 4), domain="mixed", offset=0.5),
        ]
        return tensors, (1, 2), {}, "shape_tensor_sequence"
    if entry_name == "aten::dstack.out":
        tensors = [
            make_tensor_values(dtype, device, shape=(2, 3), domain="mixed", offset=0.0),
            make_tensor_values(dtype, device, shape=(2, 3), domain="mixed", offset=0.5),
        ]
        return tensors, (), {}, "shape_tensor_sequence"
    if entry_name == "aten::column_stack.out":
        tensors = [
            make_tensor_values(dtype, device, shape=(3,), domain="mixed", offset=0.0),
            make_tensor_values(dtype, device, shape=(3,), domain="mixed", offset=0.5),
        ]
        return tensors, (), {}, "shape_tensor_sequence"
    if entry_name in {"aten::tril_indices.out", "aten::triu_indices.out"}:
        return 3, (4, 0), {}, "shape_index_output"

    raise UnsupportedSampleStrategy(f"No manual shape sample for {entry_name}")


def shape_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    input_value, args, kwargs, case_id = shape_args_for_entry(entry["name"], dtype, device=device)
    sample = _sample_input(input_value, args=args, kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_shape",
        family=strategy.get("family", entry.get("base_name")),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
        metadata={"case_id": case_id},
    )


def _structured_from_generated_sample(
    sample: GeneratedSample,
    names: Iterable[str],
    purposes: Iterable[str],
    *,
    variant: str | None = None,
    param_types: Iterable[str | None] | None = None,
) -> GeneratedOpInputs:
    values = sample.call_args()
    names = tuple(names)
    purposes = tuple(purposes)
    param_types = tuple(param_types or (None for _ in values))
    params = []
    for index, value in enumerate(values):
        name = names[index] if index < len(names) else f"arg{index}"
        purpose = purposes[index] if index < len(purposes) else "argument"
        param_type = param_types[index] if index < len(param_types) else None
        params.append(_param(name, value, purpose, index, param_type=param_type))
    for key, value in sample.kwargs.items():
        params.append(_param(key, value, "option", len(params), keyword_only=True))
    op_metadata = get_op_metadata(sample.dispatcher_name)
    variant_kind = variant or sample.metadata.get("variant_kind") or sample.metadata.get("variant", "functional")
    case_id = sample.metadata.get("case_id", f"case_{sample.sample_index}")
    return GeneratedOpInputs(
        op_name=sample.dispatcher_name,
        variant=variant_kind,
        dtype=sample.dtype,
        device=sample.device,
        params=tuple(params),
        input_condition=sample.input_condition,
        strategy_name=sample.strategy_name,
        family=sample.family,
        metadata={
            **op_metadata,
            **dict(sample.metadata),
            "dispatcher_name": sample.dispatcher_name,
            "signature_id": f"{sample.dispatcher_name}:{variant_kind}",
            "variant_kind": variant_kind,
            "case_id": case_id,
            "surface_kind": sample.metadata.get("surface_kind", op_metadata.get("surface_kind", "functional_data")),
            "input_condition": sample.input_condition,
            "source": sample.metadata.get("source", "torchcts"),
        },
    )


def _entry_arg_purpose(arg: dict, strategy_name: str) -> str:
    name = arg.get("name", "")
    if name in {"self", "input", "mat1", "batch1"}:
        return "data"
    if name in {"other", "mat2", "batch2", "tensor1", "tensor2"}:
        return "other"
    if name in {"weight"}:
        return "weight"
    if name in {"bias"}:
        return "bias"
    if name == "out":
        return "out"
    if name in {"dim", "keepdim", "dtype", "layout", "device", "pin_memory", "memory_format"}:
        return "option"
    if name in {"alpha", "beta", "value", "min", "max", "p", "correction", "unbiased"}:
        return "scalar_parameter"
    if strategy_name == "manual_factory":
        return "factory_parameter"
    return "argument"


def _structured_from_entry_sample(entry: dict, sample: GeneratedSample) -> GeneratedOpInputs:
    args = [arg for arg in entry.get("args", []) if arg.get("name") != "out"]
    values = sample.call_args()
    params = []
    for index, value in enumerate(values):
        arg = args[index] if index < len(args) else {}
        name = arg.get("name") or f"arg{index}"
        params.append(_param(
            name,
            value,
            _entry_arg_purpose(arg, sample.strategy_name),
            index,
            param_type="tensor" if arg.get("tensor") else None,
            keyword_only=bool(arg.get("kwarg_only")),
            metadata={"schema_type": arg.get("type"), "alias": arg.get("alias")},
        ))
    for index in range(len(values), len(args)):
        arg = args[index]
        if not arg.get("has_default"):
            continue
        params.append(_param(
            arg.get("name") or f"arg{index}",
            arg.get("default"),
            _entry_arg_purpose(arg, sample.strategy_name),
            index,
            keyword_only=bool(arg.get("kwarg_only")),
            metadata={"schema_type": arg.get("type"), "alias": arg.get("alias"), "defaulted": True},
        ))
    for key, value in sample.kwargs.items():
        params.append(_param(key, value, "option", len(params), keyword_only=True))
    op_metadata = get_op_metadata(entry["name"])
    variant_kind = entry.get("variant_kind") or entry.get("surface_kind") or "functional"
    case_id = sample.metadata.get("case_id", f"case_{sample.sample_index}")
    return GeneratedOpInputs(
        op_name=entry["name"],
        variant=variant_kind,
        dtype=sample.dtype,
        device=sample.device,
        params=tuple(params),
        input_condition=sample.input_condition,
        strategy_name=sample.strategy_name,
        family=sample.family,
        metadata={
            **op_metadata,
            **dict(sample.metadata),
            "schema": entry.get("schema"),
            "surface_kind": entry.get("surface_kind"),
            "variant_kind": variant_kind,
            "dispatcher_name": entry["name"],
            "signature_id": f"{entry['name']}:{variant_kind}",
            "case_id": case_id,
            "input_condition": sample.input_condition,
            "source": sample.metadata.get("source", "torchcts"),
            "base_name": entry.get("base_name"),
            "overload": entry.get("overload", ""),
        },
    )


def _without_case_id(case: dict) -> dict:
    return {
        key: value for key, value in case.items()
        if key not in {"case_id", "purpose", "required", "tags", "source", "semantic_level", "level_reason", "level_source"}
    }


def _case_family_name(op_name: str) -> str:
    normalized = op_name.removeprefix("torch.").removeprefix("aten::")
    if normalized in {"nn.functional.linear"}:
        return "linear"
    if normalized in {"elementwise_binary"}:
        return "binary"
    if "." in normalized and normalized not in DEFAULT_SHORTCUT_CASES:
        base, _overload = normalized.split(".", 1)
        return base
    return normalized


def _case_tags(family: str, case_id: str, params: dict[str, Any]) -> tuple[str, ...]:
    tags = set()
    if "broadcast" in case_id:
        tags.add("broadcast")
    if "vector" in case_id:
        tags.add("rank_polymorphic")
    if "batch" in case_id:
        tags.add("batched")
    if "bias" in case_id:
        tags.add("optional_or_broadcast_bias")
    if family in {"matmul", "mm", "bmm", "addmm", "addbmm", "baddbmm", "chain_matmul", "linear"}:
        tags.add("matmul")
    if family in {"conv2d", "convolution"}:
        tags.add("convolution")
    if family == "binary":
        tags.add("binary")
    if params.get("noncontiguous"):
        tags.add("noncontiguous")
    return tuple(sorted(tags))


def _case_spec_from_dict(family: str, case: dict, index: int, *, source: str = "torchcts") -> GeneratedCaseSpec:
    case = dict(case)
    case_id = str(case.pop("case_id", f"case_{index}"))
    purpose = str(case.pop("purpose", DEFAULT_CASE_PURPOSES.get(case_id, f"{family} generated case {index}.")))
    required = bool(case.pop("required", True))
    explicit_tags = tuple(str(tag) for tag in case.pop("tags", ()))
    case_source = str(case.pop("source", source))
    semantic_level = case.pop("semantic_level", None)
    level_reason = case.pop("level_reason", None)
    level_source = case.pop("level_source", None)
    if semantic_level is not None:
        semantic_level = validate_semantic_level(semantic_level, field_name=f"{case_id}.semantic_level")
    tags = tuple(sorted(set(explicit_tags) | set(_case_tags(family, case_id, case))))
    return GeneratedCaseSpec(
        case_id=case_id,
        purpose=purpose,
        params=case,
        required=required,
        tags=tags,
        source=case_source,
        semantic_level=semantic_level,
        level_reason=str(level_reason) if level_reason is not None else None,
        level_source=str(level_source) if level_source is not None else None,
    )


def _case_specs_from_cases(
    family: str,
    cases: Iterable[dict],
    *,
    source: str = "torchcts",
) -> tuple[GeneratedCaseSpec, ...]:
    return tuple(_case_spec_from_dict(family, case, index, source=source) for index, case in enumerate(cases))


def _shortcut_case_specs(op_name: str, explicit_cases: Iterable[dict] | None = None) -> tuple[GeneratedCaseSpec, ...]:
    family = _case_family_name(op_name)
    if explicit_cases is not None:
        return _case_specs_from_cases(family, explicit_cases)
    return _case_specs_from_cases(
        family,
        DEFAULT_SHORTCUT_CASES.get(family, ({"case_id": "default"},)),
    )


def _shortcut_strategy_name(family: str) -> str:
    if family in {"matmul", "mm", "bmm", "addmm", "addbmm", "baddbmm", "chain_matmul", "linear"}:
        return "manual_matmul"
    if family in {"conv2d", "convolution"}:
        return "manual_convolution"
    return "manual_elementwise"


def _shortcut_pseudo_entry(op_name: str) -> dict:
    normalized = op_name.removeprefix("torch.").removeprefix("aten::")
    family = _case_family_name(normalized)
    return {
        "name": op_name if op_name.startswith("aten::") else f"aten::{normalized}",
        "base_name": family,
        "surface_kind": "functional_data",
        "generated": {
            "strategy": {
                "strategy": _shortcut_strategy_name(family),
            }
        },
    }


def _shortcut_case_specs_with_levels(
    op_name: str,
    explicit_cases: Iterable[dict] | None = None,
) -> tuple[GeneratedCaseSpec, ...]:
    return _with_semantic_levels(_shortcut_pseudo_entry(op_name), _shortcut_case_specs(op_name, explicit_cases))


def _semantic_metadata_for_case_spec(case_spec: GeneratedCaseSpec, case_index: int) -> dict[str, Any]:
    return {
        "case_id": case_spec.case_id,
        "case_index": case_index,
        "case_purpose": case_spec.purpose,
        "case_required": case_spec.required,
        "case_tags": case_spec.tags,
        "case_source": case_spec.source,
        "semantic_level": case_spec.semantic_level,
        "level_reason": case_spec.level_reason,
        "level_source": case_spec.level_source,
    }


def _semantic_metadata_for_shortcut(op_name: str, sample_index: int) -> dict[str, Any]:
    specs = _shortcut_case_specs_with_levels(op_name)
    if not specs:
        return {}
    case_index = sample_index % len(specs)
    return _semantic_metadata_for_case_spec(specs[case_index], case_index)


def _semantic_metadata_for_entry(entry: dict, sample_index: int) -> dict[str, Any]:
    specs = sample_case_specs_for_entry(entry)
    if not specs:
        return {}
    case_index = sample_index % len(specs)
    return _semantic_metadata_for_case_spec(specs[case_index], case_index)


def _with_sample_metadata(result: GeneratedOpInputs, metadata: dict[str, Any]) -> GeneratedOpInputs:
    if not metadata:
        return result
    return replace(result, metadata={**result.metadata, **metadata})


def _shortcut_cases(op_name: str, explicit_cases: Iterable[dict] | None = None) -> tuple[dict, ...]:
    return tuple(
        {"case_id": spec.case_id, **dict(spec.params)}
        for spec in _shortcut_case_specs_with_levels(op_name, explicit_cases)
    )


def _strategy_case_specs(entry: dict, strategy: dict | None = None) -> tuple[GeneratedCaseSpec, ...]:
    strategy = strategy or (entry.get("generated", {}) or {}).get("strategy") or {}
    strategy_name = strategy.get("strategy")
    base_name = entry.get("base_name", "")
    logical_base_name = base_name.rstrip("_")

    if strategy_name == "manual_matmul":
        if logical_base_name == "addmm" and entry.get("surface_kind") == "mutating_or_inplace":
            cases = [case for case in DEFAULT_SHORTCUT_CASES["addmm"] if case.get("case_id") == "full_bias"]
            return _case_specs_from_cases("addmm", cases)
        if logical_base_name in {"matmul", "mm", "bmm", "addmm", "addbmm", "baddbmm", "chain_matmul", "linear"}:
            return _shortcut_case_specs(logical_base_name)
    if strategy_name in {"opinfo_out", "opinfo_inplace_unary", "opinfo_view_alias"}:
        return (GeneratedCaseSpec(
            case_id="opinfo_sample_0",
            purpose=DEFAULT_CASE_PURPOSES["opinfo_sample_0"],
            required=True,
            tags=("opinfo", strategy_name.removeprefix("opinfo_")),
            source="pytorch_opinfo",
        ),)
    if strategy_name == "manual_factory":
        return (GeneratedCaseSpec(
            case_id="factory_default",
            purpose=DEFAULT_CASE_PURPOSES["factory_default"],
            required=True,
            tags=("factory", str(strategy.get("family", "factory"))),
        ),)
    if strategy_name == "manual_factory_out":
        return (GeneratedCaseSpec(
            case_id="factory_out_default",
            purpose=DEFAULT_CASE_PURPOSES["factory_out_default"],
            required=True,
            tags=("factory", "out_variant", str(strategy.get("family", "factory"))),
        ),)
    if strategy_name == "manual_fft":
        return (GeneratedCaseSpec(
            case_id="fft_forward",
            purpose=DEFAULT_CASE_PURPOSES["fft_forward"],
            required=True,
            tags=("fft", str(strategy.get("family", logical_base_name)), entry.get("surface_kind", "surface")),
        ),)
    if strategy_name == "manual_foreach":
        family = str(strategy.get("family", "foreach"))
        overload = str(strategy.get("overload", "default") or "default")
        return (GeneratedCaseSpec(
            case_id=f"foreach_{family}_{overload}".replace(".", "_"),
            purpose=DEFAULT_CASE_PURPOSES["foreach_list"],
            required=True,
            tags=("foreach", family, overload),
        ),)
    if strategy_name == "manual_bitwise":
        overload = entry.get("overload") or ""
        if "Scalar_Tensor" in overload:
            case_id = "bitwise_scalar_tensor"
        elif "Scalar" in overload:
            case_id = "bitwise_tensor_scalar"
        elif "Tensor" in overload:
            case_id = "bitwise_tensor_tensor"
        else:
            case_id = "bitwise_unary"
        return (GeneratedCaseSpec(
            case_id=case_id,
            purpose=DEFAULT_CASE_PURPOSES[case_id],
            required=True,
            tags=("bitwise", str(strategy.get("family", "bitwise"))),
        ),)
    if strategy_name == "manual_special_math":
        return (GeneratedCaseSpec(
            case_id="special_domain",
            purpose=DEFAULT_CASE_PURPOSES["special_domain"],
            required=True,
            tags=("special_math", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_elementwise":
        return (GeneratedCaseSpec(
            case_id="elementwise_domain",
            purpose=DEFAULT_CASE_PURPOSES["elementwise_domain"],
            required=True,
            tags=("elementwise", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_reduction":
        return (GeneratedCaseSpec(
            case_id="reduction_dim",
            purpose=DEFAULT_CASE_PURPOSES["reduction_dim"],
            required=True,
            tags=("reduction", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_indexing":
        return (GeneratedCaseSpec(
            case_id="indexing_default",
            purpose=DEFAULT_CASE_PURPOSES["indexing_default"],
            required=True,
            tags=("indexing", str(strategy.get("family", logical_base_name)), entry.get("surface_kind", "surface")),
        ),)
    if strategy_name == "manual_rng":
        return (GeneratedCaseSpec(
            case_id="rng_default",
            purpose=DEFAULT_CASE_PURPOSES["rng_default"],
            required=True,
            tags=("rng", str(strategy.get("family", logical_base_name)), entry.get("surface_kind", "surface")),
        ),)
    if strategy_name == "manual_multi_output_reduction":
        return (GeneratedCaseSpec(
            case_id="multi_output_reduction_dim",
            purpose=DEFAULT_CASE_PURPOSES["multi_output_reduction_dim"],
            required=True,
            tags=("reduction", "multi_output", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_upsample":
        return (GeneratedCaseSpec(
            case_id="upsample_forward",
            purpose=DEFAULT_CASE_PURPOSES["upsample_forward"],
            required=True,
            tags=("upsample", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_pooling":
        return (GeneratedCaseSpec(
            case_id="pooling_forward",
            purpose=DEFAULT_CASE_PURPOSES["pooling_forward"],
            required=True,
            tags=("pooling", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_convolution":
        return (GeneratedCaseSpec(
            case_id="convolution_forward",
            purpose=DEFAULT_CASE_PURPOSES["convolution_forward"],
            required=True,
            tags=("convolution", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_grid_backward":
        return (GeneratedCaseSpec(
            case_id="grid_backward",
            purpose=DEFAULT_CASE_PURPOSES["grid_backward"],
            required=True,
            tags=("grid", "backward", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_loss":
        return (GeneratedCaseSpec(
            case_id="loss_forward",
            purpose=DEFAULT_CASE_PURPOSES["loss_forward"],
            required=True,
            tags=("loss", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_linalg":
        return (GeneratedCaseSpec(
            case_id="linalg_forward",
            purpose=DEFAULT_CASE_PURPOSES["linalg_forward"],
            required=True,
            tags=("linalg", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_metadata":
        return (GeneratedCaseSpec(
            case_id="metadata_behavior",
            purpose=DEFAULT_CASE_PURPOSES["metadata_behavior"],
            required=True,
            tags=("metadata", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_padding":
        return (GeneratedCaseSpec(
            case_id="padding_forward",
            purpose=DEFAULT_CASE_PURPOSES["padding_forward"],
            required=True,
            tags=("padding", str(strategy.get("family", logical_base_name))),
        ),)
    if strategy_name == "manual_shape":
        try:
            _input_value, _args, _kwargs, case_id = shape_args_for_entry(entry["name"], torch.float32, device="cpu")
        except Exception:
            case_id = "shape_default"
        surface_kind = entry.get("surface_kind", "shape")
        return (GeneratedCaseSpec(
            case_id=case_id,
            purpose=DEFAULT_CASE_PURPOSES.get(case_id, DEFAULT_CASE_PURPOSES["shape_default"]),
            required=True,
            tags=("shape", surface_kind, str(strategy.get("family", logical_base_name))),
        ),)
    return ()


def _with_semantic_levels(entry: dict, specs: tuple[GeneratedCaseSpec, ...]) -> tuple[GeneratedCaseSpec, ...]:
    enriched = []
    for spec in specs:
        if spec.semantic_level is not None:
            enriched.append(spec)
            continue
        level_info = case_level_for_entry(entry, spec.case_id, spec.tags)
        enriched.append(replace(
            spec,
            semantic_level=level_info.level,
            level_reason=level_info.reason,
            level_source=level_info.source,
        ))
    return tuple(enriched)


def sample_case_specs_for_entry(
    entry: dict,
    explicit_cases: Iterable[dict] | None = None,
) -> tuple[GeneratedCaseSpec, ...]:
    """Return semantic case families planned for a dispatcher entry."""

    if explicit_cases is not None:
        return _with_semantic_levels(
            entry,
            _case_specs_from_cases(entry.get("base_name", entry.get("name", "op")), explicit_cases),
        )
    strategy_specs = _strategy_case_specs(entry)
    if strategy_specs:
        return _with_semantic_levels(entry, strategy_specs)
    return _with_semantic_levels(entry, _metadata_case_specs_for_entry(entry))


def _metadata_case_specs_for_entry(entry: dict, explicit_cases: Iterable[dict] | None = None) -> tuple[GeneratedCaseSpec, ...]:
    if explicit_cases is not None:
        return _case_specs_from_cases(entry.get("base_name", entry.get("name", "op")), explicit_cases)
    metadata = get_op_metadata(entry["name"])
    category = metadata.get("category")
    tensor_args = [arg for arg in entry.get("args", []) if arg.get("tensor") and arg.get("name") != "out"]
    if category in {"elementwise_binary", "comparison"} and len(tensor_args) == 2:
        return _shortcut_case_specs("binary")
    if category == "matmul" and entry.get("base_name", "").rstrip("_") in DEFAULT_SHORTCUT_CASES:
        logical_base_name = entry["base_name"].rstrip("_")
        if logical_base_name == "addmm" and entry.get("surface_kind") == "mutating_or_inplace":
            cases = [case for case in DEFAULT_SHORTCUT_CASES["addmm"] if case.get("case_id") == "full_bias"]
            return _case_specs_from_cases("addmm", cases)
        return _shortcut_case_specs(logical_base_name)
    return _case_specs_from_cases(entry.get("base_name", "metadata"), ({"case_id": "metadata_default"},), source="metadata")


def sample_case_depth_for_entry(entry: dict) -> dict[str, Any]:
    """Return JSON-ready semantic case depth metadata for an entry."""

    specs = sample_case_specs_for_entry(entry)
    levels = [spec.semantic_level for spec in specs if spec.semantic_level is not None]
    return {
        "planned_count": len(specs),
        "required_count": sum(1 for spec in specs if spec.required),
        "optional_count": sum(1 for spec in specs if not spec.required),
        "case_ids": [spec.case_id for spec in specs],
        "required_case_ids": [spec.case_id for spec in specs if spec.required],
        "optional_case_ids": [spec.case_id for spec in specs if not spec.required],
        "tags": sorted({tag for spec in specs for tag in spec.tags}),
        "semantic_levels": sorted(set(levels)),
        "min_semantic_level": min(levels) if levels else None,
        "max_semantic_level": max(levels) if levels else None,
        "cases": [spec.to_dict() for spec in specs],
    }


def sample_case_specs_for_op(
    op_name: str,
    *,
    variant: str | None = None,
    audit: dict | None = None,
    cases: Iterable[dict] | None = None,
) -> tuple[GeneratedCaseSpec, ...]:
    """Return planned semantic case families for an op name or dispatcher surface."""

    normalized = op_name.removeprefix("torch.").removeprefix("aten::")
    if variant and not op_name.startswith("torch.") and "." not in normalized:
        normalized = f"{normalized}.{variant}"
    family = _case_family_name(normalized)
    if family in DEFAULT_SHORTCUT_CASES:
        return _shortcut_case_specs_with_levels(family, cases)
    dispatcher_name = op_name if op_name.startswith("aten::") else f"aten::{normalized}"
    return sample_case_specs_for_entry(dispatcher_entry(dispatcher_name, audit=audit), explicit_cases=cases)


def _metadata_cases_for_entry(entry: dict, explicit_cases: Iterable[dict] | None = None) -> tuple[dict, ...]:
    return tuple(
        {"case_id": spec.case_id, **dict(spec.params)}
        for spec in _metadata_case_specs_for_entry(entry, explicit_cases)
    )


def _split_dispatcher_name(dispatcher_name: str) -> tuple[str, str]:
    text = dispatcher_name.removeprefix("aten::")
    if "." in text:
        base, overload = text.split(".", 1)
    else:
        base, overload = text, ""
    return base, overload


def _is_tensorish_type(value) -> bool:
    return "Tensor" in str(value)


def _alias_info_dict(alias_info) -> dict | None:
    if alias_info is None:
        return None
    return {
        "is_write": bool(getattr(alias_info, "is_write", False)),
        "repr": repr(alias_info),
    }


def _schema_arg_record(arg) -> dict:
    has_default = bool(arg.has_default_value()) if hasattr(arg, "has_default_value") else False
    return {
        "name": arg.name,
        "type": str(arg.type),
        "tensor": _is_tensorish_type(arg.type),
        "alias": _alias_info_dict(getattr(arg, "alias_info", None)),
        "kwarg_only": bool(getattr(arg, "kwarg_only", False)),
        "is_out": bool(getattr(arg, "is_out", False)),
        "has_default": has_default,
        "default": getattr(arg, "default_value", None) if has_default else None,
    }


def _schema_return_record(ret) -> dict:
    return {
        "name": ret.name,
        "type": str(ret.type),
        "tensor": _is_tensorish_type(ret.type),
        "alias": _alias_info_dict(getattr(ret, "alias_info", None)),
    }


def _live_dispatcher_entry(dispatcher_name: str) -> dict:
    from torchcts.core.coverage import classify_surface

    base, overload = _split_dispatcher_name(dispatcher_name)
    schema = torch._C._dispatch_find_schema_or_throw(f"aten::{base}", overload).schema()
    args = [_schema_arg_record(arg) for arg in schema.arguments]
    returns = [_schema_return_record(ret) for ret in schema.returns]
    tensor_args = [arg for arg in args if arg["tensor"]]
    tensor_returns = [ret for ret in returns if ret["tensor"]]
    surface_kind, variant_kind = classify_surface(dispatcher_name, schema)
    return {
        "name": dispatcher_name,
        "base_name": base,
        "overload": overload,
        "schema": str(schema),
        "args": args,
        "returns": returns,
        "tensor_args": tensor_args,
        "tensor_returns": tensor_returns,
        "has_tensor_args": bool(tensor_args),
        "has_tensor_returns": bool(tensor_returns),
        "surface_kind": surface_kind,
        "variant_kind": variant_kind,
        "generated": {"strategy": None},
    }


@lru_cache(maxsize=None)
def _dispatcher_callable(dispatcher_name: str):
    base, overload = _split_dispatcher_name(dispatcher_name)
    packet = getattr(torch.ops.aten, base)
    return getattr(packet, overload or "default")


def _public_callable(op_name: str):
    normalized = op_name.removeprefix("torch.").removeprefix("aten::")
    if normalized == "linear":
        return torch.nn.functional.linear
    if normalized == "conv2d":
        return torch.nn.functional.conv2d
    if normalized == "binary":
        raise UnsupportedSampleStrategy("generic binary samples do not name a concrete reference operator")
    if hasattr(torch, normalized):
        return getattr(torch, normalized)
    return _dispatcher_callable(op_name if op_name.startswith("aten::") else f"aten::{op_name}")


def _compute_cpu_expected_for(op_inputs: GeneratedOpInputs, reference_op=None) -> GeneratedExpected:
    try:
        callable_op = reference_op or _public_callable(op_inputs.op_name)
        cpu_args = move_to_device(op_inputs.positional_args(), "cpu")
        cpu_kwargs = move_to_device(op_inputs.kwargs(), "cpu")
        value = callable_op(*cpu_args, **cpu_kwargs)
        return GeneratedExpected(
            value=value,
            type=_value_type(value),
            metadata={
                "reference": getattr(callable_op, "__name__", repr(callable_op)),
                "input_condition": op_inputs.input_condition,
            },
        )
    except Exception as exc:
        if op_inputs.metadata.get("category") == "matmul":
            try:
                from torchcts.core.reference_oracles import matmul_family_reference

                cpu_args = move_to_device(op_inputs.positional_args(), "cpu")
                cpu_kwargs = move_to_device(op_inputs.kwargs(), "cpu")
                value = matmul_family_reference(op_inputs.dispatcher_name, cpu_args, cpu_kwargs)
                return GeneratedExpected(
                    value=value,
                    type=_value_type(value),
                    metadata={
                        "reference": "torchcts.core.reference_oracles.matmul_family_reference",
                        "input_condition": op_inputs.input_condition,
                        "fallback_for": type(exc).__name__,
                    },
                )
            except Exception:
                pass
        return GeneratedExpected(
            error={
                "type": type(exc).__name__,
                "message": str(exc),
            },
            metadata={"input_condition": op_inputs.input_condition},
        )


def _metadata_inputs_for_entry(
    entry: dict,
    dtype: torch.dtype,
    *,
    device: str,
    input_condition: str,
    seed: int,
    sample_index: int,
    sample_kwargs: dict[str, Any] | None = None,
) -> GeneratedOpInputs:
    metadata = get_op_metadata(entry["name"])
    category = metadata.get("category")
    base_name = entry.get("base_name", "")
    logical_base_name = base_name.rstrip("_")
    sample_kwargs = dict(sample_kwargs or {})

    if category in {"elementwise_binary", "elementwise_unary", "comparison"}:
        tensor_args = [arg for arg in entry.get("args", []) if arg.get("tensor") and arg.get("name") != "out"]
        if category in {"elementwise_binary", "comparison"} and len(tensor_args) == 2:
            sample = make_binary_sample(
                dtype,
                device,
                shape=sample_kwargs.get("shape", DEFAULT_SAMPLE_SHAPE),
                other_shape=sample_kwargs.get("other_shape"),
                domain=sample_kwargs.get("domain", "mixed"),
                other_domain=sample_kwargs.get("other_domain"),
                input_condition=input_condition,
                seed=seed,
                sample_index=sample_index,
                distribution=sample_kwargs.get("distribution"),
                other_distribution=sample_kwargs.get("other_distribution"),
                noncontiguous=sample_kwargs.get("noncontiguous", False),
            )
        else:
            sample = elementwise_sample(entry, dtype, device=device, input_condition=input_condition, seed=seed)
        return _structured_from_entry_sample(entry, sample)

    if category == "reduction":
        sample = reduction_sample(entry, dtype, device=device, input_condition=input_condition, seed=seed)
        return _structured_from_entry_sample(entry, sample)

    if category == "matmul":
        if logical_base_name == "matmul":
            sample = make_matmul_sample(
                dtype,
                device,
                input_condition=input_condition,
                seed=seed,
                sample_index=sample_index,
                **sample_kwargs,
            )
        elif logical_base_name == "mm":
            sample = make_mm_sample(
                dtype,
                device,
                input_condition=input_condition,
                seed=seed,
                sample_index=sample_index,
                **sample_kwargs,
            )
        elif logical_base_name == "bmm":
            sample = make_bmm_sample(
                dtype,
                device,
                input_condition=input_condition,
                seed=seed,
                sample_index=sample_index,
                **sample_kwargs,
            )
        elif logical_base_name == "addmm":
            sample = make_addmm_sample(
                dtype,
                device,
                input_condition=input_condition,
                seed=seed,
                sample_index=sample_index,
                **sample_kwargs,
            )
        elif logical_base_name == "addbmm":
            sample = make_addbmm_sample(
                dtype,
                device,
                input_condition=input_condition,
                seed=seed,
                sample_index=sample_index,
                **sample_kwargs,
            )
        elif logical_base_name == "baddbmm":
            sample = make_baddbmm_sample(
                dtype,
                device,
                input_condition=input_condition,
                seed=seed,
                sample_index=sample_index,
                **sample_kwargs,
            )
        elif logical_base_name == "chain_matmul":
            sample = make_chain_matmul_sample(
                dtype,
                device,
                input_condition=input_condition,
                seed=seed,
                sample_index=sample_index,
                **sample_kwargs,
            )
        elif logical_base_name == "linear":
            sample = make_linear_sample(
                dtype,
                device,
                input_condition=input_condition,
                seed=seed,
                sample_index=sample_index,
                **sample_kwargs,
            )
        else:
            raise UnsupportedSampleStrategy(f"No metadata fallback sample generator for matmul op {entry['name']}")
        return _structured_from_entry_sample(entry, sample)

    if category == "convolution":
        if base_name in {"conv2d", "convolution"}:
            sample = make_conv2d_sample(
                dtype,
                device,
                input_condition=input_condition,
                seed=seed,
                sample_index=sample_index,
            )
            if base_name == "conv2d":
                values = sample.call_args()
                sample = replace(
                    sample,
                    dispatcher_name="aten::conv2d",
                    args=(values[1], values[2], values[3], values[4], values[5], values[8]),
                )
            else:
                sample = replace(sample, dispatcher_name="aten::convolution")
            return _structured_from_entry_sample(entry, sample)
        raise UnsupportedSampleStrategy(f"No metadata fallback sample generator for convolution op {entry['name']}")

    raise UnsupportedSampleStrategy(
        f"No TorchCTS sample generator for {entry['name']} "
        f"(metadata category {category!r})"
    )


def get_inputs_for_op(
    op_name: str,
    *,
    variant: str | None = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    audit: dict | None = None,
    generate_results: bool = False,
    reference_op=None,
    **kwargs,
) -> GeneratedOpInputs:
    """Return structured generated inputs for an op or high-level op family.

    ``op_name`` accepts exact dispatcher names such as ``aten::add.Tensor`` and
    convenience names such as ``matmul``, ``mm``, ``bmm``, ``addmm``,
    ``addbmm``, ``baddbmm``, ``chain_matmul``, ``linear``, ``conv2d``, and
    ``binary``.
    """

    normalized = op_name.removeprefix("torch.").removeprefix("aten::")
    if variant and not op_name.startswith("torch.") and "." not in normalized:
        normalized = f"{normalized}.{variant}"

    def _finish(result: GeneratedOpInputs, metadata: dict[str, Any], reference=None) -> GeneratedOpInputs:
        result = _with_sample_metadata(result, metadata)
        return result.with_expected(_compute_cpu_expected_for(result, reference)) if generate_results else result

    if normalized in {"matmul"}:
        sample = make_matmul_sample(
            dtype,
            device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            **kwargs,
        )
        result = _structured_from_generated_sample(sample, ("self", "other"), ("data", "other"), variant=variant)
        return _finish(result, _semantic_metadata_for_shortcut(normalized, sample_index), reference_op)
    if normalized in {"mm"}:
        sample = make_mm_sample(
            dtype,
            device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            **kwargs,
        )
        result = _structured_from_generated_sample(sample, ("mat1", "mat2"), ("data", "weight"), variant=variant)
        return _finish(result, _semantic_metadata_for_shortcut(normalized, sample_index), reference_op)
    if normalized in {"bmm"}:
        sample = make_bmm_sample(
            dtype,
            device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            **kwargs,
        )
        result = _structured_from_generated_sample(sample, ("batch1", "batch2"), ("data", "weight"), variant=variant)
        return _finish(result, _semantic_metadata_for_shortcut(normalized, sample_index), reference_op)
    if normalized in {"addmm"}:
        sample = make_addmm_sample(
            dtype,
            device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            **kwargs,
        )
        result = _structured_from_generated_sample(
            sample,
            ("input", "mat1", "mat2"),
            ("bias_or_input", "data", "weight"),
            variant=variant,
        )
        return _finish(result, _semantic_metadata_for_shortcut(normalized, sample_index), reference_op)
    if normalized in {"addbmm"}:
        sample = make_addbmm_sample(
            dtype,
            device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            **kwargs,
        )
        result = _structured_from_generated_sample(
            sample,
            ("input", "batch1", "batch2"),
            ("bias_or_input", "data", "weight"),
            variant=variant,
        )
        return _finish(result, _semantic_metadata_for_shortcut(normalized, sample_index), reference_op)
    if normalized in {"baddbmm"}:
        sample = make_baddbmm_sample(
            dtype,
            device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            **kwargs,
        )
        result = _structured_from_generated_sample(
            sample,
            ("input", "batch1", "batch2"),
            ("bias_or_input", "data", "weight"),
            variant=variant,
        )
        return _finish(result, _semantic_metadata_for_shortcut(normalized, sample_index), reference_op)
    if normalized in {"chain_matmul"}:
        sample = make_chain_matmul_sample(
            dtype,
            device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            **kwargs,
        )
        result = _structured_from_generated_sample(
            sample,
            ("matrices",),
            ("data",),
            variant=variant,
        )
        chain_reference = reference_op or (lambda matrices: torch.chain_matmul(*matrices))
        return _finish(result, _semantic_metadata_for_shortcut(normalized, sample_index), chain_reference)
    if normalized in {"linear", "nn.functional.linear"}:
        sample = make_linear_sample(
            dtype,
            device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            **kwargs,
        )
        result = _structured_from_generated_sample(
            sample,
            ("input", "weight", "bias"),
            ("activation", "weight", "bias"),
            variant=variant,
        )
        return _finish(result, _semantic_metadata_for_shortcut(normalized, sample_index), reference_op)
    if normalized == "conv2d":
        sample = make_conv2d_sample(
            dtype,
            device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            **kwargs,
        )
        values = sample.call_args()
        conv2d_sample = replace(
            sample,
            dispatcher_name="conv2d",
            args=(values[1], values[2], values[3], values[4], values[5], values[8]),
            metadata={**sample.metadata, **get_op_metadata("aten::conv2d")},
        )
        result = _structured_from_generated_sample(
            conv2d_sample,
            ("input", "weight", "bias", "stride", "padding", "dilation", "groups"),
            ("activation", "weight", "bias", "option", "option", "option", "option"),
            variant=variant,
        )
        return _finish(result, _semantic_metadata_for_shortcut(normalized, sample_index), reference_op)
    if normalized == "convolution":
        sample = make_conv2d_sample(
            dtype,
            device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            **kwargs,
        )
        sample = replace(sample, dispatcher_name="aten::convolution")
        result = _structured_from_generated_sample(
            sample,
            ("input", "weight", "bias", "stride", "padding", "dilation", "transposed", "output_padding", "groups"),
            ("activation", "weight", "bias", "option", "option", "option", "option", "option", "option"),
            variant=variant,
        )
        return _finish(result, _semantic_metadata_for_shortcut(normalized, sample_index), reference_op)
    if normalized in {"binary", "elementwise_binary"}:
        sample = make_binary_sample(
            dtype,
            device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            **kwargs,
        )
        result = _structured_from_generated_sample(sample, ("self", "other"), ("data", "other"), variant=variant)
        return _finish(result, _semantic_metadata_for_shortcut(normalized, sample_index), reference_op)

    if op_name.startswith("aten::"):
        dispatcher_name = op_name
        if variant and "." not in dispatcher_name.removeprefix("aten::"):
            dispatcher_name = f"{dispatcher_name}.{variant}"
    else:
        dispatcher_name = f"aten::{normalized}"
    entry = dispatcher_entry(dispatcher_name, audit=audit)
    try:
        sample = sample_for_entry(
            entry,
            dtype,
            device=device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
        )
        result = _structured_from_entry_sample(entry, sample)
    except UnsupportedSampleStrategy:
        result = _metadata_inputs_for_entry(
            entry,
            dtype,
            device=device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
            sample_kwargs=kwargs,
        )
    return _finish(result, _semantic_metadata_for_entry(entry, sample_index), reference_op)


def iter_inputs_for_op(
    op_name: str,
    *,
    variant: str | None = None,
    dtypes: Iterable[torch.dtype] = (torch.float32,),
    device: str = "cpu",
    input_conditions: Iterable[str] | None = None,
    manifest: dict | None = None,
    seed: int = DEFAULT_IEEE754_SEED,
    audit: dict | None = None,
    generate_results: bool = False,
    reference_op=None,
    cases: Iterable[dict] | None = None,
    max_opinfo_samples: int = 1,
    **kwargs,
) -> Iterator[GeneratedOpInputs]:
    """Yield the planned input cases for an op across dtypes/conditions/cases."""

    normalized = op_name.removeprefix("torch.").removeprefix("aten::")
    if normalized in DEFAULT_SHORTCUT_CASES or normalized in {
        "matmul",
        "mm",
        "bmm",
        "addmm",
        "addbmm",
        "baddbmm",
        "chain_matmul",
        "linear",
        "nn.functional.linear",
        "conv2d",
        "convolution",
        "binary",
        "elementwise_binary",
    }:
        shortcut_specs = _shortcut_case_specs_with_levels(normalized, cases)
        for dtype in dtypes:
            conditions = tuple(input_conditions or input_conditions_for(manifest, normalized, dtype))
            for input_condition in conditions:
                for case_index, case_spec in enumerate(shortcut_specs):
                    merged = {**kwargs, **dict(case_spec.params)}
                    result = get_inputs_for_op(
                        op_name,
                        variant=variant,
                        dtype=dtype,
                        device=device,
                        input_condition=input_condition,
                        seed=seed,
                        sample_index=case_index,
                        audit=audit,
                        generate_results=generate_results,
                        reference_op=reference_op,
                        **merged,
                    )
                    yield replace(
                        result,
                        metadata={
                            **result.metadata,
                            **_semantic_metadata_for_case_spec(case_spec, case_index),
                        },
                    )
        return

    dispatcher_name = op_name if op_name.startswith("aten::") else f"aten::{op_name}"
    entry = dispatcher_entry(dispatcher_name, audit=audit)
    strategy = (entry.get("generated", {}) or {}).get("strategy") or {}
    if not strategy:
        for dtype in dtypes:
            conditions = tuple(input_conditions or input_conditions_for(manifest, entry.get("base_name", op_name), dtype))
            for input_condition in conditions:
                for case_index, case_spec in enumerate(sample_case_specs_for_entry(entry, explicit_cases=cases)):
                    merged = {**kwargs, **dict(case_spec.params)}
                    result = get_inputs_for_op(
                        dispatcher_name,
                        variant=variant,
                        dtype=dtype,
                        device=device,
                        input_condition=input_condition,
                        seed=seed,
                        sample_index=case_index,
                        audit=audit,
                        generate_results=generate_results,
                        reference_op=reference_op,
                        **merged,
                    )
                    yield replace(
                        result,
                        metadata={
                            **result.metadata,
                            **_semantic_metadata_for_case_spec(case_spec, case_index),
                        },
                    )
        return
    if strategy.get("strategy") == "manual_matmul":
        for dtype in dtypes:
            conditions = tuple(input_conditions or input_conditions_for(manifest, entry.get("base_name", op_name), dtype))
            for input_condition in conditions:
                for case_index, case_spec in enumerate(sample_case_specs_for_entry(entry, explicit_cases=cases)):
                    result = get_inputs_for_op(
                        dispatcher_name,
                        variant=variant,
                        dtype=dtype,
                        device=device,
                        input_condition=input_condition,
                        seed=seed,
                        sample_index=case_index,
                        audit=audit,
                        generate_results=generate_results,
                        reference_op=reference_op,
                        **case_spec.params,
                    )
                    yield replace(
                        result,
                        metadata={
                            **result.metadata,
                            **_semantic_metadata_for_case_spec(case_spec, case_index),
                        },
                    )
        return
    for sample_index, sample in enumerate(iter_samples_for_entry(
        entry,
        manifest=manifest,
        dtypes=dtypes,
        device=device,
        seed=seed,
        max_opinfo_samples=max_opinfo_samples,
    )):
        if input_conditions is not None and sample.input_condition not in set(input_conditions):
            continue
        result = _structured_from_entry_sample(entry, sample)
        result = _with_sample_metadata(result, _semantic_metadata_for_entry(entry, sample_index))
        if generate_results:
            result = result.with_expected(_compute_cpu_expected_for(result, reference_op))
        yield result


def get_all_inputs_for_op(
    op_name: str,
    *,
    variant: str | None = None,
    dtypes: Iterable[torch.dtype] = (torch.float32,),
    device: str = "cpu",
    input_conditions: Iterable[str] | None = None,
    manifest: dict | None = None,
    seed: int = DEFAULT_IEEE754_SEED,
    audit: dict | None = None,
    generate_results: bool = False,
    reference_op=None,
    cases: Iterable[dict] | None = None,
    max_opinfo_samples: int = 1,
    **kwargs,
) -> tuple[GeneratedOpInputs, ...]:
    """Return all planned input cases for an op as a concrete tuple."""

    return tuple(iter_inputs_for_op(
        op_name,
        variant=variant,
        dtypes=dtypes,
        device=device,
        input_conditions=input_conditions,
        manifest=manifest,
        seed=seed,
        audit=audit,
        generate_results=generate_results,
        reference_op=reference_op,
        cases=cases,
        max_opinfo_samples=max_opinfo_samples,
        **kwargs,
    ))


def bitwise_dtype_supported(family: str, dtype: torch.dtype) -> bool:
    if dtype not in {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        return False
    if family in {"bitwise_left_shift", "bitwise_right_shift"}:
        return dtype != torch.bool
    return True


def bitwise_base_tensor(dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype == torch.bool:
        values = torch.tensor(
            [[True, False, True, False], [False, True, False, True]],
            dtype=dtype,
            device=device,
        )
    else:
        values = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            dtype=dtype,
            device=device,
        )
    return values.t().contiguous().t()


def bitwise_other_tensor(family: str, dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.tensor(
            [[False, True, True, False], [True, False, True, False]],
            dtype=dtype,
            device=device,
        )
    if family in {"bitwise_left_shift", "bitwise_right_shift"}:
        return torch.tensor(
            [[0, 1, 2, 3], [1, 0, 2, 1]],
            dtype=dtype,
            device=device,
        )
    return torch.tensor(
        [[3, 1, 6, 2], [4, 7, 5, 9]],
        dtype=dtype,
        device=device,
    )


def bitwise_scalar(family: str, dtype: torch.dtype):
    if dtype == torch.bool:
        if family == "bitwise_and":
            return False
        return True
    if family in {"bitwise_left_shift", "bitwise_right_shift"}:
        return 1
    if family == "bitwise_and":
        return 3
    if family == "bitwise_or":
        return 8
    if family == "bitwise_xor":
        return 5
    return 1


def bitwise_args_and_template(entry: dict, dtype: torch.dtype, device: str) -> tuple[tuple, torch.Tensor]:
    strategy = entry["generated"]["strategy"]
    family = strategy["family"]
    overload = entry.get("overload") or ""
    base = bitwise_base_tensor(dtype, device)
    other = bitwise_other_tensor(family, dtype, device)
    scalar = bitwise_scalar(family, dtype)

    if family == "bitwise_not":
        return (base,), base
    if overload in {"Scalar", "Scalar_out", "Tensor_Scalar", "Tensor_Scalar_out"}:
        return (base, scalar), base
    if overload in {"Tensor", "Tensor_out"}:
        return (base, other), base
    if overload in {"Scalar_Tensor", "Scalar_Tensor_out"}:
        return (scalar, other), other
    return (base,), base


def bitwise_sample(entry: dict, dtype: torch.dtype, device: str = "cpu") -> GeneratedSample:
    args, template = bitwise_args_and_template(entry, dtype, device)
    strategy = entry["generated"]["strategy"]
    return GeneratedSample(
        dispatcher_name=entry["name"],
        strategy_name="manual_bitwise",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=InputCondition.CLEAN,
        input=args[0] if args else None,
        args=tuple(args[1:]),
        kwargs={},
        has_input=bool(args),
        metadata={"template": template},
    )


def factory_args(entry_name: str):
    cases = {
        "aten::_efficientzerotensor": ([2, 3],),
        "aten::bartlett_window": (8,),
        "aten::bartlett_window.periodic": (8, True),
        "aten::blackman_window": (8,),
        "aten::blackman_window.periodic": (8, True),
        "aten::fft_fftfreq": (8, 0.5),
        "aten::fft_rfftfreq": (8, 0.5),
        "aten::hamming_window": (8,),
        "aten::hamming_window.periodic": (8, True),
        "aten::hamming_window.periodic_alpha": (8, True, 0.54),
        "aten::hamming_window.periodic_alpha_beta": (8, True, 0.54, 0.46),
        "aten::hann_window": (8,),
        "aten::hann_window.periodic": (8, True),
        "aten::kaiser_window": (8,),
        "aten::kaiser_window.beta": (8, True, 12.0),
        "aten::kaiser_window.periodic": (8, True),
        "aten::empty_permuted": ([2, 3], [1, 0]),
        "aten::range": (0, 6),
        "aten::range.step": (0, 6, 2),
    }
    return cases[entry_name]


def factory_dtype_supported(family: str, dtype: torch.dtype) -> bool:
    if family == "zero_tensor":
        return True
    if family == "empty":
        return True
    if family in {"window", "frequency"}:
        return bool(dtype.is_floating_point)
    if family == "range":
        return dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
    return False


def factory_sample(entry: dict, dtype: torch.dtype, device: str = "cpu") -> GeneratedSample:
    strategy = entry["generated"]["strategy"]
    return GeneratedSample(
        dispatcher_name=entry["name"],
        strategy_name="manual_factory",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=InputCondition.CLEAN,
        args=tuple(factory_args(entry["name"])),
        kwargs={"dtype": dtype, "device": device},
        has_input=False,
    )


def _factory_tensor_scalar(dtype: torch.dtype, value, device: str):
    if dtype == torch.bool:
        value = bool(value)
    elif not dtype.is_floating_point and not dtype.is_complex:
        value = int(value)
    return torch.tensor(value, dtype=dtype, device=device)


def factory_out_arg_value(arg: dict, dtype: torch.dtype, device: str):
    name = arg.get("name", "")
    if arg.get("tensor"):
        if name in {"self", "qtensor"}:
            return make_tensor_values(dtype, device, shape=(2, 3), domain="mixed")
        if name == "start":
            return _factory_tensor_scalar(dtype, 0.0, device)
        if name == "end":
            return _factory_tensor_scalar(dtype, 6.0, device)
        return _factory_tensor_scalar(dtype, 1.0, device)
    if name in {"window_length", "n"}:
        return 8
    if name == "m":
        return 5
    if name == "size":
        return [2, 3]
    if name == "physical_layout":
        return [1, 0]
    if name == "stride":
        return [3, 1]
    if name == "s":
        return 3
    if name == "start":
        return 0
    if name == "end":
        return 6
    if name == "step":
        return 2
    if name == "steps":
        return 8
    if name == "base":
        return 10.0
    if name == "d":
        return 0.5
    if name == "periodic":
        return True
    if name == "alpha":
        return 0.54
    if name == "beta":
        return 0.46
    if name == "fill_value":
        return 3
    if name == "names":
        return None
    if name == "memory_format":
        return None
    return None


def factory_out_call_parts(entry: dict, dtype: torch.dtype, device: str) -> tuple[tuple, dict]:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("name") == "out":
            continue
        value = factory_out_arg_value(arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    return tuple(args), kwargs


def factory_out_args(entry: dict, dtype: torch.dtype, device: str) -> tuple:
    args, kwargs = factory_out_call_parts(entry, dtype, device)
    ordered_kwargs = [
        kwargs[arg.get("name")]
        for arg in entry.get("args", [])
        if arg.get("name") != "out" and arg.get("kwarg_only")
    ]
    return (*args, *ordered_kwargs)


def factory_out_arg_map(entry: dict, args: tuple) -> dict:
    names = [arg.get("name") for arg in entry.get("args", []) if arg.get("name") != "out"]
    return dict(zip(names, args))


def factory_out_shape(entry: dict, args: tuple) -> tuple[int, ...]:
    values = factory_out_arg_map(entry, args)
    base_name = entry["base_name"]
    if base_name in {"linspace", "logspace"}:
        return (int(values["steps"]),)
    if base_name in {"bartlett_window", "blackman_window", "hamming_window", "hann_window", "kaiser_window"}:
        return (int(values["window_length"]),)
    if base_name in {"fft_fftfreq", "fft_rfftfreq"}:
        return (int(values["n"]),)
    if base_name == "eye":
        n = int(values["n"])
        return (n, int(values.get("m", n)))
    if base_name in {"full", "empty", "ones", "zeros"}:
        return tuple(int(dim) for dim in values["size"])
    if base_name in {"empty_permuted", "empty_strided", "new_empty", "new_empty_strided", "new_full", "new_ones", "new_zeros"}:
        return tuple(int(dim) for dim in values["size"])
    if base_name in {"empty_like", "full_like", "ones_like", "zeros_like"}:
        return tuple(values["self"].shape)
    if base_name == "scalar_tensor":
        return ()
    if base_name == "arange":
        start = values.get("start", 0)
        end = values["end"]
        step = values.get("step", 1)
        return (len(range(int(start), int(end), int(step))),)
    if base_name == "range":
        start = values["start"]
        end = values["end"]
        step = values.get("step", 1)
        distance = int(end) - int(start)
        step_int = int(step)
        if step_int == 0:
            raise ValueError("range step cannot be zero")
        if (distance < 0 and step_int > 0) or (distance > 0 and step_int < 0):
            return (0,)
        return (abs(distance) // abs(step_int) + 1,)
    raise ValueError(f"unsupported factory out shape for {entry['name']}")


def factory_out_sample(entry: dict, dtype: torch.dtype, device: str = "cpu") -> GeneratedSample:
    args = factory_out_args(entry, dtype, device)
    strategy = entry["generated"]["strategy"]
    return GeneratedSample(
        dispatcher_name=entry["name"],
        strategy_name="manual_factory_out",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=InputCondition.CLEAN,
        args=args,
        kwargs={},
        has_input=False,
        metadata={"out_shape": factory_out_shape(entry, args)},
    )


FFT_COMPLEX_INPUT_BASES = frozenset({
    "_fft_c2c",
    "_fft_c2r",
    "fft_fft",
    "fft_fft2",
    "fft_fftn",
    "fft_hfft",
    "fft_hfft2",
    "fft_hfftn",
    "fft_ifft",
    "fft_ifft2",
    "fft_ifftn",
    "fft_irfft",
    "fft_irfft2",
    "fft_irfftn",
})

FFT_REAL_INPUT_BASES = frozenset({
    "_fft_r2c",
    "fft_ihfft",
    "fft_ihfft2",
    "fft_ihfftn",
    "fft_rfft",
    "fft_rfft2",
    "fft_rfftn",
})

FFT_HERMITIAN_REAL_OUTPUT_BASES = frozenset({
    "_fft_c2r",
    "fft_hfft",
    "fft_hfft2",
    "fft_hfftn",
    "fft_irfft",
    "fft_irfft2",
    "fft_irfftn",
})


def fft_real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def fft_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.complex128
    return torch.complex64


def fft_input_shape(base_name: str) -> tuple[int, ...]:
    if base_name in FFT_HERMITIAN_REAL_OUTPUT_BASES:
        return (4, 3)
    return (4, 4)


def fft_input_tensor(base_name: str, dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    real_dtype = fft_real_dtype(dtype)
    shape = fft_input_shape(base_name)
    if base_name in FFT_COMPLEX_INPUT_BASES:
        real = make_tensor_values(real_dtype, device, shape=shape, domain="mixed", offset=0.0)
        imag = make_tensor_values(real_dtype, device, shape=shape, domain="mixed", offset=0.375)
        return torch.complex(real, imag).to(dtype=fft_complex_dtype(dtype))
    if base_name in FFT_REAL_INPUT_BASES:
        return make_tensor_values(real_dtype, device, shape=shape, domain="mixed", offset=0.0)
    raise SampleGenerationError(f"No FFT input rule for {base_name}")


def fft_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    arg_type = arg.get("type", "")
    base_name = entry["base_name"]
    if arg.get("is_out"):
        return None
    if name == "self":
        return fft_input_tensor(base_name, dtype, device)
    if name == "n":
        return 4
    if name == "last_dim_size":
        return 4
    if name == "s":
        return [4, 4]
    if name == "dim":
        if base_name.startswith("_fft_"):
            return [0, 1]
        if arg_type == "int":
            return -1
        return [-2, -1]
    if name == "normalization":
        return 0
    if name in {"forward", "onesided"}:
        return True
    if name == "norm":
        return "backward"
    return None


def fft_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    if dtype not in {torch.float32, torch.float64}:
        raise SampleGenerationError(f"FFT samples require float32 or float64, got {dtype}")
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        value = fft_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_fft",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=sample,
        sample_index=sample_index,
    )


def special_math_domain(base_name: str, arg_name: str) -> str:
    if arg_name in {"n", "ord", "p"}:
        return "integer"
    if base_name == "_dirichlet_grad":
        if arg_name == "x":
            return "probability"
        return "positive_large"
    if "ndtri" in base_name:
        return "probability"
    if "polynomial" in base_name:
        return "unit"
    if any(token in base_name for token in ("gamm", "zeta", "polygamma", "digamma")):
        return "positive_large"
    if any(token in base_name for token in ("log", "bessel_y", "modified_bessel_k")):
        return "positive"
    if base_name in {"special_xlogy", "xlogy"} and arg_name == "other":
        return "positive"
    return "mixed"


def special_tensor_values(dtype: torch.dtype, device: str, domain: str, offset: float = 0.0) -> torch.Tensor:
    if domain == "integer":
        return torch.full(DEFAULT_SAMPLE_SHAPE, 3, dtype=torch.int64, device=device)
    if domain == "probability":
        base = torch.linspace(0.1 + offset, 0.9 - offset, 12, dtype=torch.float32).reshape(DEFAULT_SAMPLE_SHAPE)
    elif domain == "positive_large":
        base = torch.linspace(2.5 + offset, 4.25 + offset, 12, dtype=torch.float32).reshape(DEFAULT_SAMPLE_SHAPE)
    else:
        base = make_tensor_values(dtype, "cpu", offset=offset, domain=domain).cpu()

    if domain != "integer":
        if dtype.is_complex:
            if base.is_complex():
                base = base.to(dtype)
            else:
                base = torch.complex(base, base / 8).to(dtype)
        elif dtype.is_floating_point:
            base = base.to(dtype)
        else:
            base = torch.round(base * 4).to(dtype)
    return base.to(device)


def special_scalar_value(arg: dict, base_name: str):
    arg_name = arg.get("name", "")
    arg_type = arg.get("type", "")
    if arg_name in {"n", "ord", "p"} or "int" in arg_type:
        return 3
    if "bool" in arg_type:
        return True
    if special_math_domain(base_name, arg_name) in {"positive", "positive_large", "probability"}:
        return 1.5
    return 0.75


def special_math_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    base_name = entry["base_name"].rstrip("_")
    args = []
    for index, arg in enumerate(entry.get("args", [])):
        if arg.get("name") == "out":
            continue
        if arg.get("tensor"):
            domain = special_math_domain(base_name, arg.get("name", ""))
            args.append(special_tensor_values(dtype, device, domain=domain, offset=0.03 * index))
        else:
            args.append(special_scalar_value(arg, base_name))

    sample = _sample_input(args[0] if args else torch.empty((), dtype=dtype, device=device), tuple(args[1:]))
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_special_math",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


def elementwise_domain(base_name: str, arg_name: str) -> str:
    if arg_name in {"condition"}:
        return "bool"
    if base_name in {"acos", "arccos", "asin", "arcsin", "atanh", "arctanh"}:
        return "unit"
    if base_name in {"erfinv", "logit"}:
        return "probability"
    if base_name in {"acosh", "arccosh"}:
        return "positive_large"
    if base_name in {"digamma", "lgamma", "mvlgamma", "polygamma"}:
        return "positive_large"
    if base_name == "polar" and arg_name == "abs":
        return "positive"
    if base_name in {"log", "log10", "log2", "reciprocal", "rsqrt", "sqrt"}:
        return "positive"
    if base_name == "log1p":
        return "nonzero"
    if base_name in {"div", "divide", "true_divide", "floor_divide", "fmod", "remainder", "addcdiv", "gcd", "lcm"}:
        if arg_name in {"other", "tensor2"}:
            return "nonzero"
    if base_name in {"pow", "float_power"}:
        if arg_name in {"self"}:
            return "positive"
        if arg_name in {"exponent"}:
            return "small"
    if base_name == "lerp" and arg_name == "weight":
        return "unit"
    if base_name in {"clamp", "clip", "clamp_min", "clamp_max"}:
        if arg_name == "min":
            return "lower_bound"
        if arg_name == "max":
            return "upper_bound"
    return "mixed"


def elementwise_tensor_values(dtype: torch.dtype, device: str, domain: str, offset: float = 0.0) -> torch.Tensor:
    if domain == "bool":
        return torch.tensor(
            [[True, False, True, False], [False, True, False, True], [True, True, False, False]],
            dtype=torch.bool,
            device=device,
        )
    if domain == "small":
        return torch.full(DEFAULT_SAMPLE_SHAPE, 2, dtype=dtype, device=device)
    if domain == "positive_large":
        base = torch.linspace(1.25 + offset, 3.25 + offset, 12, dtype=torch.float32).reshape(DEFAULT_SAMPLE_SHAPE)
        if dtype.is_complex:
            return torch.complex(base, base / 8).to(dtype).to(device)
        if dtype.is_floating_point:
            return base.to(dtype).to(device)
        return torch.round(base * 2).to(dtype).to(device)
    if domain == "probability":
        base = torch.linspace(0.1 + offset, 0.9 - offset, 12, dtype=torch.float32).reshape(DEFAULT_SAMPLE_SHAPE)
        base = base.clamp(0.05, 0.95)
        if dtype.is_complex:
            return torch.complex(base, base / 16).to(dtype).to(device)
        if dtype.is_floating_point:
            return base.to(dtype).to(device)
        return torch.round(base).to(dtype).to(device)
    if domain == "lower_bound":
        return torch.full(DEFAULT_SAMPLE_SHAPE, -0.5, dtype=dtype, device=device)
    if domain == "upper_bound":
        return torch.full(DEFAULT_SAMPLE_SHAPE, 0.5, dtype=dtype, device=device)
    return make_tensor_values(dtype, device, offset=offset, domain=domain)


def elementwise_scalar_value(arg: dict, base_name: str):
    arg_name = arg.get("name", "")
    arg_type = arg.get("type", "")
    if arg_name == "rounding_mode":
        return None
    if arg_name == "approximate":
        return "none"
    if arg_name == "condition" or "bool" in arg_type:
        return True
    if arg_name == "alpha":
        return 1
    if arg_name in {"beta", "scale", "input_scale"}:
        return 1
    if arg_name == "negative_slope":
        return 0.01
    if arg_name == "lambd":
        return 0.5
    if arg_name == "threshold":
        return 0.0
    if arg_name == "eps":
        return 1e-6
    if arg_name == "nan":
        return 0.0
    if arg_name == "posinf":
        return 1e6
    if arg_name == "neginf":
        return -1e6
    if arg_name == "value":
        return 0.5
    if arg_name == "min":
        return -0.5
    if arg_name == "max":
        return 0.5
    if arg_name in {"other", "tensor2"} and base_name in {
        "div",
        "divide",
        "true_divide",
        "floor_divide",
        "fmod",
        "remainder",
        "addcdiv",
    }:
        return 2
    if arg_name == "exponent":
        return 2
    if arg_name == "weight":
        return 0.25
    if "int" in arg_type:
        return 2
    if "float" in arg_type or arg_type == "number":
        return 1.25
    return None


def elementwise_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    base_name = entry["base_name"].rstrip("_")
    args = []
    kwargs = {}
    for index, arg in enumerate(entry.get("args", [])):
        if arg.get("name") == "out":
            continue
        if arg.get("tensor"):
            arg_name = arg.get("name", "")
            domain = elementwise_domain(base_name, arg_name)
            if base_name == "prelu" and arg_name == "weight":
                value = elementwise_tensor_values(dtype, device, domain="positive", offset=0.125 * index).reshape(-1)[:1]
            elif base_name == "fill" and arg_name == "value":
                scalar = 0.5 if dtype.is_floating_point or dtype.is_complex else 1
                value = make_scalar_tensor(dtype, scalar, device=device)
            elif base_name == "ldexp" and arg_name == "other":
                value = torch.tensor(
                    [[0, 1, -1, 2], [-2, 1, 0, -1], [2, -2, 1, 0]],
                    dtype=torch.int32,
                    device=device,
                )
            else:
                value = elementwise_tensor_values(dtype, device, domain=domain, offset=0.125 * index)
        else:
            value = elementwise_scalar_value(arg, base_name)

        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)

    sample = _sample_input(args[0] if args else torch.empty((), dtype=dtype, device=device), tuple(args[1:]), kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_elementwise",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


def reduction_arg_value(arg: dict, base_name: str = "", dtype: torch.dtype | None = None):
    name = arg.get("name", "")
    arg_type = arg.get("type", "")
    if name == "q":
        return 0.5
    if name == "dim":
        if arg_type in {"List[int]", "Optional[List[int]]"}:
            return [1]
        if arg_type in {"int", "Optional[int]"}:
            return 1
    if name == "dtype":
        if arg_type == "Optional[int]":
            return None
        return torch.float32
    if name == "input_dtype":
        return dtype or torch.float32
    if name == "p":
        return 2
    if name == "bins":
        return 5
    if name == "maxnorm":
        return 1.0
    if base_name == "histc" and name == "min":
        return -3
    if base_name == "histc" and name == "max":
        return 3
    if name == "correction":
        return 0
    if name == "unbiased":
        return False
    if name == "keepdim":
        return False
    if name == "half_to_float":
        return False
    if name == "interpolation":
        return "linear"
    if base_name in {"segment_reduce", "_segment_reduce_backward"}:
        if name == "reduce":
            return "sum"
        if name == "axis":
            return 0
        if name == "unsafe":
            return False
        if name == "initial":
            return None
    return None


def reduction_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    base_name = entry["base_name"].rstrip("_")
    args = []
    kwargs = {}
    softmax_logits = None
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        if arg.get("tensor"):
            if base_name in {"quantile", "nanquantile"} and arg.get("name") == "q":
                value = torch.tensor([0.25, 0.75], dtype=torch.float32, device=device)
            elif base_name == "segment_reduce" and arg.get("name") == "data":
                value = make_tensor_values(dtype, device, shape=(5,), domain="mixed")
            elif base_name == "segment_reduce" and arg.get("name") == "lengths":
                value = torch.tensor([2, 3], dtype=torch.long, device=device)
            elif base_name == "segment_reduce" and arg.get("name") in {"indices", "offsets"}:
                value = None
            elif base_name == "_segment_reduce_backward" and arg.get("name") in {"grad", "output"}:
                value = make_tensor_values(dtype, device, shape=(2,), domain="mixed")
            elif base_name == "_segment_reduce_backward" and arg.get("name") == "data":
                value = make_tensor_values(dtype, device, shape=(5,), domain="mixed")
            elif base_name == "_segment_reduce_backward" and arg.get("name") == "lengths":
                value = torch.tensor([2, 3], dtype=torch.long, device=device)
            elif base_name == "_segment_reduce_backward" and arg.get("name") == "offsets":
                value = None
            elif base_name in {"_softmax_backward_data", "_log_softmax_backward_data"} and arg.get("name") == "output":
                if softmax_logits is None:
                    softmax_logits = make_tensor_values(dtype, device, domain="mixed")
                if base_name == "_softmax_backward_data":
                    value = torch.softmax(softmax_logits, dim=1)
                else:
                    value = torch.log_softmax(softmax_logits, dim=1)
            else:
                value = make_tensor_values(dtype, device, domain="mixed")
        else:
            value = reduction_arg_value(arg, base_name, dtype)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)

    sample = _sample_input(args[0] if args else torch.empty((), dtype=dtype, device=device), tuple(args[1:]), kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_reduction",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


def indexing_scalar_value(dtype: torch.dtype):
    if dtype == torch.bool:
        return True
    if dtype.is_complex:
        return complex(2.0, 0.25)
    if dtype.is_floating_point:
        return 2.0
    return 2


def indexing_scalar_tensor(dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(indexing_scalar_value(dtype), dtype=dtype, device=device)


def isin_tensor_values(dtype: torch.dtype, device: str = "cpu", shape: tuple[int, ...] = (3, 4)) -> torch.Tensor:
    count = math.prod(shape)
    if dtype == torch.bool:
        values = [True, False, True, False, False, True, False, True, True, False, False, True]
        return torch.tensor(values[:count], dtype=torch.bool, device=device).reshape(shape)
    if dtype.is_complex:
        base = torch.tensor(
            [1 + 0j, 2 + 0.5j, 3 - 1j, 4 + 0j, 5 + 1j, 6 - 0.5j, 7 + 0j, 8 + 0j, 9 + 1j, 10 + 0j, 11 - 1j, 12 + 0j],
            dtype=dtype,
            device=device,
        )
        return base[:count].reshape(shape)
    if dtype.is_floating_point:
        return torch.arange(1, count + 1, dtype=dtype, device=device).reshape(shape)
    return torch.arange(1, count + 1, dtype=torch.int64, device=device).to(dtype).reshape(shape)


def isin_test_values(dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    if dtype == torch.bool:
        return torch.tensor([True], dtype=torch.bool, device=device)
    if dtype.is_complex:
        return torch.tensor([2 + 0.5j, 6 - 0.5j, 13 + 0j], dtype=dtype, device=device)
    if dtype.is_floating_point:
        return torch.tensor([2, 6, 13], dtype=dtype, device=device)
    return torch.tensor([2, 6, 13], dtype=dtype, device=device)


def isin_scalar_value(dtype: torch.dtype):
    if dtype == torch.bool:
        return True
    if dtype.is_complex:
        return complex(6.0, -0.5)
    if dtype.is_floating_point:
        return 6.0
    return 6


def indexing_index_tensor(entry: dict, device: str = "cpu") -> torch.Tensor:
    base_name = entry["base_name"].rstrip("_")
    if base_name in {"take", "put"}:
        return torch.tensor([0, 5, 9], dtype=torch.long, device=device)
    if base_name == "take_along_dim":
        return torch.tensor([[0, 2], [1, 3], [3, 0]], dtype=torch.long, device=device)
    if base_name in {"index_add", "index_copy", "index_fill", "index_reduce", "index_select"}:
        return torch.tensor([0, 2], dtype=torch.long, device=device)
    if base_name in {"embedding", "embedding_renorm"}:
        return torch.tensor([0, 2, 4, 1], dtype=torch.long, device=device)
    if base_name in {"gather", "scatter", "scatter_add", "scatter_reduce"}:
        return torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.long, device=device)
    return torch.tensor([0, 1], dtype=torch.long, device=device)


def indexing_tensor_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    base_name = entry["base_name"].rstrip("_")
    name = arg.get("name", "")
    if base_name == "bucketize" and name == "boundaries":
        boundary_dtype = torch.float32 if dtype == torch.bool or dtype.is_complex else dtype
        return torch.tensor([0, 2, 4, 6], dtype=boundary_dtype, device=device)
    if base_name == "searchsorted":
        if name == "sorted_sequence":
            sequence_dtype = torch.float32 if dtype == torch.bool or dtype.is_complex else dtype
            return torch.tensor([0, 2, 4, 6], dtype=sequence_dtype, device=device)
        if name == "sorter":
            return None
    if base_name == "bincount" and name == "weights":
        return None
    if base_name == "isin" and name in {"elements", "self", "input"}:
        return isin_tensor_values(dtype, device=device)
    if base_name == "isin" and name == "test_elements":
        return isin_test_values(dtype, device=device)
    if name in {"self", "input"}:
        if base_name == "bincount":
            return torch.tensor([0, 1, 1, 3, 2, 3], dtype=torch.long, device=device)
        if base_name == "one_hot":
            return torch.tensor([0, 1, 2, 1], dtype=torch.long, device=device)
        if base_name == "nonzero":
            return make_tensor_values(dtype, device, shape=(3, 4), domain="mixed")
        if base_name == "embedding_renorm":
            return make_tensor_values(dtype, device, shape=(5, 3), domain="mixed")
        return make_tensor_values(dtype, device, shape=(3, 4), domain="mixed")
    if name == "weight":
        return make_tensor_values(dtype, device, shape=(5, 3), domain="mixed")
    if name == "mask":
        return torch.tensor(
            [[True, False, True, False], [False, True, False, True], [True, True, False, False]],
            dtype=torch.bool,
            device=device,
        )
    if name == "indices" and arg.get("type") == "List[Optional[Tensor]]":
        return [
            torch.tensor([0, 2], dtype=torch.long, device=device),
            torch.tensor([1, 3], dtype=torch.long, device=device),
        ]
    if name in {"index", "indices"}:
        return indexing_index_tensor(entry, device=device)
    if name == "values" and base_name in {"_index_put_impl", "index_put"}:
        return make_tensor_values(dtype, device, shape=(2,), offset=0.5, domain="mixed")
    if name in {"src", "source"}:
        if base_name in {"index_add", "index_copy", "index_reduce", "scatter", "scatter_add", "scatter_reduce", "slice_scatter"}:
            shape = (3, 2)
        elif base_name in {"diagonal_scatter", "select_scatter"}:
            shape = (3,)
        elif base_name == "put":
            shape = (3,)
        else:
            shape = (3, 4)
        return make_tensor_values(dtype, device, shape=shape, offset=0.5, domain="mixed")
    if name == "value":
        return indexing_scalar_tensor(dtype, device=device)
    if name == "start":
        return torch.tensor(1, dtype=torch.long, device=device)
    return make_tensor_values(dtype, device, shape=(3, 4), offset=0.25, domain="mixed")


def indexing_arg_value(entry: dict, arg: dict, dtype: torch.dtype):
    base_name = entry["base_name"].rstrip("_")
    name = arg.get("name", "")
    if name == "dim":
        return 1
    if name == "padding_idx":
        return -1
    if name in {"scale_grad_by_freq", "sparse"}:
        return False
    if name in {"assume_unique", "invert"}:
        return False
    if name in {"out_int32", "right"}:
        return False
    if name == "max_norm":
        return 1.0
    if name == "norm_type":
        return 2.0
    if name == "num_classes":
        return 4
    if name == "sparse_grad":
        return False
    if name == "accumulate":
        return False
    if name == "include_self":
        return True
    if name == "reduce":
        if base_name == "index_reduce":
            return "amax"
        return "add" if base_name == "scatter" else "sum"
    if name == "scale":
        return 0.5
    if name == "mask_type":
        return None
    if name == "value":
        return indexing_scalar_value(dtype)
    if name in {"element", "test_element"}:
        return isin_scalar_value(dtype) if base_name == "isin" else indexing_scalar_value(dtype)
    if base_name in {"bucketize", "searchsorted"} and name == "self":
        return indexing_scalar_value(dtype)
    if name == "offset":
        return 0
    if name == "dim1":
        return 0
    if name == "dim2":
        return 1
    if name == "start":
        return 1
    if name == "end":
        return 3
    if name == "step":
        return 1
    if name == "length":
        return 2
    if name == "index":
        return 1
    if base_name == "bincount" and name == "minlength":
        return 5
    return None


def indexing_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("name") == "out":
            continue
        if arg.get("tensor"):
            value = indexing_tensor_arg_value(entry, arg, dtype, device)
        else:
            value = indexing_arg_value(entry, arg, dtype)

        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)

    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_indexing",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


CPU_GENERATOR_RNG_BASES = frozenset({"_sample_dirichlet", "binomial", "poisson"})


def rng_uses_target_device_generator(entry: dict) -> bool:
    """Return whether this RNG surface expects a generator for the output device."""

    return entry["base_name"].rstrip("_") not in CPU_GENERATOR_RNG_BASES


def rng_generator_device(entry: dict, device: str) -> str:
    if rng_uses_target_device_generator(entry):
        return device
    return "cpu"


def rng_generator(device: str, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except Exception as exc:
        raise SampleGenerationError(f"torch.Generator(device={device!r}) is not available") from exc
    generator.manual_seed(int(seed))
    return generator


def rng_tensor_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str):
    base_name = entry["base_name"].rstrip("_")
    name = arg.get("name", "")
    if base_name == "bernoulli":
        if name == "p":
            return torch.full((3, 4), 0.5, dtype=torch.float32, device=device)
        return torch.full((3, 4), 0.5, dtype=torch.float32 if dtype == torch.bool else dtype, device=device)
    if base_name == "normal":
        if name == "std":
            return torch.full((3, 4), 0.75, dtype=torch.float32 if dtype == torch.bool else dtype, device=device)
        if name in {"mean", "self"}:
            return make_tensor_values(torch.float32 if dtype == torch.bool else dtype, device, shape=(3, 4), domain="mixed")
    if base_name == "multinomial":
        return torch.full((3, 4), 0.25, dtype=torch.float32, device=device)
    if base_name == "binomial":
        if name == "count":
            return torch.full((3, 4), 10.0, dtype=torch.float32 if not dtype.is_floating_point else dtype, device=device)
        if name == "prob":
            return torch.full((3, 4), 0.5, dtype=torch.float32 if not dtype.is_floating_point else dtype, device=device)
    if base_name in {"_sample_dirichlet", "_standard_gamma"}:
        sample_dtype = dtype if dtype.is_floating_point else torch.float32
        return torch.linspace(1.0, 3.0, 12, dtype=sample_dtype, device=device).reshape(3, 4)
    if base_name == "poisson":
        return torch.full((3, 4), 4.0, dtype=torch.float32, device=device)
    if base_name == "randint_like":
        if name == "high":
            return torch.tensor(7, dtype=torch.long, device="cpu")
        return torch.empty((3, 4), dtype=dtype, device=device)
    if base_name in {"random", "random_"}:
        return torch.empty((3, 4), dtype=dtype, device=device)
    return make_tensor_values(dtype, device, shape=(3, 4), domain="mixed")


def rng_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str, seed: int):
    name = arg.get("name", "")
    base_name = entry["base_name"].rstrip("_")
    if arg.get("tensor"):
        return rng_tensor_arg_value(entry, arg, dtype, device)
    if name == "size":
        return [3, 4]
    if name == "n":
        return 12
    if name == "generator":
        return rng_generator(rng_generator_device(entry, device), seed)
    if name == "names":
        return None
    if name == "memory_format":
        return None
    if name == "low":
        return 2
    if name == "high":
        return 7
    if name == "from":
        return 2
    if name == "to":
        return 11
    if name == "mean":
        return 0.25
    if name == "median":
        return 0.0
    if name == "sigma":
        return 1.0
    if name == "std":
        return 0.75
    if name == "lambd":
        return 1.0
    if name == "p":
        return 0.5
    if name == "num_samples":
        return 2
    if name == "replacement":
        return False
    if base_name == "randint" and name in {"low", "high"}:
        return 2 if name == "low" else 7
    return None


def rng_call_parts(entry: dict, dtype: torch.dtype, device: str, seed: int = DEFAULT_IEEE754_SEED) -> tuple[tuple, dict]:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("name") == "out":
            continue
        value = rng_arg_value(entry, arg, dtype, device, seed)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    return tuple(args), kwargs


def rng_output_shape(entry: dict, args: tuple) -> tuple[int, ...]:
    base_name = entry["base_name"].rstrip("_")
    arg_names = [arg.get("name") for arg in entry.get("args", []) if arg.get("name") != "out" and not arg.get("kwarg_only")]
    arg_map = dict(zip(arg_names, args))
    if "size" in arg_map:
        return tuple(int(dim) for dim in arg_map["size"])
    if base_name == "randperm" and "n" in arg_map:
        return (int(arg_map["n"]),)
    if base_name == "multinomial":
        self_arg = arg_map.get("self")
        num_samples = int(arg_map.get("num_samples", 2))
        if isinstance(self_arg, torch.Tensor):
            if self_arg.ndim == 1:
                return (num_samples,)
            return (*tuple(self_arg.shape[:-1]), num_samples)
    for name in ("self", "mean", "std", "p"):
        value = arg_map.get(name)
        if isinstance(value, torch.Tensor):
            return tuple(value.shape)
    if base_name in {"rand", "randn", "randint", "normal"}:
        return (3, 4)
    return (3, 4)


def rng_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args, kwargs = rng_call_parts(entry, dtype, device, seed)
    strategy = entry["generated"]["strategy"]
    has_input = bool(args and isinstance(args[0], torch.Tensor))
    return GeneratedSample(
        dispatcher_name=entry["name"],
        strategy_name="manual_rng",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        input=args[0] if has_input else None,
        args=tuple(args[1:] if has_input else args),
        kwargs=kwargs,
        has_input=has_input,
        sample_index=sample_index,
        metadata={"out_shape": rng_output_shape(entry, args)},
    )


def multi_output_reduction_tensor(dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    values = torch.tensor(
        [
            [1.0, 4.0, 2.0, 8.0],
            [3.0, 7.0, 5.0, 6.0],
            [9.0, 0.0, 11.0, 10.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    if dtype == torch.bool:
        return values > 4
    if dtype.is_complex:
        return torch.complex(values, values / 8).to(dtype)
    if dtype.is_floating_point:
        return values.to(dtype)
    return values.to(torch.int64).to(dtype)


def unique_tensor(dtype: torch.dtype, device: str = "cpu", *, dim_case: bool = False) -> torch.Tensor:
    if dim_case:
        values = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
            device=device,
        )
    else:
        values = torch.tensor(
            [
                [3.0, 1.0, 3.0, 2.0],
                [2.0, 4.0, 1.0, 4.0],
                [5.0, 5.0, 6.0, 6.0],
            ],
            dtype=torch.float32,
            device=device,
        )
    if dtype == torch.bool:
        return (values.remainder(2) == 0).to(device)
    if dtype.is_complex:
        return torch.complex(values, values / 8).to(dtype)
    if dtype.is_floating_point:
        return values.to(dtype)
    return values.to(torch.int64).to(dtype)


def fake_quant_tensor(dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    if not dtype.is_floating_point:
        raise SampleGenerationError(f"fake quant samples require a floating dtype, got {dtype}")
    values = torch.linspace(-1.25, 1.25, 12, dtype=torch.float32, device=device).reshape(3, 4)
    return values.to(dtype)


def fake_quant_scale_tensor(entry: dict, dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    base_name = entry["base_name"].rstrip("_")
    if "per_channel" in base_name:
        return torch.tensor([0.05, 0.10, 0.20], dtype=torch.float32, device=device)
    return torch.tensor([0.05], dtype=torch.float32, device=device)


def fake_quant_zero_point_tensor(entry: dict, device: str = "cpu") -> torch.Tensor:
    base_name = entry["base_name"].rstrip("_")
    if "per_channel" in base_name:
        return torch.tensor([128, 127, 126], dtype=torch.int32, device=device)
    return torch.tensor([128], dtype=torch.int32, device=device)


def embedding_bag_tensor_arg(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    if name == "weight":
        return make_tensor_values(dtype, device, shape=(5, 3), domain="mixed")
    if name == "indices":
        return torch.tensor([0, 2, 4, 1], dtype=torch.long, device=device)
    if name == "offsets":
        return torch.tensor([0, 2], dtype=torch.long, device=device)
    if name == "per_sample_weights":
        return None
    return make_tensor_values(dtype, device, shape=(3, 4), domain="mixed")


def batch_norm_tensor_arg(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    if not dtype.is_floating_point:
        raise SampleGenerationError(f"batch norm samples require a floating dtype, got {dtype}")
    if name in {"input", "self", "grad_out", "grad_output"}:
        return make_tensor_values(dtype, device, shape=(2, 3, 4, 4), domain="mixed")
    if name == "weight":
        return torch.ones(3, dtype=dtype, device=device)
    if name == "bias":
        return torch.zeros(3, dtype=dtype, device=device)
    if name in {"save_mean", "save_var", "save_var_transform", "save_invstd"}:
        return torch.empty(0, dtype=dtype, device=device)
    if name in {"reserve", "reservedSpace"}:
        return torch.empty(0, dtype=torch.uint8, device=device)
    if name in {"running_mean", "mean", "sum_dy", "sum_dy_xmu"}:
        return torch.zeros(3, dtype=dtype, device=device)
    if name in {"running_var", "invstd"}:
        return torch.ones(3, dtype=dtype, device=device)
    if name == "count":
        return torch.full((1,), 32, dtype=torch.long, device=device)
    return make_tensor_values(dtype, device, shape=(3,), domain="positive")


def histogram_tensor_arg(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    base_name = entry["base_name"].rstrip("_")
    if not dtype.is_floating_point:
        raise SampleGenerationError(f"histogram samples require a floating dtype, got {dtype}")
    if name == "weight":
        return None
    if name == "bins":
        if arg.get("type") == "List[Tensor]":
            return [
                torch.tensor([0.0, 0.5, 1.5], dtype=torch.float32, device=device),
                torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32, device=device),
            ]
        return torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float32, device=device)
    if base_name.startswith("_histogramdd"):
        return torch.tensor(
            [[0.1, 0.2], [0.7, 0.4], [0.9, 0.8], [1.1, 0.6]],
            dtype=dtype,
            device=device,
        )
    return torch.tensor(
        [0.1, 0.2, 0.7, 0.9, 1.1],
        dtype=dtype,
        device=device,
    )


def multi_output_reduction_other_tensor(dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    values = torch.tensor(
        [
            [2.0, 3.0, 5.0, 1.0],
            [4.0, 6.0, 8.0, 7.0],
            [10.0, 12.0, 9.0, 11.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    if dtype == torch.bool:
        return values > 5
    if dtype.is_complex:
        return torch.complex(values, values / 8).to(dtype)
    if dtype.is_floating_point:
        return values.to(dtype)
    return values.to(torch.int64).to(dtype)


def multi_output_reduction_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    base_name = entry["base_name"].rstrip("_")
    if arg.get("tensor") or arg.get("type") == "Optional[Tensor]":
        if base_name in {"_ctc_loss", "_ctc_loss_backward"}:
            if name == "log_probs":
                return ctc_log_probs(dtype, device)
            if name == "targets":
                return ctc_targets(device)
            if name == "input_lengths":
                return ctc_input_lengths(True, device)
            if name == "target_lengths":
                return ctc_target_lengths(True, device)
            if name == "grad":
                return torch.ones(2, dtype=dtype, device=device)
            if name == "neg_log_likelihood":
                return ctc_saved_tensors(dtype, device)[0]
            if name == "log_alpha":
                return ctc_saved_tensors(dtype, device)[1]
        if base_name == "multilabel_margin_loss_forward":
            if not dtype.is_floating_point:
                raise SampleGenerationError(f"multilabel margin loss samples require a floating dtype, got {dtype}")
            if name in {"self", "input"}:
                return make_tensor_values(dtype, device, shape=(2, 4), domain="mixed")
            if name == "target":
                return torch.tensor([[0, 1, 2, -1], [1, 0, -1, -1]], dtype=torch.long, device=device)
        if "histogram" in base_name:
            return histogram_tensor_arg(entry, arg, dtype, device)
        if base_name.startswith("linalg_") or base_name.startswith("_linalg_"):
            return linalg_tensor_arg_value(entry, arg, dtype, device)
        if "batch_norm" in base_name:
            return batch_norm_tensor_arg(entry, arg, dtype, device)
        if "embedding_bag" in base_name:
            return embedding_bag_tensor_arg(entry, arg, dtype, device)
        if base_name in {"nll_loss_forward", "nll_loss2d_forward"}:
            alias = "nll_loss" if base_name == "nll_loss_forward" else "nll_loss2d"
            aliased_entry = {**entry, "base_name": alias}
            return loss_arg_value(aliased_entry, arg, dtype, device)
        if base_name == "slogdet":
            return linalg_matrix(dtype, device)
        if "fake_quantize" in base_name:
            if name in {"self", "input"}:
                return fake_quant_tensor(dtype, device)
            if name == "scale":
                return fake_quant_scale_tensor(entry, dtype, device)
            if name == "zero_point":
                return fake_quant_zero_point_tensor(entry, device)
            if name == "fake_quant_enabled":
                return torch.tensor([1], dtype=torch.uint8, device=device)
        if base_name in {"unique_dim", "unique_dim_consecutive", "unique_consecutive"}:
            return unique_tensor(dtype, device, dim_case=True)
        if base_name in {"_unique", "_unique2"}:
            return unique_tensor(dtype, device)
        if base_name in {"geqrf", "qr"}:
            return linalg_matrix(dtype, device, shape=(4, 3), positive_definite=False)
        if name == "other":
            return multi_output_reduction_other_tensor(dtype, device)
        return multi_output_reduction_tensor(dtype, device)
    if name == "dim":
        return 1
    if name == "axis":
        return 0
    if base_name in {"_ctc_loss", "_ctc_loss_backward"}:
        if name == "input_lengths":
            return ctc_input_lengths(False, device)
        if name == "target_lengths":
            return ctc_target_lengths(False, device)
        if name == "blank":
            return 0
        if name == "zero_infinity":
            return False
    if base_name == "multilabel_margin_loss_forward" and name == "reduction":
        return 1
    if name == "keepdim":
        return False
    if name == "training":
        if "no_stats" in entry.get("name", ""):
            return True
        return False
    if name == "train":
        return False
    if name == "update":
        return False
    if name == "cudnn_enabled":
        return False
    if name == "impl_index":
        return 0
    if name == "output_mask":
        return [True, True, True]
    if name in {"scale_grad_by_freq", "sparse", "include_last_offset"}:
        return False
    if name == "descending":
        return False
    if name == "stable":
        return True
    if name == "k":
        return 2
    if name == "largest":
        return True
    if name == "sorted":
        return True
    if name in {"return_inverse", "return_counts"}:
        return True
    if name == "scale":
        return 0.05
    if name == "zero_point":
        return 128
    if name == "quant_min":
        return 0
    if name == "quant_max":
        return 255
    if name == "mode":
        return 0
    if name == "padding_idx":
        return -1
    if "histogram" in base_name:
        if name == "bins":
            return [2, 3] if arg.get("type") == "List[int]" else 3
        if name == "range":
            return None
        if name == "density":
            return False
    if base_name.startswith("linalg_") or base_name.startswith("_linalg_"):
        return linalg_arg_value(entry, arg, dtype, device)
    if name == "momentum":
        return 0.1
    if name == "eps":
        return 1e-5
    if base_name in {"nll_loss_forward", "nll_loss2d_forward"}:
        alias = "nll_loss" if base_name == "nll_loss_forward" else "nll_loss2d"
        aliased_entry = {**entry, "base_name": alias}
        return loss_arg_value(aliased_entry, arg, dtype, device)
    if name in {"min", "max"} and base_name == "aminmax":
        return None
    return None


def multi_output_reduction_call_parts(entry: dict, dtype: torch.dtype, device: str = "cpu") -> tuple[tuple, dict]:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        value = multi_output_reduction_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    return tuple(args), kwargs


def multi_output_reduction_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args, kwargs = multi_output_reduction_call_parts(entry, dtype, device)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_multi_output_reduction",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


def upsample_spatial_rank(base_name: str) -> int:
    if "1d" in base_name:
        return 1
    if "2d" in base_name:
        return 2
    if "3d" in base_name:
        return 3
    return 2


def upsample_input_shape(base_name: str) -> tuple[int, ...]:
    rank = upsample_spatial_rank(base_name)
    if rank == 1:
        return (2, 3, 4)
    if rank == 2:
        return (2, 3, 4, 5)
    return (1, 2, 3, 4, 5)


def upsample_output_size(base_name: str) -> list[int]:
    rank = upsample_spatial_rank(base_name)
    if rank == 1:
        return [7]
    if rank == 2:
        return [6, 7]
    return [4, 6, 7]


def upsample_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    base_name = entry["base_name"]
    if arg.get("tensor"):
        return make_tensor_values(dtype, device, shape=upsample_input_shape(base_name), domain="mixed")
    if name == "output_size":
        return upsample_output_size(base_name)
    if name == "align_corners":
        return False
    if name in {"scales", "scales_h", "scales_w", "scales_d", "scale_factors"}:
        return None
    return None


def upsample_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        value = upsample_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_upsample",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


def pooling_spatial_rank(base_name: str) -> int:
    if "1d" in base_name:
        return 1
    if "2d" in base_name:
        return 2
    if "3d" in base_name:
        return 3
    return 2


def pooling_input_shape(base_name: str) -> tuple[int, ...]:
    rank = pooling_spatial_rank(base_name)
    if rank == 1:
        return (2, 3, 6)
    if rank == 2:
        return (2, 3, 6, 7)
    return (1, 2, 5, 6, 7)


def pooling_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    base_name = entry["base_name"]
    rank = pooling_spatial_rank(base_name)
    if arg.get("tensor"):
        if name == "random_samples":
            batch = pooling_input_shape(base_name)[0]
            channels = pooling_input_shape(base_name)[1]
            return torch.full((batch, channels, rank), 0.5, dtype=torch.float32, device=device)
        return make_tensor_values(dtype, device, shape=pooling_input_shape(base_name), domain="mixed")
    if name == "output_size":
        return [2] * rank
    if name == "kernel_size":
        return [2] * rank
    if name == "stride":
        return [2] * rank
    if name == "padding":
        return [0] * rank
    if name == "dilation":
        return [1] * rank
    if name == "ceil_mode":
        return False
    if name == "count_include_pad":
        return True
    if name == "divisor_override":
        return None
    return None


def pooling_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        value = pooling_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_pooling",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


def convolution_rank(base_name: str) -> int:
    if "3d" in base_name:
        return 3
    return 2


def convolution_group_count(base_name: str) -> int:
    if "depthwise" in base_name:
        return 2 if convolution_rank(base_name) == 3 else 4
    return 1


def convolution_input_tensor(base_name: str, dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    if convolution_rank(base_name) == 3:
        channels = convolution_group_count(base_name)
        return make_tensor_values(dtype, device, shape=(1, channels, 5, 6, 6), domain="mixed")
    channels = convolution_group_count(base_name) if "depthwise" in base_name else 3
    return make_tensor_values(dtype, device, shape=(2, channels, 8, 8), domain="mixed")


def convolution_weight_tensor(base_name: str, dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    if convolution_rank(base_name) == 3:
        groups = convolution_group_count(base_name)
        out_channels = groups if "depthwise" in base_name else 3
        in_per_group = 1 if "depthwise" in base_name else 2
        return make_weight_tensor(dtype, device=device, shape=(out_channels, in_per_group, 3, 3, 3))
    if "depthwise" in base_name:
        return make_weight_tensor(dtype, device=device, shape=(4, 1, 3, 3))
    return make_weight_tensor(dtype, device=device, shape=(4, 3, 3, 3))


def convolution_bias_tensor(base_name: str, dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    out_channels = convolution_weight_tensor(base_name, dtype, device).shape[0]
    return make_tensor_values(dtype, device, shape=(out_channels,), domain="mixed")


def convolution_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    base_name = entry["base_name"]
    rank = convolution_rank(base_name)
    groups = convolution_group_count(base_name)
    if arg.get("tensor"):
        if base_name == "col2im" and name in {"input", "self"}:
            return make_tensor_values(dtype, device, shape=(2, 27, 64), domain="mixed")
        if name in {"input", "self"}:
            return convolution_input_tensor(base_name, dtype, device)
        if name == "weight":
            return convolution_weight_tensor(base_name, dtype, device)
        if name == "bias":
            return convolution_bias_tensor(base_name, dtype, device)
        return convolution_input_tensor(base_name, dtype, device)
    if name == "kernel_size":
        return [3] * rank
    if name == "output_size":
        return [8] * rank
    if name == "stride":
        return [1] * rank
    if name == "padding":
        return [1] * rank
    if name == "dilation":
        return [1] * rank
    if name == "output_padding":
        return [0] * rank
    if name == "groups":
        return groups
    if name in {"transposed", "benchmark", "deterministic", "cudnn_enabled", "allow_tf32"}:
        return False
    return None


def convolution_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        value = convolution_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_convolution",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


def grid_sample_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    if not dtype.is_floating_point:
        raise SampleGenerationError(f"grid samples require a floating dtype, got {dtype}")

    name = arg.get("name", "")
    base_name = entry["base_name"]
    if arg.get("tensor"):
        if base_name == "affine_grid_generator" and name == "theta":
            return torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=dtype, device=device)
        if base_name == "grid_sampler_3d" and name in {"input", "self"}:
            return make_tensor_values(dtype, device, shape=(1, 1, 3, 4, 5), domain="mixed")
        if base_name == "grid_sampler_3d" and name == "grid":
            return torch.zeros((1, 2, 3, 4, 3), dtype=dtype, device=device)
        if base_name in {"grid_sampler", "grid_sampler_2d", "_grid_sampler_2d_cpu_fallback"} and name in {"input", "self"}:
            return make_tensor_values(dtype, device, shape=(1, 1, 4, 5), domain="mixed")
        if base_name in {"grid_sampler", "grid_sampler_2d", "_grid_sampler_2d_cpu_fallback"} and name == "grid":
            return torch.zeros((1, 2, 3, 2), dtype=dtype, device=device)
    if name == "size":
        return [1, 1, 4, 5]
    if name == "align_corners":
        return False
    if name in {"interpolation_mode", "padding_mode"}:
        return 0
    return None


def grid_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        value = grid_sample_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_grid",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=sample,
        sample_index=sample_index,
    )


def grid_backward_sample_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    if not dtype.is_floating_point:
        raise SampleGenerationError(f"grid backward samples require a floating dtype, got {dtype}")

    name = arg.get("name", "")
    base_name = entry["base_name"]
    is_3d = base_name == "grid_sampler_3d_backward"
    if arg.get("tensor"):
        if name in {"input", "self"}:
            shape = (1, 1, 3, 4, 5) if is_3d else (1, 1, 4, 5)
            return make_tensor_values(dtype, device, shape=shape, domain="mixed")
        if name == "grid":
            shape = (1, 2, 3, 4, 3) if is_3d else (1, 2, 3, 2)
            return torch.zeros(shape, dtype=dtype, device=device)
        if name == "grad_output":
            shape = (1, 1, 2, 3, 4) if is_3d else (1, 1, 2, 3)
            return make_tensor_values(dtype, device, shape=shape, domain="positive")
    if name in {"interpolation_mode", "padding_mode"}:
        return 0
    if name == "align_corners":
        return False
    if name == "output_mask":
        return [True, True]
    return None


def grid_backward_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        value = grid_backward_sample_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_grid_backward",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=sample,
        sample_index=sample_index,
    )


def rnn_cell_gate_count(base_name: str) -> int:
    if base_name == "lstm_cell":
        return 4
    if base_name == "gru_cell":
        return 3
    return 1


def rnn_cell_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    if not dtype.is_floating_point:
        raise SampleGenerationError(f"RNN cell samples require a floating dtype, got {dtype}")

    name = arg.get("name", "")
    base_name = entry["base_name"]
    batch = 2
    input_size = 3
    hidden_size = 4
    gate_count = rnn_cell_gate_count(base_name)
    if name == "input":
        return make_tensor_values(dtype, device, shape=(batch, input_size), domain="mixed", offset=0.0)
    if name == "hx":
        h = make_tensor_values(dtype, device, shape=(batch, hidden_size), domain="mixed", offset=0.25)
        if base_name == "lstm_cell":
            c = make_tensor_values(dtype, device, shape=(batch, hidden_size), domain="mixed", offset=0.5)
            return [h, c]
        return h
    if name == "w_ih":
        return make_weight_tensor(dtype, device=device, shape=(gate_count * hidden_size, input_size))
    if name == "w_hh":
        return make_weight_tensor(dtype, device=device, shape=(gate_count * hidden_size, hidden_size))
    if name in {"b_ih", "b_hh"}:
        return make_tensor_values(dtype, device, shape=(gate_count * hidden_size,), domain="mixed", offset=0.125)
    return None


def rnn_cell_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        value = rnn_cell_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_rnn_cell",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=sample,
        sample_index=sample_index,
    )


def loss_float_tensor(
    dtype: torch.dtype,
    device: str = "cpu",
    shape: Iterable[int] = (3, 4),
    offset: float = 0.0,
) -> torch.Tensor:
    if dtype == torch.bool or not (dtype.is_floating_point or dtype.is_complex):
        dtype = torch.float32
    return make_tensor_values(dtype, device, shape=shape, domain="mixed", offset=offset)


def ctc_log_probs(dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    if not dtype.is_floating_point:
        raise SampleGenerationError(f"CTC loss samples require a floating dtype, got {dtype}")
    logits = torch.tensor(
        [
            [[1.2, 0.1, -0.3], [0.2, 1.1, -0.4]],
            [[0.8, 0.3, -0.2], [0.4, 0.9, -0.1]],
            [[0.1, 1.0, -0.2], [0.5, 0.1, 0.8]],
            [[0.0, 0.4, 1.1], [0.3, 0.2, 0.9]],
        ],
        dtype=torch.float32,
        device=device,
    )
    return torch.log_softmax(logits, dim=2).to(dtype)


def ctc_targets(device: str = "cpu") -> torch.Tensor:
    return torch.tensor([1, 1, 2], dtype=torch.long, device=device)


def ctc_input_lengths(as_tensor: bool, device: str = "cpu"):
    values = [4, 4]
    if as_tensor:
        return torch.tensor(values, dtype=torch.long, device=device)
    return values


def ctc_target_lengths(as_tensor: bool, device: str = "cpu"):
    values = [1, 2]
    if as_tensor:
        return torch.tensor(values, dtype=torch.long, device=device)
    return values


def ctc_saved_tensors(dtype: torch.dtype, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = ctc_log_probs(dtype, device)
    return torch.ops.aten._ctc_loss(log_probs, ctc_targets(device), [4, 4], [1, 2], 0, False)


def loss_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    base_name = entry["base_name"]
    if arg.get("tensor"):
        if base_name == "ctc_loss":
            if name == "log_probs":
                return ctc_log_probs(dtype, device)
            if name == "targets":
                return ctc_targets(device)
            if name == "input_lengths":
                return ctc_input_lengths(True, device)
            if name == "target_lengths":
                return ctc_target_lengths(True, device)
        if base_name == "binary_cross_entropy":
            if name in {"self", "input"}:
                return torch.sigmoid(loss_float_tensor(dtype, device, shape=(3, 4)))
            if name == "target":
                return torch.sigmoid(loss_float_tensor(dtype, device, shape=(3, 4), offset=0.5))
            if name == "weight":
                return None
        if base_name == "binary_cross_entropy_with_logits":
            if name in {"self", "input"}:
                return loss_float_tensor(dtype, device, shape=(3, 4))
            if name == "target":
                return torch.sigmoid(loss_float_tensor(dtype, device, shape=(3, 4), offset=0.5))
            if name in {"weight", "pos_weight"}:
                return None
        if base_name == "kl_div":
            if name in {"self", "input"}:
                return torch.log_softmax(loss_float_tensor(dtype, device, shape=(3, 4)), dim=1)
            if name == "target":
                return torch.softmax(loss_float_tensor(dtype, device, shape=(3, 4), offset=0.5), dim=1)
        if base_name in {"nll_loss", "cross_entropy_loss"}:
            if name in {"self", "input"}:
                logits = loss_float_tensor(dtype, device, shape=(3, 5))
                return torch.log_softmax(logits, dim=1) if base_name == "nll_loss" else logits
            if name == "target":
                return torch.tensor([0, 2, 4], dtype=torch.long, device=device)
            if name == "weight":
                return None
        if base_name == "nll_loss2d":
            if name in {"self", "input"}:
                return torch.log_softmax(loss_float_tensor(dtype, device, shape=(2, 5, 3, 4)), dim=1)
            if name == "target":
                return torch.tensor(
                    [[[0, 1, 2, 3], [1, 2, 3, 4], [4, 3, 2, 1]], [[1, 0, 2, 4], [3, 2, 1, 0], [0, 4, 3, 2]]],
                    dtype=torch.long,
                    device=device,
                )
            if name == "weight":
                return None
        if base_name == "nll_loss_nd":
            if name in {"self", "input"}:
                return torch.log_softmax(loss_float_tensor(dtype, device, shape=(2, 5, 3)), dim=1)
            if name == "target":
                return torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long, device=device)
            if name == "weight":
                return None
        if base_name == "multi_margin_loss":
            if name in {"self", "input"}:
                return loss_float_tensor(dtype, device, shape=(3, 5))
            if name == "target":
                return torch.tensor([0, 2, 4], dtype=torch.long, device=device)
            if name == "weight":
                return None
        if base_name == "multilabel_margin_loss":
            if name in {"self", "input"}:
                return loss_float_tensor(dtype, device, shape=(2, 4))
            if name == "target":
                return torch.tensor([[0, 1, -1, -1], [2, 3, -1, -1]], dtype=torch.long, device=device)
        if base_name == "cosine_embedding_loss":
            if name == "target":
                return torch.tensor([1, -1, 1], dtype=torch.float32, device=device)
            return loss_float_tensor(dtype, device, shape=(3, 4), offset=0.25 if name == "input2" else 0.0)
        if base_name == "margin_ranking_loss":
            if name == "target":
                return torch.tensor([1, -1, 1, -1], dtype=torch.float32, device=device)
            return loss_float_tensor(dtype, device, shape=(4,), offset=0.25 if name == "input2" else 0.0)
        if base_name == "hinge_embedding_loss":
            if name == "target":
                return torch.tensor([[1, -1, 1, -1], [-1, 1, -1, 1], [1, 1, -1, -1]], dtype=torch.float32, device=device)
            return loss_float_tensor(dtype, device)
        if base_name == "triplet_margin_loss":
            offsets = {"anchor": 0.0, "positive": 0.1, "negative": 0.75}
            return loss_float_tensor(dtype, device, shape=(3, 4), offset=offsets.get(name, 0.0))
        if name == "target":
            return loss_float_tensor(dtype, device, offset=0.5)
        return loss_float_tensor(dtype, device)
    if name == "reduction":
        return 1
    if base_name == "ctc_loss":
        if name == "input_lengths":
            return ctc_input_lengths(False, device)
        if name == "target_lengths":
            return ctc_target_lengths(False, device)
        if name == "blank":
            return 0
        if name == "zero_infinity":
            return False
    if name == "ignore_index":
        return -100
    if name == "label_smoothing":
        return 0.0
    if name == "log_target":
        return False
    if name in {"margin", "delta", "beta"}:
        return 1.0
    if name == "p":
        return 1
    if name == "eps":
        return 1e-6
    if name == "swap":
        return False
    return None


def loss_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        value = loss_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_loss",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


def linalg_matrix(
    dtype: torch.dtype,
    device: str = "cpu",
    *,
    shape: Iterable[int] = (3, 3),
    positive_definite: bool = False,
    offset: float = 0.0,
) -> torch.Tensor:
    """Create a deterministic dense linalg matrix for dispatcher samples."""

    matrix = make_distribution_tensor(
        dtype,
        device,
        shape=shape,
        distribution="xavier_normal",
        domain="mixed",
        seed=DEFAULT_IEEE754_SEED + int(offset * 1000),
    )
    if positive_definite:
        if not (dtype.is_floating_point or dtype.is_complex):
            return matrix
        square = matrix
        if square.shape[-1] != square.shape[-2]:
            size = min(square.shape[-1], square.shape[-2])
            square = square[..., :size, :size]
        product = square @ square.mH
        eye = torch.eye(product.shape[-1], dtype=product.dtype, device=device)
        return product + eye.mul(product.shape[-1])
    return matrix


def linalg_vector_tensor(dtype: torch.dtype, device: str = "cpu", *, shape: Iterable[int] = (2, 3), offset: float = 0.0) -> torch.Tensor:
    return make_distribution_tensor(
        dtype,
        device,
        shape=shape,
        distribution="activation_gelu",
        domain="mixed",
        seed=DEFAULT_IEEE754_SEED + 100 + int(offset * 1000),
    )


def linalg_cholesky_factor(dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    matrix = linalg_matrix(dtype, device, positive_definite=True)
    if not (dtype.is_floating_point or dtype.is_complex):
        return matrix
    return torch.linalg.cholesky(matrix)


def linalg_geqrf_factors(dtype: torch.dtype, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    if not (dtype.is_floating_point or dtype.is_complex):
        raise SampleGenerationError(f"QR factor samples require a floating or complex dtype, got {dtype}")
    base = linalg_matrix(dtype, "cpu", shape=(4, 3), positive_definite=False)
    a, tau = torch.geqrf(base)
    return a.to(device), tau.to(device)


def linalg_triangular_matrix(dtype: torch.dtype, device: str = "cpu", *, upper: bool = True) -> torch.Tensor:
    matrix = linalg_matrix(dtype, device, positive_definite=False) + torch.eye(3, dtype=dtype, device=device).mul(4)
    return torch.triu(matrix) if upper else torch.tril(matrix)


def linalg_tensor_operator(dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    if not (dtype.is_floating_point or dtype.is_complex):
        return torch.arange(16, dtype=dtype, device=device).reshape(2, 2, 2, 2)
    operator = torch.eye(4, dtype=dtype, device=device)
    perturb = make_distribution_tensor(
        dtype,
        device,
        shape=(4, 4),
        distribution="xavier_normal",
        domain="mixed",
        seed=DEFAULT_IEEE754_SEED + 707,
    ).mul(0.05)
    return (operator + perturb).reshape(2, 2, 2, 2)


def linalg_tensor_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    base_name = entry["base_name"].rstrip("_")
    name = arg.get("name", "")
    if name in {"atol", "rtol", "tol"}:
        return torch.tensor(1e-4, dtype=torch.float32, device=device)
    if base_name in {"_cdist_forward", "_euclidean_dist"}:
        if name in {"x2", "other"}:
            return linalg_vector_tensor(dtype, device, shape=(4, 3), offset=0.25)
        return linalg_vector_tensor(dtype, device, shape=(2, 3))
    if base_name in {"_pdist_forward", "pdist"}:
        return linalg_vector_tensor(dtype, device, shape=(4, 3))
    if base_name == "pairwise_distance":
        return linalg_vector_tensor(dtype, device, shape=(3, 4), offset=0.25 if name in {"x2", "other"} else 0.0)
    if base_name in {"dot", "vdot"}:
        return linalg_vector_tensor(dtype, device, shape=(4,), offset=0.25 if name in {"other", "tensor"} else 0.0)
    if base_name in {"orgqr", "ormqr"}:
        a, tau = linalg_geqrf_factors(dtype, device)
        if name in {"self", "input"}:
            return a
        if name in {"input2", "tau"}:
            return tau
        if name in {"input3", "other"}:
            return linalg_matrix(dtype, device, shape=(4, 2), positive_definite=False, offset=0.25)
    if base_name in {"outer", "ger"}:
        return linalg_vector_tensor(dtype, device, shape=(4,) if name in {"other", "vec2"} else (3,), offset=0.25 if name in {"other", "vec2"} else 0.0)
    if base_name == "inner":
        return linalg_vector_tensor(dtype, device, shape=(2, 4) if name in {"other", "tensor"} else (3, 4), offset=0.25 if name in {"other", "tensor"} else 0.0)
    if base_name == "mv":
        if name == "vec":
            return linalg_vector_tensor(dtype, device, shape=(4,), offset=0.25)
        return linalg_matrix(dtype, device, shape=(3, 4))
    if base_name == "addmv":
        if name == "self":
            return linalg_vector_tensor(dtype, device, shape=(3,), offset=0.5)
        if name == "mat":
            return linalg_matrix(dtype, device, shape=(3, 4))
        if name == "vec":
            return linalg_vector_tensor(dtype, device, shape=(4,), offset=0.25)
    if base_name == "addr":
        if name == "self":
            return linalg_matrix(dtype, device, shape=(3, 4), offset=0.5)
        if name == "vec1":
            return linalg_vector_tensor(dtype, device, shape=(3,))
        if name == "vec2":
            return linalg_vector_tensor(dtype, device, shape=(4,), offset=0.25)
    if base_name == "tensordot":
        if name in {"other", "b"}:
            return linalg_matrix(dtype, device, shape=(4, 3, 5), offset=0.25)
        return linalg_matrix(dtype, device, shape=(2, 3, 4))
    if base_name in {"cross", "linalg_cross"}:
        return linalg_vector_tensor(dtype, device, shape=(2, 3), offset=0.25 if name in {"other", "input2"} else 0.0)
    if base_name == "vander":
        return linalg_vector_tensor(dtype, device, shape=(4,))
    if base_name in {"_cholesky_solve_helper", "cholesky_solve"}:
        if name in {"A", "input2"}:
            return linalg_cholesky_factor(dtype, device)
        return linalg_matrix(dtype, device, shape=(3, 2), positive_definite=False)
    if base_name in {"cholesky", "linalg_cholesky"}:
        return linalg_matrix(dtype, device, positive_definite=True)
    if base_name == "cholesky_inverse":
        return linalg_cholesky_factor(dtype, device)
    if base_name in {
        "_linalg_det",
        "_linalg_eigh",
        "_linalg_slogdet",
        "linalg_cholesky_ex",
        "linalg_inv_ex",
        "linalg_lu",
        "linalg_lu_factor",
        "linalg_lu_factor_ex",
        "linalg_slogdet",
    }:
        return linalg_matrix(dtype, device, positive_definite=True)
    if base_name == "linalg_solve":
        if name in {"A", "self"}:
            return linalg_matrix(dtype, device, positive_definite=True)
        return linalg_matrix(dtype, device, shape=(3, 2), positive_definite=False, offset=0.25)
    if base_name == "linalg_solve_ex":
        if name in {"A", "self"}:
            return linalg_matrix(dtype, device, positive_definite=True)
        return linalg_matrix(dtype, device, shape=(3, 2), positive_definite=False, offset=0.25)
    if base_name == "_linalg_solve_ex":
        if name in {"A", "self"}:
            return linalg_matrix(dtype, device, positive_definite=True)
        return linalg_matrix(dtype, device, shape=(3, 2), positive_definite=False, offset=0.25)
    if base_name == "linalg_solve_triangular":
        if name in {"self", "A"}:
            return linalg_triangular_matrix(dtype, device, upper=True)
        return linalg_matrix(dtype, device, shape=(3, 2), positive_definite=False, offset=0.25)
    if base_name == "linalg_tensorinv":
        return linalg_tensor_operator(dtype, device)
    if base_name == "linalg_tensorsolve":
        if name in {"other", "B"}:
            return linalg_matrix(dtype, device, shape=(2, 2), positive_definite=False, offset=0.25)
        return linalg_tensor_operator(dtype, device)
    if base_name in {"inverse", "linalg_det", "linalg_inv", "linalg_pinv", "linalg_svdvals", "linalg_matrix_exp"}:
        return linalg_matrix(dtype, device, positive_definite=False) + torch.eye(3, dtype=dtype, device=device)
    if base_name in {"linalg_matrix_power", "matrix_power"}:
        return linalg_matrix(dtype, device, positive_definite=False) + torch.eye(3, dtype=dtype, device=device)
    if base_name in {"linalg_matrix_norm", "nuclear_norm"}:
        return linalg_matrix(dtype, device, shape=(3, 4), positive_definite=False)
    if base_name == "linalg_matrix_rank":
        return linalg_matrix(dtype, device, positive_definite=True)
    if base_name == "linalg_cond":
        return linalg_matrix(dtype, device, positive_definite=True)
    if base_name == "linalg_matmul":
        if name in {"other", "mat2"}:
            return linalg_matrix(dtype, device, shape=(4, 5), offset=0.25)
        return linalg_matrix(dtype, device, shape=(3, 4))
    if base_name == "_addmm_activation":
        if name == "self":
            return linalg_matrix(dtype, device, shape=(3, 5), offset=0.5)
        if name == "mat1":
            return linalg_matrix(dtype, device, shape=(3, 4))
        if name == "mat2":
            return linalg_matrix(dtype, device, shape=(4, 5), offset=0.25)
    if base_name == "linalg_vecdot":
        return linalg_vector_tensor(dtype, device, shape=(2, 3), offset=0.25 if name in {"other", "input2"} else 0.0)
    if base_name in {"linalg_vector_norm", "linalg_norm", "native_norm"}:
        return linalg_vector_tensor(dtype, device, shape=(2, 3))
    return linalg_matrix(dtype, device)


def linalg_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    arg_type = arg.get("type", "")
    base_name = entry["base_name"].rstrip("_")
    if arg.get("tensor"):
        return linalg_tensor_arg_value(entry, arg, dtype, device)
    if name in {"ord", "p", "pow"}:
        return "fro" if arg_type in {"str", "Optional[str]"} and base_name in {"linalg_cond", "linalg_matrix_norm", "nuclear_norm"} else 2
    if name == "dim":
        if base_name == "norm_except_dim":
            return 0
        if base_name in {"linalg_matrix_norm", "nuclear_norm"}:
            return [-2, -1] if arg_type in {"List[int]", "Optional[List[int]]"} else 1
        if base_name in {"cross", "linalg_cross", "linalg_vecdot", "linalg_vector_norm", "linalg_norm", "native_norm"}:
            return [-1] if arg_type in {"List[int]", "Optional[List[int]]"} else -1
        return None
    if name == "keepdim":
        return False
    if name == "compute_mode":
        return 1
    if name in {"alpha", "beta"}:
        return 1
    if name == "dims_self":
        return [2, 1]
    if name == "dims_other":
        return [0, 1]
    if name == "use_gelu":
        return False
    if name == "eps":
        return 1e-6
    if name == "dtype":
        return None
    if name == "n":
        return 2
    if name == "diagonal":
        return 0
    if name == "N":
        return 3
    if name == "increasing":
        return False
    if name in {"atol", "rtol", "rcond", "tol"}:
        return 1e-4
    if name in {"hermitian", "upper"}:
        return True if base_name == "linalg_solve_triangular" else False
    if name in {"left", "unitriangular"}:
        return True if name == "left" else False
    if name == "transpose":
        return False
    if name == "UPLO":
        return "L"
    if name == "compute_v":
        return True
    if name == "pivot":
        return True
    if name == "check_errors":
        return False
    if name == "mode":
        return "reduced"
    return None


def linalg_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        value = linalg_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_linalg",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


def metadata_tensor_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    base_name = entry["base_name"]
    name = arg.get("name", "")
    if base_name in {"is_nonzero", "_local_scalar_dense"}:
        return make_scalar_tensor(dtype, 1, device=device)
    if base_name == "_assert_async":
        return torch.tensor(True, dtype=torch.bool, device=device)
    if base_name in {"_is_all_true", "_is_any_true"}:
        return torch.tensor([[True, True], [True, False]], dtype=torch.bool, device=device)
    if base_name == "_assert_tensor_metadata":
        return make_tensor_values(dtype, device, shape=(2, 3, 4), domain="mixed")
    if base_name in {"is_same_size", "is_set_to", "_has_compatible_shallow_copy_type"}:
        return make_tensor_values(dtype, device, shape=(2, 3, 4), domain="mixed", offset=0.5 if name in {"other", "from", "tensor"} else 0.0)
    if base_name == "result_type" and name in {"other", "tensor"}:
        return make_tensor_values(dtype, device, shape=(2, 3, 4), domain="mixed", offset=0.5)
    return make_tensor_values(dtype, device, shape=(2, 3, 4), domain="mixed", noncontiguous=base_name == "_debug_has_internal_overlap")


def metadata_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    arg_type = arg.get("type", "")
    base_name = entry["base_name"]
    if arg.get("tensor"):
        return metadata_tensor_arg_value(entry, arg, dtype, device)
    if name in {"dim", "batch_dim", "level"}:
        return 1
    if name == "memory_format":
        return torch.contiguous_format
    if name == "other":
        return 1.25 if dtype.is_floating_point or dtype.is_complex else 2
    if name == "scalar":
        return 1.25 if dtype.is_floating_point or dtype.is_complex else 2
    if name == "size":
        return [2, 3, 4]
    if name == "stride":
        return [12, 4, 1]
    if name == "dtype":
        return dtype
    if name == "device":
        return torch.device(device)
    if name == "layout":
        return torch.strided
    if arg_type == "bool":
        return False
    if arg_type == "str":
        return "TorchCTS metadata assertion sample"
    return None


def metadata_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        value = metadata_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_metadata",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


def padding_rank(base_name: str) -> int:
    if "3d" in base_name:
        return 3
    if "2d" in base_name or base_name in {"pad", "_pad_circular", "_pad_enum", "constant_pad_nd"}:
        return 2
    return 1


def padding_input_shape(base_name: str) -> tuple[int, ...]:
    rank = padding_rank(base_name)
    if rank == 3:
        return (2, 3, 4, 5, 6)
    if rank == 2:
        return (2, 3, 4, 5)
    return (2, 3, 5)


def padding_values(base_name: str) -> list[int]:
    rank = padding_rank(base_name)
    return [1, 1] * rank


def padding_arg_value(entry: dict, arg: dict, dtype: torch.dtype, device: str = "cpu"):
    name = arg.get("name", "")
    base_name = entry["base_name"]
    if arg.get("tensor"):
        return make_tensor_values(dtype, device, shape=padding_input_shape(base_name), domain="mixed")
    if name in {"pad", "padding"}:
        return padding_values(base_name)
    if name == "mode":
        if arg.get("type") == "int":
            return 0
        return "constant"
    if name == "value":
        return 0.25 if dtype.is_floating_point or dtype.is_complex else 1
    return None


def padding_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    args = []
    kwargs = {}
    for arg in entry.get("args", []):
        if arg.get("is_out"):
            continue
        value = padding_arg_value(entry, arg, dtype, device)
        if arg.get("kwarg_only"):
            kwargs[arg.get("name")] = value
        else:
            args.append(value)
    sample = _sample_input(args[0], tuple(args[1:]), kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    strategy = entry.get("generated", {}).get("strategy") or {}
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_padding",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
    )


def foreach_domain(foreach_name: str) -> str:
    if foreach_name in {"acos", "asin", "atanh"}:
        return "unit"
    if foreach_name in {"lgamma", "log", "log10", "log1p", "log2", "pow", "reciprocal", "rsqrt", "sqrt"}:
        return "positive"
    if foreach_name in {"div"}:
        return "nonzero"
    return "mixed"


def foreach_sample(
    entry: dict,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    strategy = entry["generated"]["strategy"]
    foreach_name = strategy["foreach_name"]
    domain = foreach_domain(foreach_name)
    self_tensors = [
        make_tensor_values(dtype, device, offset=0.0, domain=domain),
        make_tensor_values(dtype, device, offset=0.125, domain=domain, noncontiguous=True),
    ]
    args = ()
    kwargs = {}

    if strategy["family"] in {"binary", "extrema"}:
        overload = strategy["overload"]
        other_tensors = [
            make_tensor_values(dtype, device, offset=0.5, domain="nonzero"),
            make_tensor_values(dtype, device, offset=0.75, domain="nonzero", noncontiguous=True),
        ]
        scalar = 2 if not (dtype.is_floating_point or dtype.is_complex) else 1.25
        if overload == "Scalar":
            args = (scalar,)
        elif overload == "ScalarList":
            args = ([scalar, scalar + 1],)
        elif overload == "Tensor":
            args = (make_scalar_tensor(dtype, scalar, device=device),)
        elif overload == "List":
            args = (other_tensors,)
        if foreach_name in {"add", "sub"} and overload in {"List", "Tensor"}:
            kwargs["alpha"] = 1

    elif strategy["family"] == "copy":
        src_tensors = [
            make_tensor_values(dtype, device, offset=0.5, domain=domain),
            make_tensor_values(dtype, device, offset=0.75, domain=domain, noncontiguous=True),
        ]
        args = (src_tensors, False)

    elif strategy["family"] == "norm":
        args = (2, None)

    elif strategy["family"] == "pow":
        overload = strategy["overload"]
        exponent_tensors = [
            torch.full_like(self_tensors[0], 2),
            torch.full_like(self_tensors[1], 3),
        ]
        scalar = 2 if not (dtype.is_floating_point or dtype.is_complex) else 2.0
        if overload == "Scalar":
            args = (scalar,)
        elif overload == "ScalarList":
            args = (make_scalar_list(dtype, [2, 3]),)
        elif overload == "List":
            args = (exponent_tensors,)
        elif overload == "ScalarAndTensor":
            self_tensors = scalar
            args = (exponent_tensors,)

    elif strategy["family"] == "ternary":
        overload = strategy["overload"]
        tensor1 = [
            make_tensor_values(dtype, device, offset=0.5, domain="mixed"),
            make_tensor_values(dtype, device, offset=0.75, domain="mixed", noncontiguous=True),
        ]
        tensor2_domain = "nonzero" if foreach_name == "addcdiv" else "mixed"
        tensor2 = [
            make_tensor_values(dtype, device, offset=1.0, domain=tensor2_domain),
            make_tensor_values(dtype, device, offset=1.25, domain=tensor2_domain, noncontiguous=True),
        ]
        scalar_values = [0.5, 0.75] if (dtype.is_floating_point or dtype.is_complex) else [1, 2]
        if overload == "Scalar":
            args = (tensor1, tensor2, scalar_values[0])
        elif overload == "ScalarList":
            args = (tensor1, tensor2, make_scalar_list(dtype, scalar_values))
        elif overload == "Tensor":
            args = (tensor1, tensor2, make_packed_scalars(dtype, scalar_values, device=device))

    elif strategy["family"] == "lerp":
        overload = strategy["overload"]
        end_tensors = [
            make_tensor_values(dtype, device, offset=0.5, domain="mixed"),
            make_tensor_values(dtype, device, offset=0.75, domain="mixed", noncontiguous=True),
        ]
        scalar_values = [0.25, 0.75] if (dtype.is_floating_point or dtype.is_complex) else [0, 1]
        if overload == "Scalar":
            args = (end_tensors, scalar_values[0])
        elif overload == "ScalarList":
            args = (end_tensors, make_scalar_list(dtype, scalar_values))
        elif overload == "List":
            weight_tensors = [
                torch.full_like(self_tensors[0], 0.25),
                torch.full_like(self_tensors[1], 0.75),
            ]
            args = (end_tensors, weight_tensors)

    sample = _sample_input(self_tensors, args=args, kwargs=kwargs)
    prepared = prepare_sample(
        sample,
        input_condition,
        ieee754_seed=seed,
        sample_index=sample_index,
        op_name=entry.get("base_name") or entry.get("name"),
    )
    return _wrap_prepared_sample(
        entry=entry,
        strategy_name="manual_foreach",
        family=strategy.get("family"),
        dtype=dtype,
        device=device,
        input_condition=input_condition,
        prepared=prepared,
        sample_index=sample_index,
        metadata={"foreach_name": foreach_name, "overload": strategy.get("overload")},
    )


def opinfo_sample(
    entry: dict,
    opinfo_name: str,
    dtype: torch.dtype,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
) -> GeneratedSample:
    for index, raw_sample in enumerate(get_op_sample_inputs(opinfo_name, device, dtype)):
        if index != sample_index:
            continue
        prepared = prepare_sample(
            raw_sample,
            input_condition,
            ieee754_seed=seed,
            sample_index=sample_index,
            op_name=opinfo_name,
        )
        strategy = entry.get("generated", {}).get("strategy") or {}
        return _wrap_prepared_sample(
            entry=entry,
            strategy_name=strategy.get("strategy", "opinfo"),
            family=opinfo_name,
            dtype=dtype,
            device=device,
            input_condition=input_condition,
            prepared=prepared,
            sample_index=sample_index,
            metadata={"opinfo_name": opinfo_name},
        )
    raise SampleGenerationError(f"No OpInfo sample {sample_index} for {opinfo_name} and {dtype}")


def sample_for_entry(
    entry: dict,
    dtype: torch.dtype,
    *,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    strategy: dict | None = None,
) -> GeneratedSample:
    """Generate a sample for a coverage/generated-case entry."""

    strategy = strategy or (entry.get("generated", {}) or {}).get("strategy") or {}
    strategy_name = strategy.get("strategy")
    if strategy_name == "manual_bitwise":
        if input_condition != InputCondition.CLEAN:
            raise UnsupportedSampleStrategy("manual_bitwise does not generate NaN/Inf input tiers")
        return bitwise_sample(entry, dtype, device=device)
    if strategy_name == "manual_factory":
        if input_condition != InputCondition.CLEAN:
            raise UnsupportedSampleStrategy("manual_factory does not generate NaN/Inf input tiers")
        return factory_sample(entry, dtype, device=device)
    if strategy_name == "manual_factory_out":
        if input_condition != InputCondition.CLEAN:
            raise UnsupportedSampleStrategy("manual_factory_out does not generate NaN/Inf input tiers")
        return factory_out_sample(entry, dtype, device=device)
    if strategy_name == "manual_fft":
        if input_condition != InputCondition.CLEAN:
            raise UnsupportedSampleStrategy("manual_fft does not generate NaN/Inf input tiers")
        return fft_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_special_math":
        return special_math_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_elementwise":
        return elementwise_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_reduction":
        return reduction_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_indexing":
        if input_condition != InputCondition.CLEAN:
            raise UnsupportedSampleStrategy("manual_indexing does not generate NaN/Inf input tiers")
        return indexing_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_rng":
        if input_condition != InputCondition.CLEAN:
            raise UnsupportedSampleStrategy("manual_rng does not generate NaN/Inf input tiers")
        return rng_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_rnn_cell":
        if input_condition != InputCondition.CLEAN:
            raise UnsupportedSampleStrategy("manual_rnn_cell does not generate NaN/Inf input tiers")
        return rnn_cell_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_multi_output_reduction":
        return multi_output_reduction_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_upsample":
        return upsample_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_pooling":
        return pooling_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_convolution":
        return convolution_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_grid":
        if input_condition != InputCondition.CLEAN:
            raise UnsupportedSampleStrategy("manual_grid does not generate NaN/Inf input tiers")
        return grid_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_grid_backward":
        if input_condition != InputCondition.CLEAN:
            raise UnsupportedSampleStrategy("manual_grid_backward does not generate NaN/Inf input tiers")
        return grid_backward_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_loss":
        return loss_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_linalg":
        return linalg_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_metadata":
        if input_condition != InputCondition.CLEAN:
            raise UnsupportedSampleStrategy("manual_metadata does not generate NaN/Inf input tiers")
        return metadata_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_padding":
        return padding_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_foreach":
        return foreach_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name == "manual_matmul":
        return _metadata_inputs_for_entry(
            entry,
            dtype,
            device=device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
        ).as_generated_sample()
    if strategy_name == "manual_shape":
        return shape_sample(entry, dtype, device, input_condition, seed, sample_index)
    if strategy_name in {"opinfo_out", "opinfo_inplace_unary", "opinfo_view_alias"}:
        return opinfo_sample(
            entry,
            strategy["opinfo_name"],
            dtype,
            device=device,
            input_condition=input_condition,
            seed=seed,
            sample_index=sample_index,
        )
    raise UnsupportedSampleStrategy(f"No TorchCTS sample generator for strategy {strategy_name!r}")


def iter_samples_for_entry(
    entry: dict,
    *,
    manifest: dict | None = None,
    dtypes: Iterable[torch.dtype] | None = None,
    device: str = "cpu",
    seed: int = DEFAULT_IEEE754_SEED,
    max_opinfo_samples: int = 1,
) -> Iterator[GeneratedSample]:
    """Yield generated samples for an entry across dtypes and input tiers."""

    if dtypes is None:
        dtypes = [dtype for dtype, _dtype_str in manifest_dtype_items(manifest or {})]
    strategy = (entry.get("generated", {}) or {}).get("strategy") or {}
    for dtype in dtypes:
        for input_condition in input_conditions_for(manifest, entry.get("base_name", entry["name"]), dtype):
            sample_count = max(1, max_opinfo_samples) if str(strategy.get("strategy", "")).startswith("opinfo_") else 1
            for sample_index in range(sample_count):
                try:
                    yield sample_for_entry(
                        entry,
                        dtype,
                        device=device,
                        input_condition=input_condition,
                        seed=seed,
                        sample_index=sample_index,
                        strategy=strategy,
                    )
                except UnsupportedSampleStrategy:
                    raise
                except SampleGenerationError:
                    if sample_index == 0:
                        raise
                    break


def dispatcher_entry(dispatcher_name: str, audit: dict | None = None) -> dict:
    """Find a dispatcher entry from an audit, live schema, or generated manifest."""

    if audit is not None:
        for entry in audit.get("entries", []):
            if entry.get("name") == dispatcher_name:
                return entry

    try:
        return _live_dispatcher_entry(dispatcher_name)
    except Exception:
        pass

    from torchcts.core.coverage import SURFACE_KINDS, generated_entries_for

    for surface_kind in sorted(SURFACE_KINDS):
        for entry in generated_entries_for(surface_kind):
            if entry and entry.get("name") == dispatcher_name:
                return entry

    raise SampleGenerationError(f"Unknown dispatcher surface {dispatcher_name!r}")


def sample_for_dispatcher(
    dispatcher_name: str,
    dtype: torch.dtype,
    *,
    device: str = "cpu",
    input_condition: str = InputCondition.CLEAN,
    seed: int = DEFAULT_IEEE754_SEED,
    sample_index: int = 0,
    audit: dict | None = None,
) -> GeneratedSample:
    """Generate a sample for a dispatcher name using TorchCTS coverage metadata."""

    entry = dispatcher_entry(dispatcher_name, audit=audit)
    return sample_for_entry(
        entry,
        dtype,
        device=device,
        input_condition=input_condition,
        seed=seed,
        sample_index=sample_index,
    )


__all__ = [
    "DEFAULT_IEEE754_SEED",
    "DEFAULT_SAMPLE_SHAPE",
    "DistributionSpec",
    "GeneratedSample",
    "GeneratedOpInputs",
    "GeneratedParam",
    "GeneratedCaseSpec",
    "InputCondition",
    "REAL_WORLD_DISTRIBUTIONS",
    "SUPPORTED_SAMPLE_STRATEGIES",
    "SampleGenerationError",
    "UnsupportedSampleStrategy",
    "bitwise_args_and_template",
    "bitwise_dtype_supported",
    "bitwise_sample",
    "dispatcher_entry",
    "elementwise_domain",
    "elementwise_sample",
    "elementwise_scalar_value",
    "elementwise_tensor_values",
    "factory_args",
    "factory_dtype_supported",
    "factory_out_args",
    "factory_out_call_parts",
    "factory_out_shape",
    "factory_out_sample",
    "factory_sample",
    "fft_arg_value",
    "fft_complex_dtype",
    "fft_input_shape",
    "fft_input_tensor",
    "fft_real_dtype",
    "fft_sample",
    "foreach_domain",
    "foreach_sample",
    "get_all_inputs_for_op",
    "get_inputs_for_op",
    "grid_backward_sample",
    "grid_backward_sample_arg_value",
    "grid_sample",
    "grid_sample_arg_value",
    "ieee754_enabled",
    "input_conditions_for",
    "indexing_arg_value",
    "indexing_index_tensor",
    "indexing_sample",
    "indexing_scalar_tensor",
    "indexing_scalar_value",
    "indexing_tensor_arg_value",
    "iter_inputs_for_op",
    "iter_samples_for_entry",
    "loss_arg_value",
    "loss_float_tensor",
    "loss_sample",
    "linalg_arg_value",
    "linalg_matrix",
    "linalg_sample",
    "linalg_tensor_arg_value",
    "linalg_vector_tensor",
    "metadata_arg_value",
    "metadata_sample",
    "metadata_tensor_arg_value",
    "make_activation_tensor",
    "make_addbmm_inputs",
    "make_addbmm_sample",
    "make_addmm_inputs",
    "make_addmm_sample",
    "make_baddbmm_inputs",
    "make_baddbmm_sample",
    "make_binary_inputs",
    "make_binary_sample",
    "make_bmm_inputs",
    "make_bmm_sample",
    "make_chain_matmul_inputs",
    "make_chain_matmul_sample",
    "make_conv2d_inputs",
    "make_conv2d_sample",
    "make_distribution_tensor",
    "make_gradient_tensor",
    "make_linear_inputs",
    "make_linear_sample",
    "make_matmul_inputs",
    "make_matmul_sample",
    "make_mm_inputs",
    "make_mm_sample",
    "make_packed_scalars",
    "make_scalar_list",
    "make_scalar_tensor",
    "make_tensor_values",
    "make_weight_tensor",
    "manifest_dtype_items",
    "move_to_device",
    "convolution_arg_value",
    "convolution_bias_tensor",
    "convolution_group_count",
    "convolution_input_tensor",
    "convolution_rank",
    "convolution_sample",
    "convolution_weight_tensor",
    "multi_output_reduction_arg_value",
    "multi_output_reduction_call_parts",
    "multi_output_reduction_other_tensor",
    "multi_output_reduction_sample",
    "multi_output_reduction_tensor",
    "opinfo_sample",
    "padding_arg_value",
    "padding_input_shape",
    "padding_rank",
    "padding_sample",
    "padding_values",
    "pooling_arg_value",
    "pooling_input_shape",
    "pooling_sample",
    "pooling_spatial_rank",
    "reduction_arg_value",
    "reduction_sample",
    "rng_arg_value",
    "rng_call_parts",
    "rng_generator",
    "rng_output_shape",
    "rng_sample",
    "rng_tensor_arg_value",
    "rnn_cell_arg_value",
    "rnn_cell_gate_count",
    "rnn_cell_sample",
    "sample_for_dispatcher",
    "sample_for_entry",
    "sample_case_depth_for_entry",
    "sample_case_specs_for_entry",
    "sample_case_specs_for_op",
    "shape_args_for_entry",
    "shape_sample",
    "special_math_domain",
    "special_math_sample",
    "special_scalar_value",
    "special_tensor_values",
    "upsample_arg_value",
    "upsample_input_shape",
    "upsample_output_size",
    "upsample_sample",
    "upsample_spatial_rank",
]
