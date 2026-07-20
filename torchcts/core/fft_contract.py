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

"""Shape-aware nonfinite contribution contracts for FFT transforms."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


_ONE_DIMENSIONAL = frozenset({"fft", "ifft", "rfft", "irfft", "hfft", "ihfft"})
_TWO_DIMENSIONAL = frozenset({"fft2", "ifft2", "rfft2", "irfft2", "hfft2", "ihfft2"})
_C2C = frozenset({"fft", "ifft", "fft2", "ifft2", "fftn", "ifftn"})
_R2C = frozenset({"rfft", "rfft2", "rfftn"})
_C2R = frozenset({"irfft", "irfft2", "irfftn", "hfft", "hfft2", "hfftn"})
_R2H = frozenset({"ihfft", "ihfft2", "ihfftn"})


@dataclass(frozen=True)
class FFTContractSpec:
    family: str
    dimensions: tuple[int, ...]
    full_lengths: tuple[int, ...]
    retained_source_lengths: tuple[int, ...]
    kind: str
    half_spectrum_dimension: int | None = None


def _family(op_name: str) -> str:
    name = str(op_name).lower().removeprefix("torch.")
    if name.startswith("fft."):
        name = name.split(".")[-1]
    return name


def _dimensions(family: str, source: torch.Tensor, args: tuple, kwargs: dict) -> tuple[int, ...]:
    raw_dimensions = kwargs.get("dim", args[1] if len(args) > 1 else None)
    raw_sizes = kwargs.get("s", args[0] if args else None)
    if raw_dimensions is None:
        if family in _ONE_DIMENSIONAL:
            raw_dimensions = (-1,)
        elif family in _TWO_DIMENSIONAL:
            raw_dimensions = (-2, -1)
        elif raw_sizes is None:
            raw_dimensions = tuple(range(source.ndim))
        else:
            raw_dimensions = tuple(range(source.ndim - len(raw_sizes), source.ndim))
    elif isinstance(raw_dimensions, int):
        raw_dimensions = (raw_dimensions,)
    else:
        raw_dimensions = tuple(raw_dimensions)

    dimensions = tuple(int(dimension) % source.ndim for dimension in raw_dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("FFT dimensions must be nonempty and unique")
    return dimensions


def _requested_lengths(
    family: str,
    dimensions: tuple[int, ...],
    args: tuple,
    kwargs: dict,
) -> tuple[int | None, ...]:
    if family in _ONE_DIMENSIONAL:
        value = kwargs.get("n", args[0] if args else None)
        return (None if value is None else int(value),)
    values = kwargs.get("s", args[0] if args else None)
    if values is None:
        return (None,) * len(dimensions)
    values = tuple(int(value) for value in values)
    if len(values) != len(dimensions):
        raise ValueError("FFT size and dimension lists must have equal lengths")
    return values


def public_fft_contract_spec(
    op_name: str,
    source: torch.Tensor,
    args: tuple = (),
    kwargs: dict | None = None,
) -> FFTContractSpec:
    family = _family(op_name)
    if family in _C2C:
        kind = "c2c"
    elif family in _R2C:
        kind = "r2c"
    elif family in _C2R:
        kind = "c2r"
    elif family in _R2H:
        kind = "r2h"
    else:
        raise ValueError(f"unsupported FFT contract family {family!r}")

    kwargs = dict(kwargs or {})
    dimensions = _dimensions(family, source, args, kwargs)
    requested = _requested_lengths(family, dimensions, args, kwargs)
    full_lengths: list[int] = []
    retained_lengths: list[int] = []
    for index, (dimension, requested_length) in enumerate(zip(dimensions, requested)):
        input_length = int(source.shape[dimension])
        is_half_dimension = kind == "c2r" and index == len(dimensions) - 1
        if requested_length is None or requested_length == -1:
            full_length = 2 * (input_length - 1) if is_half_dimension else input_length
        else:
            full_length = requested_length
        if full_length <= 0:
            raise ValueError("FFT transform lengths must be positive")
        retained_limit = full_length // 2 + 1 if is_half_dimension else full_length
        full_lengths.append(full_length)
        retained_lengths.append(min(input_length, retained_limit))

    half_dimension = dimensions[-1] if kind in {"r2c", "c2r", "r2h"} else None
    return FFTContractSpec(
        family=family,
        dimensions=dimensions,
        full_lengths=tuple(full_lengths),
        retained_source_lengths=tuple(retained_lengths),
        kind=kind,
        half_spectrum_dimension=half_dimension,
    )


def generated_c2c_fft_contract_spec(
    source: torch.Tensor,
    dimensions,
) -> FFTContractSpec:
    normalized = tuple(int(dimension) % source.ndim for dimension in dimensions)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("generated c2c FFT dimensions must be nonempty and unique")
    lengths = tuple(int(source.shape[dimension]) for dimension in normalized)
    return FFTContractSpec(
        family="_fft_c2c",
        dimensions=normalized,
        full_lengths=lengths,
        retained_source_lengths=lengths,
        kind="c2c",
    )


def fft_source_contributor_mask(
    source: torch.Tensor,
    spec: FFTContractSpec,
) -> torch.Tensor:
    lane_count = 2 if source.is_complex() else 1
    mask = torch.ones((*source.shape, lane_count), dtype=torch.bool)
    for dimension, retained_length in zip(spec.dimensions, spec.retained_source_lengths):
        coordinate = torch.arange(source.shape[dimension])
        shape = [1] * source.ndim
        shape[dimension] = source.shape[dimension]
        retained = (coordinate < retained_length).reshape(shape)
        mask &= retained.unsqueeze(-1)

    if spec.kind == "c2r" and source.is_complex():
        self_conjugate = torch.ones(source.shape, dtype=torch.bool)
        for dimension, full_length in zip(spec.dimensions, spec.full_lengths):
            coordinate = torch.arange(source.shape[dimension])
            shape = [1] * source.ndim
            shape[dimension] = source.shape[dimension]
            coordinate = coordinate.reshape(shape)
            on_boundary = coordinate == 0
            if full_length % 2 == 0:
                on_boundary |= coordinate == full_length // 2
            self_conjugate &= on_boundary
        mask[..., 1] &= ~self_conjugate
    return mask


def _values_finite(tensor: torch.Tensor) -> bool:
    lanes = torch.view_as_real(tensor) if tensor.is_complex() else tensor
    return bool(torch.isfinite(lanes.detach().cpu()).all())


def compare_fft_nonfinite_groups(
    actual: torch.Tensor,
    expected: torch.Tensor,
    source: torch.Tensor,
    spec: FFTContractSpec,
    *,
    dtype: torch.dtype,
    compare,
    label: str,
) -> None:
    batch_dimensions = tuple(
        dimension for dimension in range(source.ndim) if dimension not in spec.dimensions
    )
    group_count = math.prod(source.shape[dimension] for dimension in batch_dimensions)
    contributor_mask = fft_source_contributor_mask(source.detach().cpu(), spec)
    source_lanes = (
        torch.view_as_real(source.detach().cpu())
        if source.is_complex()
        else source.detach().cpu().unsqueeze(-1)
    )
    lane_dimension = source.ndim
    source_rows = source_lanes.permute(
        *batch_dimensions, *spec.dimensions, lane_dimension
    ).reshape(group_count, -1)
    contributor_rows = contributor_mask.permute(
        *batch_dimensions, *spec.dimensions, lane_dimension
    ).reshape(group_count, -1)

    actual_rows = actual.permute(*batch_dimensions, *spec.dimensions).reshape(group_count, -1)
    expected_rows = expected.permute(*batch_dimensions, *spec.dimensions).reshape(group_count, -1)
    if actual_rows.shape != expected_rows.shape or actual_rows.shape[0] != source_rows.shape[0]:
        raise AssertionError(f"{label} FFT transform group structure is inconsistent")

    for row_index in range(group_count):
        contributing_nonfinite = bool(
            ((~torch.isfinite(source_rows[row_index])) & contributor_rows[row_index]).any()
        )
        if contributing_nonfinite:
            if _values_finite(actual_rows[row_index]):
                raise AssertionError(
                    f"{label} transform group {row_index} dropped every contributing nonfinite lane"
                )
        else:
            compare(
                actual_rows[row_index],
                expected_rows[row_index].to(actual.device),
                category="fft",
                dtype=dtype,
            )
