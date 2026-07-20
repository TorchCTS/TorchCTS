# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

"""Relationship-preserving transport for structured OpInfo samples."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any

import torch


class SampleTransportError(RuntimeError):
    """Raised when target placement cannot preserve a sample's tensor graph."""


@dataclass(frozen=True)
class _TensorDescriptor:
    path: str
    object_key: int
    storage_key: tuple[Any, ...] | None
    layout: torch.layout
    size: tuple[int, ...]
    stride: tuple[int, ...] | None
    storage_offset: int | None
    overlap: int | None
    is_conj: bool
    is_neg: bool
    requires_grad: bool
    device_type: str


def _storage_key(tensor: torch.Tensor) -> tuple[Any, ...] | None:
    if tensor.layout != torch.strided:
        return None
    try:
        storage = tensor.untyped_storage()
        return (tensor.device.type, storage._cdata)
    except Exception:
        return None


def _overlap(tensor: torch.Tensor) -> int | None:
    if tensor.layout != torch.strided:
        return None
    # Error samples are deliberately small.  Derive their overlap from shape
    # and stride so validation does not depend on backend-specific support for
    # torch._debug_has_internal_overlap (PrivateUse1 may report a stride-zero
    # view as non-overlapping even though its metadata is preserved).
    if tensor.numel() <= 4096:
        offsets: set[int] = set()
        ranges = (range(int(size)) for size in tensor.shape)
        for index in itertools.product(*ranges):
            offset = sum(component * stride for component, stride in zip(index, tensor.stride()))
            if offset in offsets:
                return 1
            offsets.add(offset)
        return 0
    try:
        return int(torch._debug_has_internal_overlap(tensor))
    except Exception:
        return None


def _walk_tensors(value, path: str, result: list[_TensorDescriptor]) -> None:
    if isinstance(value, torch.Tensor):
        result.append(
            _TensorDescriptor(
                path=path,
                object_key=id(value),
                storage_key=_storage_key(value),
                layout=value.layout,
                size=tuple(value.shape),
                stride=tuple(value.stride()) if value.layout == torch.strided else None,
                storage_offset=value.storage_offset() if value.layout == torch.strided else None,
                overlap=_overlap(value),
                is_conj=value.is_conj(),
                is_neg=value.is_neg(),
                requires_grad=value.requires_grad,
                device_type=value.device.type,
            )
        )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_tensors(item, f"{path}[{index}]", result)
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _walk_tensors(item, f"{path}[{index}]", result)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_tensors(item, f"{path}[{key!r}]", result)


def _relation_descriptor(input_value, args, kwargs) -> list[_TensorDescriptor]:
    result: list[_TensorDescriptor] = []
    _walk_tensors(input_value, "input", result)
    _walk_tensors(args, "args", result)
    _walk_tensors(kwargs, "kwargs", result)
    return result


def _same_equivalence_relation(
    before: list[_TensorDescriptor],
    after: list[_TensorDescriptor],
    key: str,
) -> bool:
    before_by_path = {item.path: item for item in before}
    after_by_path = {item.path: item for item in after}
    paths = sorted(before_by_path)
    for left_index, left_path in enumerate(paths):
        for right_path in paths[left_index:]:
            before_left = getattr(before_by_path[left_path], key)
            before_right = getattr(before_by_path[right_path], key)
            after_left = getattr(after_by_path[left_path], key)
            after_right = getattr(after_by_path[right_path], key)
            if (before_left == before_right) != (after_left == after_right):
                return False
    return True


def _validate_relations(
    before: list[_TensorDescriptor],
    after: list[_TensorDescriptor],
) -> None:
    before_by_path = {item.path: item for item in before}
    after_by_path = {item.path: item for item in after}
    if set(before_by_path) != set(after_by_path):
        raise SampleTransportError("tensor paths changed during sample transport")

    for path, source in before_by_path.items():
        target = after_by_path[path]
        metadata = (
            "layout",
            "size",
            "stride",
            "storage_offset",
            "overlap",
            "is_conj",
            "is_neg",
            "requires_grad",
        )
        for attribute in metadata:
            if getattr(source, attribute) != getattr(target, attribute):
                raise SampleTransportError(
                    f"{path} changed {attribute} during sample transport: "
                    f"{getattr(source, attribute)!r} -> {getattr(target, attribute)!r}"
                )
        if source.device_type == "meta" and target.device_type != "meta":
            raise SampleTransportError(f"{path} meta tensor was materialized during sample transport")

    if not _same_equivalence_relation(before, after, "object_key"):
        raise SampleTransportError("repeated tensor object identity changed during sample transport")
    if not _same_equivalence_relation(before, after, "storage_key"):
        raise SampleTransportError("tensor storage alias relationships changed during sample transport")


class _GraphMover:
    def __init__(self, device):
        self.device = torch.device(device)
        self.tensor_memo: dict[int, torch.Tensor] = {}
        self.storage_memo: dict[tuple[Any, ...], tuple[torch.dtype, torch.Tensor]] = {}

    def move(self, value):
        if isinstance(value, torch.Tensor):
            return self._move_tensor(value)
        if isinstance(value, list):
            return [self.move(item) for item in value]
        if isinstance(value, tuple):
            items = [self.move(item) for item in value]
            if type(value) is tuple:
                return tuple(items)
            try:
                return type(value)(*items)
            except TypeError:
                return type(value)(items)
        if isinstance(value, dict):
            return type(value)((key, self.move(item)) for key, item in value.items())
        return value

    def _move_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        memoized = self.tensor_memo.get(id(tensor))
        if memoized is not None:
            return memoized

        if tensor.device.type == "meta":
            moved = tensor
        elif tensor.layout == torch.strided:
            moved = self._move_strided(tensor)
        else:
            moved = tensor.to(self.device)

        if tensor.requires_grad and not moved.requires_grad:
            moved.requires_grad_(True)

        self.tensor_memo[id(tensor)] = moved
        return moved

    def _move_strided(self, tensor: torch.Tensor) -> torch.Tensor:
        key = _storage_key(tensor)
        if key is None:
            raise SampleTransportError("strided tensor has no transportable storage identity")
        storage = tensor.untyped_storage()
        storage_elements, remainder = divmod(storage.nbytes(), tensor.element_size())
        if remainder:
            raise SampleTransportError(
                f"storage size {storage.nbytes()} is not divisible by dtype size {tensor.element_size()}"
            )

        memoized = self.storage_memo.get(key)
        if memoized is None:
            source_storage = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
            source_storage = source_storage.set_(storage, 0, (storage_elements,), (1,))
            target_storage = source_storage.to(self.device)
            self.storage_memo[key] = (tensor.dtype, target_storage)
        else:
            storage_dtype, target_storage = memoized
            if storage_dtype != tensor.dtype:
                raise SampleTransportError(
                    "aliases with different dtypes cannot yet be transported without changing storage identity"
                )

        moved = target_storage.as_strided(
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
        )
        if tensor.is_conj():
            moved = moved.conj()
        if tensor.is_neg():
            moved = torch._neg_view(moved)
        return moved


def move_sample_preserving_relations(input_value, args, kwargs, device):
    """Move a complete sample while preserving tensor graph relationships."""

    before = _relation_descriptor(input_value, args, kwargs)
    mover = _GraphMover(device)
    moved_input = mover.move(input_value)
    moved_args = mover.move(args)
    moved_kwargs = mover.move(kwargs)
    after = _relation_descriptor(moved_input, moved_args, moved_kwargs)
    _validate_relations(before, after)
    return moved_input, moved_args, moved_kwargs


__all__ = ["SampleTransportError", "move_sample_preserving_relations"]
