# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

from __future__ import annotations

from functools import lru_cache
import json
from importlib import resources
from typing import Any

import torch

from torchcts.core.pytorch_compat import (
    collected_versions_from_metadata,
    is_runtime_version_validated,
    normalize_torch_version,
)
from torchcts.core.version_rules import parse_torch_version


@lru_cache(maxsize=1)
def load_op_metadata() -> dict:
    """Load TorchCTS-owned generic PyTorch op metadata."""

    text = resources.files("torchcts").joinpath("op_metadata.json").read_text(encoding="utf-8")
    return json.loads(text)


def _normalize_dispatcher_name(dispatcher_name: str) -> str:
    return dispatcher_name if dispatcher_name.startswith("aten::") else f"aten::{dispatcher_name}"


def get_op_metadata(dispatcher_name: str) -> dict:
    """Return metadata for an exact dispatcher name, if present."""

    name = _normalize_dispatcher_name(dispatcher_name)
    data = load_op_metadata()
    entry = dict((data.get("ops") or {}).get(name, {}))
    if not entry or not _metadata_is_v2(data) or entry.get("legacy_static_only"):
        return entry

    record = schema_record_for_runtime(name, metadata=data)
    if record is None:
        return entry
    entry.setdefault("schema", record.get("schema") or "")
    entry.setdefault("signature", record.get("schema") or "")
    entry.setdefault("args", list(record.get("args") or []))
    entry.setdefault("returns", list(record.get("returns") or []))
    entry.setdefault("base_name", record.get("base_name") or name.removeprefix("aten::").split(".", 1)[0])
    entry.setdefault("base_op", entry.get("base_name"))
    entry.setdefault("overload", record.get("overload") or "")
    entry.setdefault("surface_kind", record.get("surface_kind"))
    entry.setdefault("variant_kind", record.get("variant_kind"))
    entry.setdefault("variant", entry.get("variant_kind"))
    return entry


def _metadata_entry(dispatcher_name: str, metadata: dict | None = None) -> dict:
    name = _normalize_dispatcher_name(dispatcher_name)
    data = metadata if metadata is not None else load_op_metadata()
    entry = (data.get("ops") or {}).get(name, {})
    return entry if isinstance(entry, dict) else {}


def _version_text(runtime_version: str | None) -> str:
    return str(runtime_version or torch.__version__)


def _validation_metadata(metadata: dict | None) -> dict:
    data = metadata if metadata is not None else load_op_metadata()
    versions = set(collected_versions_from_metadata(data.get("metadata") or {}))
    for entry in (data.get("ops") or {}).values():
        if not isinstance(entry, dict):
            continue
        for version in entry.get("versions_seen") or ():
            normalized = normalize_torch_version(str(version))
            if normalized is not None:
                versions.add(normalized)
    return {"collected_versions": sorted(versions, key=lambda item: parse_torch_version(item) or (0, 0, 0))}


def _in_closed_open_range(runtime_version: str, min_version: str | None, max_version: str | None) -> bool:
    runtime = parse_torch_version(runtime_version)
    minimum = parse_torch_version(min_version)
    maximum = parse_torch_version(max_version)
    if runtime is None:
        return False
    if minimum is not None and runtime < minimum:
        return False
    if maximum is not None and runtime >= maximum:
        return False
    return True


def _metadata_is_v2(metadata: dict | None) -> bool:
    data = metadata if metadata is not None else load_op_metadata()
    return int(data.get("version", 1) or 1) >= 2


def _v1_schema_record(entry: dict) -> dict:
    return {
        "min": None,
        "max": None,
        "schema": entry.get("signature", ""),
        "args": list(entry.get("args") or []),
        "returns": list(entry.get("returns") or []),
        "surface_kind": entry.get("surface_kind"),
        "variant_kind": entry.get("variant") or entry.get("variant_kind"),
        "base_name": entry.get("base_op") or entry.get("base_name"),
        "overload": entry.get("overload") or "",
    }


def schema_record_for_runtime(
    dispatcher_name: str,
    runtime_version: str | None = None,
    *,
    metadata: dict | None = None,
) -> dict | None:
    """Return the schema-range record that applies to the requested runtime.

    Version-1 metadata had no historical range data; for that legacy shape,
    an existing metadata entry is treated as always available so current
    behavior is preserved until the tracked artifact is replaced.
    """

    entry = _metadata_entry(dispatcher_name, metadata)
    if not entry:
        return None
    if not _metadata_is_v2(metadata) or entry.get("legacy_static_only"):
        return _v1_schema_record(entry)

    version = normalize_torch_version(_version_text(runtime_version))
    if version is None or not is_runtime_version_validated(_validation_metadata(metadata), version):
        return None
    for record in entry.get("schema_ranges") or ():
        if not isinstance(record, dict):
            continue
        if _in_closed_open_range(version, record.get("min"), record.get("max")):
            return dict(record)
    return None


def op_available_in_runtime(
    dispatcher_name: str,
    runtime_version: str | None = None,
    *,
    metadata: dict | None = None,
) -> bool:
    """Return whether metadata says a dispatcher overload exists in runtime."""

    entry = _metadata_entry(dispatcher_name, metadata)
    if not entry:
        return False
    if not _metadata_is_v2(metadata) or entry.get("legacy_static_only"):
        return True
    return schema_record_for_runtime(dispatcher_name, runtime_version, metadata=metadata) is not None


def _nearest_schema_record(entry: dict, runtime_version: str) -> dict:
    ranges = [record for record in entry.get("schema_ranges") or () if isinstance(record, dict)]
    if not ranges:
        return _v1_schema_record(entry)

    runtime = parse_torch_version(runtime_version)
    if runtime is None:
        return dict(ranges[0])

    prior = [
        record for record in ranges
        if parse_torch_version(record.get("min")) is not None
        and parse_torch_version(record.get("min")) <= runtime
    ]
    if prior:
        return dict(prior[-1])
    return dict(ranges[0])


def _tensor_records(records: Any) -> list[dict]:
    return [dict(record) for record in records or () if isinstance(record, dict) and record.get("tensor")]


def runtime_unavailable_op_entries(
    *,
    metadata: dict | None = None,
    runtime_version: str | None = None,
    live_names: set[str] | frozenset[str] | None = None,
) -> list[dict]:
    """Synthesize audit entries for ops known to exist only in other versions."""

    data = metadata if metadata is not None else load_op_metadata()
    if not _metadata_is_v2(data):
        return []

    version = normalize_torch_version(_version_text(runtime_version))
    if version is None or not is_runtime_version_validated(_validation_metadata(data), version):
        return []
    live = set(live_names or ())
    entries = []
    for name, entry in sorted((data.get("ops") or {}).items()):
        if name in live or not isinstance(entry, dict):
            continue
        if entry.get("legacy_static_only"):
            continue
        if op_available_in_runtime(name, version, metadata=data):
            continue
        schema = _nearest_schema_record(entry, version)
        args = list(schema.get("args") or [])
        returns = list(schema.get("returns") or [])
        tensor_args = _tensor_records(args)
        tensor_returns = _tensor_records(returns)
        entries.append({
            "name": name,
            "base_name": schema.get("base_name") or name.removeprefix("aten::").split(".", 1)[0],
            "overload": schema.get("overload") or "",
            "schema": schema.get("schema") or "",
            "args": args,
            "returns": returns,
            "tensor_args": tensor_args,
            "tensor_returns": tensor_returns,
            "has_tensor_args": bool(tensor_args),
            "has_tensor_returns": bool(tensor_returns),
            "surface_kind": schema.get("surface_kind") or "functional_data",
            "variant_kind": schema.get("variant_kind") or "functional",
            "dispatch": {},
            "runtime_availability": {
                "status": "unavailable_in_pytorch_runtime",
                "runtime_version": version,
                "introduced": entry.get("introduced"),
                "removed": entry.get("removed"),
                "versions_seen": list(entry.get("versions_seen") or []),
                "versions_missing": list(entry.get("versions_missing") or []),
            },
        })
    return entries


__all__ = [
    "get_op_metadata",
    "load_op_metadata",
    "op_available_in_runtime",
    "runtime_unavailable_op_entries",
    "schema_record_for_runtime",
]
