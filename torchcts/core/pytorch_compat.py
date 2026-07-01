# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

from __future__ import annotations

import re
from typing import Any

import torch

from torchcts.core.version_rules import parse_torch_version


_VERSION_PREFIX_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")


def normalize_torch_version(version: str | None = None) -> str | None:
    """Return major.minor.patch for a PyTorch version string."""

    text = str(version or torch.__version__)
    match = _VERSION_PREFIX_RE.match(text)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return f"{int(major)}.{int(minor)}.{int(patch) if patch is not None else 0}"


def next_patch_upper_bound(version: str | None) -> str | None:
    parts = parse_torch_version(version)
    if parts is None:
        return None
    major, minor, patch = parts
    return f"{major}.{minor}.{patch + 1}"


def collected_versions_from_metadata(metadata: dict[str, Any] | None) -> tuple[str, ...]:
    versions = []
    for version in (metadata or {}).get("collected_versions") or ():
        normalized = normalize_torch_version(str(version))
        if normalized is not None:
            versions.append(normalized)
    return tuple(sorted(set(versions), key=lambda item: parse_torch_version(item) or (0, 0, 0)))


def validated_version_status(metadata: dict[str, Any] | None, runtime_version: str | None = None) -> dict[str, Any]:
    normalized = normalize_torch_version(runtime_version)
    versions = collected_versions_from_metadata(metadata)
    minimum = versions[0] if versions else None
    maximum = versions[-1] if versions else None
    validated = normalized in versions if normalized is not None else False
    newer_than_matrix = False
    older_than_matrix = False
    if normalized is not None and maximum is not None:
        runtime_parts = parse_torch_version(normalized)
        max_parts = parse_torch_version(maximum)
        min_parts = parse_torch_version(minimum)
        newer_than_matrix = runtime_parts is not None and max_parts is not None and runtime_parts > max_parts
        older_than_matrix = runtime_parts is not None and min_parts is not None and runtime_parts < min_parts
    return {
        "runtime_version": str(runtime_version or torch.__version__),
        "normalized_runtime_version": normalized,
        "validated": validated,
        "newer_than_matrix": newer_than_matrix,
        "older_than_matrix": older_than_matrix,
        "min_validated_version": minimum,
        "max_validated_version": maximum,
        "collected_versions": list(versions),
    }


def is_runtime_version_validated(metadata: dict[str, Any] | None, runtime_version: str | None = None) -> bool:
    return bool(validated_version_status(metadata, runtime_version).get("validated"))


def unvalidated_version_message(metadata: dict[str, Any] | None, runtime_version: str | None = None) -> str:
    status = validated_version_status(metadata, runtime_version)
    normalized = status.get("normalized_runtime_version") or str(runtime_version or torch.__version__)
    minimum = status.get("min_validated_version")
    maximum = status.get("max_validated_version")
    if minimum and maximum:
        return (
            f"PyTorch {normalized} is not in the TorchCTS validated PyTorch matrix "
            f"({minimum} through {maximum})."
        )
    return f"PyTorch {normalized} is not in the TorchCTS validated PyTorch matrix."


__all__ = [
    "collected_versions_from_metadata",
    "is_runtime_version_validated",
    "next_patch_upper_bound",
    "normalize_torch_version",
    "unvalidated_version_message",
    "validated_version_status",
]
