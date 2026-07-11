# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

"""Canonical storage for backend-specific collection evidence."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


FORMAT = "backend_scoped_evidence"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
SOURCES_NAME = "sources.jsonl"
PROMOTION_REF_PREFIX = "backend-evidence:"

_BACKEND_ALIASES = {
    "cpu_build": "cpu-build",
    "cpu-build": "cpu-build",
    "privateuseone": "privateuse1",
    "privateuse1": "privateuse1",
}
_FORBIDDEN_BACKENDS = {"", "all", "any", "cpu", "quantized"}
_FORBIDDEN_KEYS = {
    "_source_path",
    "archive",
    "archive_name",
    "archive_path",
    "cwd",
    "device_uuid",
    "environment_variables",
    "home_directory",
    "host",
    "host_id",
    "hostname",
    "machine",
    "machine_name",
    "pack_name",
    "platform",
    "processor",
    "python_executable",
    "run_name",
    "staging_dir",
    "staging_path",
    "torch_config",
    "username",
    "working_directory",
}
_POSIX_ABSOLUTE_RE = re.compile(r"(?<![A-Za-z0-9_.:/-])/(?!/)(?:[^\s\"'<>|]+)")
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)(?:^|[\s\"'])(?:[A-Z]:[\\/]|\\\\)[^\s\"']+")
_FILE_URI_RE = re.compile(r"(?i)file://")
_ARCHIVE_RE = re.compile(r"torchcts-evidence-[^\s\"']+\.tar\.gz")
_SOURCE_ID_RE = re.compile(r"source-(\d{10})\Z")
_REGISTERED_AT_RE = re.compile(r"registered at (?P<path>[^\s\[]+)")
_TRACEBACK_FILE_RE = re.compile(r'File "(?P<path>[^"]+)"')


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def canonical_backend(value: str | None) -> str:
    backend = str(value or "").strip().lower().replace(" ", "-")
    backend = _BACKEND_ALIASES.get(backend, backend.replace("_", "-"))
    if backend in _FORBIDDEN_BACKENDS:
        raise ValueError(f"Backend evidence requires an explicit backend, got {value!r}")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", backend):
        raise ValueError(f"Invalid backend evidence key: {value!r}")
    return backend


def parse_promotion_reference(value: str) -> str | None:
    if not value.startswith(PROMOTION_REF_PREFIX):
        return None
    return canonical_backend(value[len(PROMOTION_REF_PREFIX) :])


def promotion_reference(backend: str) -> str:
    return f"{PROMOTION_REF_PREFIX}{canonical_backend(backend)}"


@dataclass
class ExpandedBackendEvidence:
    """Expanded evidence keyed by backend, surface, and collection source."""

    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    observations: dict[str, dict[str, dict[str, dict[str, Any]]]] = field(default_factory=dict)
    promotions: dict[str, dict[str, set[str]]] = field(default_factory=dict)

    def clone(self) -> "ExpandedBackendEvidence":
        return ExpandedBackendEvidence(
            sources=copy.deepcopy(self.sources),
            observations=copy.deepcopy(self.observations),
            promotions={
                backend: {surface: set(source_ids) for surface, source_ids in surfaces.items()}
                for backend, surfaces in self.promotions.items()
            },
        )


def stores_semantically_equal(left: ExpandedBackendEvidence, right: ExpandedBackendEvidence) -> bool:
    def normalized_promotions(store: ExpandedBackendEvidence) -> dict[str, dict[str, set[str]]]:
        return {
            backend: {surface: set(source_ids) for surface, source_ids in surfaces.items() if source_ids}
            for backend, surfaces in store.promotions.items()
            if any(sources for sources in surfaces.values())
        }

    return (
        left.sources == right.sources
        and left.observations == right.observations
        and normalized_promotions(left) == normalized_promotions(right)
    )


def assert_store_extends(previous: ExpandedBackendEvidence, updated: ExpandedBackendEvidence) -> None:
    for source_id, source in previous.sources.items():
        if updated.sources.get(source_id) != source:
            raise ValueError(f"Existing backend evidence source changed: {source_id}")
    for backend, surfaces in previous.observations.items():
        for surface, contracts in surfaces.items():
            for source_id, contract in contracts.items():
                actual = updated.observations.get(backend, {}).get(surface, {}).get(source_id)
                if actual != contract:
                    raise ValueError(f"Existing backend observation changed: {backend}/{surface}/{source_id}")
    for backend, surfaces in previous.promotions.items():
        for surface, source_ids in surfaces.items():
            actual = updated.promotions.get(backend, {}).get(surface, set())
            if actual != source_ids:
                raise ValueError(f"Existing backend promotion selection changed: {backend}/{surface}")


def _store_root(path: str | os.PathLike[str]) -> Path:
    root = Path(path)
    return root.parent if root.name == MANIFEST_NAME else root


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Missing backend evidence file: {path}") from exc
    if text and not text.endswith("\n"):
        raise ValueError(f"Backend evidence JSONL must end with a newline: {path}")
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise ValueError(f"Blank JSONL record at {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _backend_sources(store: ExpandedBackendEvidence, backend: str) -> list[str]:
    return [
        source_id
        for source_id, source in sorted(store.sources.items())
        if backend in source.get("backends", [])
    ]


def _outcome(contract: dict[str, Any]) -> str:
    result = contract.get("oracle_result") or {}
    if result.get("ok") is True:
        return "passed"
    if result.get("ok") is False:
        return "failed"
    if result.get("skipped"):
        return "skipped"
    return "other"


def _walk_values(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_values(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_values(item, path + (str(index),))


def _validate_public_value(value: Any, *, label: str) -> None:
    errors: list[str] = []
    for path, item in _walk_values(value):
        dotted = ".".join(path) or "<root>"
        if path and path[-1] in _FORBIDDEN_KEYS:
            errors.append(f"{dotted} uses forbidden machine-local field {path[-1]!r}")
        if not isinstance(item, str):
            continue
        if _FILE_URI_RE.search(item):
            errors.append(f"{dotted} contains a file URI")
        if _WINDOWS_ABSOLUTE_RE.search(item):
            errors.append(f"{dotted} contains a Windows absolute path")
        if _POSIX_ABSOLUTE_RE.search(item):
            errors.append(f"{dotted} contains a POSIX absolute path")
        if _ARCHIVE_RE.search(item):
            errors.append(f"{dotted} contains a legacy evidence archive name")
    if errors:
        preview = "; ".join(errors[:8])
        if len(errors) > 8:
            preview += f"; and {len(errors) - 8} more"
        raise ValueError(f"{label} contains non-public machine data: {preview}")


def validate_expanded_store(store: ExpandedBackendEvidence) -> None:
    source_ids = list(store.sources)
    if source_ids != sorted(source_ids):
        raise ValueError("Backend evidence sources are not sorted by source ID")
    seen_numbers: list[int] = []
    for source_id, source in store.sources.items():
        match = _SOURCE_ID_RE.fullmatch(source_id)
        if match is None:
            raise ValueError(f"Invalid backend evidence source ID: {source_id!r}")
        seen_numbers.append(int(match.group(1)))
        if source.get("source_id") != source_id:
            raise ValueError(f"Source record ID mismatch for {source_id}")
        backends = source.get("backends")
        if not isinstance(backends, list) or not backends:
            raise ValueError(f"Source {source_id} has no explicit backends")
        canonical = [canonical_backend(backend) for backend in backends]
        if canonical != sorted(set(canonical)) or canonical != backends:
            raise ValueError(f"Source {source_id} backends are not canonical and sorted")
        _validate_public_value(source, label=f"source {source_id}")
    if seen_numbers and seen_numbers != sorted(set(seen_numbers)):
        raise ValueError("Backend evidence source numbers are duplicated or unsorted")

    for backend, surfaces in store.observations.items():
        if canonical_backend(backend) != backend:
            raise ValueError(f"Non-canonical backend key: {backend}")
        allowed_sources = set(_backend_sources(store, backend))
        if not allowed_sources:
            raise ValueError(f"Backend {backend} has observations but no declared sources")
        used_sources: set[str] = set()
        for surface, source_contracts in surfaces.items():
            if not surface:
                raise ValueError(f"Backend {backend} contains an empty surface key")
            for source_id, contract in source_contracts.items():
                if source_id not in allowed_sources:
                    raise ValueError(f"{backend}/{surface} references unknown source {source_id}")
                used_sources.add(source_id)
                if not isinstance(contract, dict):
                    raise ValueError(f"{backend}/{surface}/{source_id} contract is not an object")
                _validate_public_value(contract, label=f"{backend}/{surface}/{source_id}")
        if used_sources != allowed_sources:
            missing = sorted(allowed_sources - used_sources)
            raise ValueError(f"Backend {backend} sources have no observations: {missing}")

    for source_id, source in store.sources.items():
        for backend in source["backends"]:
            if backend not in store.observations:
                raise ValueError(f"Source {source_id} declares missing backend {backend}")

    for backend, surfaces in store.promotions.items():
        if backend not in store.observations:
            raise ValueError(f"Promotions reference missing backend {backend}")
        for surface, promotion_sources in surfaces.items():
            contracts = store.observations[backend].get(surface)
            if contracts is None:
                raise ValueError(f"Promotion references missing surface {backend}/{surface}")
            for source_id in promotion_sources:
                contract = contracts.get(source_id)
                if contract is None:
                    raise ValueError(f"Promotion {backend}/{surface} references missing source {source_id}")
                if _outcome(contract) != "passed":
                    raise ValueError(f"Promotion {backend}/{surface}/{source_id} is not passing")


def load_backend_evidence_store(
    path: str | os.PathLike[str],
    *,
    verify_canonical: bool = False,
) -> ExpandedBackendEvidence:
    root = _store_root(path)
    manifest_path = root / MANIFEST_NAME
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
    except FileNotFoundError as exc:
        raise ValueError(f"Missing backend evidence manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid backend evidence manifest: {exc}") from exc
    if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported backend evidence format: {manifest.get('format')!r} "
            f"version {manifest.get('format_version')!r}"
        )
    if manifest.get("sources_path") != SOURCES_NAME:
        raise ValueError("Backend evidence manifest has an invalid sources path")
    if verify_canonical and manifest_text != _pretty_json(manifest):
        raise ValueError("Backend evidence manifest is not canonically serialized")

    source_path = root / SOURCES_NAME
    source_records = _read_jsonl(source_path)
    if verify_canonical:
        expected_source_text = "".join(_canonical_json(record) + "\n" for record in source_records)
        if source_path.read_text(encoding="utf-8") != expected_source_text:
            raise ValueError("Backend evidence sources are not canonically serialized")
    sources: dict[str, dict[str, Any]] = {}
    for source in source_records:
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or source_id in sources:
            raise ValueError(f"Duplicate or invalid backend evidence source: {source_id!r}")
        sources[source_id] = source
    if manifest.get("source_count") != len(sources):
        raise ValueError("Backend evidence manifest source count mismatch")

    declared_entries = manifest.get("backends")
    if not isinstance(declared_entries, list) or not declared_entries:
        raise ValueError("Backend evidence manifest has no backend entries")
    declared_paths = {MANIFEST_NAME, SOURCES_NAME}
    observations: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    promotions: dict[str, dict[str, set[str]]] = {}
    seen_backends: set[str] = set()
    previous_backend = ""
    for entry in declared_entries:
        backend = canonical_backend(entry.get("backend"))
        if backend in seen_backends or backend <= previous_backend:
            raise ValueError("Backend evidence manifest entries are duplicate or unsorted")
        previous_backend = backend
        seen_backends.add(backend)
        expected_path = f"{backend}.jsonl"
        if entry.get("path") != expected_path:
            raise ValueError(f"Backend manifest path mismatch for {backend}")
        declared_paths.add(expected_path)
        records = _read_jsonl(root / expected_path)
        if not records:
            raise ValueError(f"Backend evidence file is empty: {expected_path}")
        backend_sources = [source_id for source_id, source in sources.items() if backend in source.get("backends", [])]
        expanded_surfaces: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        backend_promotions: dict[str, set[str]] = defaultdict(set)
        previous_key: tuple[str, str] | None = None
        for record in records:
            surface = record.get("surface")
            contract = record.get("contract")
            if not isinstance(surface, str) or not isinstance(contract, dict):
                raise ValueError(f"Invalid backend evidence record in {expected_path}")
            scoped_sources = record.get("sources", backend_sources)
            if not isinstance(scoped_sources, list) or not scoped_sources:
                raise ValueError(f"Empty source scope for {backend}/{surface}")
            if scoped_sources != sorted(set(scoped_sources)):
                raise ValueError(f"Unsorted or duplicate source scope for {backend}/{surface}")
            first_source = scoped_sources[0]
            sort_key = (surface, first_source)
            if previous_key is not None and sort_key <= previous_key:
                raise ValueError(f"Non-canonical record order in {expected_path}")
            previous_key = sort_key
            for source_id in scoped_sources:
                if source_id not in backend_sources:
                    raise ValueError(f"Unknown source {source_id} for backend {backend}")
                if source_id in expanded_surfaces[surface]:
                    raise ValueError(f"Overlapping source scopes for {backend}/{surface}/{source_id}")
                expanded_surfaces[surface][source_id] = contract
            promotion_sources = record.get("promotion_sources", [])
            if not isinstance(promotion_sources, list) or promotion_sources != sorted(set(promotion_sources)):
                raise ValueError(f"Invalid promotion source list for {backend}/{surface}")
            if not set(promotion_sources).issubset(scoped_sources):
                raise ValueError(f"Promotion sources exceed evidence scope for {backend}/{surface}")
            backend_promotions[surface].update(promotion_sources)
        observations[backend] = {surface: dict(values) for surface, values in expanded_surfaces.items()}
        promotions[backend] = {surface: set(values) for surface, values in backend_promotions.items() if values}
        if entry.get("source_count") != len(backend_sources):
            raise ValueError(f"Backend manifest source count mismatch for {backend}")
        if entry.get("surface_count") != len(expanded_surfaces):
            raise ValueError(f"Backend manifest surface count mismatch for {backend}")
        if entry.get("record_count") != len(records):
            raise ValueError(f"Backend manifest record count mismatch for {backend}")
        if verify_canonical:
            expected_text = "".join(_canonical_json(record) + "\n" for record in records)
            if (root / expected_path).read_text(encoding="utf-8") != expected_text:
                raise ValueError(f"Backend evidence file is not canonically serialized: {expected_path}")

    present_paths = {
        str(item.relative_to(root))
        for item in root.rglob("*")
        if item.is_file()
    }
    if present_paths != declared_paths:
        missing = sorted(declared_paths - present_paths)
        extra = sorted(present_paths - declared_paths)
        raise ValueError(f"Backend evidence store path mismatch; missing={missing}, extra={extra}")

    store = ExpandedBackendEvidence(sources=sources, observations=observations, promotions=promotions)
    validate_expanded_store(store)
    return store


def load_backend_evidence_store_or_empty(path: str | os.PathLike[str]) -> ExpandedBackendEvidence:
    root = _store_root(path)
    if not (root / MANIFEST_NAME).exists():
        return ExpandedBackendEvidence()
    return load_backend_evidence_store(root, verify_canonical=True)


def _serialized_store(store: ExpandedBackendEvidence) -> tuple[dict[str, Any], dict[str, str]]:
    validate_expanded_store(store)
    files: dict[str, str] = {}
    source_records = [store.sources[source_id] for source_id in sorted(store.sources)]
    files[SOURCES_NAME] = "".join(_canonical_json(record) + "\n" for record in source_records)
    manifest_backends: list[dict[str, Any]] = []
    for backend in sorted(store.observations):
        backend_sources = _backend_sources(store, backend)
        source_set = set(backend_sources)
        records: list[dict[str, Any]] = []
        for surface in sorted(store.observations[backend]):
            variants: dict[str, tuple[dict[str, Any], list[str]]] = {}
            for source_id, contract in sorted(store.observations[backend][surface].items()):
                key = _canonical_json(contract)
                if key not in variants:
                    variants[key] = (contract, [])
                variants[key][1].append(source_id)
            ordered_variants = sorted(variants.values(), key=lambda item: item[1][0])
            accepted = store.promotions.get(backend, {}).get(surface, set())
            for contract, scoped_sources in ordered_variants:
                record: dict[str, Any] = {"contract": contract, "surface": surface}
                if set(scoped_sources) != source_set:
                    record["sources"] = scoped_sources
                promotion_sources = sorted(accepted.intersection(scoped_sources))
                if promotion_sources:
                    record["promotion_sources"] = promotion_sources
                records.append(record)
        path = f"{backend}.jsonl"
        files[path] = "".join(_canonical_json(record) + "\n" for record in records)
        manifest_backends.append(
            {
                "backend": backend,
                "path": path,
                "record_count": len(records),
                "source_count": len(backend_sources),
                "surface_count": len(store.observations[backend]),
            }
        )
    manifest = {
        "backends": manifest_backends,
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "source_count": len(store.sources),
        "sources_path": SOURCES_NAME,
    }
    files[MANIFEST_NAME] = _pretty_json(manifest)
    return manifest, files


def write_backend_evidence_store(store: ExpandedBackendEvidence, path: str | os.PathLike[str]) -> None:
    root = _store_root(path)
    root.parent.mkdir(parents=True, exist_ok=True)
    _, files = _serialized_store(store)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
    backup: Path | None = None
    try:
        for relative, text in files.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
        reread = load_backend_evidence_store(staging, verify_canonical=True)
        if not stores_semantically_equal(reread, store):
            raise ValueError("Backend evidence changed during deterministic write/read validation")
        if root.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{root.name}.backup-", dir=root.parent))
            backup.rmdir()
            os.replace(root, backup)
        os.replace(staging, root)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if root.exists() and backup is not None and backup.exists():
            shutil.rmtree(root)
            os.replace(backup, root)
        elif backup is not None and backup.exists() and not root.exists():
            os.replace(backup, root)
        if staging.exists():
            shutil.rmtree(staging)
        raise


def next_source_id(store: ExpandedBackendEvidence) -> str:
    highest = 0
    for source_id in store.sources:
        match = _SOURCE_ID_RE.fullmatch(source_id)
        if match is None:
            raise ValueError(f"Invalid existing source ID: {source_id}")
        highest = max(highest, int(match.group(1)))
    return f"source-{highest + 1:010d}"


def _normalize_location(value: str, *, checkout: str, environment_root: str) -> str:
    path = value.replace("\\", "/")
    suffix = ""
    match = re.match(r"^(.*?)(:\d+(?::\d+)?)$", path)
    if match:
        path, suffix = match.groups()
    for prefix in (checkout, environment_root):
        normalized_prefix = prefix.replace("\\", "/").rstrip("/")
        if normalized_prefix and path.startswith(normalized_prefix + "/"):
            path = path[len(normalized_prefix) + 1 :]
            break
    if "/site-packages/" in path:
        path = path.split("/site-packages/", 1)[1]
    elif path.startswith("/pytorch/"):
        path = path[1:]
    elif "/pytorch/" in path:
        path = "pytorch/" + path.split("/pytorch/", 1)[1]
    elif path.startswith("/"):
        parts = PurePosixPath(path).parts
        torchcts_index = next((index for index, part in enumerate(parts) if part == "torchcts"), None)
        python_index = next((index for index, part in enumerate(parts) if part.startswith("python3")), None)
        if torchcts_index is not None:
            path = "/".join(parts[torchcts_index:])
        elif python_index is not None:
            path = "/".join(parts[python_index:])
        else:
            path = PurePosixPath(path).name
    return path.lstrip("/") + suffix


def _sanitize_string(value: str, *, checkout: str, environment_root: str, hostname: str) -> str:
    sanitized = value
    if hostname:
        sanitized = sanitized.replace(hostname, "backend-host")
    if checkout:
        sanitized = sanitized.replace(checkout.rstrip("/") + "/", "")
        sanitized = sanitized.replace(checkout, "")
    if environment_root:
        normalized_root = environment_root.rstrip("/")
        site_marker = normalized_root + "/"
        sanitized = sanitized.replace(site_marker, "")
        sanitized = sanitized.replace(normalized_root, "")

    def registered(match: re.Match[str]) -> str:
        location = _normalize_location(
            match.group("path"),
            checkout=checkout,
            environment_root=environment_root,
        )
        return f"registered at {location}"

    def traceback_file(match: re.Match[str]) -> str:
        location = _normalize_location(
            match.group("path"),
            checkout=checkout,
            environment_root=environment_root,
        )
        return f'File "{location}"'

    sanitized = _REGISTERED_AT_RE.sub(registered, sanitized)
    sanitized = _TRACEBACK_FILE_RE.sub(traceback_file, sanitized)
    sanitized = sanitized.replace("file://", "")
    return sanitized


def _sanitize_value(value: Any, *, checkout: str, environment_root: str, hostname: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_value(item, checkout=checkout, environment_root=environment_root, hostname=hostname)
            for key, item in value.items()
            if key not in {"_source_path"}
        }
    if isinstance(value, list):
        return [
            _sanitize_value(item, checkout=checkout, environment_root=environment_root, hostname=hostname)
            for item in value
        ]
    if isinstance(value, str):
        return _sanitize_string(value, checkout=checkout, environment_root=environment_root, hostname=hostname)
    return value


def normalize_observation(record: dict[str, Any], environment: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    payload = copy.deepcopy(record)
    try:
        surface = str(payload.pop("surface"))
        backend = canonical_backend(payload.pop("backend_gate"))
    except KeyError as exc:
        raise ValueError(f"Backend observation is missing {exc.args[0]}") from exc
    oracle = payload.get("oracle")
    if isinstance(oracle, dict):
        oracle.pop("promotion_backend", None)
        oracle.pop("promotion_evidence", None)
    checkout = str(environment.get("cwd") or "")
    executable = str(environment.get("python_executable") or "")
    environment_root = ""
    if executable:
        executable_path = Path(executable)
        environment_root = str(executable_path.parent.parent)
    hostname = str(environment.get("hostname") or "")
    payload = _sanitize_value(
        payload,
        checkout=checkout,
        environment_root=environment_root,
        hostname=hostname,
    )
    _validate_public_value(payload, label=f"normalized observation {backend}/{surface}")
    return backend, surface, payload


def normalize_source_environment(
    environment: dict[str, Any],
    *,
    source_id: str,
    backends: Iterable[str],
    runtime_modifications: Iterable[str] = (),
) -> dict[str, Any]:
    device = environment.get("device") or {}
    normalized_backends = sorted({canonical_backend(backend) for backend in backends})
    architecture = str(environment.get("machine") or "unknown")
    platform_text = str(environment.get("platform") or "unknown")
    platform_prefix = platform_text.split("-with-", 1)[0]
    architecture_suffix = f"-{architecture}"
    if architecture != "unknown" and platform_prefix.endswith(architecture_suffix):
        platform_prefix = platform_prefix[: -len(architecture_suffix)]
    if "-" in platform_prefix:
        os_name, os_release = platform_prefix.split("-", 1)
    else:
        os_name, os_release = platform_prefix, "unknown"
    capabilities = []
    normalized_modifications = set(runtime_modifications)
    accelerator_backends = {"cuda", "rocm"}
    cuda_devices = (device.get("cuda_devices") or []) if set(normalized_backends) & accelerator_backends else []
    for item in cuda_devices:
        device_name = str(item.get("name") or "unknown")
        if "LD_PRELOAD fake sm_89" in device_name:
            device_name = device_name.split(" (", 1)[0]
            normalized_modifications.add("sm89-guard-bypass")
        capabilities.append(
            {
                "compute_capability": [item.get("major"), item.get("minor")],
                "multi_processor_count": item.get("multi_processor_count"),
                "name": device_name,
                "total_memory": item.get("total_memory"),
            }
        )
    source = {
        "architecture": architecture,
        "backends": normalized_backends,
        "collected_at": environment.get("generated_at"),
        "device_capabilities": capabilities,
        "operating_system": {"name": os_name, "release": os_release},
        "python_version": str(environment.get("python") or "unknown").split()[0],
        "pytorch_version": device.get("torch_version"),
        "requested_device": device.get("requested_device"),
        "runtime_modifications": sorted(normalized_modifications),
        "source_id": source_id,
        "torch_build": {
            "cuda_available": device.get("cuda_available"),
            "cuda_version": device.get("torch_cuda"),
            "cudnn_available": device.get("cudnn_available"),
            "cudnn_version": device.get("cudnn_version"),
            "hip_version": device.get("torch_hip"),
            "mps_available": device.get("mps_available"),
        },
        "torchcts_version": environment.get("torchcts_version"),
    }
    _validate_public_value(source, label=f"normalized source {source_id}")
    return source


def add_source_observations(
    store: ExpandedBackendEvidence,
    *,
    source: dict[str, Any],
    records: Iterable[tuple[str, str, dict[str, Any]]],
) -> None:
    source_id = source["source_id"]
    if source_id in store.sources:
        raise ValueError(f"Backend evidence source already exists: {source_id}")
    store.sources[source_id] = copy.deepcopy(source)
    store.sources = dict(sorted(store.sources.items()))
    for backend, surface, contract in records:
        backend = canonical_backend(backend)
        store.observations.setdefault(backend, {}).setdefault(surface, {})[source_id] = copy.deepcopy(contract)
    store.observations = {
        backend: {surface: values for surface, values in sorted(surfaces.items())}
        for backend, surfaces in sorted(store.observations.items())
    }
    validate_expanded_store(store)


def outcome_counts(store: ExpandedBackendEvidence) -> dict[str, int]:
    counts = {"failed": 0, "other": 0, "passed": 0, "skipped": 0}
    for surfaces in store.observations.values():
        for contracts in surfaces.values():
            for contract in contracts.values():
                counts[_outcome(contract)] += 1
    return counts
