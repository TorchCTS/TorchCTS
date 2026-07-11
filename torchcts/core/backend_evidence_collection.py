# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

"""Collect backend-specific evidence directly into the canonical store."""

from __future__ import annotations

import os
import platform
import socket
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from torchcts import __version__ as torchcts_version
from torchcts.core import coverage
from torchcts.core.backend_evidence import (
    add_source_observations,
    assert_store_extends,
    canonical_backend,
    load_backend_evidence_store,
    load_backend_evidence_store_or_empty,
    next_source_id,
    normalize_observation,
    normalize_source_environment,
    write_backend_evidence_store,
)
from torchcts.core.oracles import OracleSpec, OracleUnavailable, all_oracle_specs, run_oracle_for_surface


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _safe_call(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return {"ok": True, "value": _json_safe(fn(*args, **kwargs))}
    except Exception as exc:
        return {
            "ok": False,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }


def _cuda_device_records() -> list[dict[str, Any]]:
    records = []
    if not torch.cuda.is_available():
        return records
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        record = {
            "index": index,
            "name": props.name,
            "total_memory": props.total_memory,
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count,
        }
        records.append(record)
    return records


def _device_environment(device: str) -> dict[str, Any]:
    device_obj = torch.device(device)
    payload: dict[str, Any] = {
        "requested_device": device,
        "device_type": device_obj.type,
        "torch_version": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        payload.update(
            {
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": _safe_call(torch.cuda.current_device),
                "cudnn_available": torch.backends.cudnn.is_available(),
                "cudnn_version": torch.backends.cudnn.version(),
                "cuda_devices": _cuda_device_records(),
            }
        )
    return payload


def _environment_record(device: str) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "torchcts_version": torchcts_version,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "device": _device_environment(device),
    }


def _normalize_surfaces(surfaces: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized: list[str] = []
    for item in surfaces or ():
        for part in str(item).split(","):
            surface = part.strip()
            if surface:
                normalized.append(surface)
    return sorted(dict.fromkeys(normalized))


_BACKEND_GATE_ALIASES = {
    "cuda": ("cuda",),
    "fbgemm": ("fbgemm",),
    "mps": ("mps",),
    "privateuse1": ("privateuse1",),
    "privateuseone": ("privateuse1",),
    "cpu-build": ("cpu_build",),
    "cpu_build": ("cpu_build",),
    "rocm": ("rocm",),
    "xla": ("xla",),
}


def _split_selector_values(values: list[str] | tuple[str, ...] | None) -> list[str]:
    selectors: list[str] = []
    for value in values or ():
        for comma_part in str(value).split(","):
            for plus_part in comma_part.split("+"):
                selector = plus_part.strip().lower()
                if selector:
                    selectors.append(selector)
    return selectors


def _normalize_backend_gates(gates: list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    selectors = _split_selector_values(gates)
    if not selectors:
        return None
    normalized: set[str] = set()
    for selector in selectors:
        expanded = _BACKEND_GATE_ALIASES.get(selector)
        if expanded is None:
            valid = ", ".join(sorted(_BACKEND_GATE_ALIASES))
            raise ValueError(f"Unknown backend gate selector {selector!r}; valid selectors: {valid}")
        normalized.update(expanded)
    return tuple(sorted(normalized))


def _oracle_gate_can_run_on_device(gate: str | None, device: str) -> bool:
    device_type = torch.device(device).type
    if gate == "cuda":
        return device_type == "cuda" and torch.cuda.is_available() and not bool(getattr(torch.version, "hip", None))
    if gate == "rocm":
        return device_type == "cuda" and torch.cuda.is_available() and bool(getattr(torch.version, "hip", None))
    if gate == "mps":
        return device_type == "mps" and bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if gate in {"cpu_build", "fbgemm", "quantized"}:
        return device_type == "cpu"
    if gate == "privateuse1":
        return device_type in {"privateuseone", "privateuse1"}
    if gate == "xla":
        return device_type == "xla"
    return False


def _backend_gate_for_entry(entry: dict[str, Any], spec: OracleSpec | None) -> str | None:
    if spec is not None:
        return spec.backend_gate
    pending_review = entry.get("pending_review") or {}
    if pending_review.get("backend_gate"):
        return str(pending_review["backend_gate"])
    oracle = entry.get("oracle") or {}
    if oracle.get("backend_gate"):
        return str(oracle["backend_gate"])
    return None


def _target_from_entry(entry: dict[str, Any], spec: OracleSpec | None) -> dict[str, Any]:
    return {
        "surface": entry.get("name") or (spec.surface if spec is not None else ""),
        "entry": entry,
        "spec": spec,
        "backend_gate": _backend_gate_for_entry(entry, spec),
    }


def _select_targets(
    audit: dict,
    *,
    surfaces: list[str] | tuple[str, ...] | None = None,
    backend_gates: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    requested = set(_normalize_surfaces(surfaces))
    selected_gates = _normalize_backend_gates(backend_gates)
    specs = list(all_oracle_specs())
    by_surface = {spec.surface: spec for spec in specs}
    audit_by_name = {entry.get("name"): entry for entry in audit.get("entries", [])}
    if requested:
        missing = sorted(surface for surface in requested if surface not in by_surface and surface not in audit_by_name)
        if missing:
            raise ValueError(f"No audit entries or oracle specs found for requested surfaces: {', '.join(missing)}")
        targets = [
            _target_from_entry(audit_by_name.get(surface) or {"name": surface}, by_surface.get(surface))
            for surface in sorted(requested)
        ]
        if selected_gates is not None:
            mismatched = [target["surface"] for target in targets if target["backend_gate"] not in selected_gates]
            if mismatched:
                raise ValueError(
                    "Requested surfaces do not belong to the selected backend gates: " + ", ".join(mismatched)
                )
        return targets
    if selected_gates is None:
        raise ValueError("Backend evidence collection requires --backend-gate unless exact surfaces are requested")
    selected = []
    for entry in audit.get("entries", []):
        if entry.get("coverage_kind") != "backend_pack":
            continue
        spec = by_surface.get(entry.get("name"))
        gate = _backend_gate_for_entry(entry, spec)
        if gate in selected_gates:
            selected.append(_target_from_entry(entry, spec))
    return sorted(selected, key=lambda target: target["surface"])


def _schema_evidence(surface: str) -> dict[str, Any]:
    def _record() -> dict[str, Any]:
        schema = coverage._schema_for(surface)
        return {
            "schema": str(schema),
            "args": [coverage._schema_arg_record(arg) for arg in schema.arguments],
            "returns": [coverage._schema_return_record(ret) for ret in schema.returns],
        }

    return _safe_call(_record)


def _dispatch_evidence(surface: str) -> dict[str, Any]:
    return {
        "dispatch_registration_map": _safe_call(coverage._dispatch_registration_map, surface),
        "dispatch_dump_table": _safe_call(torch._C._dispatch_dump_table, surface),
    }


def _oracle_result(
    surface: str,
    device: str,
    spec: OracleSpec | None,
    *,
    run_oracles: bool,
    run_pending_candidates: bool = False,
) -> dict[str, Any]:
    if spec is None:
        return {"ok": None, "skipped": True, "reason": "no oracle spec registered"}
    if not run_oracles:
        return {"ok": None, "skipped": True, "reason": "oracle execution disabled"}
    if spec.runner == "backend_property":
        return {"ok": None, "skipped": True, "reason": "oracle runner not implemented"}
    if spec.coverage_status.startswith("pending_") and not run_pending_candidates:
        return {"ok": None, "skipped": True, "reason": "pending candidate oracle not requested"}
    if not _oracle_gate_can_run_on_device(spec.backend_gate, device):
        return {
            "ok": None,
            "skipped": True,
            "reason": f"backend gate {spec.backend_gate!r} does not run on device {torch.device(device).type!r}",
        }
    try:
        value = run_oracle_for_surface(surface, device)
    except OracleUnavailable as exc:
        return {"ok": None, "skipped": True, "reason": str(exc)}
    except Exception as exc:
        return {
            "ok": False,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    return {"ok": True, "value": _json_safe(value)}


def _backend_pack_evidence(
    audit: dict,
    targets: list[dict[str, Any]],
    device: str,
    *,
    run_oracles: bool,
    run_pending_candidates: bool,
) -> dict[str, Any]:
    records = []
    for target in targets:
        spec = target["spec"]
        audit_entry = target["entry"]
        surface = target["surface"]
        records.append(
            {
                "surface": surface,
                "oracle": spec.metadata() if spec is not None else None,
                "coverage_status": spec.coverage_status if spec is not None else audit_entry.get("status"),
                "coverage_kind": spec.coverage_kind if spec is not None else audit_entry.get("coverage_kind"),
                "backend_gate": target["backend_gate"],
                "audit_status": audit_entry.get("status"),
                "audit_coverage_kind": audit_entry.get("coverage_kind"),
                "surface_kind": audit_entry.get("surface_kind"),
                "variant_kind": audit_entry.get("variant_kind"),
                "semantic_level": audit_entry.get("semantic_level"),
                "pending_review": audit_entry.get("pending_review"),
                "exclusion": audit_entry.get("exclusion"),
                "schema": _schema_evidence(surface),
                "dispatch": _dispatch_evidence(surface),
                "oracle_result": _oracle_result(
                    surface,
                    device,
                    spec,
                    run_oracles=run_oracles,
                    run_pending_candidates=run_pending_candidates,
                ),
            }
        )
    return {
        "metadata": {
            "device": device,
            "backend_gates": sorted({target["backend_gate"] for target in targets if target.get("backend_gate")}),
            "run_oracles": run_oracles,
            "run_pending_candidates": run_pending_candidates,
            "record_count": len(records),
        },
        "records": records,
    }


def collect_backend_evidence(
    *,
    store: str | os.PathLike,
    device: str,
    surfaces: list[str] | tuple[str, ...] | None = None,
    backend_gates: list[str] | tuple[str, ...] | None = None,
    run_oracles: bool = True,
    run_pending_candidates: bool = False,
    runtime_modifications: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    destination = Path(store)
    previous = load_backend_evidence_store_or_empty(destination)
    audit = coverage.build_audit()
    targets = _select_targets(
        audit,
        surfaces=surfaces,
        backend_gates=backend_gates,
    )
    if not targets:
        raise ValueError("Backend evidence selection did not match any surfaces")
    selected_gates = sorted({target["backend_gate"] for target in targets if target.get("backend_gate")})
    if any(gate in {None, "any"} for gate in (target.get("backend_gate") for target in targets)):
        raise ValueError("Every collected backend evidence surface must have an explicit backend gate")
    if "privateuse1" in selected_gates:
        raise ValueError(
            "PrivateUse1 evidence requires the registered backend identity; "
            "the generic privateuse1 gate is not a canonical evidence backend"
        )
    incompatible = [gate for gate in selected_gates if not _oracle_gate_can_run_on_device(gate, device)]
    if incompatible:
        raise ValueError(
            f"Backend gates {', '.join(incompatible)} cannot run on requested device {torch.device(device).type!r}"
        )
    for gate in selected_gates:
        canonical_backend(gate)

    backend_evidence = _backend_pack_evidence(
        audit,
        targets,
        device,
        run_oracles=run_oracles,
        run_pending_candidates=run_pending_candidates,
    )
    environment = _environment_record(device)
    normalized_records = [
        normalize_observation(record, environment)
        for record in backend_evidence["records"]
    ]
    updated = previous.clone()
    source_id = next_source_id(updated)
    source = normalize_source_environment(
        environment,
        source_id=source_id,
        backends={backend for backend, _, _ in normalized_records},
        runtime_modifications=runtime_modifications or (),
    )
    add_source_observations(updated, source=source, records=normalized_records)
    assert_store_extends(previous, updated)
    write_backend_evidence_store(updated, destination)
    reread = load_backend_evidence_store(destination, verify_canonical=True)
    assert_store_extends(previous, reread)

    oracle_failures = [
        record
        for record in backend_evidence["records"]
        if record.get("oracle_result", {}).get("ok") is False
    ]
    return {
        "source_id": source_id,
        "store": str(destination),
        "device": device,
        "pytorch_version": torch.__version__,
        "torchcts_version": torchcts_version,
        "backend_gates": [canonical_backend(gate) for gate in selected_gates],
        "surface_count": len(targets),
        "run_oracles": run_oracles,
        "run_pending_candidates": run_pending_candidates,
        "oracle_failure_count": len(oracle_failures),
        "oracle_success_count": sum(
            1 for record in backend_evidence["records"] if record.get("oracle_result", {}).get("ok") is True
        ),
        "oracle_skipped_count": sum(
            1 for record in backend_evidence["records"] if record.get("oracle_result", {}).get("skipped")
        ),
    }


def run_backend_evidence_collection_command(
    *,
    store: str | os.PathLike,
    device: str,
    surfaces: list[str] | tuple[str, ...] | None,
    backend_gates: list[str] | tuple[str, ...] | None,
    run_oracles: bool,
    run_pending_candidates: bool,
    runtime_modifications: list[str] | tuple[str, ...] | None = None,
    require_oracle_results: bool = False,
    fail_on_oracle_failure: bool = False,
) -> int:
    result = collect_backend_evidence(
        store=store,
        device=device,
        surfaces=surfaces,
        backend_gates=backend_gates,
        run_oracles=run_oracles,
        run_pending_candidates=run_pending_candidates,
        runtime_modifications=runtime_modifications,
    )
    print(f"Updated backend evidence store: {result['store']}")
    print(f"Added evidence source: {result['source_id']}")
    print(f"Selected backend gates: {', '.join(result['backend_gates']) or 'none'}")
    print(f"Selected oracle surfaces: {result['surface_count']}")
    if run_oracles:
        print(f"Oracle failures: {result['oracle_failure_count']}")
        print(f"Oracle successes: {result['oracle_success_count']}")
        print(f"Oracle skipped: {result['oracle_skipped_count']}")
    exit_code = 0
    if fail_on_oracle_failure and result["oracle_failure_count"]:
        exit_code = 1
    if require_oracle_results and result["oracle_success_count"] != result["surface_count"]:
        missing = result["surface_count"] - result["oracle_success_count"]
        print(f"Error: {missing} selected oracle surfaces did not produce passing oracle results")
        exit_code = 1
    return exit_code
