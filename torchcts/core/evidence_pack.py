# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

"""Build portable backend-promotion evidence archives."""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import tarfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from torchcts import __version__ as torchcts_version
from torchcts.core import coverage
from torchcts.core.oracles import OracleSpec, all_oracle_specs, run_oracle_for_surface


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value.strip())
    return "-".join(part for part in slug.split("-") if part) or "unknown"


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _unique_pack_path(output_root: Path, base_name: str) -> tuple[str, Path]:
    for index in range(1000):
        pack_name = base_name if index == 0 else f"{base_name}-{index}"
        staging = output_root / pack_name
        if not staging.exists() and not (output_root / f"{pack_name}.tar.gz").exists():
            return pack_name, staging
    raise RuntimeError(f"Could not allocate a unique evidence pack path under {output_root}")


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


def _torch_config_text() -> str:
    show = getattr(torch.__config__, "show", None)
    if show is None:
        return ""
    try:
        return str(show())
    except Exception as exc:
        return f"{exc.__class__.__name__}: {exc}"


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
        try:
            free, total = torch.cuda.mem_get_info(index)
            record["mem_get_info"] = {"free": free, "total": total}
        except Exception as exc:
            record["mem_get_info_error"] = f"{exc.__class__.__name__}: {exc}"
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
    selected_env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCHCTS_DEVICE_NAME",
        "TORCHCTS_HARDWARE_KEY",
        "TORCHCTS_RESULTS_DIR",
    )
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
        "selected_environment": {key: os.environ[key] for key in selected_env_keys if key in os.environ},
        "device": _device_environment(device),
        "torch_config": _torch_config_text(),
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
    "all": ("all",),
    "cuda": ("cuda",),
    "cuda+rocm": ("cuda", "rocm"),
    "fbgemm": ("fbgemm",),
    "mps": ("mps",),
    "privateuse1": ("privateuse1",),
    "privateuseone": ("privateuse1",),
    "cpu": ("cpu",),
    "cpu+fbgemm+cpu_build": ("cpu", "fbgemm", "cpu_build"),
    "cpu_build": ("cpu_build",),
    "rocm": ("rocm",),
    "xla": ("xla",),
    "any": ("any",),
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
        if "all" in expanded:
            return ("all",)
        normalized.update(expanded)
    normalized.add("any")
    return tuple(sorted(normalized))


def _gate_matches_device(gate: str | None, device_type: str) -> bool:
    if gate == "any":
        return True
    if gate == device_type:
        return True
    if gate == "privateuse1" and device_type in {"privateuseone", "privateuse1"}:
        return True
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
    device: str,
    *,
    surfaces: list[str] | tuple[str, ...] | None = None,
    backend_gates: list[str] | tuple[str, ...] | None = None,
    include_all_backend_packs: bool = False,
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
        return [
            _target_from_entry(audit_by_name.get(surface) or {"name": surface}, by_surface.get(surface))
            for surface in sorted(requested)
        ]
    device_type = torch.device(device).type
    selected = []
    for entry in audit.get("entries", []):
        if entry.get("coverage_kind") != "backend_pack":
            continue
        spec = by_surface.get(entry.get("name"))
        gate = _backend_gate_for_entry(entry, spec)
        if include_all_backend_packs or selected_gates == ("all",):
            selected.append(_target_from_entry(entry, spec))
        elif selected_gates is not None:
            if gate in selected_gates:
                selected.append(_target_from_entry(entry, spec))
        elif _gate_matches_device(gate, device_type):
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


def _oracle_result(surface: str, device: str, spec: OracleSpec | None, *, run_oracles: bool) -> dict[str, Any]:
    if spec is None:
        return {"ok": None, "skipped": True, "reason": "no oracle spec registered"}
    if not run_oracles:
        return {"ok": None, "skipped": True, "reason": "oracle execution disabled"}
    return _safe_call(run_oracle_for_surface, surface, device)


def _backend_pack_evidence(
    audit: dict,
    targets: list[dict[str, Any]],
    device: str,
    *,
    run_oracles: bool,
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
                "oracle_result": _oracle_result(surface, device, spec, run_oracles=run_oracles),
            }
        )
    return {
        "metadata": {
            "device": device,
            "backend_gates": sorted({target["backend_gate"] for target in targets if target.get("backend_gate")}),
            "run_oracles": run_oracles,
            "record_count": len(records),
        },
        "records": records,
    }


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# TorchCTS Backend Evidence Pack",
                "",
                "This archive is generated by `torchcts coverage evidence-pack`.",
                "It is intended to support backend-pack coverage promotion review.",
                "",
                f"- Device: `{summary['device']}`",
                f"- PyTorch: `{summary['pytorch_version']}`",
                f"- TorchCTS: `{summary['torchcts_version']}`",
                f"- Backend gates: `{', '.join(summary['backend_gates']) or 'none'}`",
                f"- Selected oracle surfaces: `{summary['surface_count']}`",
                f"- Oracle results run: `{summary['run_oracles']}`",
                f"- Oracle failures: `{summary['oracle_failure_count']}`",
                "",
                "Important files:",
                "",
                "- `environment.json`: host, torch, CUDA/MPS, and selected environment facts.",
                "- `coverage/audit.json`: full live coverage audit.",
                "- `coverage/pending_review.json`: pending and excluded coverage records.",
                "- `oracles/backend_pack_evidence.json`: schemas, dispatcher tables, specs, and oracle results.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_evidence_pack(
    *,
    device: str,
    output_dir: str | os.PathLike | None = None,
    surfaces: list[str] | tuple[str, ...] | None = None,
    backend_gates: list[str] | tuple[str, ...] | None = None,
    run_oracles: bool = True,
    include_all_backend_packs: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_dir) if output_dir is not None else coverage.DEFAULT_OUTPUT_DIR / "evidence-packs"
    stamp = _utc_stamp()
    host = _safe_slug(socket.gethostname().split(".", 1)[0])
    device_slug = _safe_slug(device)
    pack_name, staging = _unique_pack_path(output_root, f"torchcts-evidence-{host}-{device_slug}-{stamp}")
    staging.mkdir(parents=True, exist_ok=False)

    audit = coverage.build_audit()
    targets = _select_targets(
        audit,
        device,
        surfaces=surfaces,
        backend_gates=backend_gates,
        include_all_backend_packs=include_all_backend_packs,
    )
    backend_evidence = _backend_pack_evidence(audit, targets, device, run_oracles=run_oracles)
    oracle_failures = [
        record
        for record in backend_evidence["records"]
        if record.get("oracle_result", {}).get("ok") is False
    ]
    summary = {
        "device": device,
        "pytorch_version": torch.__version__,
        "torchcts_version": torchcts_version,
        "backend_gates": sorted({target["backend_gate"] for target in targets if target.get("backend_gate")}),
        "surface_count": len(targets),
        "run_oracles": run_oracles,
        "oracle_failure_count": len(oracle_failures),
        "archive_name": f"{pack_name}.tar.gz",
    }

    _write_json(staging / "environment.json", _environment_record(device))
    _write_json(staging / "summary.json", summary)
    _write_json(staging / "coverage" / "audit.json", audit)
    _write_json(staging / "coverage" / "pending_review.json", coverage.build_pending_review_artifact(audit))
    (staging / "coverage").mkdir(parents=True, exist_ok=True)
    (staging / "coverage" / "summary.md").write_text(coverage.render_summary_markdown(audit), encoding="utf-8")
    (staging / "coverage" / "pending_review.md").write_text(
        coverage.render_pending_review_markdown(audit),
        encoding="utf-8",
    )
    _write_json(staging / "oracles" / "backend_pack_evidence.json", backend_evidence)
    _write_readme(staging / "README.md", summary)

    archive_path = output_root / f"{pack_name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(staging.rglob("*")):
            archive.add(path, arcname=str(Path(pack_name) / path.relative_to(staging)))

    summary.update(
        {
            "staging_dir": str(staging),
            "archive": str(archive_path),
        }
    )
    return summary


def run_evidence_pack_command(
    *,
    device: str,
    output_dir: str | os.PathLike | None,
    surfaces: list[str] | tuple[str, ...] | None,
    backend_gates: list[str] | tuple[str, ...] | None,
    run_oracles: bool,
    include_all_backend_packs: bool,
) -> int:
    result = build_evidence_pack(
        device=device,
        output_dir=output_dir,
        surfaces=surfaces,
        backend_gates=backend_gates,
        run_oracles=run_oracles,
        include_all_backend_packs=include_all_backend_packs,
    )
    print(f"Wrote evidence directory: {result['staging_dir']}")
    print(f"Wrote evidence archive: {result['archive']}")
    print(f"Selected backend gates: {', '.join(result['backend_gates']) or 'none'}")
    print(f"Selected oracle surfaces: {result['surface_count']}")
    if run_oracles:
        print(f"Oracle failures: {result['oracle_failure_count']}")
    return 0
