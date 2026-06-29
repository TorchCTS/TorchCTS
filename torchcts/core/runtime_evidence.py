"""Results-only runtime evidence helpers.

These helpers are diagnostic sinks. They must never influence collection,
selection, skipping, xfail, pass, or failure semantics.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path


def _safe_text(value, *, limit: int = 4000) -> str:
    try:
        text = str(value)
    except Exception as exc:
        text = f"<unprintable {type(value).__name__}: {type(exc).__name__}>"
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _safe_tail(value, *, limit: int = 4000) -> str:
    try:
        text = str(value)
    except Exception as exc:
        text = f"<unprintable {type(value).__name__}: {type(exc).__name__}>"
    return text if len(text) <= limit else text[-limit:]


def _evidence_path() -> Path:
    results_dir = Path(os.environ.get("TORCHCTS_RESULTS_DIR") or "results")
    hardware_key = os.environ.get("TORCHCTS_HARDWARE_KEY") or "unknown"
    return results_dir / f"{hardware_key}_opinfo_oracle_failures_{os.getpid()}.jsonl"


def _harness_probe_evidence_path() -> Path:
    results_dir = Path(os.environ.get("TORCHCTS_RESULTS_DIR") or "results")
    hardware_key = os.environ.get("TORCHCTS_HARDWARE_KEY") or "unknown"
    return results_dir / f"{hardware_key}_harness_probe_failures_{os.getpid()}.jsonl"


def harness_probe_failure_key(record: dict) -> tuple:
    return (
        record.get("probe_kind"),
        record.get("name"),
        record.get("stage"),
        record.get("error_type"),
        record.get("error_message"),
        record.get("returncode"),
    )


def record_opinfo_oracle_failure(
    phase,
    op_name,
    dtype_str,
    stage,
    exc,
    *,
    input_condition=None,
    sample_index=None,
    nodeid=None,
) -> None:
    """Append OpInfo CPU-oracle failure evidence under the current results dir."""

    try:
        path = _evidence_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "created_at": datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "device_name": os.environ.get("TORCHCTS_DEVICE_NAME") or "unknown",
            "hardware_key": os.environ.get("TORCHCTS_HARDWARE_KEY") or "unknown",
            "pytorch_version": os.environ.get("TORCHCTS_PYTORCH_VERSION") or "unknown",
            "phase": _safe_text(phase),
            "op_name": _safe_text(op_name),
            "dtype": _safe_text(dtype_str),
            "stage": _safe_text(stage),
            "input_condition": None if input_condition is None else _safe_text(input_condition),
            "sample_index": sample_index,
            "nodeid": None if nodeid is None else _safe_text(nodeid),
            "error_type": type(exc).__name__,
            "error_message": _safe_text(exc),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        return


def record_harness_probe_failure(
    probe_kind,
    name,
    exc_or_result,
    *,
    stage=None,
) -> dict | None:
    """Append harness probe failure evidence and return the JSON-ready record."""

    try:
        evidence = exc_or_result.to_dict() if hasattr(exc_or_result, "to_dict") else {}
        if not isinstance(evidence, dict):
            evidence = {}
        error_type = evidence.get("error_type") or type(exc_or_result).__name__
        error_message = evidence.get("error_message")
        if error_message is None:
            error_message = _safe_text(exc_or_result)

        record = {
            "created_at": datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "device_name": os.environ.get("TORCHCTS_DEVICE_NAME") or "unknown",
            "hardware_key": os.environ.get("TORCHCTS_HARDWARE_KEY") or "unknown",
            "pytorch_version": os.environ.get("TORCHCTS_PYTORCH_VERSION") or "unknown",
            "probe_kind": _safe_text(probe_kind),
            "name": _safe_text(name),
            "stage": _safe_text(stage),
            "declared": True,
            "outcome": "failed",
            "error_type": _safe_text(error_type),
            "error_message": _safe_text(error_message),
            "returncode": evidence.get("returncode"),
            "timed_out": bool(evidence.get("timed_out", False)),
            "stdout_tail": _safe_tail(evidence.get("stdout_tail", evidence.get("stdout", ""))),
            "stderr_tail": _safe_tail(evidence.get("stderr_tail", evidence.get("stderr", ""))),
            "command_args": [
                _safe_text(arg, limit=1000)
                for arg in evidence.get("command_args", [])
            ],
        }
        path = _harness_probe_evidence_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record
    except Exception:
        return None
