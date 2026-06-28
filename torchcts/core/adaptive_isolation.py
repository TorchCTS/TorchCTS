# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

from __future__ import annotations

import datetime as _datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from torchcts.core.known_segfaults import canonicalize_nodeid


CRASH_SIGNALS = {"SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL"}


@dataclass(frozen=True)
class AdaptiveIsolationCandidate:
    nodeid: str
    canonical_nodeid: str
    isolation_source: str
    reason: str
    evidence_path: str
    prior_status: str | None = None
    prior_signal: str | None = None
    prior_error_type: str | None = None
    prior_timestamp: str | None = None

    def to_json(self) -> dict:
        return {
            "nodeid": self.nodeid,
            "canonical_nodeid": self.canonical_nodeid,
            "isolation_source": self.isolation_source,
            "reason": self.reason,
            "evidence_path": self.evidence_path,
            "prior_status": self.prior_status,
            "prior_signal": self.prior_signal,
            "prior_error_type": self.prior_error_type,
            "prior_timestamp": self.prior_timestamp,
        }


@dataclass(frozen=True)
class AdaptiveIsolationLoadResult:
    candidates: dict[str, AdaptiveIsolationCandidate]
    rejected: list[dict]
    warnings: list[str]
    artifacts_considered: list[str]


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, warnings: list[str]) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"{path}: ignored malformed adaptive-isolation artifact: {exc}")
        return None


def _metadata_matches(data: dict, *, device_name: str, hardware_key: str, torch_version: str) -> bool:
    metadata = data.get("metadata") or {}
    return (
        metadata.get("device_name") == device_name
        and metadata.get("hardware_key") == hardware_key
        and metadata.get("pytorch_version") == torch_version
    )


def _history_paths(results_dir: Path, hardware_key: str, limit: int) -> list[Path]:
    history_dir = results_dir / f"{hardware_key}_history"
    if not history_dir.exists():
        return []
    paths = [path for path in history_dir.glob("*.json") if path.is_file()]
    paths.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    return paths[:limit]


def _artifact_paths(results_dir: Path, hardware_key: str, history_limit: int) -> list[Path]:
    paths: list[Path] = []
    latest = results_dir / f"{hardware_key}_latest.json"
    if latest.exists():
        paths.append(latest)
    paths.extend(_history_paths(results_dir, hardware_key, history_limit))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(path)
    return ordered


def _subprocess_signal(record: dict) -> str | None:
    subprocess_record = record.get("subprocess") or {}
    signal = subprocess_record.get("signal")
    return str(signal) if signal else None


def _candidate_from_record(
    nodeid: str,
    record: dict,
    *,
    evidence_path: Path,
    prior_timestamp: str | None,
) -> AdaptiveIsolationCandidate | None:
    canonical = canonicalize_nodeid(nodeid)
    status = record.get("status")
    error_type = record.get("error_type")
    signal = _subprocess_signal(record)
    subprocess_record = record.get("subprocess") or {}

    if error_type == "ProcessCrash" or signal in CRASH_SIGNALS:
        reason = "previous run recorded a process crash"
        if signal:
            reason = f"{reason} ({signal})"
        return AdaptiveIsolationCandidate(
            nodeid=nodeid,
            canonical_nodeid=canonical,
            isolation_source="adaptive_previous_crash",
            reason=reason,
            evidence_path=str(evidence_path),
            prior_status=str(status) if status is not None else None,
            prior_signal=signal,
            prior_error_type=str(error_type) if error_type is not None else None,
            prior_timestamp=prior_timestamp,
        )

    timed_out = bool(subprocess_record.get("timed_out"))
    if error_type == "TimeoutError" and (
        timed_out
        or record.get("failure_stage") == "subprocess_timeout"
        or str(record.get("phase") or "").startswith("subprocess")
    ):
        return AdaptiveIsolationCandidate(
            nodeid=nodeid,
            canonical_nodeid=canonical,
            isolation_source="adaptive_previous_timeout",
            reason="previous subprocess isolation timed out",
            evidence_path=str(evidence_path),
            prior_status=str(status) if status is not None else None,
            prior_signal=signal,
            prior_error_type="TimeoutError",
            prior_timestamp=prior_timestamp,
        )
    return None


def _terminal_record(record: dict) -> bool:
    return bool(record) and record.get("status") in {"PASS", "FAIL", "ERROR", "SKIP"}


def _load_json_candidates(
    artifacts: Iterable[Path],
    *,
    device_name: str,
    hardware_key: str,
    torch_version: str,
    latest_artifact_name: str,
    warnings: list[str],
    rejected: list[dict],
) -> tuple[dict[str, AdaptiveIsolationCandidate], dict[str, None], dict[str, dict]]:
    candidates: dict[str, AdaptiveIsolationCandidate] = {}
    resolved: dict[str, None] = {}
    latest_matching: dict[str, dict] = {}

    for path in artifacts:
        data = _read_json(path, warnings)
        if data is None:
            continue
        if not _metadata_matches(
            data,
            device_name=device_name,
            hardware_key=hardware_key,
            torch_version=torch_version,
        ):
            rejected.append({"path": str(path), "reason": "metadata_mismatch"})
            continue
        if path.name == latest_artifact_name:
            latest_matching = {"path": str(path), "data": data}
        timestamp = (data.get("metadata") or {}).get("timestamp")
        for nodeid, record in (data.get("results") or {}).items():
            canonical = canonicalize_nodeid(nodeid)
            candidate = _candidate_from_record(
                nodeid,
                record,
                evidence_path=path,
                prior_timestamp=timestamp,
            )
            if canonical in candidates:
                continue
            if canonical in resolved:
                if candidate is not None:
                    rejected.append(
                        {
                            "nodeid": nodeid,
                            "canonical_nodeid": canonical,
                            "path": str(path),
                            "reason": "newer_nonisolating_record_resolved_candidate",
                        }
                    )
                continue
            if candidate is not None:
                candidates[canonical] = candidate
            elif _terminal_record(record):
                resolved[canonical] = None
    return candidates, resolved, latest_matching


def _parse_runlog_nodes(path: Path) -> list[str]:
    nodes: list[str] = []
    if not path.exists():
        return nodes
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\s*[\d.]+s\s+(.+)$", line)
        if match:
            nodeid = match.group(1).strip()
            if nodeid:
                nodes.append(nodeid)
    return nodes


def _add_runlog_hang_candidate(
    candidates: dict[str, AdaptiveIsolationCandidate],
    resolved: dict[str, None],
    *,
    latest_matching: dict[str, dict],
    results_dir: Path,
    hardware_key: str,
    rejected: list[dict],
) -> None:
    if not latest_matching:
        return
    runlog_path = results_dir / f"{hardware_key}_runlog.txt"
    runlog_nodes = _parse_runlog_nodes(runlog_path)
    if not runlog_nodes:
        return
    nodeid = runlog_nodes[-1]
    canonical = canonicalize_nodeid(nodeid)
    if canonical in candidates or canonical in resolved:
        return

    latest_data = latest_matching["data"]
    results = latest_data.get("results") or {}
    canonical_results = {canonicalize_nodeid(key): value for key, value in results.items()}
    if canonical in canonical_results:
        rejected.append(
            {
                "nodeid": nodeid,
                "canonical_nodeid": canonical,
                "path": str(runlog_path),
                "reason": "runlog_last_node_has_result_record",
            }
        )
        return
    metadata = latest_data.get("metadata") or {}
    if metadata.get("session_completed") is True:
        rejected.append(
            {
                "nodeid": nodeid,
                "canonical_nodeid": canonical,
                "path": str(runlog_path),
                "reason": "latest_json_reports_completed_session",
            }
        )
        return

    candidates[canonical] = AdaptiveIsolationCandidate(
        nodeid=nodeid,
        canonical_nodeid=canonical,
        isolation_source="adaptive_suspected_hang",
        reason="previous runlog ended at this node without a result record",
        evidence_path=str(runlog_path),
        prior_status=None,
        prior_signal=None,
        prior_error_type=None,
        prior_timestamp=metadata.get("timestamp"),
    )


def load_adaptive_isolation(
    results_dir: str | Path,
    *,
    hardware_key: str,
    device_name: str,
    torch_version: str,
    history_limit: int = 5,
) -> AdaptiveIsolationLoadResult:
    base = Path(results_dir)
    warnings: list[str] = []
    rejected: list[dict] = []
    artifacts = _artifact_paths(base, hardware_key, history_limit)
    candidates, resolved, latest_matching = _load_json_candidates(
        artifacts,
        device_name=device_name,
        hardware_key=hardware_key,
        torch_version=torch_version,
        latest_artifact_name=f"{hardware_key}_latest.json",
        warnings=warnings,
        rejected=rejected,
    )
    _add_runlog_hang_candidate(
        candidates,
        resolved,
        latest_matching=latest_matching,
        results_dir=base,
        hardware_key=hardware_key,
        rejected=rejected,
    )
    return AdaptiveIsolationLoadResult(
        candidates=candidates,
        rejected=rejected,
        warnings=warnings,
        artifacts_considered=[str(path) for path in artifacts],
    )


def filter_candidates_for_collection(
    load_result: AdaptiveIsolationLoadResult,
    collected_nodeids: Iterable[str],
) -> tuple[dict[str, dict], list[dict]]:
    collected = {canonicalize_nodeid(nodeid) for nodeid in collected_nodeids}
    accepted: dict[str, dict] = {}
    rejected = list(load_result.rejected)
    for canonical, candidate in load_result.candidates.items():
        if canonical in collected:
            accepted[canonical] = candidate.to_json()
        else:
            rejected.append(
                {
                    "nodeid": candidate.nodeid,
                    "canonical_nodeid": canonical,
                    "path": candidate.evidence_path,
                    "reason": "not_collected_in_current_run",
                }
            )
    return accepted, rejected


def build_adaptive_isolation_artifact(
    *,
    hardware_key: str,
    device_name: str,
    torch_version: str,
    mode: str,
    accepted: dict[str, dict],
    rejected: list[dict],
    warnings: list[str],
    artifacts_considered: list[str],
) -> dict:
    return {
        "version": 1,
        "generated_at": _utc_now(),
        "mode": mode,
        "metadata": {
            "device_name": device_name,
            "hardware_key": hardware_key,
            "pytorch_version": torch_version,
        },
        "accepted": accepted,
        "rejected": rejected,
        "warnings": warnings,
        "artifacts_considered": artifacts_considered,
    }
