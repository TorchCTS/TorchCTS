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

from __future__ import annotations

import datetime as _datetime
import fnmatch
import json
from pathlib import Path
from typing import Iterable

from torchcts.core.version_rules import parse_torch_version, version_in_range


PACKAGED_KNOWN_SEGFAULTS = Path(__file__).resolve().parents[1] / "known_segfaults.json"
PROJECT_KNOWN_SEGFAULTS = "known_segfaults.json"

VALID_SIGNALS = {"SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL"}
VALID_MATCHES = {"nodeid", "dispatcher", "coverage_id"}
VALID_EVIDENCE_SCOPES = {"exact_node", "constrained_metadata", "dispatcher_surface"}
VALID_CLASSIFICATIONS = {"confirmed_backend_crash"}
EXACT_CONSTRAINT_KEYS = {
    "suite",
    "test_kind",
    "coverage_kind",
    "surface_kind",
    "variant_kind",
    "coverage_status",
    "strategy",
    "strategy_family",
    "dtype",
}
GLOB_CONSTRAINT_KEYS = {"nodeid_glob", "coverage_id_glob"}
SEMANTIC_CONSTRAINT_KEYS = {"semantic_level"}
VALID_CONSTRAINT_KEYS = EXACT_CONSTRAINT_KEYS | GLOB_CONSTRAINT_KEYS | SEMANTIC_CONSTRAINT_KEYS
MATCH_METADATA_KEYS = (
    "dispatcher_name",
    "coverage_id",
    "suite",
    "test_kind",
    "coverage_kind",
    "surface_kind",
    "variant_kind",
    "coverage_status",
    "strategy",
    "strategy_family",
    "semantic_level",
    "dtype",
)

REQUIRED_ENTRY_KEYS = {
    "id",
    "backend",
    "match",
    "dispatcher",
    "evidence_scope",
    "classification",
    "expected_signal",
    "repro",
    "reason",
    "owner",
    "pytorch_min",
    "pytorch_max",
    "hardware",
    "review_after",
}

ALLOWED_ENTRY_KEYS = set(REQUIRED_ENTRY_KEYS) | {"nodeid", "coverage_id", "constraints"}
REQUIRED_REPRO_KEYS = {"script", "case"}
ALLOWED_REPRO_KEYS = set(REQUIRED_REPRO_KEYS)


class KnownSegfaultError(ValueError):
    """Raised when a known-segfault ledger is malformed."""


def _parse_date(value: str, path: str, entry_id: str) -> _datetime.date:
    if not isinstance(value, str):
        raise KnownSegfaultError(f"{path}: entry {entry_id} review_after must be YYYY-MM-DD")
    try:
        return _datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise KnownSegfaultError(f"{path}: entry {entry_id} review_after must be YYYY-MM-DD") from exc


def _canonical_dispatcher_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.endswith(".default"):
        return text[: -len(".default")]
    return text


def _dispatcher_matches(expected: str | None, actual: str | None) -> bool:
    return _canonical_dispatcher_name(expected) == _canonical_dispatcher_name(actual)


def _normalize_constraints(value, path: str, entry_id: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise KnownSegfaultError(f"{path}: entry {entry_id} constraints must be an object")
    unknown = sorted(set(value) - VALID_CONSTRAINT_KEYS)
    if unknown:
        raise KnownSegfaultError(
            f"{path}: entry {entry_id} constraints has unknown field(s): {', '.join(unknown)}"
        )

    normalized = {}
    for key, raw in value.items():
        raw_values = raw if isinstance(raw, list) else [raw]
        if not raw_values:
            raise KnownSegfaultError(f"{path}: entry {entry_id} constraints.{key} must not be empty")

        values = []
        if key in SEMANTIC_CONSTRAINT_KEYS:
            for item in raw_values:
                if not isinstance(item, int) or isinstance(item, bool) or not 1 <= item <= 8:
                    raise KnownSegfaultError(
                        f"{path}: entry {entry_id} constraints.{key} must contain semantic levels 1..8"
                    )
                values.append(item)
        else:
            for item in raw_values:
                if not isinstance(item, str) or not item.strip():
                    raise KnownSegfaultError(
                        f"{path}: entry {entry_id} constraints.{key} must contain non-empty strings"
                    )
                values.append(item.strip())
        normalized[key] = values
    return normalized


def _normalize_payload_item(payload_item) -> tuple[str, dict]:
    if isinstance(payload_item, tuple) and len(payload_item) == 2:
        path, payload = payload_item
        return str(path), payload
    return "<memory>", payload_item


def _validate_entry(entry: dict, path: str, seen_ids: set[str]) -> dict:
    if not isinstance(entry, dict):
        raise KnownSegfaultError(f"{path}: each known_segfaults entry must be an object")

    entry_id = entry.get("id", "<missing>")
    missing = sorted(REQUIRED_ENTRY_KEYS - set(entry))
    if missing:
        raise KnownSegfaultError(f"{path}: entry {entry_id} missing required field(s): {', '.join(missing)}")

    unknown = sorted(set(entry) - ALLOWED_ENTRY_KEYS)
    if unknown:
        raise KnownSegfaultError(f"{path}: entry {entry_id} has unknown field(s): {', '.join(unknown)}")

    if not isinstance(entry["id"], str) or not entry["id"].strip():
        raise KnownSegfaultError(f"{path}: entry id must be a non-empty string")
    if entry["id"] in seen_ids:
        raise KnownSegfaultError(f"{path}: duplicate known segfault id {entry['id']!r}")
    seen_ids.add(entry["id"])

    for key in ("backend", "dispatcher", "reason", "owner", "hardware"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise KnownSegfaultError(f"{path}: entry {entry['id']} field {key} must be a non-empty string")
    for key in ("nodeid", "coverage_id"):
        if key in entry and (not isinstance(entry[key], str) or not entry[key].strip()):
            raise KnownSegfaultError(f"{path}: entry {entry['id']} field {key} must be a non-empty string")

    if entry["match"] not in VALID_MATCHES:
        raise KnownSegfaultError(f"{path}: entry {entry['id']} match must be one of {sorted(VALID_MATCHES)}")
    if entry["evidence_scope"] not in VALID_EVIDENCE_SCOPES:
        raise KnownSegfaultError(
            f"{path}: entry {entry['id']} evidence_scope must be one of {sorted(VALID_EVIDENCE_SCOPES)}"
        )
    if entry["classification"] not in VALID_CLASSIFICATIONS:
        raise KnownSegfaultError(
            f"{path}: entry {entry['id']} classification must be one of {sorted(VALID_CLASSIFICATIONS)}"
        )
    if entry["expected_signal"] not in VALID_SIGNALS:
        raise KnownSegfaultError(f"{path}: entry {entry['id']} expected_signal must be one of {sorted(VALID_SIGNALS)}")

    if entry["pytorch_min"] is not None and parse_torch_version(entry["pytorch_min"]) is None:
        raise KnownSegfaultError(f"{path}: entry {entry['id']} pytorch_min is invalid")
    if entry["pytorch_max"] is not None and parse_torch_version(entry["pytorch_max"]) is None:
        raise KnownSegfaultError(f"{path}: entry {entry['id']} pytorch_max is invalid")

    _parse_date(entry["review_after"], path, entry["id"])

    repro = entry["repro"]
    if not isinstance(repro, dict):
        raise KnownSegfaultError(f"{path}: entry {entry['id']} repro must be an object")
    missing_repro = sorted(REQUIRED_REPRO_KEYS - set(repro))
    if missing_repro:
        raise KnownSegfaultError(
            f"{path}: entry {entry['id']} repro missing required field(s): {', '.join(missing_repro)}"
        )
    unknown_repro = sorted(set(repro) - ALLOWED_REPRO_KEYS)
    if unknown_repro:
        raise KnownSegfaultError(
            f"{path}: entry {entry['id']} repro has unknown field(s): {', '.join(unknown_repro)}"
        )
    for key in REQUIRED_REPRO_KEYS:
        if not isinstance(repro[key], str) or not repro[key].strip():
            raise KnownSegfaultError(f"{path}: entry {entry['id']} repro.{key} must be a non-empty string")

    constraints = _normalize_constraints(entry.get("constraints"), path, entry["id"])
    match_mode = entry["match"]
    evidence_scope = entry["evidence_scope"]
    if match_mode == "nodeid":
        if "nodeid" not in entry:
            raise KnownSegfaultError(f"{path}: entry {entry['id']} match=nodeid requires nodeid")
        if evidence_scope != "exact_node":
            raise KnownSegfaultError(f"{path}: entry {entry['id']} match=nodeid requires evidence_scope=exact_node")
    elif evidence_scope == "exact_node":
        raise KnownSegfaultError(f"{path}: entry {entry['id']} evidence_scope=exact_node requires match=nodeid")

    if match_mode == "dispatcher" and evidence_scope == "constrained_metadata" and not constraints:
        raise KnownSegfaultError(
            f"{path}: entry {entry['id']} constrained dispatcher rules require non-empty constraints"
        )
    if match_mode == "coverage_id":
        has_coverage_id_glob = bool(constraints.get("coverage_id_glob"))
        if "coverage_id" not in entry and not has_coverage_id_glob:
            raise KnownSegfaultError(
                f"{path}: entry {entry['id']} match=coverage_id requires coverage_id or constraints.coverage_id_glob"
            )
        if evidence_scope == "constrained_metadata" and not constraints:
            raise KnownSegfaultError(
                f"{path}: entry {entry['id']} constrained coverage_id rules require non-empty constraints"
            )

    normalized = dict(entry)
    if constraints:
        normalized["constraints"] = constraints
    else:
        normalized.pop("constraints", None)
    normalized["source_path"] = path
    return normalized


def validate_known_segfaults(payloads: Iterable[dict | tuple[str | Path, dict]]) -> list[dict]:
    entries: list[dict] = []
    seen_ids: set[str] = set()
    for payload_item in payloads:
        path, payload = _normalize_payload_item(payload_item)
        if not isinstance(payload, dict):
            raise KnownSegfaultError(f"{path}: known-segfault ledger must be a JSON object")
        unknown_top = sorted(set(payload) - {"version", "known_segfaults"})
        if unknown_top:
            raise KnownSegfaultError(f"{path}: unknown top-level field(s): {', '.join(unknown_top)}")
        if payload.get("version") != 1:
            raise KnownSegfaultError(f"{path}: version must be 1")
        raw_entries = payload.get("known_segfaults")
        if not isinstance(raw_entries, list):
            raise KnownSegfaultError(f"{path}: known_segfaults must be a list")
        for entry in raw_entries:
            entries.append(_validate_entry(entry, path, seen_ids))
    return entries


def _read_payload(path: Path) -> tuple[str, dict]:
    try:
        return str(path), json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KnownSegfaultError(f"{path}: malformed JSON: {exc}") from exc


def load_known_segfaults(project_root: str | Path = ".") -> list[dict]:
    payloads: list[tuple[str, dict]] = []
    if PACKAGED_KNOWN_SEGFAULTS.exists():
        payloads.append(_read_payload(PACKAGED_KNOWN_SEGFAULTS))

    project_path = Path(project_root) / PROJECT_KNOWN_SEGFAULTS
    if project_path.exists():
        payloads.append(_read_payload(project_path))

    return validate_known_segfaults(payloads)


def active_known_segfaults(
    entries: Iterable[dict],
    *,
    backend: str,
    torch_version: str,
    hardware_key: str,
) -> list[dict]:
    active: list[dict] = []
    for entry in entries:
        if entry["backend"] != backend:
            continue
        if entry["hardware"] != "any" and entry["hardware"] != hardware_key:
            continue
        if not version_in_range(torch_version, entry["pytorch_min"], entry["pytorch_max"]):
            continue
        active.append(dict(entry))
    return active


def canonicalize_nodeid(nodeid: str) -> str:
    """Return a package-relative node id stable across source and wheel installs."""

    text = str(nodeid).replace("\\", "/")
    path, sep, suffix = text.partition("::")
    while path.startswith("./"):
        path = path[2:]
    marker = "/torchcts/"
    if marker in path:
        path = "torchcts/" + path.rsplit(marker, 1)[1]
    return f"{path}{sep}{suffix}"


def _matched_metadata(metadata: dict | None) -> dict:
    metadata = metadata or {}
    return {key: metadata.get(key) for key in MATCH_METADATA_KEYS if metadata.get(key) is not None}


def _constraint_value(metadata: dict | None, key: str, nodeid: str):
    metadata = metadata or {}
    if key == "nodeid_glob":
        return nodeid
    if key == "coverage_id_glob":
        return metadata.get("coverage_id")
    return metadata.get(key)


def constraints_match(entry: dict, nodeid: str, metadata: dict | None = None) -> bool:
    canonical_nodeid = canonicalize_nodeid(nodeid)
    for key, expected_values in (entry.get("constraints") or {}).items():
        actual = _constraint_value(metadata, key, canonical_nodeid)
        if actual is None:
            return False
        if key in GLOB_CONSTRAINT_KEYS:
            if not any(fnmatch.fnmatch(str(actual), pattern) for pattern in expected_values):
                return False
        elif key in SEMANTIC_CONSTRAINT_KEYS:
            if actual not in expected_values:
                return False
        elif str(actual) not in expected_values:
            return False
    return True


def entry_matches(
    entry: dict,
    nodeid: str,
    metadata: dict | None = None,
    *,
    include_constraints: bool = True,
) -> bool:
    canonical_nodeid = canonicalize_nodeid(nodeid)
    metadata = metadata or {}
    match_mode = entry.get("match")
    if match_mode == "nodeid":
        primary = canonicalize_nodeid(entry.get("nodeid", "")) == canonical_nodeid
    elif match_mode == "dispatcher":
        primary = _dispatcher_matches(entry.get("dispatcher"), metadata.get("dispatcher_name"))
    elif match_mode == "coverage_id":
        coverage_id = metadata.get("coverage_id")
        primary = bool(entry.get("coverage_id") and coverage_id == entry.get("coverage_id"))
        if not primary:
            primary = any(
                fnmatch.fnmatch(str(coverage_id or ""), pattern)
                for pattern in (entry.get("constraints") or {}).get("coverage_id_glob", [])
            )
    else:
        primary = False
    if not primary:
        return False
    if include_constraints and not constraints_match(entry, canonical_nodeid, metadata):
        return False
    return True


def match_specificity(entry: dict) -> tuple[int, int, int, int]:
    constraints = entry.get("constraints") or {}
    base = {"nodeid": 300, "dispatcher": 200, "coverage_id": 100}.get(entry.get("match"), 0)
    exact_count = sum(1 for key in constraints if key not in GLOB_CONSTRAINT_KEYS)
    glob_count = sum(1 for key in constraints if key in GLOB_CONSTRAINT_KEYS)
    return (base, len(constraints), exact_count, -glob_count)


def _match_sort_key(entry: dict) -> tuple[int, int, int, int, str]:
    return (*match_specificity(entry), entry.get("id", ""))


def annotate_match(entry: dict, nodeid: str, metadata: dict | None = None) -> dict:
    annotated = dict(entry)
    annotated["matched_by"] = entry["match"]
    annotated["matched_nodeid"] = canonicalize_nodeid(nodeid)
    annotated["matched_metadata"] = _matched_metadata(metadata)
    annotated["constraints"] = dict(entry.get("constraints") or {})
    annotated["evidence_scope"] = entry["evidence_scope"]
    return annotated


def matching_known_segfaults(
    nodeid: str,
    active_entries: Iterable[dict],
    *,
    metadata: dict | None = None,
) -> list[dict]:
    matches = [
        annotate_match(entry, nodeid, metadata)
        for entry in active_entries
        if entry_matches(entry, nodeid, metadata)
    ]
    return sorted(matches, key=_match_sort_key, reverse=True)


def best_known_segfault_match(
    nodeid: str,
    active_entries: Iterable[dict],
    *,
    metadata: dict | None = None,
) -> dict | None:
    matches = matching_known_segfaults(nodeid, active_entries, metadata=metadata)
    if not matches:
        return None
    top = matches[0]
    top_score = match_specificity(top)
    tied = [match for match in matches if match_specificity(match) == top_score]
    if len(tied) > 1:
        ids = ", ".join(match["id"] for match in tied)
        raise KnownSegfaultError(
            f"ambiguous known segfault match for {canonicalize_nodeid(nodeid)}: {ids}"
        )
    return top


def match_known_segfault(item, active_entries: Iterable[dict], metadata: dict | None = None) -> dict | None:
    nodeid = canonicalize_nodeid(getattr(item, "nodeid", str(item)))
    if metadata is None:
        metadata = getattr(item, "metadata", None)
    return best_known_segfault_match(nodeid, active_entries, metadata=metadata)


def expired_known_segfault_warnings(
    entries: Iterable[dict],
    *,
    today: _datetime.date | None = None,
) -> list[str]:
    today = today or _datetime.date.today()
    warnings: list[str] = []
    for entry in entries:
        review_after = _parse_date(entry["review_after"], entry.get("source_path", "<memory>"), entry["id"])
        if review_after < today:
            warnings.append(
                f"known segfault {entry['id']} review_after={entry['review_after']} has expired"
            )
    return warnings
