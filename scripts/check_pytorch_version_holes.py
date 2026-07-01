#!/usr/bin/env python3
"""Check for missing stable PyTorch patch versions in the validated matrix."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "scripts" / "pytorch_version_matrix.json"
DEFAULT_EXCLUSIONS = REPO_ROOT / "scripts" / "pytorch_version_hole_exclusions.json"
_VERSION_RE = re.compile(r"(?<![\w.])v?(\d+\.\d+\.\d+)(?![\w.])")


def parse_version_parts(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", str(version))
    if not match:
        raise ValueError(f"Expected semantic version, got {version!r}")
    return tuple(int(part) for part in match.groups())


def normalize_version(version: str) -> str:
    major, minor, patch = parse_version_parts(version)
    return f"{major}.{minor}.{patch}"


def minor_key(version: str) -> str:
    major, minor, _patch = parse_version_parts(version)
    return f"{major}.{minor}"


def stable_versions_from_text(text: str) -> set[str]:
    versions = set()
    for match in _VERSION_RE.finditer(text):
        version = match.group(1)
        try:
            versions.add(normalize_version(version))
        except ValueError:
            continue
    return versions


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def load_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_exclusions(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    excluded = set()
    for record in data.get("exclusions") or ():
        version = record.get("version")
        family = record.get("family")
        reason = record.get("reason")
        source = record.get("source")
        checked_at = record.get("checked_at")
        if version and family and reason and source and checked_at:
            excluded.add((normalize_version(str(version)), str(family)))
    return excluded


def selected_entries(matrix: dict[str, Any], selection: str | None = None) -> list[dict[str, Any]]:
    if selection:
        return list((matrix.get("selections") or {}).get(selection) or [])
    entries: list[dict[str, Any]] = []
    for selection_entries in (matrix.get("selections") or {}).values():
        entries.extend(selection_entries or [])
    return entries


def matrix_versions_by_family(
    matrix: dict[str, Any],
    *,
    selection: str | None,
    available_versions: set[str],
) -> dict[str, set[str]]:
    by_family: dict[str, set[str]] = {}
    for entry in selected_entries(matrix, selection):
        family = str(entry.get("family") or "cpu")
        version = entry.get("version")
        if version:
            by_family.setdefault(family, set()).add(normalize_version(str(version)))
            continue
        minor = entry.get("minor")
        if minor:
            candidates = [version for version in available_versions if minor_key(version) == str(minor)]
            if candidates:
                by_family.setdefault(family, set()).add(sorted(candidates, key=parse_version_parts)[-1])
    return by_family


def _versions_in_range(versions: set[str], minimum: str, maximum: str) -> list[str]:
    min_key = parse_version_parts(minimum)
    max_key = parse_version_parts(maximum)
    return [
        version
        for version in sorted(versions, key=parse_version_parts)
        if min_key <= parse_version_parts(version) <= max_key
    ]


def find_version_holes(
    matrix: dict[str, Any],
    *,
    selection: str | None = None,
    available_versions: set[str],
    exclusions: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    exclusions = exclusions or set()
    matrix_by_family = matrix_versions_by_family(
        matrix,
        selection=selection,
        available_versions=available_versions,
    )
    families = {}
    unresolved = []
    for family, matrix_versions in sorted(matrix_by_family.items()):
        if not matrix_versions:
            continue
        minimum = sorted(matrix_versions, key=parse_version_parts)[0]
        maximum = sorted(matrix_versions, key=parse_version_parts)[-1]
        available_in_range = set(_versions_in_range(available_versions, minimum, maximum))
        holes = sorted(available_in_range - matrix_versions, key=parse_version_parts)
        excluded_holes = [version for version in holes if (version, family) in exclusions]
        unresolved_holes = [version for version in holes if (version, family) not in exclusions]
        unresolved.extend({"version": version, "family": family} for version in unresolved_holes)
        families[family] = {
            "min": minimum,
            "max": maximum,
            "matrix_versions": sorted(matrix_versions, key=parse_version_parts),
            "available_versions": sorted(available_in_range, key=parse_version_parts),
            "excluded_holes": excluded_holes,
            "unresolved_holes": unresolved_holes,
        }
    return {
        "ok": not unresolved,
        "families": families,
        "unresolved": unresolved,
    }


def resolve_available_versions(matrix: dict[str, Any], explicit_versions: list[str]) -> set[str]:
    versions = {normalize_version(version) for version in explicit_versions}
    urls = []
    source_url = matrix.get("source_url")
    if source_url:
        urls.append(str(source_url))
    for family in (matrix.get("families") or {}).values():
        index_url = family.get("index_url") if isinstance(family, dict) else None
        if index_url:
            urls.append(str(index_url))
    for url in dict.fromkeys(urls):
        versions.update(stable_versions_from_text(fetch_text(url)))
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--selection")
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--available-version", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        matrix = load_matrix(args.matrix)
        available_versions = resolve_available_versions(matrix, args.available_version)
        if not available_versions:
            raise RuntimeError("could not resolve any available PyTorch versions")
        result = find_version_holes(
            matrix,
            selection=args.selection,
            available_versions=available_versions,
            exclusions=load_exclusions(args.exclusions),
        )
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "families": {}, "unresolved": []}

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if result.get("error"):
            print(result["error"], file=sys.stderr)
        for family, info in (result.get("families") or {}).items():
            print(f"{family}: {info['min']} through {info['max']}")
            if info["unresolved_holes"]:
                print("  unresolved holes: " + ", ".join(info["unresolved_holes"]))
            if info["excluded_holes"]:
                print("  excluded holes: " + ", ".join(info["excluded_holes"]))
            if not info["unresolved_holes"] and not info["excluded_holes"]:
                print("  holes: none")
        print(f"ok: {str(bool(result.get('ok'))).lower()}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
