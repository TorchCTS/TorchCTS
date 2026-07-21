#!/usr/bin/env python3
"""Verify frozen oracle fixtures, evidence, and generator dependency pins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.oracle_fixtures.inventory import _check_reviewed_local, build_inventory
from tests.oracles.schema import (
    FixtureValidationError,
    load_generator_catalog,
    load_case_manifest,
    load_source_catalog,
    resolve_under,
)


CASES_MANIFEST = REPO_ROOT / "tests/oracles/cases/manifest.json"
SOURCE_CATALOG = REPO_ROOT / "evidence/oracles/sources.json"
EVIDENCE_MANIFEST = REPO_ROOT / "evidence/oracles/manifest.json"
EVIDENCE_ROOT = EVIDENCE_MANIFEST.parent
LOCK_PATH = REPO_ROOT / "scripts/oracle_fixtures/requirements-oracles.lock"
REQUIRED_PINS = {
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
    "mpmath": "1.3.0",
    "numpy": "2.5.0",
    "onnx": "1.22.0",
    "scipy": "1.18.0",
    "sympy": "1.14.0",
}


def _verify_lock() -> None:
    text = LOCK_PATH.read_text(encoding="ascii")
    lines = text.splitlines()
    for package, version in REQUIRED_PINS.items():
        if f"{package}=={version} \\" not in lines:
            raise FixtureValidationError(f"generator lock is missing exact pin {package}=={version}")
    # The installer, not this script, owns validation of every transitive wheel
    # hash.  This check only guards against accidentally committing an unhashed
    # lock while avoiding dependence on uv/pip's continuation formatting.
    if "--hash=sha256:" not in text:
        raise FixtureValidationError("generator dependency lock contains no package hashes")


def _verify_evidence_manifest() -> None:
    payload = json.loads(EVIDENCE_MANIFEST.read_text(encoding="ascii"))
    if payload.get("schema_version") != 1:
        raise FixtureValidationError("evidence manifest has unsupported schema_version")
    catalog_path = resolve_under(EVIDENCE_ROOT, payload.get("source_catalog", ""))
    if catalog_path != SOURCE_CATALOG.resolve():
        raise FixtureValidationError("evidence manifest points at the wrong source catalog")
    generator_catalog_path = resolve_under(
        EVIDENCE_ROOT, payload.get("generator_catalog", "")
    )
    if generator_catalog_path != (EVIDENCE_ROOT / "generators.json").resolve():
        raise FixtureValidationError("evidence manifest points at the wrong generator catalog")

    # Raw evidence hashes are owned by source records.  The evidence manifest
    # only enforces reachability, so it does not duplicate those hashes.
    listed = {catalog_path}
    catalog = json.loads(catalog_path.read_text(encoding="ascii"))
    for source in catalog.get("sources", []):
        for artifact in source.get("raw_evidence", []):
            listed.add(resolve_under(EVIDENCE_ROOT, artifact.get("path", "")))
    actual = {
        path.resolve()
        for directory in (EVIDENCE_ROOT / "raw", EVIDENCE_ROOT / "generator-records")
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".bin", ".json", ".jsonl", ".pb", ".txt"}
    }
    orphans = sorted(path.relative_to(EVIDENCE_ROOT).as_posix() for path in actual - listed)
    if orphans:
        raise FixtureValidationError(f"evidence manifest has orphan semantic artifacts: {orphans}")


def verify() -> tuple[int, int]:
    _verify_lock()
    _verify_evidence_manifest()
    sources = load_source_catalog(SOURCE_CATALOG, evidence_root=EVIDENCE_ROOT)
    load_generator_catalog(EVIDENCE_ROOT / "generators.json", repo_root=REPO_ROOT)
    cases = load_case_manifest(
        CASES_MANIFEST,
        source_catalog_path=SOURCE_CATALOG,
        repo_root=REPO_ROOT,
        evidence_root=EVIDENCE_ROOT,
    )
    inventory = build_inventory()
    reviewed_errors = _check_reviewed_local(inventory)
    if reviewed_errors:
        raise FixtureValidationError("; ".join(reviewed_errors))
    return len(cases), len(sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify accepted records read-only")
    args = parser.parse_args()
    if not args.check:
        parser.error("only the read-only --check workflow is implemented")
    try:
        case_count, source_count = verify()
    except (FixtureValidationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"oracle fixture verification failed: {error}", file=sys.stderr)
        return 1
    print(f"oracle fixtures verified: {case_count} cases, {source_count} sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
