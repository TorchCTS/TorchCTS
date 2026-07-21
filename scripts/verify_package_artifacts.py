#!/usr/bin/env python3
"""Verify TorchCTS wheel and sdist archive contents."""

from __future__ import annotations

import argparse
import email
import re
import sys
import tarfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_README = REPO_ROOT / "README.md"
REQUIRED_PACKAGE_FILES = (
    "torchcts/op_dtype_contracts.json",
    "torchcts/op_metadata.json",
    "torchcts/site_scripts/install_plan.py",
)
FORBIDDEN_FRAGMENTS = (
    "op_dtype_contract_evidence.jsonl",
    "evidence/",
    "results/",
    "scratch/",
    "scripts/oracle_fixtures/",
    "tests/oracles/",
)
FORBIDDEN_LOGICAL_PATHS = (
    "site_scripts/install_plan.py",
    "site_scripts/install.sh",
    "site_scripts/install.ps1",
)
TORCH_REQUIREMENT_RE = re.compile(r"^Requires-Dist:\s*torch(?P<constraints>.*)$", re.MULTILINE)
TORCH_UPPER_BOUND_RE = re.compile(r"<\s*\d+\.\d+\.\d+")


def _normalized_members(path: Path) -> tuple[list[str], str, str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            metadata_text = ""
            for name in members:
                if name.endswith(".dist-info/METADATA"):
                    metadata_text = archive.read(name).decode("utf-8", errors="replace")
                    break
        readme_text = str(email.message_from_string(metadata_text).get_payload())
        return members, metadata_text, readme_text
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path) as archive:
            members = archive.getnames()
            metadata_text = ""
            readme_text = ""
            for member in archive.getmembers():
                if member.name.endswith("PKG-INFO"):
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        metadata_text = extracted.read().decode("utf-8", errors="replace")
                if _logical_member(path, member.name) == "README.md":
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        readme_text = extracted.read().decode("utf-8", errors="replace")
        return members, metadata_text, readme_text
    raise ValueError(f"Unsupported artifact type: {path}")


def _contains_suffix(members: list[str], suffix: str) -> bool:
    return any(member.endswith(suffix) for member in members)


def _logical_member(path: Path, member: str) -> str:
    normalized = member.replace("\\", "/").strip("/")
    if path.suffix == ".whl":
        return normalized
    parts = normalized.split("/", 1)
    return parts[1] if len(parts) == 2 else normalized


def verify_archive(path: Path, *, expected_readme: str | None = None) -> list[str]:
    errors: list[str] = []
    members, metadata_text, packaged_readme = _normalized_members(path)
    for required in REQUIRED_PACKAGE_FILES:
        if not _contains_suffix(members, required):
            errors.append(f"{path}: missing {required}")
    for member in members:
        normalized = member.replace("\\", "/")
        for fragment in FORBIDDEN_FRAGMENTS:
            if fragment in normalized:
                errors.append(f"{path}: forbidden evidence artifact included as {member}")
        logical = _logical_member(path, member)
        if logical in FORBIDDEN_LOGICAL_PATHS:
            errors.append(f"{path}: forbidden standalone site installer artifact included as {member}")
    torch_requirement = TORCH_REQUIREMENT_RE.search(metadata_text or "")
    if not torch_requirement:
        errors.append(f"{path}: missing torch dependency")
    else:
        constraints = torch_requirement.group("constraints")
        if ">=2.7.0" not in constraints or not TORCH_UPPER_BOUND_RE.search(constraints):
            errors.append(f"{path}: missing bounded torch dependency")
    if expected_readme is not None and packaged_readme != expected_readme:
        errors.append(f"{path}: packaged README does not match the current root README.md")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    args = parser.parse_args(argv)

    errors: list[str] = []
    expected_readme = args.readme.read_text(encoding="utf-8")
    for artifact in args.artifacts:
        try:
            errors.extend(verify_archive(artifact, expected_readme=expected_readme))
        except Exception as exc:
            errors.append(f"{artifact}: {type(exc).__name__}: {exc}")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    for artifact in args.artifacts:
        print(f"{artifact}: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
