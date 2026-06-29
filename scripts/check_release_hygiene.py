#!/usr/bin/env python3
"""Release hygiene checks for paths that must never be committed or shipped."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


DENIED_COMPONENTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "reports",
    "results",
    "scratch",
}

DENIED_FILENAMES = {
    ".DS_Store",
    "createpackage.sh",
    "createvenv.sh",
    "install-in-site.sh",
    "manifest.py",
    "pushpypi.sh",
    "pypi-token.txt",
}

DENIED_PACKAGE_PATHS = {
    ("torchcts", "opinfo_cache", "known_failures.json"),
}

FORBIDDEN_TEXT_TOKENS = {
    "metal" "core",
}


def _run_git(args: list[str], repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return [line for line in result.stdout.splitlines() if line]


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "not inside a git repository")
    return Path(result.stdout.strip())


def _is_denied_path(path: str, *, artifact: bool = False) -> str | None:
    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        return None
    pure = PurePosixPath(normalized)
    parts = pure.parts
    lower_name = pure.name.lower()

    for part in parts:
        if part in DENIED_COMPONENTS:
            return f"denied component {part!r}"
        if part.endswith(".egg-info") and not artifact:
            return "local egg-info metadata"

    if pure.name in DENIED_FILENAMES:
        return f"denied filename {pure.name!r}"
    for denied_parts in DENIED_PACKAGE_PATHS:
        if len(parts) >= len(denied_parts) and tuple(parts[-len(denied_parts):]) == denied_parts:
            return "runtime known-failure cache must not be shipped"
    if lower_name.startswith(".env"):
        return "environment file"
    if lower_name.endswith((".pem", ".key")):
        return "private key/certificate material"
    if "token" in lower_name:
        return "token-like filename"
    if lower_name.endswith((".pyc", ".pyo")):
        return "bytecode artifact"
    return None


def _check_git_paths(repo: Path) -> list[str]:
    errors: list[str] = []
    denied_cache = repo / "torchcts" / "opinfo_cache" / "known_failures.json"
    if denied_cache.exists():
        errors.append("denied package file exists: torchcts/opinfo_cache/known_failures.json")

    ignored_tracked = _run_git(["ls-files", "-ci", "--exclude-standard"], repo)
    for path in ignored_tracked:
        errors.append(f"tracked ignored file: {path}")

    tracked = _run_git(["ls-files"], repo)
    staged = _run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"], repo)
    for label, paths in (("tracked", tracked), ("staged", staged)):
        for path in paths:
            reason = _is_denied_path(path)
            if reason:
                errors.append(f"{label} denied path: {path} ({reason})")
    return errors


def _is_text_bytes(data: bytes) -> bool:
    return b"\0" not in data


def _check_forbidden_text(repo: Path) -> list[str]:
    errors: list[str] = []
    tracked = _run_git(["ls-files"], repo)
    lowered_tokens = [token.encode("utf-8").lower() for token in FORBIDDEN_TEXT_TOKENS]
    for rel in tracked:
        path = repo / rel
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            continue
        except Exception as exc:
            errors.append(f"could not scan tracked file {rel}: {type(exc).__name__}: {exc}")
            continue
        if not _is_text_bytes(data):
            continue
        lowered = data.lower()
        for token in lowered_tokens:
            if token in lowered:
                errors.append(f"tracked file contains forbidden text {token.decode('utf-8')!r}: {rel}")
    return errors


def _artifact_members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with ZipFile(path) as archive:
            return archive.namelist()
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path) as archive:
            return archive.getnames()
    return []


def _check_artifacts(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not path.exists():
            errors.append(f"artifact does not exist: {path}")
            continue
        for member in _artifact_members(path):
            reason = _is_denied_path(member, artifact=True)
            if reason:
                errors.append(f"artifact {path} contains denied path: {member} ({reason})")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="*",
        type=Path,
        help="Optional wheel/sdist files to inspect for denied local artifacts.",
    )
    args = parser.parse_args(argv)

    try:
        repo = _repo_root()
        errors = _check_git_paths(repo)
        errors.extend(_check_forbidden_text(repo))
        errors.extend(_check_artifacts(args.artifacts))
    except Exception as exc:
        print(f"release hygiene check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        return 1
    print("Release hygiene checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
