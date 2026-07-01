#!/usr/bin/env python3
"""Collect raw ATen dispatcher inventory from the active PyTorch runtime."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ARTIFACT_VERSION = 1
ARTIFACT_KIND = "torch_dispatcher_inventory"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_ROOT = REPO_ROOT / "scratch" / "pytorch-2.7-compat" / "matrix"


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def normalized_torch_version(version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version))
    if match:
        return ".".join(match.groups())
    match = re.match(r"^(\d+)\.(\d+)", str(version))
    if match:
        return f"{match.group(1)}.{match.group(2)}.0"
    return str(version)


def repo_commit(repo_root: Path = REPO_ROOT) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def default_output_path(torch_version: str, family: str, matrix_root: Path = DEFAULT_MATRIX_ROOT) -> Path:
    normalized = normalized_torch_version(torch_version)
    return matrix_root / "raw" / f"torch-{normalized}-{family}-dispatcher.json"


def build_raw_artifact(
    *,
    inventory: dict[str, Any],
    torch_version: str,
    family: str,
    errors: list[dict[str, str]] | None = None,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "version": ARTIFACT_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "collection": {
            "torch_version": str(torch_version),
            "normalized_torch_version": normalized_torch_version(torch_version),
            "wheel_family": family,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "repo_root": str(repo_root),
            "repo_commit": repo_commit(repo_root),
            "generated_at": generated_at or utc_now(),
        },
        "entries": list(inventory.get("entries", [])),
        "errors": list(errors or []),
    }


def collect_raw_artifact(family: str, *, allow_partial: bool = False) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch
    from torchcts.core.coverage import build_dispatcher_inventory

    errors: list[dict[str, str]] = []
    try:
        inventory = build_dispatcher_inventory()
    except Exception as exc:
        if not allow_partial:
            raise
        inventory = {"entries": []}
        errors.append({
            "kind": "dispatcher_inventory_failed",
            "detail": f"{type(exc).__name__}: {exc}",
        })
    return build_raw_artifact(
        inventory=inventory,
        torch_version=getattr(torch, "__version__", "unknown"),
        family=family,
        errors=errors,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="cpu", help="PyTorch wheel/build family label.")
    parser.add_argument("--out", type=Path, help="Raw dispatcher artifact output path.")
    parser.add_argument("--allow-partial", action="store_true", help="Write an error artifact if inventory fails.")
    args = parser.parse_args(argv)

    artifact = collect_raw_artifact(args.family, allow_partial=args.allow_partial)
    out = args.out or default_output_path(artifact["collection"]["normalized_torch_version"], args.family)
    write_json(out, artifact)
    print(f"Wrote dispatcher inventory: {out}")
    return 0 if not artifact.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
