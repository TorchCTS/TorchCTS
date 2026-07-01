#!/usr/bin/env python3
"""Collect raw TorchCTS CPU dtype-contract evidence from the active runtime."""

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
ARTIFACT_KIND = "torch_dtype_contract_probe"
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
    return matrix_root / "raw" / f"torch-{normalized}-{family}-dtype-contracts.json"


def _load_contract_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "metadata": {}, "contracts": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"version": 1, "metadata": {}, "contracts": {}}


def build_raw_artifact(
    *,
    contract_payload: dict[str, Any],
    torch_version: str,
    family: str,
    layers: list[str],
    errors: list[dict[str, str]] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    metadata = contract_payload.get("metadata") or {}
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
            "generated_at": utc_now(),
            "probe_layers": list(layers),
            "selected_dtypes": list(metadata.get("selected_dtypes") or []),
            "version_rule": metadata.get("version_rule"),
        },
        "contracts": dict(contract_payload.get("contracts") or {}),
        "probe_counts": dict(metadata.get("last_run_probe_counts") or {}),
        "contract_counts": dict(metadata.get("contract_counts") or {}),
        "source_extraction": dict(metadata.get("source_extraction") or {}),
        "generator_metadata": metadata,
        "errors": list(errors or []),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generation_command(args, temp_contract_path: Path, version_rule: str) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_op_dtype_contracts.py"),
        "--out",
        str(temp_contract_path),
        "--version-rule",
        version_rule,
        "--timeout",
        str(args.timeout),
        "--summary-every",
        str(args.summary_every),
    ]
    for layer in args.layer or ["source"]:
        cmd.extend(["--layer", layer])
    for dtype in args.dtypes or ():
        cmd.extend(["--dtypes", dtype])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.isolated:
        cmd.append("--isolated")
    if args.quiet:
        cmd.append("--quiet")
    if args.pytorch_src:
        cmd.extend(["--pytorch-src", args.pytorch_src])
    return cmd


def collect_raw_artifact(args) -> tuple[dict[str, Any], int]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch

    normalized_version = normalized_torch_version(getattr(torch, "__version__", "unknown"))
    out = args.out or default_output_path(normalized_version, args.family)
    temp_contract_path = out.with_name(f"{out.stem}.contracts.tmp.json")
    if temp_contract_path.exists():
        temp_contract_path.unlink()

    layers = args.layer or ["source"]
    cmd = _generation_command(args, temp_contract_path, normalized_version)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    errors: list[dict[str, str]] = []
    if result.returncode != 0:
        errors.append({
            "kind": "dtype_contract_generation_failed",
            "detail": (result.stdout or "").strip()[-4000:],
        })
        if not args.allow_partial:
            if temp_contract_path.exists():
                temp_contract_path.unlink()
            return build_raw_artifact(
                contract_payload={"version": 1, "metadata": {}, "contracts": {}},
                torch_version=getattr(torch, "__version__", "unknown"),
                family=args.family,
                layers=layers,
                errors=errors,
            ), result.returncode or 1

    contract_payload = _load_contract_payload(temp_contract_path)
    if temp_contract_path.exists() and not args.keep_intermediate:
        temp_contract_path.unlink()
    return build_raw_artifact(
        contract_payload=contract_payload,
        torch_version=getattr(torch, "__version__", "unknown"),
        family=args.family,
        layers=layers,
        errors=errors,
    ), 0 if not errors else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="cpu", help="PyTorch wheel/build family label.")
    parser.add_argument("--out", type=Path, help="Raw dtype-contract artifact output path.")
    parser.add_argument(
        "--layer",
        action="append",
        choices=("source", "opinfo-forward", "opinfo-backward", "generated", "all"),
        default=None,
        help="contract layer to collect; defaults to source",
    )
    parser.add_argument("--dtypes", action="append", help="dtype or comma-separated dtypes to probe")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--summary-every", type=int, default=250)
    parser.add_argument("--pytorch-src", default=str(REPO_ROOT.parent / "pytorch-src"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    args = parser.parse_args(argv)

    artifact, status = collect_raw_artifact(args)
    out = args.out or default_output_path(artifact["collection"]["normalized_torch_version"], args.family)
    if status == 0 or args.allow_partial:
        write_json(out, artifact)
        print(f"Wrote dtype contract artifact: {out}")
    else:
        for error in artifact.get("errors", []):
            print(error["detail"], file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
