#!/usr/bin/env python3
"""Update TorchCTS PyTorch compatibility artifacts end to end."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_pytorch_version_holes
from scripts import collect_pytorch_version_matrix
from scripts import reduce_pytorch_dtype_contracts


DEFAULT_MATRIX = REPO_ROOT / "scripts" / "pytorch_version_matrix.json"
DEFAULT_MATRIX_ROOT = REPO_ROOT / "scratch" / "pytorch-2.7-compat" / "matrix"
DEFAULT_SELECTION = "torch-2.7-through-2.12-cpu"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def run_command(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed with exit {result.returncode}: {' '.join(cmd)}")
    return result


def load_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_matrix(path: Path, matrix: dict[str, Any]) -> None:
    path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_version_to_matrix(path: Path, *, selection: str, version: str, family: str) -> None:
    matrix = load_matrix(path)
    selections = matrix.setdefault("selections", {})
    entries = selections.setdefault(selection, [])
    for entry in entries:
        if entry.get("version") == version and entry.get("family", family) == family:
            return
    entries.append({"version": version, "family": family})
    entries.sort(
        key=lambda entry: (
            collect_pytorch_version_matrix.parse_version_parts(str(entry.get("version") or "9999.9999.9999")),
            str(entry.get("family") or ""),
        )
    )
    write_matrix(path, matrix)


def update_pyproject_torch_bound(upper_bound: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    replacement = f'"torch>=2.7.0,<{upper_bound}"'
    updated, count = re.subn(r'"torch>=2\.7\.0(?:,<[^"]+)?"', replacement, text)
    if count != 1:
        raise RuntimeError("could not update pyproject torch dependency bound")
    PYPROJECT.write_text(updated, encoding="utf-8")


def _artifact_paths(
    jobs: list[collect_pytorch_version_matrix.MatrixJob],
    matrix_root: Path,
    dtype_layers: tuple[str, ...],
) -> tuple[list[Path], list[Path]]:
    dispatcher_paths = [
        collect_pytorch_version_matrix.artifact_path(matrix_root, job)
        for job in jobs
    ]
    dtype_paths = [
        collect_pytorch_version_matrix.dtype_artifact_path(
            matrix_root,
            job,
            dtype_layers=dtype_layers,
        )
        for job in jobs
    ]
    return dispatcher_paths, dtype_paths


def check_version_holes(matrix_path: Path, selection: str, available_versions: list[str]) -> dict[str, Any]:
    matrix = check_pytorch_version_holes.load_matrix(matrix_path)
    available = check_pytorch_version_holes.resolve_available_versions(matrix, available_versions)
    return check_pytorch_version_holes.find_version_holes(
        matrix,
        selection=selection,
        available_versions=available,
        exclusions=check_pytorch_version_holes.load_exclusions(check_pytorch_version_holes.DEFAULT_EXCLUSIONS),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--add-version")
    parser.add_argument("--family", default="cpu")
    parser.add_argument("--update-tracked", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max-runtime-bytes", type=int, default=2_000_000)
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-selftests", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--available-version", action="append", default=[])
    args = parser.parse_args(argv)

    try:
        if args.add_version:
            add_version_to_matrix(
                args.matrix,
                selection=args.selection,
                version=args.add_version,
                family=args.family,
            )

        hole_report = check_version_holes(args.matrix, args.selection, args.available_version)
        if not hole_report.get("ok"):
            print(json.dumps(hole_report, indent=2, sort_keys=True))
            if args.update_tracked:
                raise RuntimeError("unresolved PyTorch version holes block tracked artifact update")

        matrix = load_matrix(args.matrix)
        jobs = collect_pytorch_version_matrix.plan_jobs(matrix, selection=args.selection)
        dtype_layers = ("all",)
        if not args.skip_collection:
            run_command([
                sys.executable,
                "scripts/collect_pytorch_version_matrix.py",
                "--matrix",
                str(args.matrix),
                "--matrix-root",
                str(args.matrix_root),
                "--selection",
                args.selection,
                "--artifact",
                "dispatcher",
                "--artifact",
                "dtype-contracts",
                "--dtype-layer",
                "all",
                "--skip-existing",
            ])

        dispatcher_paths, dtype_paths = _artifact_paths(jobs, args.matrix_root, dtype_layers)
        run_command([
            sys.executable,
            "scripts/reduce_pytorch_op_inventory.py",
            *[str(path) for path in dispatcher_paths],
            "--update-tracked" if args.update_tracked else "--out",
            str(REPO_ROOT / "scratch" / "pytorch-2.7-compat" / "matrix" / "reduced" / "op_metadata.preview.json"),
            "--compact",
        ] if not args.update_tracked else [
            sys.executable,
            "scripts/reduce_pytorch_op_inventory.py",
            *[str(path) for path in dispatcher_paths],
            "--update-tracked",
            "--compact",
        ])
        run_command([
            sys.executable,
            "scripts/reduce_pytorch_dtype_contracts.py",
            *[str(path) for path in dtype_paths],
            "--no-existing-contracts",
            "--verify-equivalence",
            "--max-runtime-bytes",
            str(args.max_runtime_bytes),
            "--update-tracked" if args.update_tracked else "--runtime-out",
            str(REPO_ROOT / "scratch" / "pytorch-2.7-compat" / "matrix" / "reduced" / "op_dtype_contracts.preview.json"),
        ] if not args.update_tracked else [
            sys.executable,
            "scripts/reduce_pytorch_dtype_contracts.py",
            *[str(path) for path in dtype_paths],
            "--no-existing-contracts",
            "--verify-equivalence",
            "--max-runtime-bytes",
            str(args.max_runtime_bytes),
            "--update-tracked",
        ])

        runtime = reduce_pytorch_dtype_contracts.load_existing_contracts(
            reduce_pytorch_dtype_contracts.TRACKED_RUNTIME_OUTPUT
            if args.update_tracked
            else REPO_ROOT / "scratch" / "pytorch-2.7-compat" / "matrix" / "reduced" / "op_dtype_contracts.preview.json"
        )
        upper_bound = (runtime.get("metadata") or {}).get("dependency_upper_bound")
        if args.update_tracked and upper_bound:
            update_pyproject_torch_bound(str(upper_bound))

        if args.verify:
            run_command([
                sys.executable,
                "scripts/verify_pytorch_dtype_contract_artifacts.py",
                "--max-runtime-bytes",
                str(args.max_runtime_bytes),
            ])
            if not args.skip_selftests:
                run_command([sys.executable, "-m", "pytest", "torchcts/selftest"])
            if not args.skip_build:
                run_command([sys.executable, "-m", "build"])
                artifacts = [str(path) for path in sorted((REPO_ROOT / "dist").glob("*")) if path.suffix in {".whl", ".gz"}]
                if artifacts:
                    run_command([sys.executable, "scripts/verify_package_artifacts.py", *artifacts])

        summary = {
            "versions": [job.version for job in jobs],
            "version_holes_ok": hole_report.get("ok"),
            "runtime_bytes": reduce_pytorch_dtype_contracts.TRACKED_RUNTIME_OUTPUT.stat().st_size
            if args.update_tracked and reduce_pytorch_dtype_contracts.TRACKED_RUNTIME_OUTPUT.exists()
            else None,
            "torch_dependency_upper_bound": upper_bound,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"compatibility update failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
