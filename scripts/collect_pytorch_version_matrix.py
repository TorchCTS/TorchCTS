#!/usr/bin/env python3
"""Harvest raw TorchCTS compatibility artifacts in isolated PyTorch venvs."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = REPO_ROOT / "scripts" / "pytorch_version_matrix.json"
DEFAULT_MATRIX_ROOT = REPO_ROOT / "scratch" / "pytorch-2.7-compat" / "matrix"
DEFAULT_ARTIFACTS = ("dispatcher",)
DTYPE_COLLECTION_DEPS = ("pytest>=7.0.0", "psutil>=5.0.0", "numpy", "expecttest")


@dataclass(frozen=True)
class MatrixJob:
    version: str
    family: str
    index_url: str
    labels: tuple[str, ...] = field(default_factory=tuple)
    source_url: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return self.version, self.family


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def load_matrix(path: Path = DEFAULT_MATRIX_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_version_parts(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version))
    if not match:
        raise ValueError(f"Expected semantic PyTorch version, got {version!r}")
    return tuple(int(part) for part in match.groups())


def resolve_latest_patch(minor: str, source_url: str) -> str:
    with urllib.request.urlopen(source_url, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")
    pattern = re.compile(rf"\bv({re.escape(minor)}\.\d+)\b")
    versions = sorted({match.group(1) for match in pattern.finditer(text)}, key=parse_version_parts)
    if not versions:
        raise ValueError(f"Could not resolve latest PyTorch patch for {minor} from {source_url}")
    return versions[-1]


def _job_label(entry: dict[str, Any], version: str) -> str:
    if entry.get("alias"):
        return str(entry["alias"])
    return str(entry.get("version", version))


def plan_jobs(
    matrix: dict[str, Any],
    *,
    selection: str | None = None,
    versions: Iterable[str] = (),
    family: str = "cpu",
    latest_resolver: Callable[[str, str], str] = resolve_latest_patch,
) -> list[MatrixJob]:
    source_url = str(matrix.get("source_url") or "")
    families = matrix.get("families") or {}

    raw_entries: list[dict[str, Any]] = []
    if selection:
        selections = matrix.get("selections") or {}
        if selection not in selections:
            raise ValueError(f"Unknown matrix selection {selection!r}")
        raw_entries.extend(selections[selection])
    for version in versions:
        raw_entries.append({"version": version, "family": family})
    if not raw_entries:
        raise ValueError("No matrix jobs requested; pass --selection or --version")

    jobs_by_key: dict[tuple[str, str], MatrixJob] = {}
    labels_by_key: dict[tuple[str, str], list[str]] = {}
    for entry in raw_entries:
        entry_family = str(entry.get("family") or family)
        if entry_family not in families:
            raise ValueError(f"Unknown PyTorch family {entry_family!r}")
        if entry.get("version"):
            resolved_version = str(entry["version"])
        elif entry.get("minor"):
            resolved_version = latest_resolver(str(entry["minor"]), source_url)
        else:
            raise ValueError(f"Matrix entry needs version or minor alias: {entry!r}")
        key = (resolved_version, entry_family)
        label = _job_label(entry, resolved_version)
        labels_by_key.setdefault(key, []).append(label)
        jobs_by_key[key] = MatrixJob(
            version=resolved_version,
            family=entry_family,
            index_url=str((families[entry_family] or {}).get("index_url") or ""),
            labels=tuple(labels_by_key[key]),
            source_url=source_url,
        )
    return sorted(jobs_by_key.values(), key=lambda job: (parse_version_parts(job.version), job.family))


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def artifact_path(matrix_root: Path, job: MatrixJob) -> Path:
    return matrix_root / "raw" / f"torch-{job.version}-{job.family}-dispatcher.json"


def _artifact_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.replace("torch.", "")).strip("-") or "default"


def dtype_artifact_path(
    matrix_root: Path,
    job: MatrixJob,
    *,
    dtype_layers: tuple[str, ...] = ("source",),
    dtype_values: tuple[str, ...] = (),
    dtype_limit: int = 0,
) -> Path:
    suffix = ""
    if dtype_layers != ("source",) or dtype_values or dtype_limit:
        parts = ["dtype-contracts", "layers-" + "-".join(_artifact_component(layer) for layer in dtype_layers)]
        if dtype_values:
            parts.append("dtypes-" + "-".join(_artifact_component(value) for value in dtype_values))
        if dtype_limit:
            parts.append(f"limit-{dtype_limit}")
        suffix = "-" + "-".join(parts)
        return matrix_root / "raw" / f"torch-{job.version}-{job.family}{suffix}.json"
    return matrix_root / "raw" / f"torch-{job.version}-{job.family}-dtype-contracts.json"


def requested_artifact_paths(
    matrix_root: Path,
    job: MatrixJob,
    artifacts: tuple[str, ...],
    *,
    dtype_layers: tuple[str, ...] = ("source",),
    dtype_values: tuple[str, ...] = (),
    dtype_limit: int = 0,
) -> list[Path]:
    paths = []
    if "dispatcher" in artifacts:
        paths.append(artifact_path(matrix_root, job))
    if "dtype-contracts" in artifacts:
        paths.append(dtype_artifact_path(
            matrix_root,
            job,
            dtype_layers=dtype_layers,
            dtype_values=dtype_values,
            dtype_limit=dtype_limit,
        ))
    return paths


def log_path(matrix_root: Path, job: MatrixJob) -> Path:
    return matrix_root / "logs" / f"torch-{job.version}-{job.family}.log"


def create_venv(venv_dir: Path, python_executable: str) -> None:
    result = subprocess.run(
        [python_executable, "-m", "venv", str(venv_dir)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"failed to create venv with {python_executable}: {detail}")


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_file: Path,
) -> subprocess.CompletedProcess[str]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(result.stdout or "")
        log.write(f"\n[exit {result.returncode}]\n")
    return result


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def run_job(
    job: MatrixJob,
    *,
    matrix_root: Path,
    skip_existing: bool,
    force: bool,
    keep_venvs: bool,
    clean_venvs: bool,
    dry_run: bool,
    python_executable: str = sys.executable,
    artifacts: tuple[str, ...] = DEFAULT_ARTIFACTS,
    dtype_layers: tuple[str, ...] = ("source",),
    dtype_values: tuple[str, ...] = (),
    dtype_limit: int = 0,
    dtype_timeout: float = 8.0,
    dtype_isolated: bool = False,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_command,
    venv_creator: Callable[[Path, str], None] = create_venv,
) -> dict[str, Any]:
    venv_dir = matrix_root / "venvs" / f"torch-{job.version}-{job.family}"
    artifact_paths = requested_artifact_paths(
        matrix_root,
        job,
        artifacts,
        dtype_layers=dtype_layers,
        dtype_values=dtype_values,
        dtype_limit=dtype_limit,
    )
    out = artifact_path(matrix_root, job)
    log = log_path(matrix_root, job)
    record = {
        "torch_version": job.version,
        "family": job.family,
        "labels": list(job.labels),
        "source_url": job.source_url,
        "venv": str(venv_dir),
        "commands": [],
        "artifact_types": list(artifacts),
        "artifacts": [str(path) for path in artifact_paths],
        "log": str(log),
        "python_executable": python_executable,
        "status": "pending",
        "generated_at": utc_now(),
    }

    if artifact_paths and all(path.exists() for path in artifact_paths) and skip_existing and not force:
        record["status"] = "skipped_existing"
        return record

    if clean_venvs and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if dry_run:
        record["status"] = "dry_run"
        return record

    venv_creator(venv_dir, python_executable)
    python = venv_python(venv_dir)
    env = _base_env()
    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [str(python), "-m", "pip", "install", f"torch=={job.version}"],
    ]
    if "dtype-contracts" in artifacts:
        commands.append([str(python), "-m", "pip", "install", *DTYPE_COLLECTION_DEPS])
    if "dispatcher" in artifacts:
        commands.append([
            str(python),
            str(REPO_ROOT / "scripts" / "collect_pytorch_ops.py"),
            "--family",
            job.family,
            "--out",
            str(out),
        ])
    if "dtype-contracts" in artifacts:
        dtype_cmd = [
            str(python),
            str(REPO_ROOT / "scripts" / "collect_pytorch_dtype_contracts.py"),
            "--family",
            job.family,
            "--out",
            str(dtype_artifact_path(
                matrix_root,
                job,
                dtype_layers=dtype_layers,
                dtype_values=dtype_values,
                dtype_limit=dtype_limit,
            )),
            "--timeout",
            str(dtype_timeout),
            "--quiet",
        ]
        for layer in dtype_layers:
            dtype_cmd.extend(["--layer", layer])
        for dtype_value in dtype_values:
            dtype_cmd.extend(["--dtypes", dtype_value])
        if dtype_limit:
            dtype_cmd.extend(["--limit", str(dtype_limit)])
        if dtype_isolated:
            dtype_cmd.append("--isolated")
        commands.append(dtype_cmd)
    if job.index_url:
        commands[1].extend(["--index-url", job.index_url])

    for cmd in commands:
        record["commands"].append(cmd)
        result = command_runner(cmd, cwd=REPO_ROOT, env=env, log_file=log)
        if result.returncode != 0:
            record["status"] = "failed"
            record["returncode"] = result.returncode
            if not keep_venvs and venv_dir.exists():
                shutil.rmtree(venv_dir)
            return record

    record["status"] = "passed"
    if not keep_venvs and venv_dir.exists():
        shutil.rmtree(venv_dir)
    return record


def write_manifest(matrix_root: Path, records: list[dict[str, Any]]) -> Path:
    matrix_root.mkdir(parents=True, exist_ok=True)
    path = matrix_root / "run-manifest.json"
    payload = {
        "version": 1,
        "generated_at": utc_now(),
        "runs": records,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_matrix(
    jobs: list[MatrixJob],
    *,
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
    skip_existing: bool = False,
    force: bool = False,
    keep_venvs: bool = False,
    clean_venvs: bool = False,
    dry_run: bool = False,
    python_executable: str = sys.executable,
    artifacts: tuple[str, ...] = DEFAULT_ARTIFACTS,
    dtype_layers: tuple[str, ...] = ("source",),
    dtype_values: tuple[str, ...] = (),
    dtype_limit: int = 0,
    dtype_timeout: float = 8.0,
    dtype_isolated: bool = False,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_command,
    venv_creator: Callable[[Path, str], None] = create_venv,
) -> list[dict[str, Any]]:
    records = []
    for job in jobs:
        records.append(
            run_job(
                job,
                matrix_root=matrix_root,
                skip_existing=skip_existing,
                force=force,
                keep_venvs=keep_venvs,
                clean_venvs=clean_venvs,
                dry_run=dry_run,
                python_executable=python_executable,
                artifacts=artifacts,
                dtype_layers=dtype_layers,
                dtype_values=dtype_values,
                dtype_limit=dtype_limit,
                dtype_timeout=dtype_timeout,
                dtype_isolated=dtype_isolated,
                command_runner=command_runner,
                venv_creator=venv_creator,
            )
        )
    write_manifest(matrix_root, records)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--selection")
    parser.add_argument("--version", action="append", default=[])
    parser.add_argument("--family", default="cpu")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-venvs", action="store_true")
    parser.add_argument("--clean-venvs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--artifact",
        action="append",
        choices=("dispatcher", "dtype-contracts"),
        default=None,
        help="artifact type to collect; may be passed more than once; defaults to dispatcher",
    )
    parser.add_argument(
        "--dtype-layer",
        action="append",
        choices=("source", "opinfo-forward", "opinfo-backward", "generated", "all"),
        default=None,
        help="dtype-contract layer to collect when --artifact dtype-contracts is requested; defaults to source",
    )
    parser.add_argument("--dtype", action="append", default=[], help="dtype or comma-separated dtypes for dtype probes")
    parser.add_argument("--dtype-limit", type=int, default=0, help="limit probes per dtype-contract probing layer")
    parser.add_argument("--dtype-timeout", type=float, default=8.0, help="per-probe dtype-contract timeout")
    parser.add_argument("--dtype-isolated", action="store_true", help="run dtype probes in subprocesses")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to create isolated collection venvs.",
    )
    args = parser.parse_args(argv)

    try:
        jobs = plan_jobs(
            load_matrix(args.matrix),
            selection=args.selection,
            versions=args.version,
            family=args.family,
        )
        records = run_matrix(
            jobs,
            matrix_root=args.matrix_root,
            skip_existing=args.skip_existing,
            force=args.force,
            keep_venvs=args.keep_venvs,
            clean_venvs=args.clean_venvs,
            dry_run=args.dry_run,
            python_executable=args.python,
            artifacts=tuple(args.artifact or DEFAULT_ARTIFACTS),
            dtype_layers=tuple(args.dtype_layer or ("source",)),
            dtype_values=tuple(args.dtype),
            dtype_limit=args.dtype_limit,
            dtype_timeout=args.dtype_timeout,
            dtype_isolated=args.dtype_isolated,
        )
    except Exception as exc:
        print(f"matrix collection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    for record in records:
        labels = ",".join(record.get("labels") or [])
        print(f"{record['torch_version']} {record['family']} {record['status']} {labels}")
    return 1 if any(record["status"] == "failed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
