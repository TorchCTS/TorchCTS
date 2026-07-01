# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_package_artifacts.py"
if not SCRIPT_PATH.exists():
    pytest.skip("package artifact checks require a source checkout", allow_module_level=True)

pytestmark = pytest.mark.covers_category("selftest")


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_package_artifacts", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["verify_package_artifacts"] = module
    spec.loader.exec_module(module)
    return module


def _metadata_text() -> str:
    return "Metadata-Version: 2.4\nName: torchcts\nRequires-Dist: torch>=2.7.0,<2.12.2\n"


def _write_wheel(path: Path, members: tuple[str, ...]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("torchcts-0.3.4.dist-info/METADATA", _metadata_text())
        for member in members:
            archive.writestr(member, "{}")
    return path


def _write_sdist(path: Path, members: tuple[str, ...]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        root = "torchcts-0.3.4"
        info = tarfile.TarInfo(f"{root}/PKG-INFO")
        metadata = _metadata_text().encode()
        info.size = len(metadata)
        archive.addfile(info, fileobj=io.BytesIO(metadata))
        for member in members:
            data = b"{}"
            info = tarfile.TarInfo(f"{root}/{member}")
            info.size = len(data)
            archive.addfile(info, fileobj=io.BytesIO(data))
    return path


def test_package_verifier_requires_packaged_install_planner(tmp_path):
    verifier = _load_verifier()
    members = (
        "torchcts/op_dtype_contracts.json",
        "torchcts/op_metadata.json",
        "torchcts/site_scripts/install_plan.py",
    )

    assert verifier.verify_archive(_write_wheel(tmp_path / "ok.whl", members)) == []
    assert verifier.verify_archive(_write_sdist(tmp_path / "ok.tar.gz", members)) == []


def test_package_verifier_rejects_standalone_site_installer_artifacts(tmp_path):
    verifier = _load_verifier()
    members = (
        "torchcts/op_dtype_contracts.json",
        "torchcts/op_metadata.json",
        "torchcts/site_scripts/install_plan.py",
        "site_scripts/install_plan.py",
    )

    errors = verifier.verify_archive(_write_sdist(tmp_path / "bad.tar.gz", members))

    assert any("forbidden standalone site installer artifact" in error for error in errors)
