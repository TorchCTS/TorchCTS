#!/usr/bin/env python3
# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Plan the PyTorch wheel family for TorchCTS installation.

This module is intentionally stdlib-only at import time. The planner runs before
PyTorch is installed, so do not add top-level third-party imports here.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence


TORCH_SPEC = "torch>=2.12.0"
INDEX_URLS = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda": "https://download.pytorch.org/whl/cu128",
    "rocm": "https://download.pytorch.org/whl/rocm7.2",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "mps": "",
}
DEVICE_HINTS = {
    "cpu": "cpu",
    "cuda": "cuda",
    "rocm": "cuda",
    "xpu": "xpu",
    "mps": "mps",
}
VARIANT_ALIASES = {
    "amd": "rocm",
    "hip": "rocm",
    "intel": "xpu",
    "nvidia": "cuda",
}
OVERRIDE_HELP = "TORCHCTS_TORCH_VARIANT=cuda|rocm|xpu|cpu"


@dataclass(frozen=True)
class InstallPlan:
    variant: str
    confidence: str
    torch_index_url: str
    device_hint: str
    reason: str
    warning: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "variant": self.variant,
            "confidence": self.confidence,
            "torch_index_url": self.torch_index_url,
            "device_hint": self.device_hint,
            "reason": self.reason,
            "warning": self.warning,
        }


@dataclass
class ProbeContext:
    platform_name: str
    env: Mapping[str, str]
    which: Callable[[str], str | None]
    path_exists: Callable[[str], bool]
    listdir: Callable[[str], Iterable[str]]
    read_text: Callable[[str], str]
    run_command: Callable[[Sequence[str]], str]
    pci_root: str = "/sys/bus/pci/devices"


def default_context() -> ProbeContext:
    def _listdir(path: str) -> Iterable[str]:
        return os.listdir(path)

    def _read_text(path: str) -> str:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")

    def _run_command(cmd: Sequence[str]) -> str:
        result = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout

    return ProbeContext(
        platform_name=sys.platform,
        env=os.environ,
        which=shutil.which,
        path_exists=os.path.exists,
        listdir=_listdir,
        read_text=_read_text,
        run_command=_run_command,
    )


def _normalize_variant(value: str) -> str:
    variant = value.strip().lower()
    variant = VARIANT_ALIASES.get(variant, variant)
    if variant not in INDEX_URLS:
        valid = ", ".join(sorted(INDEX_URLS))
        raise ValueError(f"Invalid TORCHCTS_TORCH_VARIANT={value!r}; expected one of: {valid}")
    return variant


def _plan_for_variant(
    variant: str,
    confidence: str,
    reason: str,
    warning: str = "",
) -> InstallPlan:
    return InstallPlan(
        variant=variant,
        confidence=confidence,
        torch_index_url=INDEX_URLS[variant],
        device_hint=DEVICE_HINTS[variant],
        reason=reason,
        warning=warning,
    )


def _is_linux(platform_name: str) -> bool:
    return platform_name.startswith("linux")


def _is_windows(platform_name: str) -> bool:
    return platform_name.startswith(("win32", "cygwin", "msys"))


def _has_tool(ctx: ProbeContext, names: Sequence[str]) -> bool:
    return any(ctx.which(name) is not None for name in names)


def _read_optional(ctx: ProbeContext, path: str) -> str:
    try:
        return ctx.read_text(path)
    except OSError:
        return ""


def _linux_pci_gpu_vendors(ctx: ProbeContext) -> set[str]:
    vendors: set[str] = set()
    if not _is_linux(ctx.platform_name) or not ctx.path_exists(ctx.pci_root):
        return vendors

    try:
        device_names = list(ctx.listdir(ctx.pci_root))
    except OSError:
        return vendors

    for name in device_names:
        base = os.path.join(ctx.pci_root, name)
        vendor_text = _read_optional(ctx, os.path.join(base, "vendor")).strip().lower()
        if not vendor_text:
            continue
        class_text = _read_optional(ctx, os.path.join(base, "class")).strip().lower()
        if class_text:
            try:
                class_value = int(class_text, 16)
            except ValueError:
                class_value = 0
            # PCI class 0x03 covers VGA, 3D, and display controllers.
            if (class_value >> 16) != 0x03:
                continue
        vendors.add(vendor_text.removeprefix("0x"))
    return vendors


def _windows_video_names(ctx: ProbeContext) -> str:
    if not _is_windows(ctx.platform_name):
        return ""
    powershell = ctx.which("powershell") or ctx.which("pwsh")
    if powershell is not None:
        output = ctx.run_command(
            [
                powershell,
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object -ExpandProperty Name",
            ]
        )
        if output.strip():
            return output
    wmic = ctx.which("wmic")
    if wmic is not None:
        return ctx.run_command(["wmic", "path", "win32_VideoController", "get", "name"])
    return ""


def _windows_vendor_flags(ctx: ProbeContext) -> dict[str, bool]:
    text = _windows_video_names(ctx).lower()
    return {
        "cuda": any(token in text for token in ("nvidia", "geforce", "quadro", "tesla")),
        "rocm": any(token in text for token in ("amd", "radeon", "ati")),
        "xpu": any(token in text for token in ("intel", "arc graphics", "iris xe")),
    }


def _detect_signals(ctx: ProbeContext) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    strong: dict[str, list[str]] = {}
    weak: dict[str, list[str]] = {}

    def add(bucket: dict[str, list[str]], variant: str, reason: str) -> None:
        bucket.setdefault(variant, []).append(reason)

    pci_vendors = _linux_pci_gpu_vendors(ctx)
    windows_vendors = _windows_vendor_flags(ctx)

    if _has_tool(ctx, ["nvidia-smi"]):
        add(strong, "cuda", "nvidia-smi found")
    if _is_linux(ctx.platform_name) and ctx.path_exists("/proc/driver/nvidia/version"):
        add(strong, "cuda", "/proc/driver/nvidia/version found")
    if "10de" in pci_vendors:
        add(strong, "cuda", "NVIDIA PCI display device found")
    if windows_vendors["cuda"]:
        add(strong, "cuda", "NVIDIA Windows display adapter found")

    if _is_linux(ctx.platform_name):
        if _has_tool(ctx, ["rocm-smi", "amd-smi", "rocminfo", "hipinfo"]):
            add(strong, "rocm", "ROCm tool found")
        if ctx.path_exists("/dev/kfd"):
            add(strong, "rocm", "/dev/kfd found")
        if ctx.path_exists("/opt/rocm"):
            add(strong, "rocm", "/opt/rocm found")
        if "1002" in pci_vendors:
            add(strong, "rocm", "AMD PCI display device found")
    elif windows_vendors["rocm"]:
        add(weak, "rocm", "AMD Windows display adapter found")

    if _has_tool(ctx, ["xpu-smi", "sycl-ls"]):
        add(strong, "xpu", "Intel XPU/SYCL tool found")
    if "8086" in pci_vendors or windows_vendors["xpu"]:
        add(weak, "xpu", "Intel display adapter found")

    return strong, weak


def _prompt_for_variant(
    reason: str,
    input_func: Callable[[str], str] = input,
    output = sys.stderr,
) -> str:
    print("", file=output)
    print("TorchCTS could not confidently choose a PyTorch build.", file=output)
    print(reason, file=output)
    print("", file=output)
    print("Select PyTorch build:", file=output)
    print("  [1] CPU only", file=output)
    print("  [2] CUDA", file=output)
    print("  [3] ROCm Linux", file=output)
    print("  [4] Intel XPU", file=output)
    print("Choice [1]: ", end="", file=output, flush=True)
    choice = input_func("").strip()
    return {
        "": "cpu",
        "1": "cpu",
        "2": "cuda",
        "3": "rocm",
        "4": "xpu",
        "cpu": "cpu",
        "cuda": "cuda",
        "rocm": "rocm",
        "xpu": "xpu",
    }.get(choice.lower(), "cpu")


def choose_install_plan(
    ctx: ProbeContext | None = None,
    *,
    prompt: bool = False,
    input_func: Callable[[str], str] = input,
) -> InstallPlan:
    ctx = ctx or default_context()

    override = ctx.env.get("TORCHCTS_TORCH_VARIANT", "").strip()
    if override:
        variant = _normalize_variant(override)
        return _plan_for_variant(
            variant,
            "override",
            f"selected by TORCHCTS_TORCH_VARIANT={override}",
        )

    if ctx.platform_name == "darwin":
        return _plan_for_variant(
            "mps",
            "strong",
            "macOS uses the default PyPI PyTorch build; MPS is verified after install",
        )

    strong, weak = _detect_signals(ctx)
    strong_variants = sorted(strong)

    if len(strong_variants) == 1:
        variant = strong_variants[0]
        return _plan_for_variant(variant, "strong", "; ".join(strong[variant]))

    if len(strong_variants) > 1:
        reason = "Multiple GPU vendors detected: " + ", ".join(
            f"{variant} ({'; '.join(strong[variant])})"
            for variant in strong_variants
        )
        if prompt:
            variant = _prompt_for_variant(reason, input_func)
            return _plan_for_variant(variant, "user", f"user selected {variant} after ambiguous detection")
        return _plan_for_variant(
            "cpu",
            "ambiguous",
            reason,
            f"Defaulting to CPU because detection was ambiguous. To force a GPU build, set {OVERRIDE_HELP}.",
        )

    weak_variants = sorted(weak)
    if weak_variants:
        reason = "Weak hardware signal detected: " + ", ".join(
            f"{variant} ({'; '.join(weak[variant])})"
            for variant in weak_variants
        )
        if prompt:
            variant = _prompt_for_variant(reason, input_func)
            return _plan_for_variant(variant, "user", f"user selected {variant} after weak detection")
        return _plan_for_variant(
            "cpu",
            "weak",
            reason,
            f"Defaulting to CPU because detection was not certain. To force a GPU build, set {OVERRIDE_HELP}.",
        )

    return _plan_for_variant("cpu", "none", "no GPU vendor signal detected")


def _sanitize_value(value: str) -> str:
    return re.sub(r"[\r\n]+", " ", value)


def emit_plan(plan: InstallPlan, output_format: str) -> None:
    data = plan.as_dict()
    if output_format == "json":
        print(json.dumps(data, sort_keys=True))
        return
    for key in ("variant", "confidence", "torch_index_url", "device_hint", "reason", "warning"):
        print(f"{key}={_sanitize_value(data[key])}")


def _torch_cuda_available(torch_module) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(is_available and is_available())


def verify_torch_install(variant: str) -> int:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"ERROR: Failed to import PyTorch after install: {exc}", file=sys.stderr)
        return 1

    version = getattr(torch, "__version__", "unknown")
    torch_version = getattr(torch, "version", None)

    if variant == "cpu":
        print(f"PyTorch verification: imported torch {version} for CPU build.")
        return 0

    if variant == "cuda":
        cuda_version = getattr(torch_version, "cuda", None)
        if not _torch_cuda_available(torch) or not cuda_version:
            print(
                "ERROR: CUDA PyTorch was selected, but torch.cuda.is_available() "
                "is false or torch.version.cuda is missing.",
                file=sys.stderr,
            )
            return 1
        print(f"PyTorch verification: CUDA available with torch {version}.")
        return 0

    if variant == "rocm":
        hip_version = getattr(torch_version, "hip", None)
        if not _torch_cuda_available(torch) or not hip_version:
            print(
                "ERROR: ROCm PyTorch was selected, but torch.cuda.is_available() "
                "is false or torch.version.hip is missing.",
                file=sys.stderr,
            )
            return 1
        print(f"PyTorch verification: ROCm/HIP available with torch {version}.")
        return 0

    if variant == "xpu":
        xpu = getattr(torch, "xpu", None)
        is_available = getattr(xpu, "is_available", None)
        if not is_available or not is_available():
            print(
                "ERROR: XPU PyTorch was selected, but torch.xpu.is_available() is false.",
                file=sys.stderr,
            )
            return 1
        print(f"PyTorch verification: XPU available with torch {version}.")
        return 0

    if variant == "mps":
        backends = getattr(torch, "backends", None)
        mps_backend = getattr(backends, "mps", None)
        torch_mps = getattr(torch, "mps", None)
        is_available = getattr(mps_backend, "is_available", None) or getattr(torch_mps, "is_available", None)
        if not is_available or not is_available():
            print(
                "WARNING: macOS PyTorch installed, but MPS is not available. "
                "TorchCTS can still run CPU tests.",
                file=sys.stderr,
            )
            return 0
        print(f"PyTorch verification: MPS available with torch {version}.")
        return 0

    print(f"ERROR: Unknown verification variant: {variant}", file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan TorchCTS PyTorch installation.")
    parser.add_argument("--format", choices=("key-value", "json"), default="key-value")
    parser.add_argument("--prompt", action="store_true", help="Prompt for weak or ambiguous detections.")
    parser.add_argument("--verify", choices=sorted(INDEX_URLS), help="Verify an installed PyTorch build.")
    args = parser.parse_args(argv)

    if args.verify:
        return verify_torch_install(args.verify)

    try:
        plan = choose_install_plan(prompt=args.prompt)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    emit_plan(plan, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
