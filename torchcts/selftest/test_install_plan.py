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

import sys
from types import SimpleNamespace

import pytest
from torchcts.site_scripts import install_plan

pytestmark = pytest.mark.covers_category("selftest")

def make_context(
    *,
    platform_name="linux",
    env=None,
    tools=(),
    paths=(),
    pci_entries=None,
    windows_output="",
):
    env = env or {}
    tool_set = set(tools)
    path_set = set(paths)
    pci_root = "/fake/sys/bus/pci/devices"
    pci_entries = pci_entries or {}
    pci_files = {}
    for name, fields in pci_entries.items():
        base = f"{pci_root}/{name}"
        pci_files[f"{base}/vendor"] = fields["vendor"]
        pci_files[f"{base}/class"] = fields.get("class", "0x030000")

    def which(name):
        return f"/usr/bin/{name}" if name in tool_set else None

    def path_exists(path):
        if path == pci_root:
            return bool(pci_entries)
        return path in path_set

    def listdir(path):
        if path == pci_root:
            return list(pci_entries)
        raise OSError(path)

    def read_text(path):
        try:
            return pci_files[path]
        except KeyError as exc:
            raise OSError(path) from exc

    def run_command(cmd):
        return windows_output

    return install_plan.ProbeContext(
        platform_name=platform_name,
        env=env,
        which=which,
        path_exists=path_exists,
        listdir=listdir,
        read_text=read_text,
        run_command=run_command,
        pci_root=pci_root,
    )


def test_explicit_variant_override_selects_requested_backend():
    ctx = make_context(env={"TORCHCTS_TORCH_VARIANT": "nvidia"})

    plan = install_plan.choose_install_plan(ctx)

    assert plan.variant == "cuda"
    assert plan.confidence == "override"
    assert plan.torch_index_url.endswith("/cu128")
    assert plan.device_hint == "cuda"


def test_macos_uses_default_pypi_torch_with_mps_hint():
    ctx = make_context(platform_name="darwin")

    plan = install_plan.choose_install_plan(ctx)

    assert plan.variant == "mps"
    assert plan.confidence == "strong"
    assert plan.torch_index_url == ""
    assert plan.device_hint == "mps"


def test_strong_nvidia_signal_selects_cuda():
    ctx = make_context(tools={"nvidia-smi"})

    plan = install_plan.choose_install_plan(ctx)

    assert plan.variant == "cuda"
    assert plan.confidence == "strong"
    assert plan.torch_index_url.endswith("/cu128")


def test_linux_amd_pci_signal_selects_rocm():
    ctx = make_context(
        pci_entries={
            "0000:01:00.0": {"vendor": "0x1002", "class": "0x030200"},
        }
    )

    plan = install_plan.choose_install_plan(ctx)

    assert plan.variant == "rocm"
    assert plan.confidence == "strong"
    assert plan.torch_index_url.endswith("/rocm7.2")
    assert plan.device_hint == "cuda"


def test_no_gpu_signal_selects_cpu_without_warning():
    ctx = make_context()

    plan = install_plan.choose_install_plan(ctx)

    assert plan.variant == "cpu"
    assert plan.confidence == "none"
    assert plan.warning == ""


def test_torch_dependency_floor_is_27():
    assert install_plan.TORCH_MIN_VERSION == "2.7.0"
    assert install_plan.TORCH_SPEC == "torch>=2.7.0"
    assert install_plan.torch_version_satisfies("2.7.0")
    assert install_plan.torch_version_satisfies("2.7.1+cpu")
    assert install_plan.torch_version_satisfies("2.12.1")
    assert not install_plan.torch_version_satisfies("2.6.9")


def test_valid_torch_install_is_kept_without_upgrade():
    status = install_plan.TorchInstallStatus(
        "valid",
        "2.7.0",
        install_plan.TORCH_MIN_VERSION,
        "already valid",
    )

    assert install_plan.torch_install_action(status, upgrade_requested=False) == "keep"


def test_missing_torch_install_is_installed():
    status = install_plan.TorchInstallStatus(
        "missing",
        "",
        install_plan.TORCH_MIN_VERSION,
        "missing",
    )

    assert install_plan.torch_install_action(status, upgrade_requested=False) == "install"


def test_old_or_broken_torch_install_fails_without_upgrade():
    too_old = install_plan.TorchInstallStatus(
        "too_old",
        "2.6.0",
        install_plan.TORCH_MIN_VERSION,
        "too old",
    )
    broken = install_plan.TorchInstallStatus(
        "broken",
        "",
        install_plan.TORCH_MIN_VERSION,
        "broken",
    )

    assert install_plan.torch_install_action(too_old, upgrade_requested=False) == "fail"
    assert install_plan.torch_install_action(broken, upgrade_requested=False) == "fail"


def test_upgrade_request_installs_even_when_torch_is_valid():
    status = install_plan.TorchInstallStatus(
        "valid",
        "2.7.0",
        install_plan.TORCH_MIN_VERSION,
        "already valid",
    )

    assert install_plan.torch_install_action(status, upgrade_requested=True) == "install"


def test_intel_vendor_only_is_weak_and_defaults_to_cpu_without_prompt():
    ctx = make_context(
        pci_entries={
            "0000:00:02.0": {"vendor": "0x8086", "class": "0x030000"},
        }
    )

    plan = install_plan.choose_install_plan(ctx, prompt=False)

    assert plan.variant == "cpu"
    assert plan.confidence == "weak"
    assert "TORCHCTS_TORCH_VARIANT" in plan.warning


def test_intel_tool_is_strong_xpu_signal():
    ctx = make_context(tools={"xpu-smi"})

    plan = install_plan.choose_install_plan(ctx)

    assert plan.variant == "xpu"
    assert plan.confidence == "strong"
    assert plan.torch_index_url.endswith("/xpu")


def test_weak_signal_prompts_when_enabled():
    ctx = make_context(
        pci_entries={
            "0000:00:02.0": {"vendor": "0x8086", "class": "0x030000"},
        }
    )

    plan = install_plan.choose_install_plan(ctx, prompt=True, input_func=lambda _: "4")

    assert plan.variant == "xpu"
    assert plan.confidence == "user"


def test_ambiguous_strong_signals_prompt_when_enabled():
    ctx = make_context(tools={"nvidia-smi", "sycl-ls"})

    plan = install_plan.choose_install_plan(ctx, prompt=True, input_func=lambda _: "3")

    assert plan.variant == "rocm"
    assert plan.confidence == "user"


def test_ambiguous_strong_signals_default_to_cpu_without_prompt():
    ctx = make_context(tools={"nvidia-smi", "sycl-ls"})

    plan = install_plan.choose_install_plan(ctx, prompt=False)

    assert plan.variant == "cpu"
    assert plan.confidence == "ambiguous"
    assert "Defaulting to CPU" in plan.warning


def test_windows_amd_adapter_is_weak_rocm_signal():
    ctx = make_context(
        platform_name="win32",
        tools={"powershell"},
        windows_output="AMD Radeon RX 7900 XTX",
    )

    plan = install_plan.choose_install_plan(ctx, prompt=False)

    assert plan.variant == "cpu"
    assert plan.confidence == "weak"
    assert "AMD Windows display adapter" in plan.reason


def test_verify_cuda_success_uses_lazy_torch_import(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda="12.8", hip=None),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert install_plan.verify_torch_install("cuda") == 0


def test_verify_rocm_rejects_non_hip_torch(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda="12.8", hip=None),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert install_plan.verify_torch_install("rocm") == 1
