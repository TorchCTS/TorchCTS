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

import os
import sys
import site
import subprocess
from dataclasses import dataclass
import torch
import importlib
import importlib.util
import re
import shutil

# Regex to parse backend name from registration calls.
# Handles: rename_privateuse1_backend("name"), register_privateuse1_backend('name'),
# torch.utils.rename_privateuse1_backend("name"), whitespace variations, etc.
_REGISTRATION_RE = re.compile(
    r'(?:rename|register)_privateuse1_backend\s*\(\s*["\']([^"\']+)["\']'
)

# Directories to skip when walking the filesystem for .py files.
_SCAN_SKIP_DIRS = frozenset({
    'torch', 'pip', 'setuptools', 'numpy', 'pytest', 'scipy', 'pandas',
    'matplotlib', 'sympy', 'networkx', 'jinja2', 'markupsafe', 'mpmath',
    'filelock', 'typing_extensions', 'psutil', 'distutils', 'wheel',
    'pkg_resources', '__pycache__', 'torchcts',
})


@dataclass(frozen=True)
class CapabilityProbeResult:
    device_name: str
    capability: str
    supported: bool
    returncode: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    command_args: tuple[str, ...] = ()

    def __bool__(self):
        return self.supported

    def to_dict(self):
        return {
            "device_name": self.device_name,
            "capability": self.capability,
            "supported": self.supported,
            "returncode": self.returncode,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stdout_tail": _probe_text_tail(self.stdout),
            "stderr_tail": _probe_text_tail(self.stderr),
            "timed_out": self.timed_out,
            "command_args": list(self.command_args),
        }


def _probe_text_tail(value, limit=2000):
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[-limit:]


def _build_scan_paths():
    """Build the list of directories to scan for custom backend .py files.
    
    Includes site-packages directories and any additional paths added by
    .pth files (which catches editable installs).
    """
    scan_paths = []
    
    # Start with site-packages directories
    try:
        for sp in site.getsitepackages():
            if os.path.isdir(sp):
                scan_paths.append(os.path.abspath(sp))
    except Exception:
        pass
    
    # Also check user site-packages
    try:
        user_sp = site.getusersitepackages()
        if user_sp and os.path.isdir(user_sp):
            scan_paths.append(os.path.abspath(user_sp))
    except Exception:
        pass
    
    # Read .pth files in each site-packages dir to find editable install paths
    for sp in list(scan_paths):
        try:
            for entry in os.listdir(sp):
                if entry.endswith('.pth') and os.path.isfile(os.path.join(sp, entry)):
                    with open(os.path.join(sp, entry), 'r', errors='ignore') as f:
                        for line in f:
                            line = line.strip()
                            # Skip empty lines, comments, and import statements
                            if not line or line.startswith('#') or line.startswith('import '):
                                continue
                            # Resolve the path (could be absolute or relative)
                            if os.path.isabs(line):
                                candidate = line
                            else:
                                candidate = os.path.join(sp, line)
                            if os.path.isdir(candidate):
                                scan_paths.append(os.path.abspath(candidate))
        except Exception:
            pass
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in scan_paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _scan_for_privateuse1_backends():
    """Scan Python source files for PrivateUse1 backend registration calls.
    
    Returns a list of (backend_name, import_path) tuples, where import_path
    is the dotted module path that triggers registration.
    """
    scan_paths = _build_scan_paths()
    candidates = []
    scanned_files = set()
    
    for search_root in scan_paths:
        try:
            for dirpath, dirnames, filenames in os.walk(search_root, followlinks=True):
                # Prune irrelevant directories
                dirnames[:] = [
                    d for d in dirnames
                    if d.lower() not in _SCAN_SKIP_DIRS
                    and not d.endswith('.dist-info')
                    and not d.endswith('.egg-info')
                    and not d.startswith('.')
                ]
                
                for filename in filenames:
                    if not filename.endswith('.py'):
                        continue
                    
                    filepath = os.path.join(dirpath, filename)
                    abs_filepath = os.path.abspath(filepath)
                    
                    # Don't scan the same file twice
                    if abs_filepath in scanned_files:
                        continue
                    scanned_files.add(abs_filepath)
                    
                    # Skip files larger than 1MB
                    try:
                        if os.path.getsize(abs_filepath) > 1024 * 1024:
                            continue
                    except OSError:
                        continue
                    
                    try:
                        with open(abs_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    except Exception:
                        continue
                    
                    # Quick check before running the regex
                    if 'privateuse1_backend' not in content:
                        continue
                    
                    matches = _REGISTRATION_RE.findall(content)
                    for backend_name in matches:
                        # Derive dotted import path from file path
                        rel_path = os.path.relpath(abs_filepath, search_root)
                        # Convert to dotted module path
                        module_path = rel_path.replace(os.sep, '.')
                        # Strip .py suffix
                        if module_path.endswith('.py'):
                            module_path = module_path[:-3]
                        # Strip trailing .__init__ for package inits
                        if module_path.endswith('.__init__'):
                            module_path = module_path[:-9]
                        
                        candidates.append((backend_name, module_path))
        except Exception:
            continue
    
    # Deduplicate by backend_name (first occurrence wins)
    seen_names = set()
    unique = []
    for backend_name, import_path in candidates:
        if backend_name not in seen_names:
            seen_names.add(backend_name)
            unique.append((backend_name, import_path))
    return unique


def _validate_backend_import(import_path, expected_name, timeout=10):
    """Validate that importing a backend module is safe by testing in a subprocess.
    
    Returns True if the subprocess exits cleanly and registers the expected
    backend name. Returns False if it segfaults, times out, or fails.
    """
    _SENTINEL = "BVRESULT:"
    script = (
        f"import {import_path}; "
        f"import torch; "
        f"print('{_SENTINEL}' + torch._C._get_privateuse1_backend_name())"
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            registered_name = None
            for line in result.stdout.splitlines():
                if line.startswith(_SENTINEL):
                    registered_name = line[len(_SENTINEL):].strip()
                    break
            if registered_name is None:
                print(
                    f"Warning: Backend '{expected_name}' import produced no "
                    f"result marker",
                    file=sys.stderr
                )
                return False
            if registered_name == expected_name:
                return True
            else:
                print(
                    f"Warning: Backend '{expected_name}' import succeeded but "
                    f"registered name is '{registered_name}'",
                    file=sys.stderr
                )
                return True  # Still valid, just unexpected name
        else:
            print(
                f"Warning: Backend '{expected_name}' import failed "
                f"(exit code {result.returncode})",
                file=sys.stderr
            )
            return False
    except subprocess.TimeoutExpired:
        print(
            f"Warning: Backend '{expected_name}' import timed out after {timeout}s",
            file=sys.stderr
        )
        return False
    except Exception as e:
        print(
            f"Warning: Backend '{expected_name}' validation error: {e}",
            file=sys.stderr
        )
        return False


def _detect_amd_gpu_windows():
    """Detect AMD GPU on Windows via WMI (Win32_VideoController).
    
    Returns True if at least one AMD/Radeon/ATI GPU is found.
    Only meaningful on Windows; returns False on other platforms.
    """
    if sys.platform != "win32":
        return False
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_VideoController | Where-Object {"
                " $_.Name -match 'AMD|Radeon|ATI' -or"
                " $_.AdapterCompatibility -match 'AMD|Advanced Micro Devices' -or"
                " $_.PNPDeviceID -match 'VEN_1002'"
                "}) -ne $null"
            ],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip().lower() == "true"
    except Exception:
        return False


def _detect_amd_gpu_linux():
    """Detect AMD GPU on Linux via CLI tools or sysfs.
    
    Returns True if rocm-smi, amd-smi, /dev/kfd, or lspci reports an AMD GPU.
    """
    if not sys.platform.startswith("linux"):
        return False
    # Check CLI tools
    for tool in ("rocm-smi", "amd-smi", "rocminfo", "hipinfo"):
        if shutil.which(tool) is not None:
            return True
    # Check kernel device
    if os.path.exists("/dev/kfd"):
        return True
    # Check lspci
    if shutil.which("lspci") is not None:
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    low = line.lower()
                    if ("amd" in low or "radeon" in low) and any(
                        tag in low for tag in ("vga", "3d", "display")
                    ):
                        return True
        except Exception:
            pass
    return False


def _check_hardware_alignment():
    """Detect OS and check if PyTorch is compiled with support for present hardware backends."""
    ok = True
    # 1. macOS / MPS check
    if sys.platform == "darwin":
        has_mps = (
            (hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) or
            (hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available())
        )
        if not has_mps:
            print("\n" + "="*80, file=sys.stderr)
            print("WARNING: Running on macOS, but the installed PyTorch does not have MPS (Metal)", file=sys.stderr)
            print("support enabled. You may be running an x86_64 or CPU-only PyTorch build.", file=sys.stderr)
            print("To use GPU acceleration, please install a native macOS PyTorch build.", file=sys.stderr)
            print("="*80 + "\n", file=sys.stderr)

    # 2. Windows / Linux checks — each hardware class is independent
    if sys.platform == "win32" or sys.platform.startswith("linux"):
        # Check NVIDIA (CUDA)
        if shutil.which("nvidia-smi") is not None:
            if not torch.cuda.is_available():
                print("\n" + "="*80, file=sys.stderr)
                print("ERROR: NVIDIA GPU hardware detected via nvidia-smi, but the installed PyTorch", file=sys.stderr)
                print("version is not compiled with CUDA support (it is likely CPU-only).", file=sys.stderr)
                print("Please install a PyTorch build with CUDA enabled.", file=sys.stderr)
                print("="*80 + "\n", file=sys.stderr)
                ok = False

        # Check AMD (ROCm) — platform-specific detection
        amd_detected = _detect_amd_gpu_linux() or _detect_amd_gpu_windows()
        if amd_detected:
            if not torch.cuda.is_available():
                print("\n" + "="*80, file=sys.stderr)
                print("ERROR: AMD GPU hardware detected, but the installed PyTorch version is", file=sys.stderr)
                print("not compiled with ROCm/CUDA support (it is likely CPU-only).", file=sys.stderr)
                print("Please install a PyTorch build with ROCm/CUDA enabled.", file=sys.stderr)
                print("="*80 + "\n", file=sys.stderr)
                ok = False

        # Check Intel (XPU)
        if shutil.which("sycl-ls") is not None or shutil.which("xpu-smi") is not None:
            has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
            if not has_xpu:
                print("\n" + "="*80, file=sys.stderr)
                print("ERROR: Intel GPU hardware detected, but the installed PyTorch version is not", file=sys.stderr)
                print("compiled with XPU support (it is likely CPU-only).", file=sys.stderr)
                print("Please install a PyTorch build with XPU enabled.", file=sys.stderr)
                print("="*80 + "\n", file=sys.stderr)
                ok = False

    return ok


def detect_backends(non_interactive=False):
    """Detect available device backends (in-tree and custom PrivateUse1).
    
    Returns a list of (name, type) tuples for in-tree backends, and
    (name, type, import_path) tuples for custom backends. No custom
    backends are imported in-process during detection.
    """
    _check_hardware_alignment()
    print("Probing backends...", flush=True)
    backends = []
    
    # 1. Check in-tree backends
    if torch.cuda.is_available():
        backends.append(("cuda", "in-tree"))
    # In some PyTorch versions, torch.backends.mps.is_available() is the check
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backends.append(("mps", "in-tree"))
    elif hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
        backends.append(("mps", "in-tree"))
        
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        backends.append(("xpu", "in-tree"))

    # 2. Scan for custom PrivateUse1 backends and validate in subprocesses
    pu1_candidates = _scan_for_privateuse1_backends()
    for backend_name, import_path in pu1_candidates:
        if _validate_backend_import(import_path, backend_name):
            backends.append((backend_name, "privateuse1", import_path))

    # Deduplicate by backend name
    unique_backends = []
    seen = set()
    for entry in backends:
        name = entry[0]
        if name not in seen:
            unique_backends.append(entry)
            seen.add(name)
            
    return unique_backends

def get_device_backend(device_name="auto", backend_import=None, non_interactive=False):
    # Import specified backend if provided
    if backend_import:
        try:
            importlib.import_module(backend_import)
        except Exception as e:
            raise ImportError(f"Failed to import backend module '{backend_import}': {e}")

    if device_name == "auto":
        detected = detect_backends(non_interactive)
        if len(detected) == 1:
            entry = detected[0]
            name, btype = entry[0], entry[1]
            # Import custom backend in-process now that it's selected
            if len(entry) == 3:
                import_path = entry[2]
                try:
                    importlib.import_module(import_path)
                except Exception as e:
                    raise ImportError(f"Failed to import backend '{name}' via '{import_path}': {e}")
            print(f"Auto-detected backend: {name} ({btype})")
            return name
        elif len(detected) > 1:
            print("\nAvailable backends detected:")
            for idx, entry in enumerate(detected, 1):
                name, btype = entry[0], entry[1]
                print(f"  [{idx}] {name} ({btype})")
            
            is_non_interactive = non_interactive or os.environ.get("BACKEND_VALIDATOR_NON_INTERACTIVE") == "1"
            if is_non_interactive:
                backend_list = ", ".join(f"{e[0]} ({e[1]})" for e in detected)
                raise ValueError(f"Ambiguous device selection in non-interactive mode. Detected backends: {backend_list}")
            
            try:
                choice = input(f"Select a backend [1-{len(detected)}]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(detected):
                    entry = detected[int(choice) - 1]
                    name = entry[0]
                    # Import custom backend in-process now that it's selected
                    if len(entry) == 3:
                        import_path = entry[2]
                        try:
                            importlib.import_module(import_path)
                        except Exception as e:
                            raise ImportError(f"Failed to import backend '{name}' via '{import_path}': {e}")
                    return name
                else:
                    raise ValueError("Invalid backend selection choice.")
            except (KeyboardInterrupt, EOFError):
                raise KeyboardInterrupt("Interrupted backend selection.")
        else:
            # No backends detected via torch — check if AMD hardware is present.
            # ROCm maps to torch.device("cuda") but torch.cuda.is_available()
            # can return False in some ROCm configurations. Suggest explicit flag.
            if _detect_amd_gpu_linux() or _detect_amd_gpu_windows():
                raise ValueError(
                    "No device backend auto-detected, but AMD GPU hardware was found.\n"
                    "PyTorch ROCm uses the 'cuda' device. Try: torchcts run --device cuda"
                )
            raise ValueError("No device backend detected. Install a backend package or set device_name explicitly in manifest.py.")
    # For non-in-tree backends passed via --device, we still need to
    # import the backend module so it registers with PyTorch.
    _IN_TREE = frozenset({"cpu", "cuda", "mps", "xpu", "meta"})
    if device_name not in _IN_TREE and not backend_import:
        candidates = _scan_for_privateuse1_backends()
        imported = False
        for bname, bpath in candidates:
            if bname == device_name:
                importlib.import_module(bpath)
                imported = True
                break
        if not imported:
            # Fallback: try importing top-level package
            try:
                importlib.import_module(device_name)
            except ImportError:
                pass

    return device_name

def get_device_module(device_name):
    # Check for module registered under torch.<device_name> (works for
    # in-tree backends and custom backends that register there)
    mod = getattr(torch, device_name, None)
    if mod is not None:
        return mod
    # Fallback: try importing the top-level package matching device_name.
    # Some custom backends expose synchronize/empty_cache/memory_allocated
    # on their own package rather than registering under torch.<name>.
    try:
        return importlib.import_module(device_name)
    except ImportError:
        return None

def create_stream(device_name):
    try:
        return torch.Stream(device=device_name)
    except Exception as exc:
        raise RuntimeError(f"stream API unavailable for device {device_name}: {type(exc).__name__}: {exc}") from exc

def create_event(device_name, *, enable_timing=False):
    try:
        return torch.Event(device=device_name, enable_timing=enable_timing)
    except TypeError:
        if enable_timing:
            raise RuntimeError(f"event timing API unavailable for device {device_name}")
        try:
            return torch.Event(device=device_name)
        except Exception as exc:
            raise RuntimeError(f"event API unavailable for device {device_name}: {type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"event API unavailable for device {device_name}: {type(exc).__name__}: {exc}") from exc

def stream_context(stream):
    if hasattr(stream, "__enter__") and hasattr(stream, "__exit__"):
        return stream
    if hasattr(torch, "stream"):
        return torch.stream(stream)
    stream_device = getattr(stream, "device", None)
    device_type = getattr(stream_device, "type", None)
    if device_type:
        mod = get_device_module(device_type)
        if mod is not None and hasattr(mod, "stream"):
            return mod.stream(stream)
    raise RuntimeError(f"stream context API unavailable for stream {stream!r}")

def synchronize(device_name):
    if device_name == "cpu":
        return
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps":
        # MPS synchronize lives under torch.mps (not torch.backends.mps)
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    else:
        mod = get_device_module(device_name)
        if mod and hasattr(mod, "synchronize"):
            mod.synchronize()
        elif hasattr(torch, "xpu") and device_name == "xpu" and hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()

def empty_cache(device_name):
    if device_name == "cpu":
        return
    if device_name == "cuda":
        torch.cuda.empty_cache()
    elif device_name == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    else:
        mod = get_device_module(device_name)
        if mod and hasattr(mod, "empty_cache"):
            mod.empty_cache()
        elif hasattr(torch, "xpu") and device_name == "xpu" and hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()

def memory_allocated(device_name, device_idx=0):
    if device_name == "cpu":
        return 0
    if device_name == "cuda":
        return torch.cuda.memory_allocated(device_idx)
    elif device_name == "mps":
        # torch.mps.current_allocated_memory() returns memory allocated
        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory()
        return 0
    else:
        mod = get_device_module(device_name)
        if mod and hasattr(mod, "memory_allocated"):
            return mod.memory_allocated(device_idx)
        elif hasattr(torch, "xpu") and device_name == "xpu" and hasattr(torch.xpu, "memory_allocated"):
            return torch.xpu.memory_allocated(device_idx)
        return 0

def memory_reserved(device_name, device_idx=0):
    if device_name == "cpu":
        return 0
    if device_name == "cuda":
        return torch.cuda.memory_reserved(device_idx)
    elif device_name == "mps":
        # MPS doesn't have reserved memory API in all versions, returning 0
        return 0
    else:
        mod = get_device_module(device_name)
        if mod and hasattr(mod, "memory_reserved"):
            return mod.memory_reserved(device_idx)
        elif hasattr(torch, "xpu") and device_name == "xpu" and hasattr(torch.xpu, "memory_reserved"):
            return torch.xpu.memory_reserved(device_idx)
        return 0

def get_device_total_memory(device_name, device_idx=0):
    """Return total device memory in bytes. Returns None if unavailable."""
    if device_name == "cuda":
        return torch.cuda.get_device_properties(device_idx).total_memory
    elif device_name == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "recommended_max_memory"):
            return torch.mps.recommended_max_memory()
        # Fallback for older PyTorch versions
        import psutil
        return psutil.virtual_memory().total
    elif device_name == "xpu":
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "get_device_properties"):
            return torch.xpu.get_device_properties(device_idx).total_memory
    else:
        mod = get_device_module(device_name)
        if mod and hasattr(mod, "get_device_properties"):
            props = mod.get_device_properties(device_idx)
            if hasattr(props, "total_memory"):
                return props.total_memory
    return None

def _capability_probe_script(device_name, capability):
    if capability == "pinned_memory":
        return (
            "import torch\n"
            f"x = torch.randn(2, 2)\n"
            f"p = x.pin_memory(device='{device_name}')\n"
            f"assert p.is_pinned()\n"
            "print('SUCCESS')\n"
        )
    if capability == "sparse":
        return (
            "import torch\n"
            "i = torch.tensor([[0, 1, 1], [2, 0, 2]])\n"
            "v = torch.tensor([3, 4, 5], dtype=torch.float32)\n"
            f"s = torch.sparse_coo_tensor(i, v, (2, 3), device='{device_name}')\n"
            "dense = s.to_dense()\n"
            "print('SUCCESS')\n"
        )
    if capability == "nested":
        return (
            "import torch\n"
            f"a = torch.randn(2, 3, device='{device_name}')\n"
            f"b = torch.randn(1, 3, device='{device_name}')\n"
            f"nt = torch.nested.nested_tensor([a, b], device='{device_name}')\n"
            "padded = nt.to_padded_tensor(padding=0.0)\n"
            f"assert nt.is_nested and padded.device.type == '{device_name}'\n"
            "print('SUCCESS')\n"
        )
    if capability == "named_tensor":
        return (
            "import torch\n"
            f"x = torch.empty((2, 3), device='{device_name}', names=('rows', 'cols'))\n"
            "y = x.align_to('cols', 'rows')\n"
            f"assert y.names == ('cols', 'rows') and y.device.type == '{device_name}'\n"
            "print('SUCCESS')\n"
        )
    if capability == "fp8":
        return (
            "import torch\n"
            f"x = torch.zeros(4, dtype=torch.float8_e4m3fn, device='{device_name}')\n"
            f"y = torch.ones(4, dtype=torch.float32, device='{device_name}').to(torch.float8_e5m2)\n"
            f"assert x.device.type == '{device_name}' and y.device.type == '{device_name}'\n"
            "print('SUCCESS')\n"
        )
    return None


def probe_capability_result(device_name, capability, timeout=10):
    """Probe a declared capability and preserve subprocess failure evidence."""

    script = _capability_probe_script(device_name, capability)
    if script is None:
        return CapabilityProbeResult(
            device_name=device_name,
            capability=capability,
            supported=False,
            error_type="UnknownCapability",
            error_message=f"no probe is defined for capability {capability!r}",
        )

    cmd = [sys.executable, "-c", script]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return CapabilityProbeResult(
            device_name=device_name,
            capability=capability,
            supported=False,
            error_type="TimeoutExpired",
            error_message=f"probe exceeded {timeout:g} seconds",
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
            command_args=tuple(cmd),
        )
    except Exception as exc:
        return CapabilityProbeResult(
            device_name=device_name,
            capability=capability,
            supported=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            command_args=tuple(cmd),
        )

    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    supported = result.returncode == 0 and "SUCCESS" in stdout
    return CapabilityProbeResult(
        device_name=device_name,
        capability=capability,
        supported=supported,
        returncode=result.returncode,
        error_type=None if supported else "CapabilityProbeFailed",
        error_message=None if supported else _probe_text_tail(stderr or stdout),
        stdout=stdout,
        stderr=stderr,
        command_args=tuple(cmd),
    )


def probe_capability(device_name, capability, timeout=10):
    """Return True if the backend supports a capability."""
    return probe_capability_result(device_name, capability, timeout=timeout).supported
