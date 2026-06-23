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
#
# TorchCTS — Global Installer (Windows)
#
# Usage:
#   irm https://torchcts.ai/scripts/install.ps1 | iex
#
# Or to uninstall:
#   powershell -ExecutionPolicy Bypass -Command "& { irm https://torchcts.ai/scripts/install.ps1 | iex } -Uninstall"
#
# Installs TorchCTS from PyPI into a centralized venv.

param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$InstallDir = Join-Path $HOME ".torchcts"
$VenvDir = Join-Path $InstallDir "venv"
$MinMajor = 3
$MinMinor = 10

# ── Uninstall ────────────────────────────────────────────────────────────────

if ($Uninstall) {
    Write-Host "[..] Uninstalling TorchCTS..." -ForegroundColor Cyan
    if (Test-Path $InstallDir) {
        Remove-Item -Recurse -Force $InstallDir
        Write-Host "[OK] Removed $InstallDir" -ForegroundColor Green
        Write-Host "[OK] TorchCTS uninstalled." -ForegroundColor Green
    } else {
        Write-Host "[..] Nothing to remove - TorchCTS is not installed." -ForegroundColor Yellow
    }
    exit 0
}

# ── Banner ───────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "TorchCTS Installer" -ForegroundColor White
Write-Host ""

# ── Locate Python ───────────────────────────────────────────────────────────

$Python = $null
$PythonArgs = @()

foreach ($candidate in @("python", "python3", "py")) {
    try {
        $null = & $candidate --version 2>&1
        $Python = $candidate
        break
    } catch {}
}

if (-not $Python) {
    Write-Host "ERROR: Python not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "  Install Python from https://python.org or the Microsoft Store."
    Write-Host "  Make sure to check 'Add python.exe to PATH' during installation."
    exit 1
}

# For 'py' launcher, use 'py -3' to ensure Python 3
if ($Python -eq "py") {
    $PythonArgs = @("-3")
}

# ── Verify version ──────────────────────────────────────────────────────────

$versionInfo = & $Python @PythonArgs -c "import sys; print(f'{sys.version_info.major} {sys.version_info.minor}')" 2>&1
$parts = $versionInfo -split " "
$pyMajor = [int]$parts[0]
$pyMinor = [int]$parts[1]
$pyVersion = "$pyMajor.$pyMinor"

if ($pyMajor -lt $MinMajor -or ($pyMajor -eq $MinMajor -and $pyMinor -lt $MinMinor)) {
    Write-Host "ERROR: Python ${MinMajor}.${MinMinor}+ required, found ${pyVersion}." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Found Python ${pyVersion}" -ForegroundColor Green

# ── Detect GPU for PyTorch wheel selection ───────────────────────────────────

$TorchIndexArgs = @()
$GpuType = "cpu"

# Check for NVIDIA GPU
try {
    $null = & nvidia-smi 2>&1
    $GpuType = "cuda"
} catch {
    # Check for AMD GPU (ROCm)
    try {
        $null = & rocm-smi 2>&1
        $GpuType = "rocm"
        $TorchIndexArgs = @("--extra-index-url", "https://download.pytorch.org/whl/rocm6.3")
    } catch {
        # CPU-only
        $GpuType = "cpu"
        $TorchIndexArgs = @("--extra-index-url", "https://download.pytorch.org/whl/cpu")
    }
}

Write-Host "[OK] GPU detection: ${GpuType}" -ForegroundColor Green

# ── Create or reuse venv ────────────────────────────────────────────────────

if (Test-Path $VenvDir) {
    Write-Host "[..] Existing installation found - upgrading." -ForegroundColor Cyan
} else {
    Write-Host "[..] Creating virtual environment in ${VenvDir}..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    & $Python @PythonArgs -m venv $VenvDir
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
}

# ── Install / upgrade ───────────────────────────────────────────────────────

$Pip = Join-Path $VenvDir "Scripts\pip.exe"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "[..] Upgrading pip..." -ForegroundColor Cyan
& $Pip install --upgrade pip --quiet

Write-Host "[..] Installing TorchCTS (PyTorch variant: ${GpuType})..." -ForegroundColor Cyan
& $Pip install --upgrade torchcts @TorchIndexArgs --quiet

$installedVersion = & $VenvPython -c "import torchcts; print(torchcts.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) { $installedVersion = "unknown" }

$torchVersion = & $VenvPython -c "import torch; print(torch.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) { $torchVersion = "unknown" }

Write-Host "[OK] Installed TorchCTS ${installedVersion} (PyTorch ${torchVersion})" -ForegroundColor Green

# ── Summary ──────────────────────────────────────────────────────────────────

$torchctsExe = Join-Path $VenvDir "Scripts\torchcts.exe"

Write-Host ""
Write-Host "TorchCTS ${installedVersion} installed successfully." -ForegroundColor Green
Write-Host ""
Write-Host "  Version:    ${installedVersion}"
Write-Host "  PyTorch:    ${torchVersion} (${GpuType})"
Write-Host "  Venv:       ${VenvDir}"
Write-Host ""

switch ($GpuType) {
    "cuda" { Write-Host "  Run:        ${torchctsExe} run --device cuda" }
    "rocm" { Write-Host "  Run:        ${torchctsExe} run --device cuda" }
    default { Write-Host "  Run:        ${torchctsExe} run --device cpu" }
}

Write-Host ""
