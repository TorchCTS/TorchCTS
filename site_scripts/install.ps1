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
# TorchCTS - Global Installer (Windows)
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
$TorchSpec = "torch>=2.12.0"

function Read-InstallPlan {
    param([string[]]$Lines)

    $plan = @{}
    foreach ($line in $Lines) {
        $idx = $line.IndexOf("=")
        if ($idx -le 0) {
            continue
        }
        $key = $line.Substring(0, $idx)
        $value = $line.Substring($idx + 1)
        $plan[$key] = $value
    }
    return $plan
}

function Write-EmbeddedInstallPlan {
    param([string]$Path)

    @'
__TORCHCTS_INSTALL_PLAN_PY__
'@ | Set-Content -LiteralPath $Path -Encoding UTF8
}

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
    return
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
    Write-Host ""
    return
}

# For 'py' launcher, use 'py -3' to ensure Python 3.
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
    Write-Host ""
    return
}

Write-Host "[OK] Found Python ${pyVersion}" -ForegroundColor Green

# ── Verify venv module ──────────────────────────────────────────────────────

& $Python @PythonArgs -m venv --help *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python venv module is not available." -ForegroundColor Red
    Write-Host "  Reinstall Python or install the venv package for your distribution."
    return
}

# ── Create or reuse venv ────────────────────────────────────────────────────

if (Test-Path $VenvDir) {
    Write-Host "[..] Existing installation found - upgrading." -ForegroundColor Cyan
} else {
    Write-Host "[..] Creating virtual environment in ${VenvDir}..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    & $Python @PythonArgs -m venv $VenvDir
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
}

# ── Write embedded planner ──────────────────────────────────────────────────

$PlanFile = Join-Path ([System.IO.Path]::GetTempPath()) ("torchcts-install-plan-{0}.py" -f ([guid]::NewGuid()))

try {
    Write-Host "[..] Preparing install planner..." -ForegroundColor Cyan
    Write-EmbeddedInstallPlan -Path $PlanFile

    # ── Plan PyTorch install ────────────────────────────────────────────────────

    Write-Host "[..] Selecting PyTorch build..." -ForegroundColor Cyan
    $PromptArgs = @()
    if (-not $env:TORCHCTS_NON_INTERACTIVE -and [Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
        $PromptArgs = @("--prompt")
    }
    $planOutput = & $Python @PythonArgs $PlanFile --format key-value @PromptArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Install planner failed."
    }
    $plan = Read-InstallPlan -Lines $planOutput

    $GpuType = $plan["variant"]
    $GpuConfidence = $plan["confidence"]
    $TorchIndexUrl = $plan["torch_index_url"]
    $DeviceHint = $plan["device_hint"]
    $TorchReason = $plan["reason"]
    $TorchWarning = $plan["warning"]

    if (-not $GpuType -or -not $DeviceHint) {
        throw "Install planner did not return a usable PyTorch plan."
    }

    Write-Host "[OK] PyTorch selection: ${GpuType} (${GpuConfidence})" -ForegroundColor Green
    Write-Host "[..] $TorchReason" -ForegroundColor Cyan
    if ($TorchWarning) {
        Write-Host "[..] $TorchWarning" -ForegroundColor Yellow
    }

    # ── Install / upgrade ───────────────────────────────────────────────────────

    $Pip = Join-Path $VenvDir "Scripts\pip.exe"
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"

    Write-Host "[..] Upgrading pip..." -ForegroundColor Cyan
    & $Pip install --upgrade pip --quiet

    Write-Host "[..] Installing PyTorch (${GpuType})..." -ForegroundColor Cyan
    $torchInstallArgs = @("install", "--upgrade", $TorchSpec)
    if ($TorchIndexUrl) {
        $torchInstallArgs += @("--index-url", $TorchIndexUrl)
    }
    $torchInstallArgs += "--quiet"
    & $Pip @torchInstallArgs

    Write-Host "[..] Installing TorchCTS..." -ForegroundColor Cyan
    & $Pip install --upgrade torchcts --quiet

    Write-Host "[..] Verifying PyTorch install..." -ForegroundColor Cyan
    & $VenvPython $PlanFile --verify $GpuType
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch verification failed."
    }

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
    Write-Host "  Run:        ${torchctsExe} run --device ${DeviceHint}"
    Write-Host ""
} finally {
    if (Test-Path $PlanFile) {
        Remove-Item -Force $PlanFile
    }
}
