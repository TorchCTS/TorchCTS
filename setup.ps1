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
# TorchCTS - Repository Setup Script (Windows)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# Creates a local .venv and installs TorchCTS in editable mode
# for development and contribution.

$ErrorActionPreference = "Stop"

$VenvDir = ".venv"
$MinMajor = 3
$MinMinor = 10
$PlanFile = Join-Path "site_scripts" "install_plan.py"
$TorchMinVersion = "2.7.0"
$TorchSpec = "torch>=$TorchMinVersion"

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

# ── Locate Python ───────────────────────────────────────────────────────────

$Python = $null
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
    return
}

# For 'py' launcher, use 'py -3' to ensure Python 3.
if ($Python -eq "py") {
    $PythonArgs = @("-3")
} else {
    $PythonArgs = @()
}

# ── Verify version ──────────────────────────────────────────────────────────

$versionInfo = & $Python @PythonArgs -c "import sys; print(f'{sys.version_info.major} {sys.version_info.minor}')" 2>&1
$parts = $versionInfo -split " "
$pyMajor = [int]$parts[0]
$pyMinor = [int]$parts[1]
$pyVersion = "$pyMajor.$pyMinor"

if ($pyMajor -lt $MinMajor -or ($pyMajor -eq $MinMajor -and $pyMinor -lt $MinMinor)) {
    Write-Host "ERROR: Python ${MinMajor}.${MinMinor}+ required, found ${pyVersion}." -ForegroundColor Red
    return
}

Write-Host "[OK] Found Python ${pyVersion}" -ForegroundColor Green

& $Python @PythonArgs -m venv --help *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python venv module is not available." -ForegroundColor Red
    return
}

if (-not (Test-Path $PlanFile)) {
    Write-Host "ERROR: Install planner not found: ${PlanFile}" -ForegroundColor Red
    return
}

# ── Create or reuse venv ────────────────────────────────────────────────────

if (Test-Path $VenvDir) {
    Write-Host "[..] Existing ${VenvDir} found - reusing it." -ForegroundColor Cyan
} else {
    Write-Host "[..] Creating virtual environment in ${VenvDir}..." -ForegroundColor Cyan
    & $Python @PythonArgs -m venv $VenvDir
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
}

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

# ── Upgrade pip and install ─────────────────────────────────────────────────

$Pip = Join-Path $VenvDir "Scripts\pip.exe"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "[..] Upgrading pip, setuptools, wheel..." -ForegroundColor Cyan
& $Pip install --upgrade pip setuptools wheel --quiet

Write-Host "[..] Checking PyTorch install..." -ForegroundColor Cyan
$torchStatusOutput = & $VenvPython $PlanFile --torch-status --format key-value
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch status check failed."
}
$torchStatusPlan = Read-InstallPlan -Lines $torchStatusOutput
$TorchStatus = $torchStatusPlan["status"]
$TorchVersion = $torchStatusPlan["version"]
$TorchDetail = $torchStatusPlan["detail"]
$UpgradeTorch = $env:TORCHCTS_UPGRADE_TORCH -eq "1"

if ($TorchStatus -eq "valid" -and -not $UpgradeTorch) {
    Write-Host "[OK] Keeping existing PyTorch ${TorchVersion}." -ForegroundColor Green
} elseif ($TorchStatus -eq "too_old" -and -not $UpgradeTorch) {
    throw "$TorchDetail Install a PyTorch ${TorchMinVersion}+ build manually, or set TORCHCTS_UPGRADE_TORCH=1 to let setup upgrade it."
} elseif ($TorchStatus -eq "broken" -and -not $UpgradeTorch) {
    throw "$TorchDetail Fix the PyTorch install manually, or set TORCHCTS_UPGRADE_TORCH=1 to let setup reinstall it."
} else {
    Write-Host "[..] Installing PyTorch (${GpuType})..." -ForegroundColor Cyan
    $torchInstallArgs = @("install")
    if ($UpgradeTorch) {
        $torchInstallArgs += "--upgrade"
    }
    $torchInstallArgs += $TorchSpec
    if ($TorchIndexUrl) {
        $torchInstallArgs += @("--index-url", $TorchIndexUrl)
    }
    $torchInstallArgs += "--quiet"
    & $Pip @torchInstallArgs
}

Write-Host "[..] Installing TorchCTS in editable mode..." -ForegroundColor Cyan
& $Pip install -e . --quiet

Write-Host "[..] Verifying PyTorch install..." -ForegroundColor Cyan
& $VenvPython $PlanFile --verify $GpuType
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch verification failed."
}

$installedVersion = & $VenvPython -c "import torchcts; print(torchcts.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) { $installedVersion = "unknown" }

# ── Summary ─────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "TorchCTS development environment ready." -ForegroundColor Green
Write-Host ""
Write-Host "  Version:    ${installedVersion}"
Write-Host "  Python:     ${pyVersion}"
Write-Host "  PyTorch:    ${GpuType}"
Write-Host "  Venv:       $(Get-Location)\${VenvDir}"
Write-Host ""
Write-Host "  Activate:   .\.venv\Scripts\Activate.ps1"
Write-Host "  Run:        torchcts run --device ${DeviceHint}"
Write-Host ""
