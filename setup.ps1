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
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Yes
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -DryRun
#
# Creates a local .venv and installs TorchCTS in editable mode
# for development and contribution.

param(
    [switch]$Yes,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$VenvDir = ".venv"
$MinMajor = 3
$MinMinor = 10
$PlanFile = Join-Path (Join-Path "torchcts" "site_scripts") "install_plan.py"
$TorchMinVersion = "2.7.0"
$TorchMaxExclusiveVersion = "2.12.2"
$TorchMaxValidatedVersion = "2.12.1"
$TorchSpec = "torch>=$TorchMinVersion,<$TorchMaxExclusiveVersion"
$AutoYes = $Yes -or $env:TORCHCTS_YES -eq "1"
$DryRunMode = $DryRun -or $env:TORCHCTS_DRY_RUN -eq "1"

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

function Get-TorchBuildLabel {
    param([string]$Variant)
    switch ($Variant) {
        "cpu" { "CPU build"; break }
        "cuda" { "NVIDIA CUDA build"; break }
        "rocm" { "AMD ROCm build"; break }
        "xpu" { "Intel XPU build"; break }
        "mps" { "Apple Metal/MPS build"; break }
        default { "${Variant} build" }
    }
}

function Get-TorchSourceLabel {
    param([string]$TorchIndexUrl)
    if ($TorchIndexUrl) {
        return $TorchIndexUrl
    }
    return "PyPI default index"
}

function Get-TorchActionLabel {
    param(
        [string]$TorchStatus,
        [string]$TorchVersion,
        [bool]$VenvCreated
    )
    if ($DryRunMode) {
        switch ($TorchStatus) {
            "valid" { return "Would keep installed PyTorch ${TorchVersion}" }
            "missing" {
                if (Test-Path $VenvDir) {
                    return "Would install validated PyTorch ${TorchMinVersion}-${TorchMaxValidatedVersion}, then install TorchCTS"
                }
                return "Would create the venv, install validated PyTorch ${TorchMinVersion}-${TorchMaxValidatedVersion}, then install TorchCTS"
            }
            "too_old" { return "Would ask before replacing PyTorch ${TorchVersion}" }
            "too_new" { return "Would ask before replacing PyTorch ${TorchVersion}" }
            "broken" { return "Would stop because installed PyTorch cannot be imported" }
            default { return "Would install validated PyTorch ${TorchMinVersion}-${TorchMaxValidatedVersion}, then install TorchCTS" }
        }
    }
    switch ($TorchStatus) {
        "valid" { return "Keep installed PyTorch ${TorchVersion}" }
        "missing" { return "Install validated PyTorch ${TorchMinVersion}-${TorchMaxValidatedVersion}" }
        "too_old" {
            if ($VenvCreated) {
                return "Stop because the new venv has unexpected PyTorch ${TorchVersion}"
            }
            return "Ask before replacing PyTorch ${TorchVersion}"
        }
        "too_new" {
            if ($VenvCreated) {
                return "Stop because the new venv has unexpected PyTorch ${TorchVersion}"
            }
            return "Ask before replacing PyTorch ${TorchVersion}"
        }
        "broken" { return "Stop because installed PyTorch cannot be imported" }
        default { return "Install validated PyTorch ${TorchMinVersion}-${TorchMaxValidatedVersion}" }
    }
}

function Write-InstallPlanSummary {
    Write-Host ""
    Write-Host "Install plan" -ForegroundColor White
    Write-Host "  Package:    TorchCTS editable checkout"
    if ($DryRunMode) {
        Write-Host "  Mode:       Dry run (no files will be changed)"
    } else {
        Write-Host "  Mode:       Install"
    }
    Write-Host "  Location:   $(Get-Location)\${VenvDir}"
    Write-Host "  Python:     ${pyVersion} (${Python})"
    Write-Host "  PyTorch:    $(Get-TorchBuildLabel -Variant $GpuType)"
    Write-Host "  Validated:  ${TorchMinVersion}-${TorchMaxValidatedVersion} (${TorchSpec})"
    Write-Host "  Torch src:  $(Get-TorchSourceLabel -TorchIndexUrl $TorchIndexUrl)"
    Write-Host "  Device:     ${DeviceHint}"
    Write-Host "  Detection:  ${TorchReason}"
    Write-Host "  Action:     $(Get-TorchActionLabel -TorchStatus $TorchStatus -TorchVersion $TorchVersion -VenvCreated $VenvCreated)"
    Write-Host ""
}

function Confirm-WrongTorchInstall {
    param(
        [string]$TorchDetail,
        [string]$TorchVersion,
        [string]$ValidatedRange,
        [string]$TorchSpec
    )

    Write-Host "[..] $TorchDetail" -ForegroundColor Yellow
    Write-Host "     Installed PyTorch: ${TorchVersion}; validated PyTorch: ${ValidatedRange} (${TorchSpec})." -ForegroundColor Yellow
    if ($AutoYes) {
        Write-Host "[..] Auto-approved PyTorch replacement via -Yes/TORCHCTS_YES=1." -ForegroundColor Cyan
        return
    }
    if ($env:TORCHCTS_NON_INTERACTIVE -eq "1" -or -not [Environment]::UserInteractive -or [Console]::IsInputRedirected) {
        throw "PyTorch version is not in the validated range. Run setup interactively to approve installing a validated PyTorch build, or install ${TorchSpec} manually first."
    }
    $answer = Read-Host "Install validated PyTorch and continue? [y/N]"
    if ($answer -notmatch '^(?i:y|yes)$') {
        throw "Aborted before changing PyTorch."
    }
}

# ── Locate Python ───────────────────────────────────────────────────────────

$Python = $null
$ExistingVenvPython = Join-Path $VenvDir "Scripts\python.exe"
$PythonCandidates = @()
if (Test-Path $ExistingVenvPython) {
    $PythonCandidates += $ExistingVenvPython
}
$PythonCandidates += @("python", "python3", "py")

foreach ($candidate in $PythonCandidates) {
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

$VenvCreated = $false
if (Test-Path $VenvDir) {
    Write-Host "[..] Existing ${VenvDir} found - reusing it." -ForegroundColor Cyan
} elseif ($DryRunMode) {
    Write-Host "[..] No existing ${VenvDir} found - dry run will not create it." -ForegroundColor Cyan
} else {
    Write-Host "[..] Creating virtual environment in ${VenvDir}..." -ForegroundColor Cyan
    & $Python @PythonArgs -m venv $VenvDir
    $VenvCreated = $true
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

Write-Host "[OK] PyTorch build: $(Get-TorchBuildLabel -Variant $GpuType)" -ForegroundColor Green
Write-Host "[..] Detection: $TorchReason" -ForegroundColor Cyan
if ($TorchWarning) {
    Write-Host "[..] $TorchWarning" -ForegroundColor Yellow
}

# ── Upgrade pip and install ─────────────────────────────────────────────────

$Pip = Join-Path $VenvDir "Scripts\pip.exe"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if ($DryRunMode -and -not (Test-Path $VenvDir)) {
    $TorchStatus = "missing"
    $TorchVersion = ""
    $TorchDetail = "No existing TorchCTS virtual environment was found."
    Write-InstallPlanSummary
    Write-Host "[..] Dry run complete. No files were changed." -ForegroundColor Cyan
    return
}

if (-not $DryRunMode) {
    Write-Host "[..] Upgrading pip, setuptools, wheel..." -ForegroundColor Cyan
    & $Pip install --upgrade pip setuptools wheel --quiet
}

Write-Host "[..] Checking PyTorch install..." -ForegroundColor Cyan
$torchStatusOutput = & $VenvPython $PlanFile --torch-status --format key-value
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch status check failed."
}
$torchStatusPlan = Read-InstallPlan -Lines $torchStatusOutput
$TorchStatus = $torchStatusPlan["status"]
$TorchVersion = $torchStatusPlan["version"]
$TorchDetail = $torchStatusPlan["detail"]

Write-InstallPlanSummary

if ($DryRunMode) {
    switch ($TorchStatus) {
        "valid" { Write-Host "[OK] Existing PyTorch ${TorchVersion} is in the validated range." -ForegroundColor Green }
        "missing" { Write-Host "[..] PyTorch is not installed in the existing venv." -ForegroundColor Yellow }
        "too_old" { Write-Host "[..] $TorchDetail" -ForegroundColor Yellow }
        "too_new" { Write-Host "[..] $TorchDetail" -ForegroundColor Yellow }
        "broken" { Write-Host "[..] $TorchDetail" -ForegroundColor Yellow }
        default { Write-Host "[..] PyTorch status is unknown." -ForegroundColor Yellow }
    }
    Write-Host "[..] Dry run complete. No files were changed." -ForegroundColor Cyan
    return
}

$TorchInstallAttempted = $false
if ($TorchStatus -eq "valid") {
    Write-Host "[OK] Keeping existing PyTorch ${TorchVersion}." -ForegroundColor Green
} elseif (($TorchStatus -eq "too_old" -or $TorchStatus -eq "too_new") -and $VenvCreated) {
    throw "$TorchDetail Installer-created venv contains PyTorch ${TorchVersion}, but TorchCTS requires ${TorchMinVersion}-${TorchMaxValidatedVersion} (${TorchSpec}). Refusing to continue."
} elseif ($TorchStatus -eq "too_old" -or $TorchStatus -eq "too_new") {
    Confirm-WrongTorchInstall -TorchDetail $TorchDetail -TorchVersion $TorchVersion -ValidatedRange "${TorchMinVersion}-${TorchMaxValidatedVersion}" -TorchSpec $TorchSpec
    Write-Host "[..] Installing validated PyTorch (${GpuType})..." -ForegroundColor Cyan
    $TorchInstallAttempted = $true
    $torchInstallArgs = @("install", "--upgrade", $TorchSpec)
    if ($TorchIndexUrl) {
        $torchInstallArgs += @("--index-url", $TorchIndexUrl)
    }
    $torchInstallArgs += "--quiet"
    & $Pip @torchInstallArgs
} elseif ($TorchStatus -eq "broken") {
    throw "$TorchDetail Fix the PyTorch install manually before running setup again."
} else {
    Write-Host "[..] Installing PyTorch (${GpuType})..." -ForegroundColor Cyan
    $TorchInstallAttempted = $true
    $torchInstallArgs = @("install")
    $torchInstallArgs += $TorchSpec
    if ($TorchIndexUrl) {
        $torchInstallArgs += @("--index-url", $TorchIndexUrl)
    }
    $torchInstallArgs += "--quiet"
    & $Pip @torchInstallArgs
}

if ($TorchInstallAttempted) {
    Write-Host "[..] Checking installed PyTorch version..." -ForegroundColor Cyan
    $torchStatusOutput = & $VenvPython $PlanFile --torch-status --format key-value
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch status check failed."
    }
    $torchStatusPlan = Read-InstallPlan -Lines $torchStatusOutput
    $TorchStatus = $torchStatusPlan["status"]
    $TorchVersion = $torchStatusPlan["version"]
    $TorchDetail = $torchStatusPlan["detail"]
    if ($TorchStatus -ne "valid") {
        throw "$TorchDetail Installer-managed PyTorch install produced ${TorchVersion}; expected ${TorchMinVersion}-${TorchMaxValidatedVersion} (${TorchSpec})."
    }
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

$torchVersion = & $VenvPython -c "import torch; print(torch.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) { $torchVersion = "unknown" }

# ── Summary ─────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "TorchCTS development environment ready." -ForegroundColor Green
Write-Host ""
Write-Host "  Version:    ${installedVersion}"
Write-Host "  Python:     ${pyVersion}"
Write-Host "  PyTorch:    ${torchVersion} ($(Get-TorchBuildLabel -Variant $GpuType))"
Write-Host "  Venv:       $(Get-Location)\${VenvDir}"
Write-Host ""
Write-Host "  Activate:   .\.venv\Scripts\Activate.ps1"
Write-Host "  Run:        torchcts run --device ${DeviceHint}"
Write-Host ""
