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
# TorchCTS — Repository Setup Script (Windows)
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
    exit 1
}

# For 'py' launcher, use 'py -3' to ensure Python 3
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
    exit 1
}

Write-Host "[OK] Found Python ${pyVersion}" -ForegroundColor Green

# ── Create or reuse venv ────────────────────────────────────────────────────

if (Test-Path $VenvDir) {
    Write-Host "[..] Existing ${VenvDir} found - reusing it." -ForegroundColor Cyan
} else {
    Write-Host "[..] Creating virtual environment in ${VenvDir}..." -ForegroundColor Cyan
    & $Python @PythonArgs -m venv $VenvDir
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
}

# ── Upgrade pip and install ─────────────────────────────────────────────────

$Pip = Join-Path $VenvDir "Scripts\pip.exe"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "[..] Upgrading pip, setuptools, wheel..." -ForegroundColor Cyan
& $Pip install --upgrade pip setuptools wheel --quiet

Write-Host "[..] Installing TorchCTS in editable mode..." -ForegroundColor Cyan
& $Pip install -e . --quiet

$installedVersion = & $VenvPython -c "import torchcts; print(torchcts.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) { $installedVersion = "unknown" }

# ── Summary ─────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "TorchCTS development environment ready." -ForegroundColor Green
Write-Host ""
Write-Host "  Version:    ${installedVersion}"
Write-Host "  Python:     ${pyVersion}"
Write-Host "  Venv:       $(Get-Location)\${VenvDir}"
Write-Host ""
Write-Host "  Activate:   .\.venv\Scripts\Activate.ps1"
Write-Host "  Run:        torchcts run --device cuda"
Write-Host ""
