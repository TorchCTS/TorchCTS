#!/bin/sh
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
# TorchCTS — Global Installer (macOS / Linux)
#
# Usage:
#   curl -fsSL https://torchcts.ai/scripts/install.sh | sh
#   curl -fsSL https://torchcts.ai/scripts/install.sh | sh -s -- --uninstall
#
# Installs TorchCTS from PyPI into ~/.torchcts/venv.

set -eu

INSTALL_DIR="$HOME/.torchcts"
VENV_DIR="$INSTALL_DIR/venv"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

# ── Colors (only if stdout is a terminal) ────────────────────────────────────

if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' NC=''
fi

info()  { printf "${CYAN}▸${NC} %s\n" "$*"; }
ok()    { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}⚠${NC} %s\n" "$*"; }
err()   { printf "${RED}✗${NC} %s\n" "$*" >&2; }

# ── Uninstall ────────────────────────────────────────────────────────────────

if [ "${1:-}" = "--uninstall" ]; then
    info "Uninstalling TorchCTS..."
    if [ -d "$INSTALL_DIR" ]; then
        rm -rf "$INSTALL_DIR"
        ok "Removed $INSTALL_DIR"
        ok "TorchCTS uninstalled."
    else
        warn "Nothing to remove — TorchCTS is not installed."
    fi
    exit 0
fi

# ── Banner ───────────────────────────────────────────────────────────────────

printf "\n${BOLD}TorchCTS Installer${NC}\n\n"

# ── Detect OS ────────────────────────────────────────────────────────────────

OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)
        err "Unsupported OS: $OS"
        echo "  This installer supports macOS and Linux."
        echo "  For Windows, use install.ps1 instead."
        exit 1
        ;;
esac

# ── Locate Python ───────────────────────────────────────────────────────────

PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python not found."
    echo ""
    case "$PLATFORM" in
        macos)
            echo "  Install via Homebrew:  brew install python"
            ;;
        linux)
            if [ -f /etc/debian_version ]; then
                echo "  Install via apt:      sudo apt update && sudo apt install python3 python3-venv"
            elif [ -f /etc/fedora-release ] || [ -f /etc/redhat-release ]; then
                echo "  Install via dnf:      sudo dnf install python3"
            elif [ -f /etc/arch-release ]; then
                echo "  Install via pacman:   sudo pacman -S python"
            else
                echo "  Install Python 3.10+ from https://python.org"
            fi
            ;;
    esac
    exit 1
fi

# ── Verify version ──────────────────────────────────────────────────────────

PYTHON_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt "$MIN_PYTHON_MAJOR" ] || \
   { [ "$PYTHON_MAJOR" -eq "$MIN_PYTHON_MAJOR" ] && [ "$PYTHON_MINOR" -lt "$MIN_PYTHON_MINOR" ]; }; then
    err "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ required, found ${PYTHON_VERSION}."
    exit 1
fi

ok "Found Python ${PYTHON_VERSION} ($(command -v "$PYTHON"))"

# ── Verify venv module ──────────────────────────────────────────────────────

if ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
    err "Python venv module is not available."
    if [ -f /etc/debian_version ]; then
        echo "  Fix:  sudo apt install python3-venv"
    else
        echo "  Your Python installation is missing the venv module."
        echo "  Reinstall Python or install the venv package for your distribution."
    fi
    exit 1
fi

# ── Detect GPU for PyTorch wheel selection ───────────────────────────────────

TORCH_INDEX_ARGS=""
GPU_TYPE="cpu"

if [ "$PLATFORM" = "macos" ]; then
    # macOS: default PyPI wheel includes MPS support, no extra index needed.
    GPU_TYPE="mps"
elif [ "$PLATFORM" = "linux" ]; then
    # Check for NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1 || [ -f /proc/driver/nvidia/version ]; then
        GPU_TYPE="cuda"
        # Default PyPI wheel includes CUDA, no extra index needed.
    # Check for AMD GPU (ROCm)
    elif command -v rocm-smi >/dev/null 2>&1 || [ -d /opt/rocm ]; then
        GPU_TYPE="rocm"
        TORCH_INDEX_ARGS="--extra-index-url https://download.pytorch.org/whl/rocm6.3"
    else
        # CPU-only: use lightweight wheel to avoid downloading ~2.5GB CUDA build
        GPU_TYPE="cpu"
        TORCH_INDEX_ARGS="--extra-index-url https://download.pytorch.org/whl/cpu"
    fi
fi

ok "GPU detection: ${GPU_TYPE}"

# ── Create or reuse venv ────────────────────────────────────────────────────

if [ -d "$VENV_DIR" ]; then
    info "Existing installation found — upgrading."
else
    info "Creating virtual environment in ${VENV_DIR}..."
    mkdir -p "$INSTALL_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created."
fi

# ── Install / upgrade ───────────────────────────────────────────────────────

PIP="${VENV_DIR}/bin/pip"
VENV_PYTHON="${VENV_DIR}/bin/python"

info "Upgrading pip..."
"$PIP" install --upgrade pip --quiet

if [ -n "$TORCH_INDEX_ARGS" ]; then
    info "Installing TorchCTS (PyTorch variant: ${GPU_TYPE})..."
    # shellcheck disable=SC2086
    "$PIP" install --upgrade torchcts $TORCH_INDEX_ARGS --quiet
else
    info "Installing TorchCTS (PyTorch variant: ${GPU_TYPE})..."
    "$PIP" install --upgrade torchcts --quiet
fi

INSTALLED_VERSION=$("$VENV_PYTHON" -c "import torchcts; print(torchcts.__version__)" 2>/dev/null || echo "unknown")
TORCH_VERSION=$("$VENV_PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")

ok "Installed TorchCTS ${INSTALLED_VERSION} (PyTorch ${TORCH_VERSION})"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
printf "${BOLD}${GREEN}TorchCTS ${INSTALLED_VERSION} installed successfully.${NC}\n"
echo ""
echo "  Version:    ${INSTALLED_VERSION}"
echo "  PyTorch:    ${TORCH_VERSION} (${GPU_TYPE})"
echo "  Venv:       ${VENV_DIR}"
echo ""

case "$GPU_TYPE" in
    mps)  echo "  Run:        ${VENV_DIR}/bin/torchcts run --device mps" ;;
    cuda) echo "  Run:        ${VENV_DIR}/bin/torchcts run --device cuda" ;;
    rocm) echo "  Run:        ${VENV_DIR}/bin/torchcts run --device cuda" ;;
    cpu)  echo "  Run:        ${VENV_DIR}/bin/torchcts run --device cpu" ;;
esac

echo "  Uninstall:  curl -fsSL https://torchcts.ai/scripts/install.sh | sh -s -- --uninstall"
echo ""
