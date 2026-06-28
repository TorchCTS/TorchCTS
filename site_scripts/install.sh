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
# TorchCTS - Global Installer (macOS / Linux)
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
TORCH_SPEC="torch>=2.12.0"

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

PLAN_FILE=""
PLAN_OUTPUT_FILE=""
cleanup() {
    if [ -n "$PLAN_FILE" ]; then
        rm -f "$PLAN_FILE"
    fi
    if [ -n "$PLAN_OUTPUT_FILE" ]; then
        rm -f "$PLAN_OUTPUT_FILE"
    fi
}
trap cleanup EXIT INT TERM

write_install_plan() {
    cat > "$PLAN_FILE" <<'__TORCHCTS_INSTALL_PLAN_PAYLOAD__'
__TORCHCTS_INSTALL_PLAN_PY__
__TORCHCTS_INSTALL_PLAN_PAYLOAD__
}

# ── Uninstall ────────────────────────────────────────────────────────────────

if [ "${1:-}" = "--uninstall" ]; then
    info "Uninstalling TorchCTS..."
    if [ -d "$INSTALL_DIR" ]; then
        rm -rf "$INSTALL_DIR"
        ok "Removed $INSTALL_DIR"
        ok "TorchCTS uninstalled."
    else
        warn "Nothing to remove - TorchCTS is not installed."
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

# ── Create or reuse venv ────────────────────────────────────────────────────

if [ -d "$VENV_DIR" ]; then
    info "Existing installation found - upgrading."
else
    info "Creating virtual environment in ${VENV_DIR}..."
    mkdir -p "$INSTALL_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created."
fi

# ── Write embedded planner ──────────────────────────────────────────────────

PLAN_FILE=$(mktemp "${TMPDIR:-/tmp}/torchcts_install_plan.XXXXXX")
PLAN_OUTPUT_FILE=$(mktemp "${TMPDIR:-/tmp}/torchcts_install_plan_output.XXXXXX")

info "Preparing install planner..."
write_install_plan

# ── Plan PyTorch install ────────────────────────────────────────────────────

info "Selecting PyTorch build..."
if [ "${TORCHCTS_NON_INTERACTIVE:-}" != "1" ] && [ -r /dev/tty ]; then
    "$PYTHON" "$PLAN_FILE" --format key-value --prompt < /dev/tty > "$PLAN_OUTPUT_FILE"
else
    "$PYTHON" "$PLAN_FILE" --format key-value > "$PLAN_OUTPUT_FILE"
fi

TORCH_VARIANT=""
TORCH_CONFIDENCE=""
TORCH_INDEX_URL=""
TORCH_DEVICE_HINT=""
TORCH_REASON=""
TORCH_WARNING=""

while IFS='=' read -r key value; do
    case "$key" in
        variant) TORCH_VARIANT=$value ;;
        confidence) TORCH_CONFIDENCE=$value ;;
        torch_index_url) TORCH_INDEX_URL=$value ;;
        device_hint) TORCH_DEVICE_HINT=$value ;;
        reason) TORCH_REASON=$value ;;
        warning) TORCH_WARNING=$value ;;
    esac
done < "$PLAN_OUTPUT_FILE"

if [ -z "$TORCH_VARIANT" ] || [ -z "$TORCH_DEVICE_HINT" ]; then
    err "Install planner did not return a usable PyTorch plan."
    exit 1
fi

ok "PyTorch selection: ${TORCH_VARIANT} (${TORCH_CONFIDENCE})"
info "$TORCH_REASON"
if [ -n "$TORCH_WARNING" ]; then
    warn "$TORCH_WARNING"
fi

# ── Install / upgrade ───────────────────────────────────────────────────────

PIP="${VENV_DIR}/bin/pip"
VENV_PYTHON="${VENV_DIR}/bin/python"

info "Upgrading pip..."
"$PIP" install --upgrade pip --quiet

info "Installing PyTorch (${TORCH_VARIANT})..."
if [ -n "$TORCH_INDEX_URL" ]; then
    "$PIP" install --upgrade "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL" --quiet
else
    "$PIP" install --upgrade "$TORCH_SPEC" --quiet
fi

info "Installing TorchCTS..."
"$PIP" install --upgrade torchcts --quiet

info "Verifying PyTorch install..."
"$VENV_PYTHON" "$PLAN_FILE" --verify "$TORCH_VARIANT"

INSTALLED_VERSION=$("$VENV_PYTHON" -c "import torchcts; print(torchcts.__version__)" 2>/dev/null || echo "unknown")
TORCH_VERSION=$("$VENV_PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")

ok "Installed TorchCTS ${INSTALLED_VERSION} (PyTorch ${TORCH_VERSION})"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
printf "${BOLD}${GREEN}TorchCTS ${INSTALLED_VERSION} installed successfully.${NC}\n"
echo ""
echo "  Version:    ${INSTALLED_VERSION}"
echo "  PyTorch:    ${TORCH_VERSION} (${TORCH_VARIANT})"
echo "  Venv:       ${VENV_DIR}"
echo ""
echo "  Run:        ${VENV_DIR}/bin/torchcts run --device ${TORCH_DEVICE_HINT}"
echo "  Uninstall:  curl -fsSL https://torchcts.ai/scripts/install.sh | sh -s -- --uninstall"
echo ""
