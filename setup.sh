#!/usr/bin/env bash
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
# TorchCTS - Repository Setup Script (macOS / Linux)
#
# Usage:
#   ./setup.sh
#
# Creates a local .venv and installs TorchCTS in editable mode
# for development and contribution.

set -euo pipefail

VENV_DIR=".venv"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10
PLAN_FILE="torchcts/site_scripts/install_plan.py"
TORCH_MIN_VERSION="2.7.0"
TORCH_SPEC="torch>=${TORCH_MIN_VERSION}"

# ── Colors ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { printf "${CYAN}▸${NC} %s\n" "$*"; }
ok()    { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}⚠${NC} %s\n" "$*"; }
err()   { printf "${RED}✗${NC} %s\n" "$*" >&2; }

# ── Locate Python ───────────────────────────────────────────────────────────

PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python not found."
    echo ""
    case "$(uname -s)" in
        Darwin)
            echo "  Install via Homebrew:  brew install python" ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                echo "  Install via apt:      sudo apt update && sudo apt install python3 python3-venv"
            elif [ -f /etc/fedora-release ] || [ -f /etc/redhat-release ]; then
                echo "  Install via dnf:      sudo dnf install python3"
            else
                echo "  Install Python 3.10+ from https://python.org"
            fi ;;
        *)
            echo "  Install Python 3.10+ from https://python.org" ;;
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

if ! "$PYTHON" -m venv --help &>/dev/null; then
    err "Python venv module is not available."
    if [ -f /etc/debian_version ]; then
        echo "  Fix:  sudo apt install python3-venv"
    else
        echo "  Your Python installation is missing the venv module."
    fi
    exit 1
fi

if [ ! -f "$PLAN_FILE" ]; then
    err "Install planner not found: ${PLAN_FILE}"
    exit 1
fi

# ── Create or reuse venv ────────────────────────────────────────────────────

if [ -d "$VENV_DIR" ]; then
    info "Existing ${VENV_DIR} found - reusing it."
else
    info "Creating virtual environment in ${VENV_DIR}..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created."
fi

# ── Plan PyTorch install ────────────────────────────────────────────────────

PLAN_OUTPUT_FILE=$(mktemp "${TMPDIR:-/tmp}/torchcts_setup_plan.XXXXXX")
trap 'rm -f "$PLAN_OUTPUT_FILE"' EXIT INT TERM

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

# ── Upgrade pip and install ─────────────────────────────────────────────────

PIP="${VENV_DIR}/bin/pip"
VENV_PYTHON="${VENV_DIR}/bin/python"

info "Upgrading pip and wheel..."
"$PIP" install --upgrade pip setuptools wheel --quiet

info "Checking PyTorch install..."
TORCH_STATUS_OUTPUT=$("$VENV_PYTHON" "$PLAN_FILE" --torch-status --format key-value)
TORCH_STATUS=""
TORCH_VERSION=""
TORCH_DETAIL=""
while IFS='=' read -r key value; do
    case "$key" in
        status) TORCH_STATUS=$value ;;
        version) TORCH_VERSION=$value ;;
        detail) TORCH_DETAIL=$value ;;
    esac
done <<EOF
$TORCH_STATUS_OUTPUT
EOF

TORCH_UPGRADE_REQUESTED="${TORCHCTS_UPGRADE_TORCH:-0}"
if [ "$TORCH_STATUS" = "valid" ] && [ "$TORCH_UPGRADE_REQUESTED" != "1" ]; then
    ok "Keeping existing PyTorch ${TORCH_VERSION}."
elif [ "$TORCH_STATUS" = "too_old" ] && [ "$TORCH_UPGRADE_REQUESTED" != "1" ]; then
    err "$TORCH_DETAIL"
    echo "  Install a PyTorch ${TORCH_MIN_VERSION}+ build manually, or set TORCHCTS_UPGRADE_TORCH=1 to let setup upgrade it."
    exit 1
elif [ "$TORCH_STATUS" = "broken" ] && [ "$TORCH_UPGRADE_REQUESTED" != "1" ]; then
    err "$TORCH_DETAIL"
    echo "  Fix the PyTorch install manually, or set TORCHCTS_UPGRADE_TORCH=1 to let setup reinstall it."
    exit 1
else
    info "Installing PyTorch (${TORCH_VARIANT})..."
    if [ "$TORCH_UPGRADE_REQUESTED" = "1" ]; then
        if [ -n "$TORCH_INDEX_URL" ]; then
            "$PIP" install --upgrade "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL" --quiet
        else
            "$PIP" install --upgrade "$TORCH_SPEC" --quiet
        fi
    else
        if [ -n "$TORCH_INDEX_URL" ]; then
            "$PIP" install "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL" --quiet
        else
            "$PIP" install "$TORCH_SPEC" --quiet
        fi
    fi
fi

info "Installing TorchCTS in editable mode..."
"$PIP" install -e . --quiet

info "Verifying PyTorch install..."
"$VENV_PYTHON" "$PLAN_FILE" --verify "$TORCH_VARIANT"

INSTALLED_VERSION=$("${VENV_PYTHON}" -c "import torchcts; print(torchcts.__version__)" 2>/dev/null || echo "unknown")

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
printf "${BOLD}${GREEN}TorchCTS development environment ready.${NC}\n"
echo ""
echo "  Version:    ${INSTALLED_VERSION}"
echo "  Python:     ${PYTHON_VERSION}"
echo "  PyTorch:    ${TORCH_VARIANT}"
echo "  Venv:       $(pwd)/${VENV_DIR}"
echo ""
echo "  Activate:   source ${VENV_DIR}/bin/activate"
echo "  Run:        torchcts run --device ${TORCH_DEVICE_HINT}"
echo ""
