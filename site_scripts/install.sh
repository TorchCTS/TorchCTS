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
#   curl -fsSL https://torchcts.ai/scripts/install.sh | sh -s -- --yes
#   curl -fsSL https://torchcts.ai/scripts/install.sh | sh -s -- --dry-run
#   curl -fsSL https://torchcts.ai/scripts/install.sh | sh -s -- --uninstall
#
# Installs TorchCTS from PyPI into ~/.torchcts/venv.

set -eu

INSTALL_DIR="$HOME/.torchcts"
VENV_DIR="$INSTALL_DIR/venv"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10
TORCH_MIN_VERSION="2.7.0"
TORCH_MAX_EXCLUSIVE_VERSION="2.12.2"
TORCH_MAX_VALIDATED_VERSION="2.12.1"
TORCH_SPEC="torch>=${TORCH_MIN_VERSION},<${TORCH_MAX_EXCLUSIVE_VERSION}"
AUTO_YES="${TORCHCTS_YES:-0}"
DRY_RUN="${TORCHCTS_DRY_RUN:-0}"

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

can_prompt() {
    [ "${TORCHCTS_NON_INTERACTIVE:-}" != "1" ] || return 1
    [ -r /dev/tty ] || return 1
    [ -w /dev/tty ] || return 1
    ( : < /dev/tty ) 2>/dev/null
}

torch_build_label() {
    case "$1" in
        cpu) echo "CPU build" ;;
        cuda) echo "NVIDIA CUDA build" ;;
        rocm) echo "AMD ROCm build" ;;
        xpu) echo "Intel XPU build" ;;
        mps) echo "Apple Metal/MPS build" ;;
        *) echo "$1 build" ;;
    esac
}

torch_source_label() {
    if [ -n "$TORCH_INDEX_URL" ]; then
        echo "$TORCH_INDEX_URL"
    else
        echo "PyPI default index"
    fi
}

torch_action_label() {
    if [ "$DRY_RUN" = "1" ]; then
        case "$TORCH_STATUS" in
            valid) echo "Would keep installed PyTorch ${TORCH_VERSION:-unknown}" ;;
            missing)
                if [ -d "$VENV_DIR" ]; then
                    echo "Would install validated PyTorch ${TORCH_MIN_VERSION}-${TORCH_MAX_VALIDATED_VERSION}, then install TorchCTS"
                else
                    echo "Would create the venv, install validated PyTorch ${TORCH_MIN_VERSION}-${TORCH_MAX_VALIDATED_VERSION}, then install TorchCTS"
                fi
                ;;
            too_old|too_new) echo "Would ask before replacing PyTorch ${TORCH_VERSION:-unknown}" ;;
            broken) echo "Would stop because installed PyTorch cannot be imported" ;;
            *) echo "Would install validated PyTorch ${TORCH_MIN_VERSION}-${TORCH_MAX_VALIDATED_VERSION}, then install TorchCTS" ;;
        esac
        return
    fi
    case "$TORCH_STATUS" in
        valid) echo "Keep installed PyTorch ${TORCH_VERSION:-unknown}" ;;
        missing) echo "Install validated PyTorch ${TORCH_MIN_VERSION}-${TORCH_MAX_VALIDATED_VERSION}" ;;
        too_old|too_new)
            if [ "$VENV_CREATED" = "1" ]; then
                echo "Stop because the new venv has unexpected PyTorch ${TORCH_VERSION:-unknown}"
            else
                echo "Ask before replacing PyTorch ${TORCH_VERSION:-unknown}"
            fi
            ;;
        broken) echo "Stop because installed PyTorch cannot be imported" ;;
        *) echo "Install validated PyTorch ${TORCH_MIN_VERSION}-${TORCH_MAX_VALIDATED_VERSION}" ;;
    esac
}

print_install_plan() {
    echo ""
    printf "${BOLD}Install plan${NC}\n"
    echo "  Package:    TorchCTS from PyPI"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  Mode:       Dry run (no files will be changed)"
    else
        echo "  Mode:       Install"
    fi
    echo "  Location:   ${VENV_DIR}"
    echo "  Python:     ${PYTHON_VERSION} ($(command -v "$PYTHON"))"
    echo "  PyTorch:    $(torch_build_label "$TORCH_VARIANT")"
    echo "  Validated:  ${TORCH_MIN_VERSION}-${TORCH_MAX_VALIDATED_VERSION} (${TORCH_SPEC})"
    echo "  Torch src:  $(torch_source_label)"
    echo "  Device:     ${TORCH_DEVICE_HINT}"
    echo "  Detection:  ${TORCH_REASON}"
    echo "  Action:     $(torch_action_label)"
    echo ""
}

confirm_wrong_torch_install() {
    warn "$TORCH_DETAIL"
    echo "  Installed PyTorch: ${TORCH_VERSION:-unknown}; validated PyTorch: ${TORCH_MIN_VERSION}-${TORCH_MAX_VALIDATED_VERSION} (${TORCH_SPEC})."
    if [ "$AUTO_YES" = "1" ]; then
        info "Auto-approved PyTorch replacement via --yes/TORCHCTS_YES=1."
        return
    fi
    if ! can_prompt; then
        err "PyTorch version is not in the validated range."
        echo "  Run the installer interactively to approve installing a validated PyTorch build, or install ${TORCH_SPEC} manually first."
        exit 1
    fi
    printf "  Install validated PyTorch and continue? [y/N] " > /dev/tty
    read -r answer < /dev/tty || answer=""
    case "$answer" in
        y|Y|yes|YES|Yes) ;;
        *)
            err "Aborted before changing PyTorch."
            exit 1
            ;;
    esac
}

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

UNINSTALL=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --uninstall) UNINSTALL=1 ;;
        --yes) AUTO_YES=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            echo "Usage: install.sh [--yes] [--dry-run] [--uninstall]"
            exit 0
            ;;
        *)
            err "Unknown option: $1"
            echo "Usage: install.sh [--yes] [--dry-run] [--uninstall]"
            exit 1
            ;;
    esac
    shift
done

if [ "$UNINSTALL" = "1" ]; then
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
for candidate in "${VENV_DIR}/bin/python" python3 python; do
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

VENV_CREATED=0
if [ -d "$VENV_DIR" ]; then
    info "Existing installation found - upgrading."
elif [ "$DRY_RUN" = "1" ]; then
    info "No existing installation found - dry run will not create ${VENV_DIR}."
else
    info "Creating virtual environment in ${VENV_DIR}..."
    mkdir -p "$INSTALL_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
    VENV_CREATED=1
    ok "Virtual environment created."
fi

# ── Write embedded planner ──────────────────────────────────────────────────

PLAN_FILE=$(mktemp "${TMPDIR:-/tmp}/torchcts_install_plan.XXXXXX")
PLAN_OUTPUT_FILE=$(mktemp "${TMPDIR:-/tmp}/torchcts_install_plan_output.XXXXXX")

info "Preparing install planner..."
write_install_plan

# ── Plan PyTorch install ────────────────────────────────────────────────────

info "Selecting PyTorch build..."
if can_prompt; then
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

ok "PyTorch build: $(torch_build_label "$TORCH_VARIANT")"
info "Detection: $TORCH_REASON"
if [ -n "$TORCH_WARNING" ]; then
    warn "$TORCH_WARNING"
fi

# ── Install / upgrade ───────────────────────────────────────────────────────

PIP="${VENV_DIR}/bin/pip"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [ "$DRY_RUN" = "1" ] && [ ! -d "$VENV_DIR" ]; then
    TORCH_STATUS="missing"
    TORCH_VERSION=""
    TORCH_DETAIL="No existing TorchCTS virtual environment was found."
    print_install_plan
    info "Dry run complete. No files were changed."
    exit 0
fi

if [ "$DRY_RUN" != "1" ]; then
    info "Upgrading pip..."
    "$PIP" install --upgrade pip --quiet
fi

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

print_install_plan

if [ "$DRY_RUN" = "1" ]; then
    case "$TORCH_STATUS" in
        valid) ok "Existing PyTorch ${TORCH_VERSION} is in the validated range." ;;
        missing) warn "PyTorch is not installed in the existing venv." ;;
        too_old|too_new|broken) warn "$TORCH_DETAIL" ;;
        *) warn "PyTorch status is unknown." ;;
    esac
    info "Dry run complete. No files were changed."
    exit 0
fi

TORCH_INSTALL_ATTEMPTED=0
if [ "$TORCH_STATUS" = "valid" ]; then
    ok "Keeping existing PyTorch ${TORCH_VERSION}."
elif { [ "$TORCH_STATUS" = "too_old" ] || [ "$TORCH_STATUS" = "too_new" ]; } && [ "$VENV_CREATED" = "1" ]; then
    err "$TORCH_DETAIL"
    echo "  Installer-created venv contains PyTorch ${TORCH_VERSION:-unknown}, but TorchCTS requires ${TORCH_MIN_VERSION}-${TORCH_MAX_VALIDATED_VERSION} (${TORCH_SPEC}). Refusing to continue."
    exit 1
elif [ "$TORCH_STATUS" = "too_old" ] || [ "$TORCH_STATUS" = "too_new" ]; then
    confirm_wrong_torch_install
    info "Installing validated PyTorch (${TORCH_VARIANT})..."
    TORCH_INSTALL_ATTEMPTED=1
    if [ -n "$TORCH_INDEX_URL" ]; then
        "$PIP" install --upgrade "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL" --quiet
    else
        "$PIP" install --upgrade "$TORCH_SPEC" --quiet
    fi
elif [ "$TORCH_STATUS" = "broken" ]; then
    err "$TORCH_DETAIL"
    echo "  Fix the PyTorch install manually before running the installer again."
    exit 1
else
    info "Installing PyTorch (${TORCH_VARIANT})..."
    TORCH_INSTALL_ATTEMPTED=1
    if [ -n "$TORCH_INDEX_URL" ]; then
        "$PIP" install "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL" --quiet
    else
        "$PIP" install "$TORCH_SPEC" --quiet
    fi
fi

if [ "$TORCH_INSTALL_ATTEMPTED" = "1" ]; then
    info "Checking installed PyTorch version..."
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
    if [ "$TORCH_STATUS" != "valid" ]; then
        err "$TORCH_DETAIL"
        echo "  Installer-managed PyTorch install produced ${TORCH_VERSION:-unknown}; expected ${TORCH_MIN_VERSION}-${TORCH_MAX_VALIDATED_VERSION} (${TORCH_SPEC})."
        exit 1
    fi
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
echo "  PyTorch:    ${TORCH_VERSION} ($(torch_build_label "$TORCH_VARIANT"))"
echo "  Venv:       ${VENV_DIR}"
echo ""
echo "  Run:        ${VENV_DIR}/bin/torchcts run --device ${TORCH_DEVICE_HINT}"
echo "  Uninstall:  curl -fsSL https://torchcts.ai/scripts/install.sh | sh -s -- --uninstall"
echo ""
