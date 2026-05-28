#!/usr/bin/env bash
set -euo pipefail

echo ""
echo " ███╗   ███╗███╗   ██╗██╗███████╗████████╗"
echo " ████╗ ████║████╗  ██║██║██╔════╝╚══██╔══╝"
echo " ██╔████╔██║██╔██╗ ██║██║███████╗   ██║   "
echo " ██║╚██╔╝██║██║╚██╗██║██║╚════██║   ██║   "
echo " ██║ ╚═╝ ██║██║ ╚████║██║███████║   ██║   "
echo " ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝╚══════╝   ╚═╝  "
echo " Visualizer"
echo ""

# ── Locate Python ──────────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python not found."
    echo "        Please install Python 3.10+ from https://python.org"
    exit 1
fi

echo "[INFO] Found: $($PYTHON --version)"

# ── Create venv if it does not exist ───────────────────────────────────────────
if [ ! -f ".venv/bin/activate" ]; then
    echo "[INFO] Creating virtual environment..."
    $PYTHON -m venv .venv
fi

# ── Activate venv ──────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source .venv/bin/activate

# ── Install / update dependencies ──────────────────────────────────────────────
echo "[INFO] Checking dependencies (first run may take a minute)..."
pip install --upgrade pip --quiet
pip install -e . --quiet

# ── Launch ─────────────────────────────────────────────────────────────────────
echo "[INFO] Launching MNIST Visualizer..."
echo ""
mnist-visualizer
