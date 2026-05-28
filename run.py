#!/usr/bin/env python3
"""
One-click launcher for MNIST Visualizer.

Works on Windows, Linux, and macOS:
    python run.py

What it does:
  1. Creates a .venv virtual environment if one does not exist yet.
  2. Installs / updates all dependencies (pip install -e .).
  3. Launches the mnist-visualizer application.
"""

import subprocess
import sys
import os
from pathlib import Path

HERE   = Path(__file__).parent
VENV   = HERE / ".venv"
# platform-specific paths inside the venv
if sys.platform == "win32":
    PYTHON = VENV / "Scripts" / "python.exe"
    PIP    = VENV / "Scripts" / "pip.exe"
    EXE    = VENV / "Scripts" / "mnist-visualizer.exe"
else:
    PYTHON = VENV / "bin" / "python"
    PIP    = VENV / "bin" / "pip"
    EXE    = VENV / "bin" / "mnist-visualizer"


def run(cmd: list, **kwargs) -> None:
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    print("\n  MNIST Visualizer — launcher\n")

    # ── 1. Create venv if missing ─────────────────────────────────────────────
    if not PYTHON.exists():
        print("[INFO] Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV)])

    # ── 2. Install / update deps ──────────────────────────────────────────────
    print("[INFO] Checking dependencies (first run may take a minute)...")
    # use 'python -m pip' so pip can upgrade itself on Windows
    subprocess.run(
        [str(PYTHON), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        check=False,   # non-fatal if it fails
    )
    run([str(PYTHON), "-m", "pip", "install", "-e", str(HERE), "--quiet"])

    # ── 3. Launch ─────────────────────────────────────────────────────────────
    print("[INFO] Launching...\n")
    os.execv(str(PYTHON), [str(PYTHON), "-m", "mnist_visualizer"])


if __name__ == "__main__":
    main()
