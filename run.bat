@echo off
setlocal EnableDelayedExpansion

echo.
echo  ███╗   ███╗███╗   ██╗██╗███████╗████████╗
echo  ████╗ ████║████╗  ██║██║██╔════╝╚══██╔══╝
echo  ██╔████╔██║██╔██╗ ██║██║███████╗   ██║
echo  ██║╚██╔╝██║██║╚██╗██║██║╚════██║   ██║
echo  ██║ ╚═╝ ██║██║ ╚████║██║███████║   ██║
echo  ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝╚══════╝   ╚═╝
echo  Visualizer
echo.

:: ── Locate Python ────────────────────────────────────────────────────────────
set PYTHON=
for %%P in (python python3) do (
    if not defined PYTHON (
        where %%P >nul 2>&1 && set PYTHON=%%P
    )
)

if not defined PYTHON (
    echo [ERROR] Python not found.
    echo         Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=*" %%V in ('%PYTHON% --version 2^>^&1') do set PY_VER=%%V
echo [INFO] Found: %PY_VER%

:: ── Create venv if it does not exist ─────────────────────────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: ── Activate venv ─────────────────────────────────────────────────────────────
call .venv\Scripts\activate.bat

:: ── Install / update dependencies ─────────────────────────────────────────────
echo [INFO] Checking dependencies (first run may take a minute)...
python -m pip install --upgrade pip --quiet
python -m pip install -e . --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

:: ── Launch ────────────────────────────────────────────────────────────────────
echo [INFO] Launching MNIST Visualizer...
echo.
mnist-visualizer

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
endlocal
