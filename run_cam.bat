@echo off
REM Run cam.py using the .venv Python 3.11 environment

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Please ensure .venv is set up.
    pause
    exit /b
)

echo Starting Hand Tracking...
.venv\Scripts\python.exe cam.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
