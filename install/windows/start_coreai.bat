@echo off
title CoreAI — Local AI Platform
cd /d "%~dp0..\.."

:: Auto-setup on first run
if not exist "venv\Scripts\python.exe" (
    echo [INFO] First run — creating virtual environment...
    python -m venv venv
    echo [INFO] Installing dependencies...
    venv\Scripts\pip install -r requirements.txt --quiet
)

call venv\Scripts\activate.bat

echo.
echo   +---------------------------------------------+
echo   ^|   CoreAI v1.0  ^|  http://localhost:8080     ^|
echo   +---------------------------------------------+
echo.

:: Open browser after 3s delay
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8080"

python server.py
pause
