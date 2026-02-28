@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: CoreAI — Build standalone Windows .exe with PyInstaller
:: Requirements: pip install pyinstaller
:: Output: dist\CoreAI\CoreAI.exe  (folder distribution)
::         dist\CoreAI.exe         (single-file, slower startup)
:: ─────────────────────────────────────────────────────────────────────────────
title CoreAI EXE Builder

cd /d "%~dp0..\.."

echo [INFO] Checking PyInstaller...
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing PyInstaller...
    pip install pyinstaller
)

echo [INFO] Building CoreAI.exe (one-folder mode)...
pyinstaller --noconfirm ^
  --name CoreAI ^
  --add-data "web;web" ^
  --add-data "linux_ai.py;." ^
  --hidden-import uvicorn.logging ^
  --hidden-import uvicorn.loops ^
  --hidden-import uvicorn.loops.auto ^
  --hidden-import uvicorn.protocols ^
  --hidden-import uvicorn.protocols.http ^
  --hidden-import uvicorn.protocols.http.auto ^
  --hidden-import uvicorn.protocols.websockets ^
  --hidden-import uvicorn.protocols.websockets.auto ^
  --hidden-import uvicorn.lifespan ^
  --hidden-import uvicorn.lifespan.on ^
  --hidden-import fastapi ^
  --collect-all fastapi ^
  --collect-all starlette ^
  server.py

echo.
echo [OK] Build complete: dist\CoreAI\CoreAI.exe
echo      Run CoreAI.exe to start the server.
echo.
pause
