@echo off
setlocal enabledelayedexpansion
title CoreAI Installer

:: ─────────────────────────────────────────────────────────────────────────────
:: CoreAI — Windows Installer
:: Requires: Python 3.10+ in PATH  (https://www.python.org/downloads/)
:: Usage: Double-click install.bat  OR  run as Administrator for system install
:: ─────────────────────────────────────────────────────────────────────────────

set APP_NAME=CoreAI
set APP_VERSION=1.0.0
set INSTALL_DIR=%LOCALAPPDATA%\CoreAI
set SCRIPT_DIR=%~dp0
set SRC_DIR=%SCRIPT_DIR%..\..

echo.
echo   +===========================================+
echo   ^|   CoreAI Windows Installer v%APP_VERSION%       ^|
echo   ^|   Local AI Platform                      ^|
echo   +===========================================+
echo.

:: ── Check Python ─────────────────────────────────────────────────────────────
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH.
    echo.
    echo  Please download and install Python 3.10+ from:
    echo  https://www.python.org/downloads/
    echo.
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK]  Python %PY_VER% found

:: ── Copy files ────────────────────────────────────────────────────────────────
echo [INFO] Installing to %INSTALL_DIR% ...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

:: Use robocopy (built-in Windows) for reliable copy
robocopy "%SRC_DIR%" "%INSTALL_DIR%" /E /XD __pycache__ .git venv build_deb /XF "*.pyc" /NFL /NDL /NJH /NJS >nul 2>&1
if %ERRORLEVEL% gtr 7 (
    echo [WARN] Some files may not have copied correctly.
)
echo [OK]  Files copied

:: ── Create virtual environment ────────────────────────────────────────────────
echo [INFO] Creating Python virtual environment...
python -m venv "%INSTALL_DIR%\venv"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause & exit /b 1
)
echo [OK]  Virtual environment created

:: ── Install dependencies ──────────────────────────────────────────────────────
echo [INFO] Installing dependencies (this may take a few minutes)...
"%INSTALL_DIR%\venv\Scripts\pip.exe" install --upgrade pip --quiet
"%INSTALL_DIR%\venv\Scripts\pip.exe" install -r "%INSTALL_DIR%\requirements.txt" --quiet
if %ERRORLEVEL% neq 0 (
    echo [WARN] Some dependencies failed. You can retry later.
) else (
    echo [OK]  Dependencies installed
)

:: ── Create Desktop shortcut ───────────────────────────────────────────────────
echo [INFO] Creating Desktop shortcut...
set SHORTCUT=%USERPROFILE%\Desktop\CoreAI.lnk
set LAUNCHER=%INSTALL_DIR%\start_coreai.bat
powershell -Command "$ws=New-Object -ComObject WScript.Shell; $sc=$ws.CreateShortcut('%SHORTCUT%'); $sc.TargetPath='%LAUNCHER%'; $sc.WorkingDirectory='%INSTALL_DIR%'; $sc.Description='CoreAI Local AI Platform'; $sc.Save()"
echo [OK]  Desktop shortcut created

:: ── Create Start Menu entry ───────────────────────────────────────────────────
set STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs\CoreAI
if not exist "%STARTMENU%" mkdir "%STARTMENU%"
copy "%SHORTCUT%" "%STARTMENU%\CoreAI.lnk" >nul 2>&1

:: ── Write launcher batch ──────────────────────────────────────────────────────
(
echo @echo off
echo title CoreAI
echo cd /d "%INSTALL_DIR%"
echo call venv\Scripts\activate.bat
echo echo.
echo echo   Starting CoreAI at http://localhost:8080
echo echo   Opening browser in 3 seconds...
echo echo.
echo start "" cmd /c "timeout /t 3 /nobreak ^>nul ^&^& start http://localhost:8080"
echo python server.py
echo pause
) > "%INSTALL_DIR%\start_coreai.bat"

:: ── Write PowerShell launcher (nicer) ────────────────────────────────────────
(
echo # CoreAI Launcher
echo Set-Location '%INSTALL_DIR%'
echo ^& '%INSTALL_DIR%\venv\Scripts\Activate.ps1'
echo Write-Host "Starting CoreAI at http://localhost:8080" -ForegroundColor Cyan
echo Start-Process "http://localhost:8080" -ErrorAction SilentlyContinue ^| Out-Null
echo Start-Sleep 2
echo python server.py
) > "%INSTALL_DIR%\start_coreai.ps1"

:: ── Write uninstaller ────────────────────────────────────────────────────────
(
echo @echo off
echo title CoreAI Uninstaller
echo echo Removing CoreAI...
echo taskkill /f /im python.exe /fi "WINDOWTITLE eq CoreAI*" ^>nul 2^>^&1
echo rmdir /s /q "%INSTALL_DIR%"
echo del /f "%USERPROFILE%\Desktop\CoreAI.lnk" ^>nul 2^>^&1
echo rmdir /s /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\CoreAI" ^>nul 2^>^&1
echo echo CoreAI uninstalled.
echo pause
) > "%USERPROFILE%\Desktop\CoreAI Uninstall.bat"

echo.
echo   +===========================================+
echo   ^|   CoreAI installed successfully!         ^|
echo   +===========================================+
echo   ^|                                          ^|
echo   ^|  Launch: Desktop shortcut "CoreAI"       ^|
echo   ^|  URL:    http://localhost:8080           ^|
echo   ^|                                          ^|
echo   +===========================================+
echo.
set /p OPEN="Open CoreAI now? [Y/n]: "
if /i not "!OPEN!"=="n" (
    start "" "%INSTALL_DIR%\start_coreai.bat"
)
pause
endlocal
