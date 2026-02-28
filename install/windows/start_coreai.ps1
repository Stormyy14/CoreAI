# CoreAI — PowerShell Launcher
# Usage: Right-click -> Run with PowerShell
#        Or: powershell -ExecutionPolicy Bypass -File start_coreai.ps1

$Root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Root

Write-Host ""
Write-Host "  CoreAI v1.0 — Local AI Platform" -ForegroundColor Cyan
Write-Host "  http://localhost:8080" -ForegroundColor White
Write-Host ""

# Auto-setup on first run
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "[INFO] First run — creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "[INFO] Installing dependencies..." -ForegroundColor Yellow
    & "venv\Scripts\pip.exe" install -r requirements.txt --quiet
}

# Activate venv
& "venv\Scripts\Activate.ps1"

# Open browser after 3 seconds
Start-Job {
    Start-Sleep 3
    Start-Process "http://localhost:8080"
} | Out-Null

# Start server
python server.py
