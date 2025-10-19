# 05_train_gsn.ps1
# Train GSN (Gain Scheduling Network) baseline

$ErrorActionPreference = "Stop"
$rootDir = Split-Path -Parent $PSScriptRoot
Set-Location $rootDir
$pythonExe = ".\.venv\Scripts\python.exe"

Write-Host "`n=== GSN (Gain Scheduling Network) Training ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Collect data (if not exists)
if (!(Test-Path "05_gsn\data\gsn_dataset.npz")) {
    Write-Host "[1/2] Collecting training data..." -ForegroundColor Yellow
    & $pythonExe 05_gsn\collect_data.py `
        --duration-sec 10 `
        --episodes 6 `
        --output "05_gsn\data\gsn_dataset.npz"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[✗] Data collection failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[1/2] Using existing data: 05_gsn\data\gsn_dataset.npz" -ForegroundColor Green
}

Write-Host ""

# Step 2: Train GSN
Write-Host "[2/2] Training GSN model..." -ForegroundColor Yellow
& $pythonExe 05_gsn\train_gsn.py `
    --data "05_gsn\data\gsn_dataset.npz" `
    --epochs 100 `
    --bs 256 `
    --lr 3e-4 `
    --save-best "05_gsn\results\gsn_best.pt"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[✗] Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Training Complete! ===" -ForegroundColor Green
Write-Host "  Model saved to: 05_gsn\results\gsn_best.pt" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: Evaluate with main_no_gui.py --mode gsn_only" -ForegroundColor Yellow
