# 06_train_attn.ps1
# Train Attention-based Gain Scheduling Network

$ErrorActionPreference = "Stop"
$rootDir = Split-Path -Parent $PSScriptRoot
Set-Location $rootDir
$pythonExe = ".\.venv\Scripts\python.exe"

Write-Host "`n=== Attention Network Training ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Collect data (reuse GSN data or collect new)
$dataPath = "05_gsn\data\gsn_dataset.npz"
if (!(Test-Path $dataPath)) {
    Write-Host "[1/2] Collecting training data..." -ForegroundColor Yellow
    & $pythonExe 05_gsn\collect_data.py `
        --duration-sec 10 `
        --episodes 6 `
        --output $dataPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[✗] Data collection failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[1/2] Using existing data: $dataPath" -ForegroundColor Green
}

Write-Host ""

# Step 2: Train Attention model
Write-Host "[2/2] Training Attention model..." -ForegroundColor Yellow
& $pythonExe 06_attn\train_attn.py `
    --data $dataPath `
    --epochs 100 `
    --seq_len 8 `
    --bs 128 `
    --lr 3e-4 `
    --save-best "06_attn\results\attn_best.pt"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[✗] Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Training Complete! ===" -ForegroundColor Green
Write-Host "  Model saved to: 06_attn\results\attn_best.pt" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: Evaluate with main_no_gui.py --mode attn_only" -ForegroundColor Yellow
