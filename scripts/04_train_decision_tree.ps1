# 04_train_decision_tree.ps1
# Complete pipeline: Collect data -> Train model -> Evaluate

$ErrorActionPreference = "Stop"
$rootDir = Split-Path -Parent $PSScriptRoot
Set-Location $rootDir
$pythonExe = ".\.venv\Scripts\python.exe"

Write-Host "`n=== Decision Tree Baseline Training Pipeline ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Collect training data
Write-Host "[1/3] Collecting training data from PI-Flight..." -ForegroundColor Yellow
& $pythonExe 04_decision_tree\collect_data.py `
    --program "01_pi_flight\results\best_program.json" `
    --traj_list zigzag3d lemniscate3d random_wp spiral_in_out stairs coupled_surface `
    --duration 10 `
    --output "04_decision_tree\data\dt_training_data.npz" `
    --log-skip 2 `
    --disturbance mild_wind

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[✗] Data collection failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 2: Train decision tree
Write-Host "[2/3] Training decision tree model..." -ForegroundColor Yellow
& $pythonExe 04_decision_tree\train_dt.py `
    --data "04_decision_tree\data\dt_training_data.npz" `
    --output "04_decision_tree\results\dt_model.pkl" `
    --max-depth 10 `
    --min-samples-split 20 `
    --min-samples-leaf 10 `
    --test-split 0.2

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[✗] Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 3: Evaluate on test trajectories
Write-Host "[3/3] Evaluating on test set..." -ForegroundColor Yellow
Write-Host "  (Use main_no_gui.py --mode dt_only for full evaluation)" -ForegroundColor Gray

Write-Host ""
Write-Host "=== Training Complete! ===" -ForegroundColor Green
Write-Host "  Model saved to: 04_decision_tree\results\dt_model.pkl" -ForegroundColor Cyan
Write-Host "  Training log:   04_decision_tree\results\dt_model.json" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Integrate DT controller into main_no_gui.py" -ForegroundColor Gray
Write-Host "  2. Run comparison: python main_no_gui.py --mode compare_all" -ForegroundColor Gray
