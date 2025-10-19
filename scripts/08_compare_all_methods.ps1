# 08_compare_all_methods.ps1
# Comprehensive comparison: CMA-ES vs DT vs GSN vs ATTN vs PI-Flight

$ErrorActionPreference = "Stop"
$rootDir = Split-Path -Parent $PSScriptRoot
Set-Location $rootDir
$pythonExe = ".\.venv\Scripts\python.exe"

Write-Host "`n" -NoNewline
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Comprehensive Baseline Comparison" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "[Checks] Verifying required files..." -ForegroundColor Yellow
$requiredFiles = @{
    "CMA-ES baseline" = "03_CMA-ES\results\best_program.json"
    "PI-Flight program" = "01_pi_flight\results\best_program.json"
    "Decision Tree model" = "04_decision_tree\results\dt_model.pkl"
    "GSN model" = "05_gsn\results\gsn_best.pt"
    "Attention model" = "06_attn\results\attn_best.pt"
}

$missing = @()
foreach ($name in $requiredFiles.Keys) {
    $path = $requiredFiles[$name]
    if (Test-Path $path) {
        Write-Host "  ✓ $name" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $name (missing: $path)" -ForegroundColor Red
        $missing += $name
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "[Warning] Missing models:" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
    Write-Host ""
    Write-Host "Run training scripts first:" -ForegroundColor Yellow
    Write-Host "  .\scripts\04_train_decision_tree.ps1" -ForegroundColor Gray
    Write-Host "  .\scripts\05_train_gsn.ps1" -ForegroundColor Gray
    Write-Host "  .\scripts\06_train_attn.ps1" -ForegroundColor Gray
    Write-Host ""
    
    $response = Read-Host "Continue with available models? (y/n)"
    if ($response -ne 'y') {
        exit 0
    }
}

Write-Host ""

# Configuration
$trainTraj = "zigzag3d lemniscate3d random_wp spiral_in_out stairs coupled_surface"
$testTraj = "zigzag3d lemniscate3d random_wp spiral_in_out stairs coupled_surface"  # Can use different test set
$aggregate = "harmonic"
$duration = 20
$disturbance = "mild_wind"
$rewardProfile = "pilight_boost"

Write-Host "[Config] Evaluation settings:" -ForegroundColor Yellow
Write-Host "  Train trajectories: $trainTraj" -ForegroundColor Gray
Write-Host "  Test trajectories:  $testTraj" -ForegroundColor Gray
Write-Host "  Aggregate:          $aggregate" -ForegroundColor Gray
Write-Host "  Duration:           ${duration}s" -ForegroundColor Gray
Write-Host "  Disturbance:        $disturbance" -ForegroundColor Gray
Write-Host "  Reward profile:     $rewardProfile" -ForegroundColor Gray
Write-Host ""

# Run comparison
Write-Host "[Evaluation] Running comprehensive comparison..." -ForegroundColor Cyan
Write-Host "  This will evaluate ALL methods on train + test sets" -ForegroundColor Gray
Write-Host "  Estimated time: ~30-60 minutes" -ForegroundColor Gray
Write-Host ""

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$summaryPath = "results\summaries\comparison_$timestamp.json"

& $pythonExe main_no_gui.py `
    --mode compare_all `
    --traj_list $trainTraj.Split(' ') `
    --test_traj_list $testTraj.Split(' ') `
    --aggregate $aggregate `
    --test_aggregate $aggregate `
    --duration_eval $duration `
    --reward_profile $rewardProfile `
    --compose-by-gain `
    --clip-D 1.2 `
    --deep-quiet `
    --save_summary $summaryPath `
    --gsn_ckpt "05_gsn\results\gsn_best.pt" `
    --attn_ckpt "06_attn\results\attn_best.pt"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " Evaluation Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Results saved to: $summaryPath" -ForegroundColor Cyan
    Write-Host ""
    
    # Display summary
    if (Test-Path $summaryPath) {
        Write-Host "Quick Summary:" -ForegroundColor Yellow
        $summary = Get-Content $summaryPath | ConvertFrom-Json
        foreach ($result in $summary.summary) {
            $method = $result.controller
            $reward = [math]::Round($result.reward, 4)
            Write-Host "  $method: $reward" -ForegroundColor Cyan
        }
    }
} else {
    Write-Host ""
    Write-Host "[✗] Evaluation failed (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Analyze results in: results\summaries\" -ForegroundColor Gray
Write-Host "  2. Generate comparison plots" -ForegroundColor Gray
Write-Host "  3. Create result table for paper" -ForegroundColor Gray
