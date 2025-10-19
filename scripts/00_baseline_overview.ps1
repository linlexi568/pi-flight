# 00_baseline_overview.ps1
# Overview of all baseline methods and training scripts

$rootDir = Split-Path -Parent $PSScriptRoot
Set-Location $rootDir

Write-Host "`n" -NoNewline
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  PI-Flight Baseline Comparison Framework" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Write-Host "Available Methods:" -ForegroundColor Yellow
Write-Host ""

# Method 1: CMA-ES
Write-Host "1. " -NoNewline -ForegroundColor White
Write-Host "CMA-ES Baseline" -ForegroundColor Green
Write-Host "   Type:        Evolution Strategy (global PID optimization)" -ForegroundColor Gray
Write-Host "   Script:      " -NoNewline -ForegroundColor Gray
Write-Host ".\03_CMA-ES\cma_pid_search.py" -ForegroundColor Cyan
Write-Host "   Output:      03_CMA-ES\results\best_program.json" -ForegroundColor Gray
Write-Host "   Train time:  ~10 minutes" -ForegroundColor Gray
Write-Host ""

# Method 2: Decision Tree
Write-Host "2. " -NoNewline -ForegroundColor White
Write-Host "Decision Tree (CART)" -ForegroundColor Green
Write-Host "   Type:        Supervised Learning (symbolic rules)" -ForegroundColor Gray
Write-Host "   Script:      " -NoNewline -ForegroundColor Gray
Write-Host ".\scripts\04_train_decision_tree.ps1" -ForegroundColor Cyan
Write-Host "   Output:      04_decision_tree\results\dt_model.pkl" -ForegroundColor Gray
Write-Host "   Train time:  ~1 minute (after data collection)" -ForegroundColor Gray
Write-Host "   Pros:        ✓ Interpretable  ✓ Fast" -ForegroundColor Gray
Write-Host "   Cons:        ✗ Needs labels  ✗ Greedy splits" -ForegroundColor Gray
Write-Host ""

# Method 3: GSN
Write-Host "3. " -NoNewline -ForegroundColor White
Write-Host "GSN (Gain Scheduling Network)" -ForegroundColor Green
Write-Host "   Type:        Neural Network (MLP)" -ForegroundColor Gray
Write-Host "   Script:      " -NoNewline -ForegroundColor Gray
Write-Host ".\scripts\05_train_gsn.ps1" -ForegroundColor Cyan
Write-Host "   Output:      05_gsn\results\gsn_best.pt" -ForegroundColor Gray
Write-Host "   Train time:  ~10 minutes" -ForegroundColor Gray
Write-Host "   Pros:        ✓ Flexible  ✓ Non-linear" -ForegroundColor Gray
Write-Host "   Cons:        ✗ Black box  ✗ Needs data" -ForegroundColor Gray
Write-Host ""

# Method 4: Attention
Write-Host "4. " -NoNewline -ForegroundColor White
Write-Host "AttnGainNet (Transformer)" -ForegroundColor Green
Write-Host "   Type:        Neural Network (Self-Attention)" -ForegroundColor Gray
Write-Host "   Script:      " -NoNewline -ForegroundColor Gray
Write-Host ".\scripts\06_train_attn.ps1" -ForegroundColor Cyan
Write-Host "   Output:      06_attn\results\attn_best.pt" -ForegroundColor Gray
Write-Host "   Train time:  ~15 minutes" -ForegroundColor Gray
Write-Host "   Pros:        ✓ Temporal modeling  ✓ SOTA architecture" -ForegroundColor Gray
Write-Host "   Cons:        ✗ Black box  ✗ Computational cost" -ForegroundColor Gray
Write-Host ""

# Method 5: PI-Flight
Write-Host "5. " -NoNewline -ForegroundColor White
Write-Host "PI-Flight (Ours)" -ForegroundColor Magenta
Write-Host "   Type:        Program Synthesis (MCTS + symbolic rules)" -ForegroundColor Gray
Write-Host "   Script:      " -NoNewline -ForegroundColor Gray
Write-Host ".\01_pi_flight\train_pi_flight.py" -ForegroundColor Cyan
Write-Host "   Output:      01_pi_flight\results\best_program.json" -ForegroundColor Gray
Write-Host "   Train time:  ~8 hours (5000 iterations)" -ForegroundColor Gray
Write-Host "   Pros:        ✓ Interpretable  ✓ No labels  ✓ Global search" -ForegroundColor Magenta
Write-Host "   Cons:        Longer training time" -ForegroundColor Gray
Write-Host ""

Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Training pipeline
Write-Host "Training Pipeline:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Step 1: Train CMA-ES baseline" -ForegroundColor White
Write-Host "    python 03_CMA-ES\cma_pid_search.py --iters 30" -ForegroundColor Gray
Write-Host ""
Write-Host "  Step 2: Train PI-Flight (your method)" -ForegroundColor White
Write-Host "    python 01_pi_flight\train_pi_flight.py --iters 5000 ..." -ForegroundColor Gray
Write-Host ""
Write-Host "  Step 3: Train Decision Tree" -ForegroundColor White
Write-Host "    .\scripts\04_train_decision_tree.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "  Step 4: Train GSN" -ForegroundColor White
Write-Host "    .\scripts\05_train_gsn.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "  Step 5: Train Attention" -ForegroundColor White
Write-Host "    .\scripts\06_train_attn.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "  Step 6: Compare all methods" -ForegroundColor White
Write-Host "    " -NoNewline
Write-Host ".\scripts\08_compare_all_methods.ps1" -ForegroundColor Cyan
Write-Host ""

Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Quick actions
Write-Host "Quick Actions:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  [1] Train all baselines (DT + GSN + ATTN)" -ForegroundColor White
Write-Host "  [2] Run comparison evaluation" -ForegroundColor White
Write-Host "  [3] View existing results" -ForegroundColor White
Write-Host "  [4] Check model files" -ForegroundColor White
Write-Host "  [Q] Quit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Select option (1-4, Q)"

switch ($choice) {
    "1" {
        Write-Host "`nTraining all baselines..." -ForegroundColor Cyan
        & ".\scripts\04_train_decision_tree.ps1"
        & ".\scripts\05_train_gsn.ps1"
        & ".\scripts\06_train_attn.ps1"
    }
    "2" {
        Write-Host "`nRunning comparison..." -ForegroundColor Cyan
        & ".\scripts\08_compare_all_methods.ps1"
    }
    "3" {
        Write-Host "`nExisting results:" -ForegroundColor Cyan
        Get-ChildItem "results\summaries\*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 5 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Gray
        }
    }
    "4" {
        Write-Host "`nChecking model files:" -ForegroundColor Cyan
        $models = @(
            "03_CMA-ES\results\best_program.json",
            "01_pi_flight\results\best_program.json",
            "04_decision_tree\results\dt_model.pkl",
            "05_gsn\results\gsn_best.pt",
            "06_attn\results\attn_best.pt"
        )
        foreach ($model in $models) {
            if (Test-Path $model) {
                $size = (Get-Item $model).Length / 1KB
                Write-Host "  ✓ $model ($([math]::Round($size, 2)) KB)" -ForegroundColor Green
            } else {
                Write-Host "  ✗ $model (not found)" -ForegroundColor Red
            }
        }
    }
    default {
        Write-Host "`nExiting..." -ForegroundColor Gray
    }
}

Write-Host ""
