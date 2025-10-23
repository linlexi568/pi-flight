# 一键测试和可视化所有checkpoint

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Checkpoint 测试与可视化工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否存在checkpoint目录
$checkpointDir = "01_pi_flight\results\checkpoints"
if (-not (Test-Path $checkpointDir)) {
    Write-Host "❌ 错误: 找不到checkpoint目录: $checkpointDir" -ForegroundColor Red
    exit 1
}

$checkpointCount = (Get-ChildItem "$checkpointDir\best_program_iter_*.json" | Measure-Object).Count
Write-Host "✅ 找到 $checkpointCount 个checkpoint文件`n" -ForegroundColor Green

# 询问用户选择
Write-Host "请选择操作:" -ForegroundColor Yellow
Write-Host "  1. 测试所有checkpoint（跳过已测试）" -ForegroundColor White
Write-Host "  2. 强制重新测试所有checkpoint" -ForegroundColor White
Write-Host "  3. 只测试前10个checkpoint（快速验证）" -ForegroundColor White
Write-Host "  4. 只生成可视化图表（需要先运行测试）" -ForegroundColor White
Write-Host "  5. 完整流程：测试 + 可视化" -ForegroundColor White
Write-Host "  0. 退出" -ForegroundColor White
Write-Host ""

$choice = Read-Host "请输入选项 (0-5)"

switch ($choice) {
    "1" {
        Write-Host "`n开始测试所有checkpoint..." -ForegroundColor Cyan
        & .venv\Scripts\python.exe analysis\test_all_checkpoints.py
    }
    "2" {
        Write-Host "`n强制重新测试所有checkpoint..." -ForegroundColor Cyan
        & .venv\Scripts\python.exe analysis\test_all_checkpoints.py --force
    }
    "3" {
        Write-Host "`n测试前10个checkpoint..." -ForegroundColor Cyan
        & .venv\Scripts\python.exe analysis\test_all_checkpoints.py --max 10
    }
    "4" {
        Write-Host "`n生成可视化图表..." -ForegroundColor Cyan
        if (-not (Test-Path "analysis\checkpoint_test_results.csv")) {
            Write-Host "❌ 错误: 找不到测试结果文件，请先运行测试" -ForegroundColor Red
            exit 1
        }
        & .venv\Scripts\python.exe analysis\visualize_checkpoint_results.py
    }
    "5" {
        Write-Host "`n【步骤1/2】测试所有checkpoint..." -ForegroundColor Cyan
        & .venv\Scripts\python.exe analysis\test_all_checkpoints.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n【步骤2/2】生成可视化图表..." -ForegroundColor Cyan
            & .venv\Scripts\python.exe analysis\visualize_checkpoint_results.py
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "`n✅ 全部完成！" -ForegroundColor Green
                Write-Host "结果文件:" -ForegroundColor Yellow
                Write-Host "  - CSV: analysis\checkpoint_test_results.csv" -ForegroundColor White
                Write-Host "  - 图表: analysis\checkpoint_figures\*.png" -ForegroundColor White
                
                # 询问是否打开结果
                $open = Read-Host "`n是否打开图表目录? (y/n)"
                if ($open -eq "y") {
                    explorer "analysis\checkpoint_figures"
                }
            }
        }
    }
    "0" {
        Write-Host "退出" -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "❌ 无效选项" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n完成！" -ForegroundColor Green
