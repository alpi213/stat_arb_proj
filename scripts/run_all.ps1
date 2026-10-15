# Equity pipeline, in order. Stops on the first failure.
# Usage:  .\scripts\run_all.ps1  [-Config configs/base.yaml]
param([string]$Config = "configs/base.yaml")

$ErrorActionPreference = "Stop"
$steps = @(
    "scripts/01_download_data.py",
    "scripts/02_select_pairs.py",
    "scripts/03_backtest.py",
    "scripts/04_capacity.py",
    "scripts/05_walkforward.py",
    "scripts/06_validate.py"
)
foreach ($step in $steps) {
    Write-Host "`n=== $step ===" -ForegroundColor Cyan
    python $step --config $Config
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $step" }
}
Write-Host "`nDone. See results/ and results/figures/." -ForegroundColor Green
