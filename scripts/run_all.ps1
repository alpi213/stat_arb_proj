# Full pipeline, in order. Stops on first failure.
# Usage:  .\scripts\run_all.ps1  [-Config configs/base.yaml]
param([string]$Config = "configs/base.yaml")

$ErrorActionPreference = "Stop"
$steps = @(
    "scripts/01_download_data.py",
    "scripts/02_select_pairs.py",
    "scripts/03_backtest.py --compare",
    "scripts/04_capacity.py",
    "scripts/05_walkforward.py",
    "scripts/06_validate.py"
)
foreach ($step in $steps) {
    Write-Host "`n=== $step ===" -ForegroundColor Cyan
    $parts = $step -split " "
    python $parts --config $Config
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $step" }
}
Write-Host "`nPipeline complete. See results/ and results/figures/." -ForegroundColor Green
