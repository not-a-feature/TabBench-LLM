[CmdletBinding()]
param(
    [string]$Models = "GEMMA,QWEN",
    [string]$Output = "results/grid_all_systems",
    [string]$Datasets = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project environment not found. Run .\scripts\setup_control_pc.ps1 first."
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "configs\llm_key.json"))) {
    throw "configs\llm_key.json is missing. Copy configs\llm_key.example.json and add the private MLCloud key."
}

Set-Location -LiteralPath $projectRoot
& $python -c "import openml"
if ($LASTEXITCODE -ne 0) {
    throw "The full grid needs the OpenML loader. Run '.\scripts\setup_control_pc.ps1' once, then retry."
}
& $python scripts/capture_run_manifest.py `
    --config configs/grid_all_systems.json --output $Output --models $Models
if ($LASTEXITCODE -ne 0) { throw "Manifest capture failed with code $LASTEXITCODE." }

$gridArgs = @(
    "scripts/grid.py",
    "--config", "configs/grid_all_systems.json",
    "--output", $Output,
    "--models", $Models
)
if ($Datasets) {
    $gridArgs += @("--datasets", $Datasets)
}
& $python @gridArgs
if ($LASTEXITCODE -ne 0) { throw "MLCloud grid exited with code $LASTEXITCODE." }
