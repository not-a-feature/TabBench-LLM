# Aggregate and preview whatever results currently exist here: one machine or merged bundles.
# This is a diagnostic preview, not the publication-complete leaderboard.
#
#   .\snapshot_results.ps1
#   .\snapshot_results.ps1 -Push
#   .\snapshot_results.ps1 -ExportBundle -BundleLabel ollama-pc
#   .\snapshot_results.ps1 -ImportBundle C:\tmp\ollama.zip,C:\tmp\h100.zip
[CmdletBinding()]
param(
    [switch]$Push,
    [switch]$RecomputeMetrics,
    [switch]$ExportBundle,
    [string]$BundleLabel = $env:COMPUTERNAME,
    [string[]]$ImportBundle,
    [string]$Output = "results/grid_all_systems"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project environment not found. Run .\scripts\setup_windows.ps1 first."
}

if ($ImportBundle) {
    $importArgs = @("scripts/result_bundle.py", "import", "--dest", $Output) + $ImportBundle
    & $python @importArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Result-bundle import failed with code $LASTEXITCODE."
    }
}

$previewArgs = @("scripts/preview_current_grid.py", "--output", $Output, "--site-out", "site_data")
if ($RecomputeMetrics -or $ImportBundle) {
    $previewArgs += "--recompute-metrics"
}
& $python @previewArgs
if ($LASTEXITCODE -ne 0) {
    throw "Current-state preview build failed with code $LASTEXITCODE."
}

if ($ExportBundle) {
    & $python scripts/result_bundle.py export --source $Output --label $BundleLabel
    if ($LASTEXITCODE -ne 0) {
        throw "Result-bundle export failed with code $LASTEXITCODE."
    }
}

if ($Push) {
    & git add -- site_data
    if ($LASTEXITCODE -ne 0) { throw "Could not stage site_data/." }

    & git diff --cached --quiet -- site_data
    $dataChanged = $LASTEXITCODE -eq 1
    if ($LASTEXITCODE -gt 1) { throw "Could not inspect staged site_data/ changes." }

    if ($dataChanged) {
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
        & git commit --only -m "Publish current-state preview ($stamp)" -- site_data
        if ($LASTEXITCODE -ne 0) { throw "Could not commit the current-state preview." }
        & git push origin HEAD:main
        if ($LASTEXITCODE -ne 0) { throw "Could not push the current-state preview to origin/main." }
        Write-Host "Pushed to origin/main (derived site_data only); the upload workflow ships it."
    } else {
        Write-Host "The current-state preview is unchanged; nothing to push."
    }
}

Write-Host "Current-state payload ready under site_data/data."
Write-Host "To preview the page, copy leaderboard.json into the juleskreuer.eu repo at"
Write-Host "  content/projects/files/tabbench-llm/ and run npx quartz build."
