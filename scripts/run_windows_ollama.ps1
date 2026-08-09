[CmdletBinding()]
param(
    [string]$Models = "OLLAMA-GEMMA3-4B,OLLAMA-QWEN3-8B,OLLAMA-LLAMA3.1-8B,OLLAMA-MISTRAL-7B,OLLAMA-PHI4-MINI",
    [string]$Output,
    [switch]$Pilot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project environment not found. Run .\scripts\setup_windows.ps1 first."
}

try {
    Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:11434/api/version" | Out-Null
} catch {
    throw "The Ollama API is not responding at http://127.0.0.1:11434. Start Ollama and retry."
}

Set-Location -LiteralPath $projectRoot
$env:TABBENCH_LLM_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
$env:TABBENCH_LLM_OLLAMA_API_KEY = "ollama"

$config = if ($Pilot) { "configs/grid_windows_ollama.json" } else { "configs/grid_all_systems.json" }
if (-not $Output) {
    $Output = if ($Pilot) { "results/windows_ollama" } else { "results/grid_all_systems" }
}
if (-not $Pilot) {
    & $python -c "import openml"
    if ($LASTEXITCODE -ne 0) {
        throw "The full grid needs the OpenML loader. Run '.\scripts\setup_windows.ps1 -SkipModels -SkipTests' once, then retry."
    }
}
$arguments = @(
    "scripts/grid.py",
    "--config", $config,
    "--output", $Output
)
if ($Models) {
    $arguments += @("--models", $Models)
}

& $python scripts/capture_run_manifest.py --config $config --output $Output --models $Models
if ($LASTEXITCODE -ne 0) {
    throw "Manifest capture failed with code $LASTEXITCODE."
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "The Ollama grid exited with code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Ollama slice complete. Inspect $Output and run 'ollama ps' to verify GPU offload."
