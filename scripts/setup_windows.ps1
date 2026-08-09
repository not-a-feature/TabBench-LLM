[CmdletBinding()]
param(
    [string[]]$Models = @(
        "gemma3:4b",
        "qwen3:8b",
        "llama3.1:8b",
        "mistral:7b-instruct",
        "phi4-mini:3.8b"
    ),
    [switch]$SkipModels,
    [switch]$SkipTests,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "uv is missing and WinGet is unavailable. Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun this script."
    }
    winget install --id=astral-sh.uv -e --source winget --accept-package-agreements --accept-source-agreements
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uvPath = $uvCommand.Source
} else {
    $wingetUv = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uv.exe"
    if (Test-Path -LiteralPath $wingetUv) {
        $uvPath = $wingetUv
    } else {
        throw "uv was installed but is not visible in this PowerShell session. Open a new PowerShell window and rerun the script."
    }
}

& $uvPath python install 3.12
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if ($Recreate) {
    & $uvPath venv --clear --python 3.12 .venv
} elseif (-not (Test-Path -LiteralPath $venvPython)) {
    & $uvPath venv --python 3.12 .venv
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The virtual environment was not created at $venvPython."
}

& $uvPath pip install --python $venvPython -r requirements-dev.txt
& $uvPath pip check --python $venvPython
$env:PYTHONPATH = (Join-Path $projectRoot "src")

if (-not $SkipModels) {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        throw "Ollama is not on PATH. Install/start Ollama for Windows, then rerun this script."
    }
    try {
        Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:11434/api/version" | Out-Null
    } catch {
        throw "The Ollama API is not responding at http://127.0.0.1:11434. Start the Ollama tray application, then rerun."
    }

    $modelFiles = @{
        "gemma3:4b" = "Modelfile.gemma3-4b"
        "qwen3:8b" = "Modelfile.qwen3-8b"
        "llama3.1:8b" = "Modelfile.llama3.1-8b"
        "mistral:7b-instruct" = "Modelfile.mistral-7b"
        "phi4-mini:3.8b" = "Modelfile.phi4-mini"
    }
    $aliases = @{
        "gemma3:4b" = "tabarena-gemma3-4b"
        "qwen3:8b" = "tabarena-qwen3-8b"
        "llama3.1:8b" = "tabarena-llama31-8b"
        "mistral:7b-instruct" = "tabarena-mistral-7b"
        "phi4-mini:3.8b" = "tabarena-phi4-mini"
    }

    foreach ($model in $Models) {
        & ollama pull $model
        if ($LASTEXITCODE -ne 0) { throw "Could not pull Ollama model $model." }
        if (-not $modelFiles.ContainsKey($model)) {
            throw "No TabArena Modelfile is registered for $model."
        }
        $modelFile = Join-Path $projectRoot ("configs\ollama\" + $modelFiles[$model])
        & ollama create $aliases[$model] -f $modelFile
        if ($LASTEXITCODE -ne 0) { throw "Could not create the TabArena alias for $model." }
    }
}

if (-not $SkipTests) {
    & $venvPython -m pytest -q
}

Write-Host ""
Write-Host "Setup complete. Run the workstation pilot with:"
Write-Host "  .\scripts\run_windows_ollama.ps1"
Write-Host "During inference, use 'ollama ps' in another terminal to confirm the model is GPU-resident."
