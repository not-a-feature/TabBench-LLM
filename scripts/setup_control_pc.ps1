[CmdletBinding()]
param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$setup = Join-Path $PSScriptRoot "setup_windows.ps1"
$arguments = @{
    SkipModels = $true
    Recreate = $Recreate
}
& $setup @arguments

Write-Host ""
Write-Host "Control-PC setup complete (no CUDA or local model installation)."
Write-Host "For MLCloud, copy configs\llm_key.example.json to configs\llm_key.json,"
Write-Host "insert the private key, then run .\scripts\run_mlcloud.ps1."
