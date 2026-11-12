<#
.SYNOPSIS
    Start the local PLaMo-3-Translate server for the tpdk_overlay `plamo_llama`
    MT provider.

.DESCRIPTION
    Runs the official llama CLI's `serve` command with the PLaMo-3-Translate
    0.8B GGUF model (Q4_K_M) and exposes its OpenAI-compatible API on
    127.0.0.1:8082. The overlay backend (`mt.provider: plamo_llama`) posts one
    translation request per text to this process; `main.py` starts it
    automatically. Leave this window running while the backend is up.

    The server and the ASR server are separate processes: PLaMo adds roughly
    0.7 GiB of VRAM to the ASR server's ~1.7 GiB - both fit the 4 GB card.

    Equivalent raw command:
      llama serve -m data/models/plamo-3-translate/plamo-3-translate-0.8b-Q4_K_M.gguf
        --host 127.0.0.1 --port 8082 --alias plamo-translate
        -ngl all -c 16384 -np 8 --jinja --no-prefill-assistant

.PARAMETER Llama
    Path to llama.exe, or `llama` to resolve the official CLI from PATH.

.PARAMETER Model
    Local GGUF path or `llama -hf` spec. Defaults to the downloaded Q4_K_M file
    under data/models; falls back to pfnet/plamo-3-translate-0.8b-GGUF:Q4_K_M.

.PARAMETER Port
    HTTP port to listen on. Must match `mt.plamo_llama.base_url` (default 8082).

.PARAMETER GpuLayers
    `all` (default, GPU), a layer count, or `0` for CPU-only.

.PARAMETER Parallel
    llama.cpp server slots. Keep `mt.plamo_llama.concurrency` at or below this.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\serve_mt_plamo.ps1

.EXAMPLE
    powershell -File scripts\serve_mt_plamo.ps1 -GpuLayers 0 -Parallel 8
#>
[CmdletBinding()]
param(
    [string]$Llama = "llama",
    [string]$Model = "",
    [int]$Port = 8082,
    [string]$GpuLayers = "all",
    [int]$Parallel = 8,
    [int]$ContextSize = 16384,
    [string]$Alias = "plamo-translate"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path

if ($Llama -eq "llama") {
    $command = Get-Command "llama" -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "llama CLI not found. Install it with: irm https://llama.app/install.ps1 | iex"
    }
    $LlamaExe = if ($command.Path) { $command.Path } else { $command.Source }
}
elseif (Test-Path -LiteralPath $Llama) {
    $LlamaExe = (Resolve-Path -LiteralPath $Llama).Path
}
else {
    throw "llama CLI not found at '$Llama'."
}

if (-not $Model) {
    $local = Join-Path $root "data\models\plamo-3-translate\plamo-3-translate-0.8b-Q4_K_M.gguf"
    $Model = if (Test-Path -LiteralPath $local) { $local } else { "pfnet/plamo-3-translate-0.8b-GGUF:Q4_K_M" }
}

if (Test-Path -LiteralPath $Model) {
    $modelArgs = @("-m", (Resolve-Path -LiteralPath $Model).Path)
}
else {
    $modelArgs = @("-hf", $Model)
}

Write-Host "llama:        $LlamaExe"
Write-Host "model:        $Model"
Write-Host "endpoint:     http://127.0.0.1:$Port (health check: GET /health)"
Write-Host "gpu layers:   $GpuLayers   slots: $Parallel   context: $ContextSize"
Write-Host "Leave this window open; the backend (mt.provider: plamo_llama) connects to it."

Set-Location $root
& $LlamaExe serve `
    @modelArgs `
    --host 127.0.0.1 `
    --port $Port `
    --alias $Alias `
    -ngl $GpuLayers `
    -c $ContextSize `
    -np $Parallel `
    --jinja `
    --no-prefill-assistant

exit $LASTEXITCODE
