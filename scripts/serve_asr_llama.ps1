<#
.SYNOPSIS
    Start the local Qwen3-ASR server for the tpdk_overlay `llama_cpp` backend.

.DESCRIPTION
    Runs the official llama CLI's `serve` command with the Qwen3-ASR GGUF model
    and exposes its OpenAI-compatible API on 127.0.0.1:8081. The overlay backend
    (`asr.backend: llama_cpp`) posts utterance audio to this process, so leave
    this window running while the backend is up. The model and its mmproj
    download on the first run.

    Install the CLI once with:
      irm https://llama.app/install.ps1 | iex

    Equivalent raw command:
      llama serve -hf ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0 --host 127.0.0.1
        --port 8081 -ngl all -c 4096 -np 1 -n 128
        --cache-type-k q8_0 --cache-type-v q8_0

.PARAMETER Llama
    Path to llama.exe, or `llama` to resolve the official CLI from PATH.

.PARAMETER Model
    GGUF repo id for llama's -hf flag (default the 0.6B Q8_0 checkpoint).

.PARAMETER Port
    HTTP port to listen on. Must match `asr.llama_cpp.base_url` (default 8081).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\serve_asr_llama.ps1

.EXAMPLE
    powershell -File scripts\serve_asr_llama.ps1 -Model ggml-org/Qwen3-ASR-1.7B-GGUF:Q8_0 -Port 8081
#>
[CmdletBinding()]
param(
    [string]$Llama = "llama",
    [string]$Model = "ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0",
    [int]$Port = 8081
)

$ErrorActionPreference = "Stop"

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

Write-Host "llama:        $LlamaExe"
Write-Host "model:        $Model"
Write-Host "endpoint:     http://127.0.0.1:$Port (health check: GET /health)"
Write-Host "Leave this window open; the backend (asr.backend: llama_cpp) connects to it."

& $LlamaExe serve `
    -hf $Model `
    --host 127.0.0.1 `
    --port $Port `
    -ngl all `
    -c 4096 `
    -np 1 `
    -n 128 `
    --cache-type-k q8_0 `
    --cache-type-v q8_0

exit $LASTEXITCODE
