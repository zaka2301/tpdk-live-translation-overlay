# Requires uv on PATH: https://docs.astral.sh/uv/
$ErrorActionPreference = "Stop"

$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv with Python 3.12..."
    py -3.12 -m venv .venv
}

$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

Write-Host "Installing CUDA 13.0 torch (must be installed before the ml extra)..."
& uv pip install --python $py "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130

Write-Host "Installing project extras (ui, ml, ocr, summarize, ct2, dev)..."
& uv pip install --python $py -e ".[ui,ml,ocr,summarize,ct2,dev]"

Write-Host "Installing PaddlePaddle GPU (CUDA 13) and the venv DLL hook..."
& powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\setup_paddle_gpu.ps1"

Write-Host "Checking environment..."
& $py "scripts\check_env.py"
