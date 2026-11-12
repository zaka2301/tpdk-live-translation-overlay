# Installs the CUDA 13 Paddle build and the venv-local DLL hook used by the
# default GPU OCR configuration. Run from bootstrap.ps1 or manually.
$ErrorActionPreference = "Stop"

$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "Run scripts\bootstrap.ps1 first (missing .venv)"
}

$cpu = & $py -m pip list --format=freeze | Select-String '^paddlepaddle=='
if ($cpu) {
    Write-Warning "CPU paddlepaddle is installed; the GPU wheel will share the 'paddle' package files. Prefer a fresh venv or remove the CPU package first."
}

Write-Host "Installing paddlepaddle-gpu 3.3.1 (CUDA 13)..."
# --no-deps is deliberate: the nvidia-* wheels are ~1 GB+, ship a different
# CUDA 13.5 runtime, and sitecustomize already points Paddle at torch's working
# CUDA 13/cuDNN DLLs. `pip check` will list those metadata deps as missing.
& $py -m pip install --no-deps --upgrade "paddlepaddle-gpu==3.3.1" `
    --index-url "https://www.paddlepaddle.org.cn/packages/stable/cu130/"

$site = & $py -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
$hook = Join-Path $site "sitecustomize.py"
if (Test-Path $hook) {
    Write-Host "sitecustomize.py already exists at $hook; leaving it untouched."
} else {
    Copy-Item (Join-Path $PSScriptRoot "paddle_sitecustomize.py") $hook
    Write-Host "Installed $hook"
}
