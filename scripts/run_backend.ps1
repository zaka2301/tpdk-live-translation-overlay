$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root
& "$root\.venv\Scripts\python.exe" -m tpdk_overlay.server.app
