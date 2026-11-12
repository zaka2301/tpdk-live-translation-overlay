"""Register CUDA/cuDNN DLL directories for a venv's Paddle GPU build.

Copied to ``<venv>/Lib/site-packages/sitecustomize.py`` by
``scripts/setup_paddle_gpu.ps1``. Paddle's Windows GPU wheel does not bundle
CUDA; the torch cu130 wheel carries the CUDA 13 runtime DLLs this machine needs.
"""

try:
    import contextlib
    import os
    import sysconfig
    from pathlib import Path

    _TPDK_DLL_HANDLES = []
    if os.name == "nt":
        _candidates = []
        _cuda = os.environ.get("CUDA_PATH")
        if _cuda:
            _candidates.append(Path(_cuda) / "bin")
        with contextlib.suppress(Exception):
            _candidates.append(
                Path(sysconfig.get_paths()["purelib"]) / "torch" / "lib"
            )
        for _path in _candidates:
            if _path.is_dir():
                with contextlib.suppress(OSError):
                    _TPDK_DLL_HANDLES.append(os.add_dll_directory(str(_path)))
except Exception:  # noqa: BLE001 - never break interpreter startup
    pass
