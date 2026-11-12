"""One-command launcher for the whole TPdK overlay stack.

Starts and supervises the four local processes the prototype needs:

1. the Qwen3-ASR GGUF server (``llama serve``, used by ``asr.backend:
   llama_cpp``),
2. the PLaMo-3-Translate GGUF server (``llama serve``, used by
   ``mt.provider: plamo_llama``),
3. the FastAPI backend (``tpdk_overlay.server.app``), and
4. the PySide6 overlay (``tpdk_overlay.ui.app``).

The overlay is the foreground UI. Quit it from the tray (Quit) or press Ctrl+C
in this console; the backend and the llama servers this launcher started are
stopped afterwards. Services that were already healthy before the launch are
reused and left running.

Usage::

    python main.py                 # starts everything
    python main.py --no-asr        # reuse an ASR server you started yourself
    python main.py --no-mt         # reuse a PLaMo server you started yourself
    python main.py --no-backend    # reuse a backend you started yourself
    python main.py --model ggml-org/Qwen3-ASR-1.7B-GGUF:Q8_0

The file restarts itself with ``.venv\\Scripts\\python.exe`` when the current
interpreter is not the project virtualenv, so a plain ``python main.py`` works
after ``scripts\\bootstrap.ps1``.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DEFAULT_ASR_MODEL = "ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0"

ASR_START_TIMEOUT_S = 300.0  # first run downloads the GGUF checkpoint
MT_START_TIMEOUT_S = 300.0  # same, for the PLaMo translation server
BACKEND_START_TIMEOUT_S = 60.0
SERVICE_GRACE_S = 20.0  # another instance may still be loading
STOP_TIMEOUT_S = 5.0
REPORT_INTERVAL_S = 15.0


class LauncherError(RuntimeError):
    """Startup failure with a message meant for the console."""


@dataclass(slots=True)
class Service:
    """A configured service endpoint plus the process owned by this launcher."""

    health_url: str
    proc: subprocess.Popen[bytes] | None = None


def _is_project_python() -> bool:
    try:
        return os.path.samefile(sys.executable, VENV_PYTHON)
    except OSError:
        return os.path.normcase(os.path.abspath(sys.executable)) == os.path.normcase(
            os.path.abspath(VENV_PYTHON)
        )


def _relaunch_in_venv() -> int | None:
    """Re-run this file with the project venv; return ``None`` if already there."""

    if not VENV_PYTHON.is_file() or _is_project_python():
        return None
    print(f"[launcher] restarting with the project virtualenv: {VENV_PYTHON}", flush=True)
    proc = subprocess.Popen(
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=str(ROOT),
    )
    try:
        return proc.wait()
    except KeyboardInterrupt:
        # The venv launcher received the same Ctrl+C and cleans up its children.
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=STOP_TIMEOUT_S)
        return 130


def _load_config():
    sys.path.insert(0, str(ROOT / "src"))
    from tpdk_overlay.config import load_config

    return load_config()


def _endpoint(base: str, path: str) -> tuple[str, str, int]:
    """Return ``(url, connect_host, port)`` for one configured base URL."""

    parsed = urlparse(base)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    return f"{scheme}://{connect_host}:{port}{path}", connect_host, port


def _http_ok(url: str, timeout_s: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return False


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def _wait_for_health(
    name: str,
    url: str,
    proc: subprocess.Popen[bytes] | None,
    timeout_s: float,
) -> None:
    started = time.monotonic()
    next_report = 0.0
    while True:
        if proc is not None and proc.poll() is not None:
            raise LauncherError(
                f"{name} exited with code {proc.returncode} before becoming healthy"
            )
        if _http_ok(url):
            print(f"[launcher] {name} ready: {url}", flush=True)
            return
        elapsed = time.monotonic() - started
        if elapsed >= timeout_s:
            raise LauncherError(
                f"{name} did not become healthy within {timeout_s:.0f}s ({url})"
            )
        if elapsed >= next_report:
            print(f"[launcher] waiting for {name} ({elapsed:.0f}s)...", flush=True)
            next_report = elapsed + REPORT_INTERVAL_S
        time.sleep(1.0)


def _asr_command(config, llama_exe: str, model: str) -> tuple[list[str], str, str, int]:
    base = config.asr.llama_cpp.base_url
    health_url, host, port = _endpoint(base, "/health")
    bind_host = urlparse(base).hostname or "127.0.0.1"
    command = [
        llama_exe,
        "serve",
        "-hf",
        model,
        "--host",
        bind_host,
        "--port",
        str(port),
        "-ngl",
        "all",
        "-c",
        "4096",
        "-np",
        "1",
        "-n",
        str(config.asr.llama_cpp.max_new_tokens),
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
    ]
    return command, health_url, host, port


def _start_asr(config, model: str) -> Service:
    """Spawn ``llama serve`` when needed; a reused service has ``proc=None``."""

    if config.asr.backend != "llama_cpp":
        print(
            "[launcher] asr.backend is 'http'; expecting the WSL ASR service, "
            "not starting llama serve",
            flush=True,
        )
        return Service(health_url="")

    command, health_url, host, port = _asr_command(config, "llama", model)

    if _http_ok(health_url):
        print(f"[launcher] ASR server already healthy at {health_url}; reusing it", flush=True)
        return Service(health_url=health_url)
    if _port_open(host, port):
        print(f"[launcher] port {port} is already in use; waiting for {health_url}", flush=True)
        _wait_for_health("ASR server", health_url, None, SERVICE_GRACE_S)
        return Service(health_url=health_url)

    llama_exe = shutil.which("llama")
    if llama_exe is None:
        raise LauncherError(
            "llama CLI not found on PATH. Install it with "
            "`irm https://llama.app/install.ps1 | iex`, or start the ASR service "
            "yourself and use asr.backend: http."
        )
    command[0] = llama_exe
    print(f"[launcher] starting ASR server on {health_url}", flush=True)
    print("[launcher] (the first run downloads the GGUF model)", flush=True)
    proc = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL)
    return Service(health_url=health_url, proc=proc)


def _mt_command(config, llama_exe: str) -> tuple[list[str], str, str, int]:
    plamo = config.mt.plamo_llama
    base = plamo.base_url
    health_url, host, port = _endpoint(base, "/health")
    bind_host = urlparse(base).hostname or "127.0.0.1"
    model_path = config.resolve(Path(plamo.model_path))
    model_args = (
        ["-m", str(model_path)]
        if model_path.is_file()
        else ["-hf", plamo.hf_model]
    )
    command = [
        llama_exe,
        "serve",
        *model_args,
        "--host",
        bind_host,
        "--port",
        str(port),
        "--alias",
        plamo.model,
        "-ngl",
        "all",
        "-c",
        str(plamo.context_size),
        "-np",
        str(plamo.concurrency),
        "--jinja",
        "--no-prefill-assistant",
    ]
    return command, health_url, host, port


def _start_mt(config) -> Service:
    """Spawn the PLaMo ``llama serve`` process when the provider needs it."""

    if config.mt.provider != "plamo_llama":
        print(
            f"[launcher] mt.provider is {config.mt.provider!r}; "
            "not starting the PLaMo server",
            flush=True,
        )
        return Service(health_url="")

    command, health_url, host, port = _mt_command(config, "llama")

    if _http_ok(health_url):
        print(
            f"[launcher] PLaMo server already healthy at {health_url}; reusing it",
            flush=True,
        )
        return Service(health_url=health_url)
    if _port_open(host, port):
        print(f"[launcher] port {port} is already in use; waiting for {health_url}", flush=True)
        _wait_for_health("PLaMo server", health_url, None, SERVICE_GRACE_S)
        return Service(health_url=health_url)

    llama_exe = shutil.which("llama")
    if llama_exe is None:
        raise LauncherError(
            "llama CLI not found on PATH. It is required by mt.provider: "
            "plamo_llama; install it with `irm https://llama.app/install.ps1 | iex` "
            "or switch mt.provider to nllb_ct2."
        )
    command[0] = llama_exe
    print(f"[launcher] starting PLaMo translation server on {health_url}", flush=True)
    if "-hf" in command:
        print("[launcher] (the first run downloads the GGUF model)", flush=True)
    proc = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL)
    return Service(health_url=health_url, proc=proc)


def _start_backend(config) -> Service:
    base = f"http://{config.server.host}:{config.server.port}"
    health_url, host, port = _endpoint(base, "/api/health")

    if _http_ok(health_url):
        print(f"[launcher] backend already healthy at {health_url}; reusing it", flush=True)
        return Service(health_url=health_url)
    if _port_open(host, port):
        print(f"[launcher] port {port} is already in use; waiting for {health_url}", flush=True)
        _wait_for_health("backend", health_url, None, SERVICE_GRACE_S)
        return Service(health_url=health_url)

    print(f"[launcher] starting backend on {health_url}", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "tpdk_overlay.server.app"],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
    )
    return Service(health_url=health_url, proc=proc)


def _stop(proc: subprocess.Popen[bytes] | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"[launcher] stopping {name}...", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=STOP_TIMEOUT_S)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start the TPdK ASR server, FastAPI backend, and overlay as one stack."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_ASR_MODEL,
        help=f"llama -hf model spec (default: {DEFAULT_ASR_MODEL})",
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        help="do not start llama serve; reuse an ASR server that is already running",
    )
    parser.add_argument(
        "--no-mt",
        action="store_true",
        help="do not start the PLaMo server; reuse one that is already running",
    )
    parser.add_argument(
        "--no-backend",
        action="store_true",
        help="do not start the backend; reuse one that is already running",
    )
    args = parser.parse_args(argv)

    spawned: dict[str, subprocess.Popen[bytes] | None] = {
        "overlay": None,
        "backend": None,
        "PLaMo server": None,
        "ASR server": None,
    }
    try:
        config = _load_config()

        if not args.no_asr:
            asr = _start_asr(config, args.model)
            spawned["ASR server"] = asr.proc
            if asr.proc is not None:
                _wait_for_health(
                    "ASR server", asr.health_url, asr.proc, ASR_START_TIMEOUT_S
                )

        if not args.no_mt:
            mt = _start_mt(config)
            spawned["PLaMo server"] = mt.proc
            if mt.proc is not None:
                _wait_for_health(
                    "PLaMo server", mt.health_url, mt.proc, MT_START_TIMEOUT_S
                )

        if not args.no_backend:
            backend = _start_backend(config)
            spawned["backend"] = backend.proc
            if backend.proc is not None:
                _wait_for_health(
                    "backend", backend.health_url, backend.proc, BACKEND_START_TIMEOUT_S
                )

        print("[launcher] starting overlay...", flush=True)
        spawned["overlay"] = subprocess.Popen(
            [sys.executable, "-m", "tpdk_overlay.ui.app"],
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
        )
        print(
            "[launcher] stack ready. Quit from the tray (Quit) or press Ctrl+C here.",
            flush=True,
        )
        try:
            return spawned["overlay"].wait()
        except KeyboardInterrupt:
            print("\n[launcher] interrupting...", flush=True)
            return 130
    except KeyboardInterrupt:
        print("\n[launcher] interrupted.", flush=True)
        return 130
    except LauncherError as exc:
        print(f"[launcher] error: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:  # noqa: BLE001 - startup failure is a console message
        print(f"[launcher] unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        # Overlay first, then the backend, then the model servers.
        for name in ("overlay", "backend", "PLaMo server", "ASR server"):
            _stop(spawned[name], name)


if __name__ == "__main__":
    _relaunch_code = _relaunch_in_venv()
    if _relaunch_code is not None:
        raise SystemExit(_relaunch_code)
    raise SystemExit(main())
