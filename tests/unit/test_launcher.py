"""Launcher helpers: endpoint normalization, ASR command, health probes."""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

from tpdk_overlay.config import load_config

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("tpdk_launcher_main", ROOT / "main.py")
launcher = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = launcher
_SPEC.loader.exec_module(launcher)


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_endpoint_maps_wildcard_host_to_loopback() -> None:
    url, host, port = launcher._endpoint("http://0.0.0.0:8765", "/api/health")
    assert url == "http://127.0.0.1:8765/api/health"
    assert host == "127.0.0.1"
    assert port == 8765


def test_endpoint_keeps_default_https_port() -> None:
    url, host, port = launcher._endpoint("https://example.test", "/health")
    assert url == "https://example.test:443/health"
    assert port == 443


def test_asr_command_uses_configured_endpoint_and_model() -> None:
    config = load_config()
    command, health_url, _host, _port = launcher._asr_command(
        config, "llama.exe", "repo/model:Q8_0"
    )
    assert command[:4] == ["llama.exe", "serve", "-hf", "repo/model:Q8_0"]
    assert "--port" in command
    assert str(config.asr.llama_cpp.max_new_tokens) in command
    assert health_url.endswith("/health")


def test_health_and_port_probes_fail_for_closed_port() -> None:
    port = _closed_port()
    assert launcher._port_open("127.0.0.1", port) is False
    assert launcher._http_ok(f"http://127.0.0.1:{port}/health", timeout_s=0.2) is False
