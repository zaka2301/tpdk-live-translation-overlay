from __future__ import annotations

import json

from fastapi.testclient import TestClient

from tpdk_overlay.server.app import create_app


def test_health() -> None:
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["protocol"] == "1.0"


def test_websocket_greets_with_idle_state() -> None:
    client = TestClient(create_app())
    with client.websocket_connect("/ws") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "session.state"
        assert hello["state"] == "idle"


def test_websocket_ping_pong() -> None:
    client = TestClient(create_app())
    with client.websocket_connect("/ws") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "ping", "client_ts_ms": 1234}))
        pong = json.loads(ws.receive_text())
        assert pong["type"] == "pong"
        assert pong["echo_ts_ms"] == 1234


def test_invalid_command_returns_notice() -> None:
    client = TestClient(create_app())
    with client.websocket_connect("/ws") as ws:
        ws.receive_text()
        ws.send_text("{not json")
        notice = json.loads(ws.receive_text())
        assert notice["type"] == "notice"
        assert notice["code"] == "invalid_command"


def test_stop_without_session_returns_notice() -> None:
    client = TestClient(create_app())
    with client.websocket_connect("/ws") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "session.stop"}))
        notice = json.loads(ws.receive_text())
        assert notice["type"] == "notice"
        assert notice["code"] == "session_not_running"
