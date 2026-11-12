"""Qwen3-ASR streaming HTTP service (vLLM backend). Linux/WSL only.

Vendored from ``qwen_asr.cli.demo_streaming`` (Apache-2.0) to add forced-language
support and a JSON-only surface for the Windows backend. The upstream demo's
``/api/start`` accepts no arguments and always auto-detects; this service passes
``language`` and ``context`` through to ``init_streaming_state``.

Run inside the WSL ASR venv (see ``scripts/setup_asr_wsl.sh``):

    python services/asr_service.py --asr-model-path Qwen/Qwen3-ASR-0.6B
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from qwen_asr import Qwen3ASRModel

SESSION_TTL_SEC = 10 * 60


@dataclass(slots=True)
class _Session:
    state: Any
    last_seen: float


app = Flask(__name__)
_sessions: dict[str, _Session] = {}
_sessions_lock = threading.Lock()

_asr: Qwen3ASRModel | None = None
_model_name = ""
_stream_kwargs: dict[str, Any] = {
    "unfixed_chunk_num": 4,
    "unfixed_token_num": 5,
    "chunk_size_sec": 1.0,
}


def _require_model() -> Qwen3ASRModel:
    if _asr is None:
        raise RuntimeError("ASR model is not loaded")
    return _asr


def _gc_locked() -> None:
    now = time.time()
    dead = [sid for sid, session in _sessions.items() if now - session.last_seen > SESSION_TTL_SEC]
    for session_id in dead:
        session = _sessions.pop(session_id)
        with suppress(Exception):
            _require_model().finish_streaming_transcribe(session.state)


def _get_session(session_id: str) -> _Session | None:
    with _sessions_lock:
        _gc_locked()
        session = _sessions.get(session_id)
        if session is not None:
            session.last_seen = time.time()
        return session


def _snapshot(state: Any) -> dict[str, str]:
    return {
        "language": getattr(state, "language", "") or "",
        "text": getattr(state, "text", "") or "",
    }


@app.get("/")
def index() -> Any:
    with _sessions_lock:
        active = len(_sessions)
    return jsonify(
        {
            "service": "tpdk-asr",
            "model": _model_name,
            "active_sessions": active,
            "sample_rate": 16_000,
        }
    )


@app.post("/api/start")
def api_start() -> Any:
    language = (request.args.get("language") or "").strip() or None
    context = request.args.get("context", "")
    state = _require_model().init_streaming_state(
        context=context,
        language=language,
        unfixed_chunk_num=_stream_kwargs["unfixed_chunk_num"],
        unfixed_token_num=_stream_kwargs["unfixed_token_num"],
        chunk_size_sec=_stream_kwargs["chunk_size_sec"],
    )
    session_id = uuid.uuid4().hex
    with _sessions_lock:
        _sessions[session_id] = _Session(state=state, last_seen=time.time())
    return jsonify({"session_id": session_id})


@app.post("/api/chunk")
def api_chunk() -> Any:
    session_id = request.args.get("session_id", "")
    session = _get_session(session_id)
    if session is None:
        return jsonify({"error": "invalid session_id"}), 400
    if request.mimetype != "application/octet-stream":
        return jsonify({"error": "expect application/octet-stream"}), 400
    raw = request.get_data(cache=False)
    if not raw or len(raw) % 4 != 0:
        return jsonify({"error": "float32 bytes required"}), 400
    pcm = np.frombuffer(raw, dtype=np.float32)
    _require_model().streaming_transcribe(pcm, session.state)
    return jsonify(_snapshot(session.state))


@app.post("/api/finish")
def api_finish() -> Any:
    session_id = request.args.get("session_id", "")
    session = _get_session(session_id)
    if session is None:
        return jsonify({"error": "invalid session_id"}), 400
    _require_model().finish_streaming_transcribe(session.state)
    with _sessions_lock:
        _sessions.pop(session_id, None)
    return jsonify(_snapshot(session.state))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPdK Qwen3-ASR streaming service")
    parser.add_argument("--asr-model-path", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--unfixed-chunk-num", type=int, default=4)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    global _asr, _model_name, _stream_kwargs

    args = _parse_args()
    _model_name = args.asr_model_path
    _stream_kwargs = {
        "unfixed_chunk_num": args.unfixed_chunk_num,
        "unfixed_token_num": args.unfixed_token_num,
        "chunk_size_sec": args.chunk_size_sec,
    }
    _asr = Qwen3ASRModel.LLM(
        model=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Model loaded: {args.asr_model_path}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
