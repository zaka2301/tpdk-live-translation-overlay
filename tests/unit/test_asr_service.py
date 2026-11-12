from __future__ import annotations

import importlib
import sys
import types

import pytest

FAKE_FLOAT32_ONE = b"\x00\x00\x80?"


class _FakeState:
    def __init__(self, language: str | None) -> None:
        self.language = language or "English"
        self.text = ""


class _FakeModel:
    def __init__(self) -> None:
        self.init_args: dict[str, object] = {}

    def init_streaming_state(
        self,
        context: str = "",
        language: str | None = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ) -> _FakeState:
        self.init_args = {
            "context": context,
            "language": language,
            "unfixed_chunk_num": unfixed_chunk_num,
            "unfixed_token_num": unfixed_token_num,
            "chunk_size_sec": chunk_size_sec,
        }
        return _FakeState(language)

    def streaming_transcribe(self, pcm, state: _FakeState) -> None:
        state.text += f"[{pcm.size}]"

    def finish_streaming_transcribe(self, state: _FakeState) -> None:
        state.text += "|final"


class _FakeQwen3ASRModel:
    @classmethod
    def LLM(cls, **kwargs) -> _FakeModel:
        model = _FakeModel()
        model.load_kwargs = kwargs
        return model


@pytest.fixture()
def service(monkeypatch: pytest.MonkeyPatch):
    fake_module = types.ModuleType("qwen_asr")
    fake_module.Qwen3ASRModel = _FakeQwen3ASRModel
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_module)
    sys.modules.pop("services.asr_service", None)

    module = importlib.import_module("services.asr_service")
    module._asr = _FakeQwen3ASRModel.LLM(model="fake")
    module._model_name = "fake"
    module._sessions.clear()
    yield module
    module._sessions.clear()


def test_start_passes_language_and_context(service) -> None:
    response = service.app.test_client().post("/api/start?language=Indonesian&context=lecture")
    assert response.status_code == 200
    session_id = response.get_json()["session_id"]

    assert service._asr.init_args["language"] == "Indonesian"
    assert service._asr.init_args["context"] == "lecture"
    assert service._sessions[session_id].state.language == "Indonesian"


def test_start_without_language_autodetects(service) -> None:
    response = service.app.test_client().post("/api/start")
    assert response.status_code == 200
    assert service._asr.init_args["language"] is None


def test_chunk_is_cumulative_and_finish_closes(service) -> None:
    client = service.app.test_client()
    session_id = client.post("/api/start").get_json()["session_id"]

    chunk = client.post(
        f"/api/chunk?session_id={session_id}",
        data=FAKE_FLOAT32_ONE * 10,
        content_type="application/octet-stream",
    )
    assert chunk.status_code == 200
    assert chunk.get_json()["text"] == "[10]"

    finished = client.post(f"/api/finish?session_id={session_id}")
    assert finished.status_code == 200
    assert finished.get_json()["text"] == "[10]|final"
    assert session_id not in service._sessions


def test_invalid_session_and_payload_are_rejected(service) -> None:
    client = service.app.test_client()
    assert client.post("/api/finish?session_id=missing").status_code == 400

    session_id = client.post("/api/start").get_json()["session_id"]
    bad = client.post(
        f"/api/chunk?session_id={session_id}",
        data=b"\x00",
        content_type="application/octet-stream",
    )
    assert bad.status_code == 400


def test_index_reports_service_metadata(service) -> None:
    body = service.app.test_client().get("/").get_json()
    assert body["service"] == "tpdk-asr"
    assert body["model"] == "fake"
