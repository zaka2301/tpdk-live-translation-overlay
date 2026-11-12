"""llama.cpp adapter: Qwen3-ASR GGUF behind a local ``llama serve`` process.

The model runs in a separate Windows process (``llama serve -hf
ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0 --port 8081``); this adapter posts each
window as base64 WAV to the OpenAI-compatible ``/v1/chat/completions``
endpoint and parses the ``language <NAME><asr_text>transcription`` reply. No
torch or transformers is needed in this process, which makes this the machine
default backend. The server always auto-detects the language.

Partials are cumulative previews of the utterance so far, posted every
``partial_interval_ms``; each preview replaces the previous one. Set
``partial_tail_sec`` above zero to preview only the recent tail instead. The
final always transcribes the whole utterance.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import string
import wave
import zlib
from typing import Any

import httpx
import numpy as np

from tpdk_overlay.asr.errors import AsrTimeout, AsrUnavailable
from tpdk_overlay.config import AsrConfig, LlamaCppAsrConfig
from tpdk_overlay.ports import AsrStream, AsrUpdate
from tpdk_overlay.utils.audio import SAMPLE_RATE_16K, rms, to_pcm16

logger = logging.getLogger(__name__)

_WARMUP_TIMEOUT_S = 5.0
_MIN_AUDIO_SEC = 0.2
_PARTIAL_MAX_TOKENS = 64  # previews are replaced by the final; cap their decode budget
_LOOP_MIN_BYTES = 200  # shorter outputs are truncated harmlessly; only guard real runaways
_LOOP_MAX_RATIO = 0.2  # exact-repeat loops compress to <0.06; natural text stays >0.3
_UTTERANCE_MAX_CHARS = 450  # ~3x the fastest human speech over a 12 s window

# Full language names (as emitted by Qwen3-ASR) -> ISO 639 codes.
_NAME_TO_CODE = {
    "japanese": "ja",
    "english": "en",
    "chinese": "zh",
    "cantonese": "yue",
    "korean": "ko",
    "arabic": "ar",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "portuguese": "pt",
    "indonesian": "id",
    "italian": "it",
    "russian": "ru",
    "thai": "th",
    "vietnamese": "vi",
    "turkish": "tr",
    "hindi": "hi",
    "malay": "ms",
    "dutch": "nl",
    "swedish": "sv",
    "danish": "da",
    "finnish": "fi",
    "polish": "pl",
    "czech": "cs",
    "filipino": "fil",
    "persian": "fa",
    "greek": "el",
    "hungarian": "hu",
    "macedonian": "mk",
    "romanian": "ro",
}

_STRIP_CHARS = str.maketrans("", "", string.punctuation + "。、！？!?,.\"'`~…-")

# Encoder-decoder ASR models emit these phrases when fed instrumental music or
# silence (artifacts of end-of-video training data). Normalized exact matches
# are dropped before text reaches the overlay.
_MUSIC_HALLUCINATIONS = frozenset(
    phrase.lower().translate(_STRIP_CHARS).strip()
    for phrase in (
        "Thank you for watching",
        "Thanks for watching",
        "Thank you for viewing",
        "Thank you for your viewing",
        "Thanks for viewing",
        "Thanks for watching!",
        "Please subscribe",
        "Please subscribe to my channel",
        "See you in the next video",
        "Subscribe to my channel",
    )
)


def normalize_language(raw: str | None) -> str:
    """Map a model-reported language name or code onto an ISO 639 code.

    Accepts ``"English"``, ``"en"``, ``"en-US"``, etc. Unknown values pass
    through lowercased (primary subtag only) so callers still get a tag.
    """

    if not raw or not raw.strip():
        return ""
    value = raw.strip().lower()
    mapped = _NAME_TO_CODE.get(value)
    if mapped is not None:
        return mapped
    return value.replace("_", "-").split("-", 1)[0]


def drop_known_hallucinations(text: str) -> str:
    """Drop an exact-match music/silence hallucination, else return text unchanged."""

    normalized = " ".join(text.lower().translate(_STRIP_CHARS).split())
    if normalized and normalized in _MUSIC_HALLUCINATIONS:
        logger.debug("Dropped music hallucination: %r", text)
        return ""
    return text


def _is_runaway_output(text: str) -> bool:
    """Detect greedy-decode runaways: repetition loops and impossibly long outputs.

    Language-agnostic: exact-repeat loops compress far better than natural
    text, and a 450+ character "transcription" of a short window is
    physically impossible at human speech rates.
    """

    if len(text) > _UTTERANCE_MAX_CHARS:
        return True
    encoded = text.encode("utf-8")
    if len(encoded) < _LOOP_MIN_BYTES:
        return False
    return len(zlib.compress(encoded)) / len(encoded) < _LOOP_MAX_RATIO


def _energy_dbfs(audio: np.ndarray) -> float:
    """Root-mean-square level of a float32 PCM window in dBFS."""

    level = rms(audio)
    if level <= 1e-9:
        return float("-inf")
    return 20.0 * math.log10(level)


def _wav_bytes(audio: np.ndarray) -> bytes:
    """Wrap 16 kHz mono float32 PCM in a 16-bit WAV container."""

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE_16K)
            wav.writeframes(to_pcm16(audio))
        return buffer.getvalue()


def _request_json(
    client: httpx.Client, method: str, path: str, **kwargs: Any
) -> dict[str, Any]:
    try:
        response = client.request(method, path, **kwargs)
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise AsrTimeout(f"llama serve request timed out: {path}") from exc
    except httpx.HTTPError as exc:
        raise AsrUnavailable(f"llama serve request failed: {path}: {exc}") from exc
    except ValueError as exc:
        raise AsrUnavailable(f"llama serve returned non-JSON for {path}") from exc
    if not isinstance(payload, dict):
        raise AsrUnavailable(f"llama serve returned an unexpected payload for {path}")
    return payload


def _parse_output(content: str) -> tuple[str, str]:
    """Parse ``language <NAME><asr_text>transcription`` server output."""

    text = (content or "").strip()
    if "<asr_text>" not in text:
        return "", text
    prefix, transcription = text.split("<asr_text>", 1)
    language = ""
    for line in prefix.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("language ") and len(line) > len("language "):
            language = line[len("language ") :].strip()
        else:
            language = line
        break
    return normalize_language(language), transcription.strip()


class LlamaCppAsrClient:
    """Talk to the OpenAI-compatible ``llama serve`` chat completions API."""

    def __init__(
        self,
        config: AsrConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._llama = config.llama_cpp
        self._client = httpx.Client(
            base_url=self._llama.base_url.rstrip("/"),
            timeout=config.timeout_s,
            transport=transport,
        )

    def warmup(self) -> None:
        """Fail fast when the server is down and warn about ignored language settings."""

        try:
            response = self._client.get("/health", timeout=_WARMUP_TIMEOUT_S)
            response.raise_for_status()
            payload = response.json() if response.content.strip() else {}
        except httpx.TimeoutException as exc:
            raise AsrTimeout(
                f"llama serve warmup timed out at {self._llama.base_url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise AsrUnavailable(
                f"llama serve is not reachable at {self._llama.base_url} ({exc}). "
                "Start it with scripts\\serve_asr_llama.ps1 or: "
                "llama serve -hf ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0 --port 8081"
            ) from exc
        except ValueError as exc:
            raise AsrUnavailable("llama serve returned non-JSON for /health") from exc
        if isinstance(payload, dict):
            status = payload.get("status")
            if status not in (None, "ok"):
                raise AsrUnavailable(
                    f"llama serve at {self._llama.base_url} is not ready: {payload}"
                )
        if self._config.language:
            logger.warning(
                "asr.language is ignored by the llama_cpp backend; "
                "llama serve always auto-detects the language"
            )
        logger.info("llama.cpp ASR server reachable at %s", self._llama.base_url)

    def open(self, *, language: str | None = None) -> AsrStream:
        if language:
            logger.debug(
                "llama_cpp auto-detects language; ignoring requested language %r", language
            )
        return _LlamaCppAsrStream(self, self._llama, language)

    def close(self) -> None:
        self._client.close()

    def transcribe(self, audio: np.ndarray, max_new_tokens: int) -> tuple[str, str]:
        """Send one window to the server; return ``(iso_language, text)``."""

        payload = _request_json(
            self._client,
            "POST",
            "/v1/chat/completions",
            json=self._request_payload(audio, max_new_tokens),
        )
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise AsrUnavailable(
                f"llama serve returned an unexpected reply: {payload!r}"
            ) from exc
        if not isinstance(content, str):
            raise AsrUnavailable("llama serve returned non-text content")
        if finish_reason == "length":
            # Hit the token cap instead of EOS: usually the start of a
            # repetition loop pushing against the per-request budget.
            logger.warning(
                "llama serve hit max_tokens (%d) without EOS; output may be truncated",
                max_new_tokens,
            )
        language, transcription = _parse_output(content)
        transcription = drop_known_hallucinations(transcription)
        if transcription and _is_runaway_output(transcription):
            logger.warning(
                "llama serve returned a likely hallucination (%d chars); dropping",
                len(transcription),
            )
            return language, ""
        return language, transcription

    def _request_payload(self, audio: np.ndarray, max_new_tokens: int) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self._llama.system_prompt:
            messages.append({"role": "system", "content": self._llama.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(_wav_bytes(audio)).decode("ascii"),
                            "format": "wav",
                        },
                    }
                ],
            }
        )
        return {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "repeat_penalty": 1.1,  # break greedy repetition loops before they fill the slot
            "cache_prompt": False,  # every utterance is new audio; don't pin it
            "stream": False,
        }


class _LlamaCppAsrStream:
    def __init__(
        self,
        client: LlamaCppAsrClient,
        llama: LlamaCppAsrConfig,
        language: str | None,
    ) -> None:
        self._client = client
        self._llama = llama
        self._language = normalize_language(language)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._since_partial = 0
        self._interval_samples = max(
            1, int(round(llama.partial_interval_ms * SAMPLE_RATE_16K / 1000.0))
        )
        self._min_samples = int(round(_MIN_AUDIO_SEC * SAMPLE_RATE_16K))
        self._latest = AsrUpdate(language=self._language, text="")
        self._final: AsrUpdate | None = None
        self._closed = False

    def push(self, pcm16k: np.ndarray) -> AsrUpdate:
        if self._closed:
            raise AsrUnavailable("ASR stream is closed")
        pcm = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
        if pcm.size:
            self._buffer = np.concatenate((self._buffer, pcm))
            self._since_partial += pcm.size
        if self._since_partial < self._interval_samples:
            return self._latest
        self._since_partial = 0
        audio = self._tail_window()
        if audio.size < self._min_samples:
            return self._latest
        level = _energy_dbfs(audio)
        if level < self._llama.min_energy_dbfs:
            # VAD false-fired on room tone: the server can only answer
            # music-credit hallucinations to near-silence.
            logger.debug("Dropping near-silent partial window (%.1f dBFS)", level)
            return self._latest
        language, text = self._client.transcribe(
            audio, min(_PARTIAL_MAX_TOKENS, self._llama.max_new_tokens)
        )
        if text:
            # An empty partial is usually a dropped hallucination; keep the
            # previous preview instead of clearing the overlay.
            self._latest = AsrUpdate(language=language or self._language, text=text)
        return self._latest

    def finish(self) -> AsrUpdate:
        if self._final is not None:
            return self._final
        if self._closed:
            raise AsrUnavailable("ASR stream is closed")
        audio = self._buffer
        self._buffer = np.zeros(0, dtype=np.float32)
        self._since_partial = 0
        try:
            level = _energy_dbfs(audio)
            if audio.size < self._min_samples or level < self._llama.min_energy_dbfs:
                logger.debug("Dropping near-silent final window (%.1f dBFS)", level)
                self._final = AsrUpdate(
                    language=self._latest.language or self._language, text=""
                )
            else:
                language, text = self._client.transcribe(audio, self._llama.max_new_tokens)
                self._final = AsrUpdate(
                    language=language or self._latest.language or self._language,
                    text=text,
                )
        except Exception:
            # Don't retry ever-longer audio after a failure; drop the
            # utterance so the next one starts from a clean buffer.
            self._closed = True
            raise
        self._closed = True
        return self._final

    def close(self) -> None:
        self._closed = True
        self._buffer = np.zeros(0, dtype=np.float32)
        self._since_partial = 0

    def _tail_window(self) -> np.ndarray:
        if self._llama.partial_tail_sec <= 0:
            # Cumulative preview: the model sees everything heard so far, so
            # successive previews grow instead of replacing unrelated tails.
            return self._buffer
        tail = max(1, int(round(self._llama.partial_tail_sec * SAMPLE_RATE_16K)))
        return self._buffer[-tail:]


__all__ = [
    "LlamaCppAsrClient",
    "drop_known_hallucinations",
    "normalize_language",
]
