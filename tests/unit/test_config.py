from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tpdk_overlay.config import load_config, project_root


def test_default_config_loads() -> None:
    config = load_config()
    assert config.server.port == 8765
    assert config.asr.model == "Qwen/Qwen3-ASR-0.6B"
    assert config.asr.backend == "llama_cpp"
    assert config.asr.llama_cpp.base_url == "http://127.0.0.1:8081"
    assert config.asr.llama_cpp.max_new_tokens == 128
    assert config.asr.llama_cpp.partial_interval_ms == 5000
    assert config.asr.llama_cpp.partial_tail_sec == 4.0
    assert config.asr.llama_cpp.min_energy_dbfs == -40.0
    assert config.ocr.det_model == "PP-OCRv6_small_det"
    assert config.data_dir == project_root() / "data"


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TPDK__SERVER__PORT", "9999")
    monkeypatch.setenv("TPDK__OCR__ENABLED", "false")
    config = load_config()
    assert config.server.port == 9999
    assert config.ocr.enabled is False


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    default_yaml = project_root() / "configs" / "default.yaml"
    defaults = yaml.safe_load(default_yaml.read_text(encoding="utf-8"))
    defaults["server"]["typo"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(defaults), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_data_dir_resolves_against_root(tmp_path: Path) -> None:
    default_yaml = project_root() / "configs" / "default.yaml"
    defaults = yaml.safe_load(default_yaml.read_text(encoding="utf-8"))
    defaults["app"]["data_dir"] = "runtime-data"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(defaults), encoding="utf-8")
    config = load_config(path)
    assert config.data_dir == project_root() / "runtime-data"
