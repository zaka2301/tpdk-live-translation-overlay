"""Process-wide logging setup.

Console output is always enabled. A rotating file handler is added when
``log_dir`` is given, so a failed lecture session can be inspected afterwards.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    *,
    force: bool = False,
) -> None:
    root = logging.getLogger()
    numeric = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric)
    if root.handlers and not force:
        return
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["get_logger", "setup_logging"]
