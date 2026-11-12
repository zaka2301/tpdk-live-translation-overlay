"""Clocks used for latency measurement.

``now_ms`` is wall clock and is comparable across processes on the same machine;
``monotonic_ms`` never jumps when the system clock changes and is what session
timestamps must use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def now_ms() -> int:
    """Wall-clock Unix epoch in milliseconds."""

    return int(time.time() * 1000)


def monotonic_ms() -> int:
    """Monotonic milliseconds; only differences are meaningful."""

    return int(time.monotonic() * 1000)


@dataclass(slots=True)
class Stopwatch:
    """Measure elapsed monotonic time for one stage of the pipeline."""

    _start: float = field(default_factory=time.perf_counter)

    def reset(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)
