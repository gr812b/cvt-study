"""Small dependency-free progress and ETA reporter for framework runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Callable


@dataclass(slots=True)
class ProgressReporter:
    total: int
    label: str = "cases"
    enabled: bool = True
    emit: Callable[[str], None] = print
    completed: int = 0
    _started: float = field(default_factory=monotonic)
    _lock: Lock = field(default_factory=Lock)

    def advance(self, message: str = "") -> None:
        if not self.enabled:
            return
        with self._lock:
            self.completed = min(self.total, self.completed + 1)
            elapsed = max(monotonic() - self._started, 1e-9)
            rate = self.completed / elapsed
            remaining = max(self.total - self.completed, 0)
            eta = remaining / rate if rate > 0.0 else 0.0
            suffix = f" — {message}" if message else ""
            self.emit(
                f"[{self.completed}/{self.total} {self.label}] "
                f"elapsed {_duration(elapsed)}, ETA {_duration(eta)}{suffix}"
            )


def _duration(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"
