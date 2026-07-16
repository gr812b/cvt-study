"""Structured diagnostics used by project loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Diagnostic:
    severity: Severity
    code: str
    message: str
    path: str = ""
    hint: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if value != ""}

    def format(self) -> str:
        location = f" {self.path}" if self.path else ""
        source = f" ({self.source})" if self.source else ""
        first = f"[{self.severity.value.upper()}] {self.code}{location}{source}: {self.message}"
        return first if not self.hint else f"{first}\n        Hint: {self.hint}"


class DiagnosticBag:
    """Mutable collector with convenient severity checks."""

    def __init__(self, diagnostics: Iterable[Diagnostic] = ()) -> None:
        self._items = list(diagnostics)

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        *,
        path: str = "",
        hint: str = "",
        source: str = "",
    ) -> None:
        self._items.append(
            Diagnostic(
                severity=severity,
                code=code,
                message=message,
                path=path,
                hint=hint,
                source=source,
            )
        )

    def error(self, code: str, message: str, **kwargs: str) -> None:
        self.add(Severity.ERROR, code, message, **kwargs)

    def warning(self, code: str, message: str, **kwargs: str) -> None:
        self.add(Severity.WARNING, code, message, **kwargs)

    def info(self, code: str, message: str, **kwargs: str) -> None:
        self.add(Severity.INFO, code, message, **kwargs)

    @property
    def items(self) -> tuple[Diagnostic, ...]:
        return tuple(self._items)

    @property
    def error_count(self) -> int:
        return sum(item.severity is Severity.ERROR for item in self._items)

    @property
    def warning_count(self) -> int:
        return sum(item.severity is Severity.WARNING for item in self._items)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        self._items.extend(diagnostics)
