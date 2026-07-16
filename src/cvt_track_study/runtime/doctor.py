"""Installation and project readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

from cvt_track_study.config import ProjectLoader


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


def run_doctor(project: Path | None = None) -> tuple[DoctorCheck, ...]:
    checks: list[DoctorCheck] = []
    python_ok = sys.version_info >= (3, 10)
    checks.append(
        DoctorCheck(
            "Python",
            "pass" if python_ok else "fail",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )
    for module in ("defusedxml", "matplotlib", "numpy", "pandas", "scipy"):
        try:
            imported = importlib.import_module(module)
            detail = str(getattr(imported, "__version__", "installed"))
            status = "pass"
        except Exception as exc:  # pragma: no cover - depends on installation
            detail, status = str(exc), "fail"
        checks.append(DoctorCheck(module, status, detail))
    if project is not None:
        try:
            resolved = ProjectLoader().resolve(project)
            status = "fail" if resolved.error_count else (
                "warning" if resolved.warning_count else "pass"
            )
            detail = (
                f"{resolved.error_count} error(s), {resolved.warning_count} warning(s)"
            )
        except Exception as exc:
            status, detail = "fail", str(exc)
        checks.append(DoctorCheck("Project", status, detail))
    return tuple(checks)
