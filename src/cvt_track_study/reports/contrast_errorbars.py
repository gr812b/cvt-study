"""Install a report-wide, high-contrast Matplotlib error-bar policy.

The framework commonly draws uncertainty intervals over Matplotlib's default
blue bars.  Matplotlib otherwise reuses the active colour cycle for the error
bars, which can make those intervals nearly invisible.  Installing this patch
at package import keeps every report error bar black without requiring each
plotting call to remember ``ecolor`` independently.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from matplotlib.axes import Axes

_PATCH_FLAG = "_cvt_black_errorbars_installed"
_ORIGINAL_ATTR = "_cvt_original_errorbar"


def install_contrast_errorbars() -> None:
    """Make every Matplotlib ``Axes.errorbar`` interval black and readable.

    The installation is idempotent, so importing or reloading the reports
    package cannot wrap ``Axes.errorbar`` more than once.
    """

    if bool(getattr(Axes, _PATCH_FLAG, False)):
        return

    original = Axes.errorbar

    @wraps(original)
    def errorbar_with_contrast(
        self: Axes,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Deliberately override any inherited/explicit series colour.  Error
        # intervals are an annotation layer in these reports and must remain
        # visually distinct from the bars or lines beneath them.
        kwargs["ecolor"] = "black"
        kwargs.setdefault("elinewidth", 1.8)
        kwargs.setdefault("capthick", 1.8)
        return original(self, *args, **kwargs)

    setattr(Axes, _ORIGINAL_ATTR, original)
    Axes.errorbar = errorbar_with_contrast  # type: ignore[method-assign]
    setattr(Axes, _PATCH_FLAG, True)


__all__ = ["install_contrast_errorbars"]
