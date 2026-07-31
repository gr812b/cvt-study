from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from cvt_track_study.reports.contrast_errorbars import install_contrast_errorbars


def test_report_errorbars_are_forced_to_black() -> None:
    install_contrast_errorbars()

    figure, axis = plt.subplots()
    container = axis.errorbar(
        np.array([0.0, 1.0]),
        np.array([1.0, 2.0]),
        yerr=np.array([0.25, 0.4]),
        fmt="none",
        ecolor="tab:blue",  # The report policy must override this.
        capsize=3,
    )

    bar_collections = [
        child for child in axis.get_children() if isinstance(child, LineCollection)
    ]
    assert bar_collections
    assert all(
        tuple(collection.get_colors()[0][:3]) == (0.0, 0.0, 0.0)
        for collection in bar_collections
        if len(collection.get_colors())
    )
    assert container is not None
    plt.close(figure)
