from __future__ import annotations

import numpy as np
import pandas as pd

from cvt_track_study.track.laps import _deduplicate_axis


def test_consensus_axis_arrays_are_writable_under_copy_on_write() -> None:
    previous = pd.options.mode.copy_on_write
    pd.options.mode.copy_on_write = True
    try:
        frame = pd.DataFrame(
            {
                "source": [0.0, 0.5, 1.0],
                "x_m": [0.0, 1.0, 0.0],
                "y_m": [0.0, 1.0, 0.0],
                "elevation_m": [100.0, 101.0, 100.0],
            }
        )
        arrays = _deduplicate_axis(
            frame["source"].to_numpy(),
            frame["x_m"].to_numpy(),
            frame["y_m"].to_numpy(),
            frame["elevation_m"].to_numpy(),
        )
        assert all(values.flags.writeable for values in arrays)
        source, x, y, elevation = arrays
        source[0] = 0.0
        source[-1] = 1.0
        x[0] = x[-1]
        y[0] = y[-1]
        elevation[0] = elevation[-1]
    finally:
        pd.options.mode.copy_on_write = previous
