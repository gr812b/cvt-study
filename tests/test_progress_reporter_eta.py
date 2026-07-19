from __future__ import annotations

from cvt_track_study.runtime.progress import (
    ProgressReporter,
)


def test_progress_prints_start_eta_and_finish():
    messages = []
    reporter = ProgressReporter(
        total=2,
        label="parameter levels",
        emit=messages.append,
    )
    reporter.begin("3 parameters")
    reporter.advance("vehicle.mass nominal")
    reporter.advance("vehicle.mass q0.95")
    reporter.finish("done")

    assert "started" in messages[0]
    assert "ETA" in messages[1]
    assert "1/2 parameter levels" in messages[1]
    assert "complete" in messages[-1]
