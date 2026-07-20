from __future__ import annotations

import pandas as pd

from cvt_track_study.reports.html import dataframe_table, render_page


def test_table_keeps_identity_columns_left_and_supports_three_state_sorting() -> None:
    frame = pd.DataFrame(
        [
            {"score": 2.0, "gate_name": "Beta", "gate_id": "E02"},
            {"score": 1.0, "gate_name": "Alpha", "gate_id": "E01"},
        ]
    )
    table = dataframe_table(
        frame,
        sticky_columns=("gate_id", "gate_name"),
        table_id="gate-table",
    )
    page = render_page(
        title="test",
        subtitle="test",
        body=table,
        report_key="test",
    )

    assert page.index("Gate Id") < page.index("Gate Name") < page.index("Score")
    assert 'id="gate-table"' in page
    assert page.count("sticky-col") >= 4
    assert 'data-sort-state="none"' in page
    assert "previous === 'none' ? 'asc'" in page
    assert "previous === 'asc' ? 'desc' : 'none'" in page
    assert "originalRows.slice()" in page
