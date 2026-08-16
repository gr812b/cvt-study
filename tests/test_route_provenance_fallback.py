from types import SimpleNamespace

import pandas as pd

# Match normal CLI import order; the package has legacy lazy-import cycles.
import cvt_track_study.cli  # noqa: F401
from cvt_track_study.track.family import _shared_gate_evidence


def test_single_route_shared_gate_evidence_does_not_require_source_variant_column():
    # Single-route nominal builds legitimately predate route-family provenance
    # columns.  The shared-evidence audit must fall back to the target route ID.
    passes = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "event_name": "test event",
                "lap_id": 1,
                "eligible": True,
            }
        ]
    )
    output = _shared_gate_evidence(
        {"route_001": SimpleNamespace(event_passes=passes)}
    )
    assert output.loc[0, "contributing_source_route_variant_ids"] == "route_001"
