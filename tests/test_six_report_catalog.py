from cvt_track_study.reports.catalog import REPORTS, canonical_report_key


def test_exactly_six_canonical_reports() -> None:
    assert tuple(REPORTS) == (
        "track_evidence",
        "nominal_simulation",
        "track_robustness",
        "structural_sensitivity",
        "full_uncertainty",
        "design_comparison",
    )
    assert len({item.html_filename for item in REPORTS.values()}) == 6


def test_legacy_names_route_to_new_contract() -> None:
    assert canonical_report_key("baseline") == "nominal_simulation"
    assert canonical_report_key("uncertainty") == "full_uncertainty"
    assert canonical_report_key("sweep") == "design_comparison"
