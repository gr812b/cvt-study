from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cvt_track_study.reports.uncertainty_mechanism import (
    enhance_full_uncertainty_report,
)


def test_mechanism_overlay_writes_intervals_and_exact_nominal(tmp_path: Path):
    rng = np.random.default_rng(12)
    rows = pd.DataFrame(
        {
            "bounded_drivetrain_loss_energy_kj": rng.normal(2.0, 0.2, 100),
            "bounded_clutch_loss_energy_kj": rng.normal(1.0, 0.1, 100),
            "bounded_tire_slip_loss_energy_kj": rng.normal(0.5, 0.08, 100),
            "bounded_brake_loss_energy_kj": rng.normal(3.0, 0.4, 100),
            "bounded_rolling_loss_energy_kj": rng.normal(2.5, 0.2, 100),
            "bounded_aerodynamic_loss_energy_kj": rng.normal(0.4, 0.05, 100),
            "bounded_obstacle_loss_energy_kj": rng.normal(4.0, 0.8, 100),
            "finite_ratio_opportunity_loss_energy_kj": rng.normal(6.0, 1.0, 100),
            "lap_time_penalty_vs_infinite_s": rng.normal(8.0, 1.5, 100),
        }
    )
    rows.to_csv(tmp_path / "replicate_results.csv", index=False)
    nominal = {column: float(values.median()) for column, values in rows.items()}
    (tmp_path / "nominal_reference.json").write_text(
        json.dumps([nominal]), encoding="utf-8"
    )
    target = tmp_path / "full_uncertainty_report.html"
    target.write_text(
        "<html><body><main><h1>Report</h1><h2>Existing</h2></main></body></html>",
        encoding="utf-8",
    )

    enhance_full_uncertainty_report(tmp_path, target)

    assert (tmp_path / "full_uncertainty_mechanism_summary.csv").is_file()
    assert (
        tmp_path / "report_plots" / "physical_losses_opportunity_and_time.png"
    ).is_file()
    document = target.read_text(encoding="utf-8")
    assert "cvt-mechanism-overlay:start" in document
    assert "exact nominal" in document.lower()
