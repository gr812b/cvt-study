# Traffic reporting update

This patch changes reporting only. It does **not** change traffic generation, the traffic speed ceiling, vehicle dynamics, gate physics, or the paired scenario scheduler.

## Full uncertainty

The traffic audit is now placed directly after the physical-mechanism uncertainty section and linked from the report navigation.

New artifacts:

- `full_uncertainty_traffic_world_summary.csv`
- `full_uncertainty_traffic_masking_summary.csv`
- `report_plots/traffic_lap_time_impact_distribution.png`
- `report_plots/traffic_exposure_vs_time_cost.png`
- `report_plots/traffic_vs_finite_ratio_penalty.png`

Traffic plots and statistics first collapse crossed route cases to one row per independent `base_draw_id`. This prevents the 14 track reconstructions from being treated as 14 independent traffic observations.

## Design comparison

When traffic is enabled, design reports now add a `Traffic sensitivity of the design conclusion` section.

New artifacts:

- `design_traffic_world_summary.csv`
- `design_traffic_world_performance.csv`
- `design_traffic_stratified_summary.csv`
- `design_traffic_winner_by_severity.csv`
- `report_plots/design_traffic_regret_heatmap.png`
- `report_plots/design_advantage_vs_traffic.png`
- `report_plots/design_finite_ratio_penalty_vs_traffic.png`

The report states whether the overall preferred design remains preferred in low, medium, and high traffic-severity thirds. These thirds are descriptive sensitivity strata formed from the independent traffic-world reference penalty; they are not calibrated probability classes.

## Regenerating an existing result

After installing the source overlay/editable package, no simulation rerun is needed to update an existing report:

```powershell
drivetrain-study report .\path\to\existing\result
```

## Verification

```powershell
python -m pytest .\tests\test_traffic_reporting.py -q
```
