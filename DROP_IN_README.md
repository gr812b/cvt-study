# Superseded by DROP_IN_README_V2.md

# Drop-in telemetry cleanup replacement

This archive contains only new or replaced repository files.

## Apply

From the repository root, copy the archive contents over the existing checkout
while preserving paths. On Windows Explorer, extract directly into the
`cvt-study` repository and approve file replacement.

Then reinstall the editable package and run tests:

```powershell
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"
pytest -q
```

## Enable it in an existing project

The project templates are updated, but an existing project is not rewritten.
Add this block to `projects/arizona/track/track.toml`:

```toml
[track.telemetry_cleanup]
enabled = true
maximum_excursion_points = 3
minimum_excursion_leg_m = 35.0
impossible_speed_multiplier = 1.5
maximum_bridge_speed_multiplier = 1.0
maximum_bridge_gap_s = 8.0
maximum_auto_removed_fraction = 0.005
maximum_auto_removed_points = 25
isolated_map_error_m = 40.0
maximum_isolated_map_outlier_points = 3
```

Run:

```powershell
drivetrain-study validate .\projects\arizona
drivetrain-study ingest .\projects\arizona
drivetrain-study build-track .\projects\arizona
```

Inspect the newest:

- `results/ingestion/<run>/rejected_telemetry_points.csv`
- `results/ingestion/<run>/telemetry_cleanup_map.png`
- `results/track_build/<run>/track/rejected_map_points.csv`
- `results/track_build/<run>/review/telemetry_cleanup_map.png`

The cleanup does not alter source FIT/GPX files, interpolate coordinates, or
silently repair sustained excursions.
