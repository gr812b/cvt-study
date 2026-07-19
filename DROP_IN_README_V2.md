# Telemetry cleanup + consensus centreline drop-in

This is a self-contained replacement bundle. It includes the previous
telemetry-cleanup files plus the new iterative multi-lap centreline logic.

Extract the archive directly into the repository root and approve replacing
existing files.

```powershell
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"
pytest -q
```

Add the `[track.centreline_consensus]` block from
`docs/CONSENSUS_CENTRELINE.md` to an existing project's `track.toml`.
Defaults work without the block, but declaring it makes the policy explicit.

Then rerun:

```powershell
drivetrain-study validate .\projects\arizona
drivetrain-study ingest .\projects\arizona
drivetrain-study build-track .\projects\arizona
```

The main map retains the same layout. Its bold line is now a robust
consensus across retained cleaned laps. The fastest lap is no longer used
as geometry. `lap_quality.csv` contains the leave-one-out metrics and
explicit consensus exclusion fields.
