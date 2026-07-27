# ÉTS spatial lap-segmentation hotfix

This replaces the lap-time reconstruction boundary detector and alignment flow.

## What was wrong

The previous detector required a retained GPX **point** to fall inside the
start/finish radius. Spatially simplified GPX traces can cross the start line
between two retained points. Those missed crossings merged multiple physical
laps into one detected chunk.

Excluded pit/service CSV rows were also removed before alignment, which shifted
later lap assignments.

## What changes

1. Start/finish visits are detected from the shortest distance between each
   consecutive GPX segment and the gate anchor.
2. Physical laps are created from those spatial crossings before CSV matching.
3. Every CSV row remains in chronological alignment, including excluded
   pit/service laps.
4. Excluded rows consume their matching spatial lap but do not create
   pointwise speed evidence.
5. One GPX lap can match at most one CSV row and one CSV row can match at most
   one GPX lap.

## Arizona configuration

In `projects/arizona/track/runs.toml`, use:

```toml
[runs.lap_time_reconstruction]
alignment_mode = "strict_index"
gate_radius_m = 8.0
allow_unmatched_laps = true
```

Keep the existing CSV duration filters. For the supplied ÉTS data, the expected
alignment is:

- 45 start/finish visits
- 44 complete spatial laps
- 43 CSV rows aligned in order
- CSV rows 16, 24, 30, and 32 retained in alignment but excluded from speed
- 39 race laps reconstructed
- final GPX lap left unmatched and visible in the audit

## Apply

Extract this ZIP over the repository root, replacing the included Python file.

Then run:

```powershell
drivetrain-study validate .\projects\arizona
drivetrain-study ingest .\projects\arizona --run ets_78_maryland_reconstructed
```

Review `lap_time_reconstruction.csv` before using the reconstructed speed as
gate evidence.
