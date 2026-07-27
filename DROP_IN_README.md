# Obstacle-severity, Latin-hypercube, and ÉTS speed-intake drop-in

Base repository observed: `gr812b/cvt-study` at commit
`2dcc0f1de685ad3411d76808c5b389e114fa091c`.

Extract this ZIP over the repository root and approve replacement. It implements
the approved uncertainty and Maryland/ÉTS intake work. **It does not implement
stop-on-convergence.** Every configured replicate still runs.

## Implemented

### 1. Course-wide obstacle energy severity

A new structural input, `track.obstacle_energy_severity`, coherently multiplies
only the dissipative resistance/work returned by every obstacle model.

- low: `0.6`
- nominal: `1.0`
- high: `1.4`

The scale is applied inside the runtime track before integration. It does not
scale obstacle geometry, elevation, grade, normal-load response, traction,
model inclusion, model choice, or any per-feature parameter. Those existing
uncertainties remain separate.

The default is a triangular low/nominal/high screening family. A baseline may
override the contract using the snippet in
`examples/obstacle_energy_severity.toml`.

### 2. Latin-hypercube uncertainty sampling

Scalar structural inputs now use Latin-hypercube marginals by default. Declared
correlation groups are imposed by rank reordering, so each marginal keeps one
sample in every stratum while coherent families retain their requested rank
structure. Paired gate-lap sampling remains intact.

### 3. Speed-source uncertainty

Every eligible pass still counts once. Speed provenance changes the measurement
error attached to that pass rather than silently deleting or duplicating it.

- FIT native speed: `native_high`, nominal sigma `0.15 m/s`
- reported GPX speed: `reported_medium`, sigma `0.30 m/s`
- timestamp-derived position speed: `derived_lower`, sigma `0.50 m/s`
- ÉTS lap-time reconstruction: sigma `0.75–1.50 m/s`, depending on cadence fit

When one measured traversal is replayed, one coherent normal error draw is used
across all its gates. This preserves the shape of the traversal rather than
injecting unrelated noise at every obstacle.

### 4. Full-uncertainty mechanism report

The canonical HTML report gains a mechanism-first section and CSV:

```text
full_uncertainty_mechanism_summary.csv
report_plots/physical_losses_opportunity_and_time.png
```

The plot shows:

1. physical loss medians with p10–p90 error bars;
2. finite-ratio opportunity loss and its p90-minus-p10 span;
3. finite-ratio lap-time penalty and its p90-minus-p10 span;
4. exact nominal diamonds from `nominal_reference.json`.

The overlay is regenerable from saved artifacts and does not rerun simulations.

### 5. ÉTS untimed GPX + race-export CSV intake

The parser now understands the supplied ÉTS export with metadata rows before the
actual `timestamp,lap_time_s` header. It detects spatial laps in the untimed GPX,
filters declared pit/stoppage durations without deleting them from the audit, and
aligns remaining CSV laps to spatial loops by expected 1 Hz cadence.

Use the ready configuration in `examples/maryland_untimed_run.toml` and copy:

```text
examples/maryland/track/lap_times/78_Baja_ETS.csv
```

into the Maryland project’s `track/lap_times/` directory. The GPX itself remains
the original `ETS.gpx` file in `track/telemetry/`.

## ÉTS validation result

The supplied files were exercised directly with an 8 m start-gate radius:

- 49 complete spatial loops detected;
- 43 CSV rows parsed;
- 4 long pit/stoppage rows excluded but retained in the audit;
- 39 race laps matched and reconstructed;
- 10 additional spatial loops retained as unmatched evidence;
- 9,711 reconstructed points;
- median inferred point period: `1.1218 s`;
- inferred range: `0.6745–1.3167 s`;
- all 39 matched laps received `reconstructed_low_medium`, below FIT certainty.

See `ETS_VALIDATION.json` and `docs/LAP_TIME_RECONSTRUCTION.md`.

## Apply and run

```powershell
Expand-Archive .\cvt-study-obstacle-severity-ets-speed-dropin.zip `
  -DestinationPath .\cvt-study -Force

cd .\cvt-study
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"
pytest -q tests/test_lap_time_reconstruction.py `
  tests/test_lap_time_run_config.py `
  tests/test_obstacle_severity.py `
  tests/test_latin_hypercube_and_gate_certainty.py `
  tests/test_uncertainty_mechanism_report.py
```

Then add the ÉTS `[[runs]]` block from `examples/maryland_untimed_run.toml`,
copy the CSV, and run:

```powershell
drivetrain-study validate .\projects\maryland
drivetrain-study ingest .\projects\maryland --run ets_78_maryland_reconstructed
drivetrain-study build-track .\projects\maryland
```

Review `lap_time_reconstruction.csv` before treating the reconstructed run as
gate evidence. Long pit/stoppage rows and unmatched spatial loops are expected to
remain visible.

## Validation performed here

- static compilation of every included Python file;
- direct reconstruction smoke test on the supplied ÉTS GPX and CSV;
- augmented GPX XML parse and 39-segment check;
- obstacle energy scale isolation test;
- Latin-hypercube one-per-stratum test;
- correlated-family rank-correlation test;
- source-specific gate-error test showing reconstructed sigma much wider than FIT;
- full-uncertainty mechanism plot/HTML regeneration smoke test.

The complete repository test suite could not be run because the GitHub
integration exposed repository reads but denied Git tree/write access, and a full
checkout was not available in the runtime. The drop-in therefore includes focused
tests and the exact validation outputs used here.
