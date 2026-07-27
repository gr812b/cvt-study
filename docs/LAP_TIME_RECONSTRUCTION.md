# Optional 1 Hz speed reconstruction for untimed GPX

## Purpose and evidence boundary

Some third-party GPX exports preserve an ordered spatial trace and elevation but
remove timestamps and speed. When an independent lap-time export exists, the
framework can reconstruct a supplemental, approximately 1 Hz timed trace.

This is not native speed telemetry. The hierarchy is:

1. native FIT speed (`native_high`);
2. explicitly reported GPX speed (`reported_medium`);
3. timestamp-derived GPX position speed (`derived_lower`);
4. lap-time-reconstructed speed (`reconstructed_*`);
5. untimed geometry/elevation only.

Every reconstructed point and gate sample carries its provenance and certainty.
The downstream uncertainty sampler gives reconstructed speed a wider measurement
error than FIT rather than pretending both are equally precise.

## Reconstruction method

For each configured run:

1. parse and conservatively clean the original GPX;
2. refuse reconstruction when useful native timestamps already exist;
3. detect repeated visits to the configured lap-gate coordinate using point order;
4. parse the external lap CSV, including exports with metadata rows before the
   actual header;
5. retain all CSV rows in the audit, but mark rows outside configured race-lap
   duration bounds as excluded;
6. align included CSV rows and detected spatial loops using dynamic programming;
7. minimize mismatch between inferred point period and the expected export
   cadence while allowing explicit unmatched GPX loops or CSV rows;
8. assign uniform within-lap timestamps:

   `dt = declared_lap_time / (point_count - 1)`

9. derive local speed from each GPX step distance divided by reconstructed `dt`;
10. export canonical points, a lap audit, and an augmented GPX.

The dynamic alignment does not hide uncertainty. Unmatched loops and excluded
rows remain in `lap_time_reconstruction.csv`.

## ÉTS race-export format

The supplied file begins with metadata:

```csv
Car Number,78
Team Name,Baja ÉTS
timestamp,lap_time_s
...
```

Header discovery recognizes `lap_time_s`, and the skipped metadata is retained as
`csv_metadata_json`. The timestamp text serves as a source-row identity; the
actual local timing comes from `lap_time_s`.

## Maryland configuration

```toml
[[runs]]
run_id = "ets_78_maryland_reconstructed"
vehicle_id = "ets_78"
driver_id = "ets_driver_unknown"
file = "telemetry/ETS.gpx"
use_for_centreline = true
use_for_gate_evidence = true

[runs.lap_time_reconstruction]
enabled = true
lap_times_file = "lap_times/78_Baja_ETS.csv"
lap_time_column = "lap_time_s"
expected_point_period_s = 1.0
alignment_mode = "cadence_dynamic_programming"
minimum_csv_lap_time_s = 180.0
maximum_csv_lap_time_s = 420.0
maximum_point_period_error_fraction = 0.35
maximum_alignment_error_fraction = 0.65
skip_detected_lap_penalty = 1.0
skip_csv_lap_penalty = 1.5
minimum_matched_laps = 20
allow_unmatched_laps = true
gate_radius_m = 8.0
export_augmented_gpx = true
```

`allow_unmatched_laps=true` is appropriate here only because every mismatch is
explicitly exported. It must not be used to suppress review.

## Certainty mapping

The inferred point-period error controls the reconstructed label:

- within the configured band: `reconstructed_low_medium`, sigma `0.75 m/s`;
- within twice the band: `reconstructed_low`, sigma `1.00 m/s`;
- farther outside: `reconstructed_very_low`, sigma `1.50 m/s`.

By comparison, FIT native speed uses sigma `0.15 m/s`. These are transparent
engineering measurement-error contracts, not posterior calibration claims.

## Outputs

```text
results/ingestion/<timestamp>/
├── lap_time_reconstruction.csv
└── runs/ets_78_maryland_reconstructed/
    ├── canonical_points.csv
    ├── lap_time_reconstruction.csv
    ├── reconstructed_timed.gpx
    ├── segments.csv
    ├── summary.json
    └── diagnostics.json
```

The lap audit includes detected and CSV identities, matching state, point count,
declared duration, inferred point period, cadence error, path distance, average
lap speed, certainty, and quality flags.

Do not declare `reconstructed_timed.gpx` as a second physical run. It is a review
artifact derived from the original untimed GPX plus the CSV.

## Supplied-data result

Using the Maryland start gate at approximately
`38.40368056, -76.84270833` and an 8 m radius:

- 49 spatial loops were detected;
- 39 race-duration CSV rows were matched;
- 4 long pit/stoppage rows were excluded but audited;
- 10 spatial loops remained unmatched;
- the inferred cadence was `0.6745–1.3167 s`, median `1.1218 s`.

This supports using the reconstruction as supplemental speed evidence, not as a
replacement for the McMaster FIT trace.
