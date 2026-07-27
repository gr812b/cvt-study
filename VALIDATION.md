# Focused validation

## Supplied ÉTS data

See `ETS_VALIDATION.json`.

- 49 spatial loops detected from `ETS.gpx`.
- 43 rows parsed from `78_Baja_ÉTS.csv`.
- 4 long pit/stoppage rows excluded and retained in the audit.
- 39 race laps aligned and reconstructed.
- 10 extra spatial loops retained as unmatched.
- 9,711 reconstructed points.
- augmented GPX parsed as valid XML with 39 track segments.

## Mechanism and sampling tests

See `MECHANISM_VALIDATION.json`.

- obstacle energy scale changed resistance from 100 N to 60 N at severity 0.6;
  geometry, grade, normal-load scale, and traction were unchanged;
- every Latin-hypercube stratum was occupied exactly once for two test inputs;
- requested 0.8 family correlation produced Spearman correlation 0.773 in the
  deterministic test;
- FIT gate-error standard deviation was 0.0955 m/s;
- reconstructed gate-error standard deviation was 0.9978 m/s;
- mechanism plot, summary CSV, exact nominal marker, and HTML injection passed.

## Static validation

`python -m compileall` passed for all included source and test files.

The full repository test suite was not available because the runtime did not have
a complete checkout and the GitHub integration denied Git tree/write operations.
