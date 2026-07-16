# Phase 2 checkpoint — GPX ingestion and canonical telemetry

## Status

Phase 2 is complete. The clean pipeline accepts GPX tracks as its only raw
telemetry format. CSV telemetry compatibility was intentionally not retained.

## User-facing entry point

```powershell
cvt-study ingest .\projects\my_track
```

A run is declared in `track/runs.toml`; its GPX file remains inside
`track/gpx/`. Run, vehicle, and driver identity are project metadata rather than
recorder-dependent GPX conventions.

## Implemented contract

The parser supports GPX 1.0 and 1.1 track content and preserves:

- all `<trk>` and `<trkseg>` boundaries;
- latitude, longitude, timestamps, and elevation;
- direct speed and course fields when present;
- fix type, satellite count, HDOP, VDOP, and PDOP;
- leaf extension values as provenance-preserving JSON;
- source file path and SHA-256 hash.

Routes and waypoints are counted and reported but are not treated as driven
telemetry. A route-only file is rejected. XML entity expansion and external
entities are rejected by the secure parser.

## Canonical speed contract

Each point retains three distinct speed fields:

1. recorder-reported speed;
2. speed derived from within-segment point spacing and positive timestamp
   spacing;
3. analysis speed, which prefers a valid reported value and otherwise uses the
   derived value.

No distance, time step, or derived speed is calculated across a GPX segment
boundary.

## Diagnostics and corrections completed before Phase 3

The Phase 2 review identified and corrected:

- segment durations that could appear valid after timestamp regression;
- conflation of missing and syntactically invalid timestamps;
- optional numeric columns becoming object-typed when entirely absent;
- documentation that implied routes or waypoints were retained as telemetry;
- weak export-boundary type annotations;
- potential packaged-template drift.

Backward timestamps are now fatal evidence errors for track reconstruction.
Missing and invalid timestamps remain separately counted, while segment duration
is omitted whenever timing is incomplete or non-monotonic.

## Outputs

```text
results/ingestion/<timestamp>/
├── canonical_points.csv
├── segments.csv
├── run_summaries.csv
├── diagnostics.json
├── ingestion_manifest.json
├── configuration/
└── runs/<run_id>/
```

Every output is written to a temporary directory and atomically published only
after the complete artifact tree succeeds.

## Elevation boundary

GPX elevation survives ingestion unchanged and is available to Phase 3. It is
not interpreted as road grade and does not generate vehicle force. This avoids
turning noisy GNSS altitude into false precision before a validated elevation
processing method exists.
