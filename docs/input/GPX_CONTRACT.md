# GPX raw-run contract

The clean pipeline accepts **GPX tracks only**. A declared run must reference a
`.gpx` file inside its project directory, and the file must contain at least one
`<trk>` element with at least one valid `<trkpt>`.

## Supported GPX content

GPX 1.0 and 1.1 namespaces are parsed by local element name. The parser retains:

- every GPX track and track-segment boundary;
- latitude and longitude;
- timestamp, normalized to UTC;
- elevation exactly as reported;
- direct GPX speed and course fields when present;
- fix type, satellite count, HDOP, VDOP, and PDOP;
- leaf values under `<extensions>` as JSON for provenance and later adapters.

Routes (`<rte>`) and waypoints (`<wpt>`) are counted and reported but are not
converted into driven telemetry. A route-only GPX file is rejected rather than
silently treated as a measured run.

The XML parser rejects entity expansion and external-entity payloads.

## Run metadata

`track/runs.toml` owns metadata that GPX commonly does not contain consistently:

```toml
[[runs]]
file = "gpx/vehicle_A_run_01.gpx"
vehicle_id = "vehicle_A"
run_id = "A01"
driver_id = "driver_1"
use_for_centreline = true
use_for_gate_evidence = true
```

One `run_id` identifies one logical recording and must be unique in the project.
The same vehicle or driver may appear in several runs.

## Canonical speed fields

Three speed fields are kept:

- `reported_speed_mps`: a direct GPX or extension speed;
- `derived_speed_mps`: great-circle point spacing divided by positive timestamp
  spacing within one GPX segment;
- `analysis_speed_mps`: reported speed when valid, otherwise derived speed.

The raw reported and derived values remain separate. Track reconstruction may
replace isolated analysis-speed spikes, but the canonical ingestion output is
not globally smoothed.

## Segment and timestamp rules

Point-to-point distance and time are never calculated across a GPX segment
boundary. Segment duration is exported only when all timestamps exist and no
timestamp regression occurs. A backward timestamp therefore cannot produce a
plausible-looking but meaningless duration.

Missing timestamps and syntactically invalid timestamps are counted separately.
Both are unusable for time-derived quantities. Duplicate timestamps, long sampling
gaps, missing elevation, and invalid coordinates are reported with structured
diagnostic codes. A backward timestamp is a fatal ingestion error for track
reconstruction because it can corrupt lap ordering and speed-derived progress;
segment duration is omitted rather than made to look plausible.

## Elevation scope

Elevation is retained through ingestion and track reconstruction, including
between-lap p10, median, and p90 profiles. It is **not** differentiated into
road grade and does not create a gravitational vehicle force in Phases 2 or 3.
Raw GNSS altitude can contain large low-frequency and pointwise errors, so using
it dynamically requires a separate validated method.
