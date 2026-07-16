# GPX and FIT raw-run contract

The pipeline accepts `.gpx` and `.fit` activity recordings inside the project.
GPX files must contain at least one `<trk>` with a valid `<trkpt>`; FIT files must
contain positioned record messages. Both formats produce the same canonical point
table, so reconstruction does not need format-specific logic.

## Supported FIT content

The official Garmin FIT SDK decodes record messages with CRC checking. The adapter
prefers enhanced speed and altitude fields, then their standard equivalents, and
retains device cumulative distance, timestamps, positions, heading, GPS accuracy,
and the decoded record as provenance JSON. FIT semicircle coordinates are converted
to degrees.

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

One `run_id` identifies one physical recording and must be unique in the project.
If FIT and GPX are two exports of the same recording, declare the FIT version once
and retain the GPX as a support copy, not as independent lap evidence. The same
vehicle or driver may appear in several genuinely separate runs.

## Canonical speed fields

The canonical speed fields preserve the source hierarchy:

- `device_speed_mps`: native FIT enhanced/standard device speed;
- `reported_speed_mps`: a direct GPX or extension speed;
- `derived_speed_mps`: position/distance spacing divided by positive timestamp spacing;
- `analysis_speed_mps`: device speed, else reported speed, else derived speed;
- `analysis_speed_source` and `speed_certainty`: explicit provenance and certainty tier.

Native, reported, and derived values remain separate. Device distance is also
preserved and used for step distance when valid. Track reconstruction may
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
road grade and does not automatically create a gravitational vehicle force.
Every bundle includes a low-cost coverage/repeatability/materiality screen. Only a
material, repeatable profile earns a paired with/without-grade sensitivity; grade
force remains disabled until that sensitivity changes the design conclusion.
