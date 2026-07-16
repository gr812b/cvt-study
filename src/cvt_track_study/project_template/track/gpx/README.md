Place raw `.gpx` recordings here and declare each file in `../runs.toml`.

Only GPX `<trk>/<trkseg>/<trkpt>` telemetry is ingested. Routes and waypoints are
reported but are not converted to driven runs. Elevation is retained for review,
while grade force remains disabled until altitude processing is validated.
