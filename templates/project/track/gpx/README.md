Place raw `.gpx` or `.fit` recordings here and declare each file in `../runs.toml`.

FIT preserves native device speed, cumulative distance, enhanced altitude, and GPS
accuracy when present. GPX `<trk>/<trkseg>/<trkpt>` telemetry remains supported;
routes and waypoints are reported but not converted to driven runs. If FIT and GPX
are exports of the same recording, declare the FIT file once under that run ID so
the session is not counted as independent evidence twice.
