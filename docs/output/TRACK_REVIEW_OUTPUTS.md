# Phase 3 output reference

A `build-track` run writes one atomic result directory.

```text
results/track_build/<timestamp>/
├── ingestion/
│   ├── canonical_points.csv
│   ├── segments.csv
│   └── run_summaries.json
├── track/
│   ├── centreline.csv
│   ├── lap_quality.csv
│   ├── map_matched_points.csv
│   ├── track_profile.csv
│   ├── event_projection.csv
│   ├── response_features.csv
│   ├── event_passes.csv
│   ├── gate_evidence.csv
│   └── gate_review.csv
├── review/
│   ├── track_map.png
│   ├── elevation_profile.png
│   ├── track_review.html
│   └── REVIEW_SUMMARY.md
├── diagnostics.json
└── track_build_manifest.json
```

The map labels every physical anchor and marks physical start, physical end,
entry window, median observed minimum-speed location, and median recovery point.
Accepted/review/rejected marker shapes are based on response-group evidence.

`gate_review.csv` is the first table a user should open. It is sorted by required
attention and includes a suggested action. Its map-quality fields distinguish:

- anchor projection error and declared anchor uncertainty;
- whether physical start/end came from explicit coordinates or configured extents;
- start/end source and provenance;
- effective physical-start and physical-end errors;
- assumptions and overlap/branch flags retained after acceptance.

`event_projection.csv` is the source for checking every physical feature and its
geometry. `response_features.csv` shows the analysis intervals formed from one or
more physical features. `event_passes.csv` is the auditable lap-by-lap evidence
behind every aggregate score.

`lap_quality.csv` reports both initial and post-map-match validity. `quality_flags`
explains exclusions such as incomplete timing, implausible distance, excessive
stationary time, insufficient speed coverage, sampling gaps, timestamp regressions,
high map error, or backward
progress jumps. Excluded laps remain visible rather than disappearing from the
review record.

`REVIEW_SUMMARY.md` lists required fixes and recommended reviews first. It also
lists accepted gates together with retained assumptions, and identifies excluded
laps. Acceptance means the evidence meets configured review rules; it does not
mean the geometry or empirical speed distribution is exact.


## Phase 4 bundle

The same build directory now also contains `track_bundle.json` and
`track_bundle.sha256`. Review tables remain the human-facing evidence; the bundle is
the machine-facing boundary. See `TRACK_BUNDLE_CONTRACT.md`.
