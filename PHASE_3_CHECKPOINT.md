# Phase 3 checkpoint — track reconstruction, evidence, and review

## Status

Phase 3 is complete. It converts validated GPX runs and uncertainty-aware event
geometry into a common closed-course coordinate, lap-by-lap event evidence,
decomposed speed-gate confidence, and a formal review package. It does not yet
produce the simulation-ready bundle planned for Phase 4.

## User-facing entry points

```powershell
cvt-study build-track .\projects\my_track
cvt-study review .\projects\my_track
```

Both commands use the project directory as the single lookup root. Results go to
that project's `results/track_build/` directory unless an explicit output path is
provided.

## Reconstruction flow

1. Canonical GPX points are mapped into a local metre-based tangent frame.
2. Isolated speed spikes are replaced without globally smoothing the trace.
3. Visits to the declared lap-gate event form complete laps.
4. Laps are checked for duration, distance consistency, stationary fraction,
   speed coverage, sampling gaps, and timestamp regressions.
5. The fastest valid centreline-enabled lap creates the reference centreline.
6. Every lap is progress-aware map matched to a single along-track coordinate
   `s`; high-error and backward-jump laps are retained in output but excluded
   from evidence.
7. Valid evidence laps form p10/median/p90 speed and elevation profiles.
8. Physical features are projected in declared course order.
9. Response groups preserve separate physical features while combining GPS
   responses that cannot be separated defensibly.
10. Per-lap approach, entry, minimum, exit, braking, elevation, and recovery
    metrics are extracted.
11. Speed-gate evidence is scored and converted into actionable review status.

## Uncertainty-first map contract

Anchor coordinates, optional explicit start/end coordinates, and fallback
before/after-anchor extents each carry their own source and uncertainty.

For an extent-derived physical start, the review error is

\[
e_{\mathrm{start}} =
\sqrt{e_{\mathrm{anchor,projection}}^2 +
      u_{\mathrm{anchor}}^2 + u_{\mathrm{before}}^2}.
\]

For an explicit start coordinate,

\[
e_{\mathrm{start}} =
\sqrt{e_{\mathrm{start,projection}}^2 + u_{\mathrm{start}}^2}.
\]

The same contract applies to the physical end. Gate coordinate quality uses the
physical-start error because the entry-speed window is defined relative to that
boundary. Anchor projection remains a mandatory geometry check even for events
that are not nominated as speed gates.

Explicit start and end coordinates require no redundant extent table. A fallback
extent is required only for a missing side.

## Gate evidence

The score combines separately exported components for:

- valid pass count;
- entry-speed repeatability;
- braking evidence;
- pace independence;
- physical-start coordinate quality;
- cross-vehicle agreement.

The weighted score is an evidence summary, not a calibrated probability. With
one vehicle, cross-vehicle agreement is explicitly neutral rather than claimed
as measured. A lap gate is not automatically promoted to a speed gate.

Review states are:

- `must_fix`;
- `recommended_review`;
- `accepted`;
- `rejected`;
- `not_a_candidate`.

A geometry failure takes precedence over `not_a_candidate`, so a bad lap-gate or
map reference cannot disappear merely because it is not a simulation speed gate.

## Review outputs

```text
results/track_build/<timestamp>/
├── ingestion/
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
├── configuration/
├── diagnostics.json
└── track_build_manifest.json
```

`gate_review.csv` is the first review table. It contains empirical gate-speed
statistics, each evidence component, start/end geometry provenance, separate
projection and declared uncertainty terms, effective errors, reasons, and a
suggested next action.

## Synthetic reference result

The packaged reference project resolves with zero errors and zero warnings. Its
expected Phase 3 result is:

- 1,081 valid GPX points in one segment;
- nine complete, valid laps;
- reconstructed length about 599.9 m;
- three accepted candidate gates;
- start/finish retained only as the lap separator;
- elevation profile exported with grade force disabled.

## Phase boundary

Phase 3 prepares empirical and declared uncertainty evidence but does not sample
it jointly. It also does not assign obstacle force or energy equations. Phase 4
will freeze the versioned track-bundle boundary before simulation migration.
