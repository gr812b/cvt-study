# Route-variant detection drop-in

Base repository state reviewed: `gr812b/cvt-study` main commit
`97cf8e84b44b1411538502552aeb24f193158bf0`.

## Purpose

Detect and separate repeatedly driven, genuinely different closed-course routes
before centreline consensus. Alternate routes are not averaged into one track and
are not mislabeled as telemetry errors.

## Changed files

- `src/cvt_track_study/config/__init__.py`
- `src/cvt_track_study/config/project_v12.py`
- `src/cvt_track_study/track/export.py`
- `src/cvt_track_study/track/model.py`
- `src/cvt_track_study/track/reconstruction.py`
- `src/cvt_track_study/track/route_variants.py`
- `src/cvt_track_study/track/service.py`
- `src/cvt_track_study/project_template/track/track.toml`
- `tests/test_route_variants.py`
- `docs/ROUTE_VARIANTS.md`
- `docs/MARYLAND_REVIEW_QUESTIONS.md`

## Apply

Extract this ZIP over the repository root. Then reinstall editable code:

```powershell
py -m pip install -e ".[dev]"
```

Run focused checks:

```powershell
py -m pytest -q tests/test_route_variants.py
drivetrain-study validate .\projects\maryland
```

## Maryland first pass

Add this to `projects/maryland/track/track.toml`:

```toml
[track.route_variants]
enabled = true
minimum_supported_laps = 2
sample_spacing_m = 5.0
same_variant_p95_distance_m = 10.0
divergence_distance_m = 15.0
maximum_divergent_fraction = 0.03
maximum_length_relative_difference = 0.08
maximum_within_variant_length_deviation_fraction = 0.15
selection = "require_explicit"
reference_run_id = ""
selected_variant_id = ""
```

Run `build-track`. When multiple supported routes are found, the build stops and
prints their IDs, support counts and median lengths. Review the route-variant
audit rather than changing thresholds to obtain a preferred result.

After confirming a reference recording for the intended nominal route, use:

```toml
selection = "reference_run"
reference_run_id = "confirmed_run_id"
```

## Outputs

- `track/lap_quality.csv`
- `track/route_variant_summary.csv`
- `track/route_variant_pairwise.csv`
- route-variant policy and full inventory in `track_build_manifest.json`

## Validation performed here

- Static compilation of every included Python module: passed.
- TOML parsing of the updated project template: passed.
- Four focused synthetic tests: passed.
  - repeated long and short paths become two supported variants;
  - reference-run selection selects the intended route;
  - ambiguous multiple routes fail under `require_explicit`;
  - telemetry-invalid laps do not create route variants.

A full repository test suite and a Maryland-data end-to-end run were not executed
in this runtime because the repository and raw telemetry were not available in
the local container. The implementation therefore includes conservative defaults
and machine-readable pairwise evidence specifically so the first real build can
be reviewed before use.
