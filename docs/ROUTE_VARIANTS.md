# Genuine route variants

The route-variant stage runs after complete-lap detection and telemetry-quality
screening, but before centreline consensus.

It exists for cases where cars repeatedly took genuinely different paths around
the same closed course. Those paths must not be averaged into one centreline and
must not be mislabeled as GPS outliers.

## Detection contract

Each data-quality-valid lap is spatially resampled. Every pair is compared using:

- symmetric p95 nearest-path distance;
- fraction of each path farther than the declared divergence distance;
- relative path-length difference.

Complete-link clustering is used. A cluster can merge only when every lap in one
cluster remains compatible with every lap in the other, which prevents chaining
several progressively different lines into one route.

A cluster is a supported route only after `minimum_supported_laps` repeated laps.
Singletons and weak clusters remain visible but do not become nominal routes.

## Selection is separate from detection

When only one supported route exists, it is selected automatically. When several
supported routes exist, the default `require_explicit` policy stops the build.
Available explicit policies are:

- `reference_run`: select the route containing one declared run;
- `variant_id`: select an ID from a prior, unchanged audit;
- `largest_supported`: explicit policy to choose the most repeated route;
- `longest_supported`: explicit policy to choose the longest repeated route.

`reference_run` is preferred when an independent recording is known to represent
the target course configuration.

## Outputs

- `track/lap_quality.csv` contains route membership, support and exclusion reason.
- `track/route_variant_summary.csv` contains one row per detected route cluster.
- `track/route_variant_pairwise.csv` contains the pairwise geometric evidence.
- `track_build_manifest.json` records thresholds, selection policy and the full
  route inventory.

Supported alternate routes remain valid evidence. They are excluded only from the
selected route's centreline, event projection and speed-gate evidence.

## Maryland configuration

Start with the safe audit-only policy:

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
```

Run `build-track` once and inspect the variant summary. After confirming which
recording represents the intended nominal route, prefer:

```toml
selection = "reference_run"
reference_run_id = "the_confirmed_reference_run_id"
```

Threshold changes are analysis-policy changes and belong in track robustness;
they should not be tuned merely to force the desired grouping.
