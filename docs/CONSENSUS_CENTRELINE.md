# Robust multi-lap centreline consensus

The track centreline is no longer copied from the fastest or any other
individual lap.

## Processing order

1. Parse FIT/GPX and remove only clearly isolated, bracketed coordinate
   excursions.
2. Detect complete laps and apply non-spatial quality checks.
3. Build an initial pointwise-median centreline from every provisionally
   valid lap enabled for centreline evidence.
4. Map each lap to that shared geometry.
5. Remove only isolated post-map coordinate outliers.
6. For every candidate lap, rebuild the centreline without that lap and
   measure:
   - median, p95, and maximum map error;
   - the fraction of track with sustained disagreement;
   - backward map progression;
   - the p95 and maximum centreline movement caused by including the lap.
7. Exclude only clear sustained or influential geometry outliers, subject
   to an iteration removal cap and a minimum retained-lap count.
8. Rebuild and repeat until stable or the iteration limit is reached.

## Important meanings

`reference_lap` is now only the retained lap with the lowest final p95 map
error. It is a diagnostic representative and does not define the geometry.

`centreline_included` identifies laps used in the final consensus.

`consensus_excluded` and `consensus_exclusion_reason` identify laps removed
by leave-one-out geometry screening. These laps remain visible in
`lap_quality.csv` and are never silently deleted.

The main track map retains the same visual format. Its bold line is now the
robust multi-lap consensus, while the faint trajectories show the cleaned
valid lap cloud.

## Existing-project configuration

```toml
[track.centreline_consensus]
minimum_laps = 3
maximum_iterations = 6
convergence_tolerance_m = 0.10
smoothing_window_nodes = 5
leave_one_out_p95_limit_m = 15.0
sustained_error_threshold_m = 15.0
minimum_sustained_outlier_fraction = 0.05
strong_sustained_outlier_fraction = 0.15
maximum_leave_one_out_shift_m = 5.0
robust_mad_multiplier = 3.5
maximum_outlier_fraction_per_iteration = 0.20
```

Defaults are active even when this block is absent, but declaring the block
makes the analysis policy explicit and preserves it with exported
configuration.
