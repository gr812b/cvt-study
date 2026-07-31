# Multi-variable design grids

A design sweep may use either the historical one-dimensional form:

```toml
[design_variable]
path = "drivetrain.final_drive_ratio"
values = [5.0, 5.5, 6.0]
```

or a Cartesian grid:

```toml
[[design_variables]]
path = "drivetrain.cvt.maximum_reduction_ratio"
values = [2.5, 3.0, 3.5, 4.0]

[[design_variables]]
path = "drivetrain.final_drive_ratio"
values = [4.5, 5.0, 5.5, 6.0]

[design_grid]
maximum_candidates = 100
```

Every combination is evaluated on the same paired scenario. In uncertainty-informed
mode, all design-controlled paths are removed from the saved source draw before the
candidate overrides are applied.

The design report keeps the existing bar plots and adds heatmaps when exactly two
variables are configured. It also writes `design_grid_summary.csv` beside the HTML.

The currently approved multi-variable paths are:

- `drivetrain.final_drive_ratio`
- `drivetrain.cvt.maximum_reduction_ratio`
- `drivetrain.cvt.minimum_reduction_ratio`

These paths may share one infinite-CVT reference because the bounded transmission
limits are intentionally excluded from the shared reference case.
