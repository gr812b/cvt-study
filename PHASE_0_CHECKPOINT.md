# Phase 0 checkpoint

> Historical checkpoint. Phase 1 has since implemented the approved loaders,
> profile resolution, validation, exports, and CLI.

## Implemented

- GPX-only raw telemetry contract.
- Elevation retained but not yet used for grade force.
- Self-contained project/track/vehicle/study/results layout.
- Reusable profile locations and inheritance intent.
- Uncertainty colocated with physical parameters.
- Explicit fixed-value reason requirement.
- Normal, truncated-normal, uniform, triangular, empirical, and discrete contracts.
- User-friendly normal uncertainty from a confidence half-width.
- Provenance and future correlation-group fields.
- Working contract tests.
- Existing functional repository preserved under `prototype/` for staged migration.

## Deliberately not implemented yet

- TOML project loading and inheritance resolution.
- Unit conversion.
- GPX parsing.
- Event-schema validation.
- Numerical sampling.
- Obstacle coefficient defaults before obstacle equations are frozen.
- Clean CLI.

## Approval questions before Phase 1

1. Is the project directory arrangement natural for how tracks and vehicles will be reused?
2. Is requiring `distribution = "fixed"` plus a reason the desired zero-uncertainty behavior?
3. Should project event definitions remain one `events.toml`, or eventually support one file per event group?
4. Are the profile precedence rules acceptable?

## Verification completed

- Clean Phase 0 contract tests: 12 passed.
- Preserved prototype integration tests: 5 passed.
- Preserved GPS-analysis tests: 11 passed.
