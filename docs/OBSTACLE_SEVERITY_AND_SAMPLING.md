# Obstacle severity and uncertainty sampling

## Course-wide obstacle energy severity

`track.obstacle_energy_severity` is a dimensionless structural input applied
inside the runtime simulation. For every active obstacle model:

`F_obstacle,realized(s) = severity × F_obstacle,base(s)`

Because obstacle work is the line integral of resistance force, the same scale
multiplies dissipative obstacle energy. The wrapper deliberately leaves these
unchanged:

- obstacle inclusion and model choice;
- vertical profile geometry and elevation;
- conservative grade contribution;
- normal-load response;
- traction multiplier;
- per-feature energy/impact/roughness parameters.

The default low/nominal/high values are 0.6, 1.0, and 1.4. They are screening
cases for a poorly calibrated global burden, not claimed real-world
probabilities. Structural sensitivity uses exactly those three levels. Full
uncertainty samples the triangular family while preserving each feature’s own
uncertainty declarations.

## Latin-hypercube design

All non-fixed scalar inputs use Latin-hypercube marginals. With `N` requested
replicates, each input receives exactly one draw in every interval
`[i/N, (i+1)/N)`. This improves marginal coverage over independent random draws
without changing the declared distributions.

Declared correlation matrices are imposed by Iman-Conover-style rank reordering:

1. generate every scalar marginal as a Latin hypercube;
2. draw correlated Gaussian scores for each declared family;
3. reorder each marginal according to those score ranks.

The result preserves both one-per-stratum marginal coverage and the requested
rank dependence. Inputs outside a declared correlation family remain separately
stratified.

## Measured traversal and speed error

Gate targets continue to be sampled as coherent measured traversals identified by
`run_id`, `lap_id`, `vehicle_id`, and `driver_id`. Every eligible pass counts once.
The selected traversal receives one source-specific measurement-error draw shared
across its gates:

`v_gate,realized = max(0, v_gate,observed + sigma_source × z_traversal)`

A shared `z_traversal` avoids physically implausible independent jitter at every
gate. Reconstructed speed has a larger sigma than FIT, so its lower certainty is
propagated into answer uncertainty rather than merely written as a note.

## Reporting

The full-uncertainty report separates:

- dissipative physical losses;
- finite-ratio opportunity loss;
- finite-ratio lap-time penalty.

Each is shown with p10–p90 ranges, and the exact nominal case is taken from
`nominal_reference.json`. Opportunity loss is not stacked with physical losses
because it is a counterfactual limitation, not dissipated energy.
