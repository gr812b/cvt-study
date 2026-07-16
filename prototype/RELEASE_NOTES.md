# Release notes

## Tire uncertainty update

The host tire model now exposes two independent engineering-decision inputs:

- `peak_traction_scale`: scales the maximum longitudinal force available from the track friction coefficient.
- `slip_stiffness_n_per_mps`: controls how quickly force builds with tire-surface slip speed.

`models.tire_model_from_levels` provides low/medium/high choices for each axis. The structural sensitivity runner now sweeps peak traction and slip stiffness separately. Rolling resistance remains independent.

The standard vehicle remains CVT 3.5–0.9, final drive 7.556, 22 in tire diameter (11 in radius), and 0.001 s integration.
