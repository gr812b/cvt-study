# Reconstruction-to-simulation boundary

`track_bundle.json` is the only supported input from track reconstruction to the
simulation package. The simulator imports no GPX parser, lap detector, map
matcher, event spreadsheet, or pandas reconstruction frame.

Schema `1.2.x` carries explicit uncertainty-aware obstacle declarations and semantic uncertainty roles to every
physical feature. A bundle is simulation-ready only when:

- geometry and interval references are valid;
- accepted speed gates have empirical distributions and confidence evidence;
- every physical feature explicitly declares a supported model, including
  `none`;
- `capabilities.obstacle_models_ready` is true;
- `capabilities.uncertainty_roles_ready` is true.

Vehicle, engine, tire, driver, CVT, and study settings remain outside the bundle.
This permits one reviewed track bundle to be used with multiple vehicles and
designs. The runtime combines the bundle with one fully resolved scenario.

Raw GPX/FIT elevation is retained and screened in the bundle, but `grade_force_ready` remains
false. Explicit smooth obstacle profiles may still contribute local conservative
height/grade because their geometry is separately declared.
