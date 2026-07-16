# Uncertainty propagation and study contracts

Phase 6 turns the uncertainty declarations already attached to physical inputs
into reproducible simulation scenarios. It does not infer uncertainty from the
spread of final outputs. The direction of information is:

```text
input contract + empirical gate evidence
→ sampled physical scenario
→ bounded design and matched infinite reference
→ paired result row
→ output distribution and decision statistics
```

## One scenario, every compared design

A scenario is one complete plausible realization of the declared physical
inputs. The scenario contains sampled SI values, categorical model choices,
empirical speed-gate targets, and the random seed that produced them.

Every design candidate in a sweep receives the same scenario. This common-random-
number pairing prevents a design from appearing better simply because it was
simulated on an easier sampled track. The matched infinite reference receives the
same scenario as its bounded case.

## Supported input distributions

Numeric quantities support:

- `fixed`
- `normal`
- `truncated_normal`
- `uniform`
- `triangular`
- `empirical`

Categorical model choices support `fixed` and `discrete`. Empirical numeric inputs
resample actual observations rather than inventing values between observed
points. Normal and truncated-normal transforms use SciPy. All sampled numeric
values are converted to SI before entering the simulator.

A bounded uncertainty declaration may not include impossible physical support.
For example, mass uncertainty may not cross zero and efficiency uncertainty must
remain in `(0, 1]`. An unbounded normal that has material probability outside a
physical domain produces a validation error requiring a truncated normal.
Obstacle coefficients that must remain non-negative require explicit bounded
support.

## Gate-speed sampling

Accepted gates carry empirical samples labelled by run, lap, vehicle, and driver.
The default `paired_lap` policy selects one identity shared by all active gates and
uses that lap's speed at every gate. This preserves observed within-lap pace and
driver behaviour.

When no identity covers every gate, the sampler chooses an identity with maximum
coverage, retains it wherever possible, independently samples only the missing
gates, and lists those gate IDs in the manifest. The alternative `independent`
policy is available but must be selected consciously because it destroys observed
within-lap pairing.

## Correlated inputs

Correlation groups use a Gaussian copula. The study declares member paths and a
correlation matrix. Validation requires that the matrix is finite, symmetric, has
unit diagonal, is positive semidefinite, and does not overlap another group.
Marginal distributions remain exactly those declared by each input. The matrix
controls correlation in the latent Gaussian variables. It equals ordinary Pearson
correlation only for normal marginals; for bounded, empirical, or discrete
marginals it should be interpreted as a dependence parameter rather than a promise
about the final sampled Pearson coefficient.

Correlation is not a general constraint solver. Relationships that must hold for
every realization—such as maximum CVT ratio remaining above minimum CVT ratio—
require non-overlapping supports or a future constrained parameterization.

## Study types

### Design sweep

A design variable is evaluated at explicit values while a declared uncertainty
mode is sampled. The output reports physical p10–p90 bands, bootstrap estimation
intervals, paired win fractions, paired regret, and bootstrap intervals on those
ranking statistics. The design variable
itself is excluded from random sampling.

### Measured-track robustness

Only measured-track variability is sampled. Empirical gate targets are always in
this category. An obstacle coefficient or model alternative joins this study only
when its uncertainty declaration explicitly contains:

```toml
role = "measured_track"
```

Broad defaults and uncertain calibration coefficients such as an estimated
impact coefficient use `role = "structural"` and therefore remain nominal in a
measured-track robustness run. This prevents uncertainty about the model from
being mislabeled as lap-to-lap track variation.

This study answers whether a design conclusion survives plausible realizations
of the measured track. It does not establish robustness to uncertain drag, engine
power, tire properties, or structurally uncertain obstacle coefficients.

### Structural sensitivity

One declared structural input is changed at a time. Numeric inputs evaluate the
exact nominal value separately from selected distribution quantiles. Discrete
model-form choices evaluate the nominal choice and each declared alternative.
These cases are not Monte Carlo replicates and do not receive probability
confidence intervals.

This answers which model assumptions affect the result and in what direction. It
does not claim that the tested quantiles are equally probable operating cases.

### Full uncertainty propagation

All stochastic inputs are sampled jointly, including measured-track, structural,
and uncertain initial-condition inputs. This produces the overall output
distribution implied by the declared model. `selected_structural` may instead
sample a named subset through `sampling.paths`; every selected path must actually
have `role = "structural"`.

## Physical variation versus estimation error

For each output, the p10, median, and p90 describe variation across plausible
physical scenarios. Bootstrap intervals describe finite-sample uncertainty in
those estimated statistics. They are reported separately and must not be added
together.

The convergence report also includes sample count, split-half median stability,
and Monte Carlo standard error of the mean. Fewer than 20 scenarios is explicitly
reported as a quick check rather than a converged production study.

## Reference caching

An infinite reference is reused only when the swept design path is mathematically
absent from the reference mechanism. The current safe shared path is bounded-CVT
minimum reduction ratio. Final drive and maximum reduction ratio are not shared
because the finite launch-torque contract depends on them even for the otherwise
unbounded reference.

The manifest records bounded runs, reference runs, cache hits, and a fingerprint
of each reference case. A speed optimization may never alter the physical
comparison contract.

## Numerical quality gate

Every study checks:

- completion of all bounded and reference cases;
- infinite-reference dominance;
- gate compliance within 0.5 km/h;
- vehicle energy-balance residual;
- engine-to-wheel energy-balance residual.

`numerical_quality.valid_for_decision` becomes true only when every check passes.
A short coarse-step run may validate code paths while correctly remaining invalid
for engineering decisions.

## Current uncertainty boundary

Phase 6 propagates all declared scalar and categorical simulation inputs plus
empirical gate targets. It does not yet perturb physical feature coordinates,
centreline geometry, or GPX elevation. The manifest lists these omissions. Grade
force remains disabled.


## Semantic uncertainty roles

A distribution describes *how* a value varies. Its role describes *why* it is
being varied and therefore which study is allowed to sample it. The supported
roles are:

- `structural`: uncertainty in a vehicle parameter, model coefficient, or model
  form;
- `measured_track`: repeatable variation evidenced by track observations;
- `initial_condition`: uncertainty in the state from which a simulation begins.

Vehicle, drivetrain, driver, and surface quantities default to structural when
the role is omitted. Initial conditions default to `initial_condition`. Ambiguous
stochastic obstacle inputs must declare their role explicitly. Fixed values do
not need a role because they are never sampled.

The distinction is deliberately semantic rather than statistical. A triangular
distribution can represent either structural uncertainty or measured-track
variation; the shape alone cannot tell the software which interpretation is
correct.
