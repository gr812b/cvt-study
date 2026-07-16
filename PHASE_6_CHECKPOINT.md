# Phase 6 checkpoint — paired uncertainty studies

Phase 6 turns the uncertainty declarations already attached to project inputs into reproducible, paired simulation studies. It completes the uncertainty engine and the four core study runners. Phase 7 uncertainty attribution has not been started.

## Implemented study entry points

```powershell
cvt-study run sweep <project>
cvt-study run track-robustness <project>
cvt-study run structural-sensitivity <project>
cvt-study run uncertainty <project>
```

The runners consume the reviewed `track_bundle.json`, one resolved vehicle, and one study configuration. They do not read GPX or reconstruction internals.

## Uncertainty roles

Distribution shape and engineering meaning are now separate contracts.

- `structural`: uncertainty in a vehicle parameter, model coefficient, or model form;
- `measured_track`: repeatable variation supported by track observations;
- `initial_condition`: uncertainty in the initial simulation state.

Vehicle, drivetrain, driver, surface, and broad obstacle priors are structural by default. Initial-state quantities default to `initial_condition`. A stochastic obstacle input must explicitly choose `structural` or `measured_track`; the software does not infer its role from the fact that the obstacle appears in the track bundle.

This distinction prevents uncertain calibration coefficients such as an estimated impact coefficient from being mislabeled as lap-to-lap track variation.

## Supported uncertainty contracts

Numeric quantities support:

- fixed;
- normal;
- truncated normal;
- uniform;
- triangular;
- empirical samples.

Categorical choices support:

- fixed;
- discrete alternatives.

Normal and truncated-normal transforms use SciPy. Every sampled numeric value is converted to SI before simulation.

Validation rejects uncertainty support that permits impossible physical values. Examples include negative mass, efficiency outside `(0, 1]`, crossing CVT ratio bounds, negative obstacle coefficients, and materially invalid tails from an unbounded normal. Invalid support is not silently clipped or repeatedly resampled; the user must declare an appropriate bounded distribution.

## Paired scenario sampling

One scenario is a complete realization of all inputs selected by the study mode. Every drivetrain candidate and its matched infinite reference receive exactly the same scenario.

Accepted gate targets use a paired-lap policy by default. One run/lap/vehicle/driver identity is selected and its observed speed is used at every gate for which that identity exists. When no identity covers every active gate, the maximum-coverage identity is retained and only missing gates fall back to independent empirical draws. The fallback gate IDs are written to the manifest.

This preserves observed within-lap pace relationships instead of independently constructing a physically artificial mixture of gate speeds.

## Correlated inputs

Declared correlation groups use a Gaussian copula. Validation requires:

- known stochastic member paths;
- members sampled by the selected study mode;
- no overlap between groups;
- finite, symmetric correlation matrices;
- unit diagonal;
- positive-semidefinite matrices;
- agreement between the group declared by each input and the study group.

The matrix controls latent Gaussian dependence. It is exactly the final Pearson correlation only for normal marginals. It is not a general constraint solver, so relationships that must hold for every draw still require non-overlapping support or a future constrained parameterization.

## Four engineering studies

### Design sweep

Evaluates explicit design values under paired uncertainty scenarios. The design variable is excluded from random sampling. Outputs include physical p10/median/p90 bands, finite-sample bootstrap intervals, paired win fractions, paired regret, and bootstrap intervals on ranking statistics.

### Measured-track robustness

Samples empirical accepted-gate speeds and only obstacle inputs explicitly marked `role = "measured_track"`. Structural vehicle assumptions and broad obstacle-model priors remain nominal.

This answers whether a design conclusion survives plausible realizations of the measured track. It does not claim robustness to uncertain drag, power, efficiency, tire properties, or structural obstacle coefficients.

### Structural sensitivity

Changes one structural input at a time while all other inputs remain nominal.

Numeric parameters evaluate the exact nominal value and configured distribution quantiles. Discrete model choices evaluate the nominal choice and each declared alternative. These are deterministic sensitivity cases, not equally probable Monte Carlo outcomes, so probability confidence intervals are not attached to them.

### Full uncertainty propagation

Jointly samples every declared stochastic measured-track, structural, and initial-condition input. A selected-structural mode may instead sample an explicit subset of structural paths.

This produces the overall output distribution implied by the current declared model. It does not make undeclared uncertainty disappear; omitted geometry and elevation uncertainty are listed in the manifest.

## Output statistics

Physical scenario variation and Monte Carlo estimation error are kept separate.

For each output, the study reports:

- count, mean, and standard deviation;
- p10, median, and p90 across physical scenarios;
- bootstrap 95% intervals on p10, median, and p90;
- probability-like paired win fraction, with exact ties split equally;
- bootstrap 95% interval on paired win fraction;
- paired mean, median, and p90 regret with bootstrap intervals;
- threshold exceedance fractions with Wilson 95% intervals;
- split-half median stability;
- Monte Carlo standard error of the mean.

Fewer than 20 scenarios is explicitly labelled `more_replicates_recommended`. A one- or two-scenario smoke test can verify the mechanism path but cannot masquerade as a converged distribution.

## Infinite-reference caching

Reference reuse is allowed only when the swept design variable is mathematically absent from the infinite-reference mechanism.

The current safe shared path is:

```text
drivetrain.cvt.minimum_reduction_ratio
```

Final-drive ratio and maximum CVT reduction are not shared because the Phase 5 finite launch-torque contract depends on them even for the otherwise unbounded reference. The manifest records bounded simulations, reference simulations, cache hits, and the reference-cache policy.

## Numerical decision gate

Every study checks:

- completion of every bounded and reference case;
- infinite-reference dominance;
- accepted-gate compliance within 0.5 km/h;
- vehicle energy-balance residual;
- engine-to-wheel energy-balance residual.

`numerical_quality.valid_for_decision` is true only when every check passes. Coarse short runs are retained as useful integration tests but correctly remain invalid for engineering decisions when numerical tolerances fail.

## Bundle contract update

The track-bundle schema is now `1.2.0`. The supported family is `1.2.x`.

The bump records that uncertainty roles are part of the published track/simulation contract. Older bundle minors are rejected rather than silently reinterpreted. Projects must rebuild their track bundle after upgrading.

## Study outputs

A study result directory contains:

```text
REPORT.md
run_manifest.json
input_contracts.json
scenario_draws.jsonl
replicate_results.csv
summary.json
summary.csv
convergence.json
track_bundle.json
track_bundle.sha256
resolved_inputs/
*.png
```

The scenario file records every sampled SI quantity, categorical choice, gate target, paired gate identity, fallback gate, replicate number, and deterministic seed. The result directory is staged atomically so a failed run does not publish a partial result as complete.

## Validation performed

### Test suites

- 140 clean-package tests passed across grouped suites;
- 16 preserved prototype regression tests passed;
- source compilation passed;
- public runtime annotations resolved.

### Short code-path studies

At a deliberately coarse 10 ms step, all four study types were exercised using one or two scenarios. The runs produced complete outputs and correctly marked themselves invalid for decisions where energy closure or reference dominance failed. This tested failure reporting rather than hiding numerical shortcomings.

### Decision-quality numerical checks

At a 2 ms step:

- a two-value minimum-CVT-ratio sweep completed two bounded cases with one shared infinite reference and one cache hit;
- every completion, dominance, gate, vehicle-energy, and powertrain-energy check passed;
- a real Arizona full-uncertainty scenario sampled 43 declared physical inputs and all 13 active gates;
- the Arizona gate draw used one of 11 complete paired lap identities with no independent fallback;
- the Arizona run passed every numerical quality check.

These one-scenario checks validate the complete mechanism and uncertainty path. They are not presented as converged physical distributions, and their convergence reports explicitly request more replicates.

### Categorical structural check

A discrete obstacle-model sensitivity was run for a nominal `none` model and a `distributed_resistance` alternative. The output retained categorical values and did not invent numeric quantiles or probability intervals.

### Installed-wheel check

A fresh wheel was built and installed in an isolated virtual environment. The packaged commands `init`, `validate`, `build-track`, `validate-bundle`, and `run track-robustness` completed outside the source tree. SciPy is an explicit project dependency.

A provenance-ordering defect discovered during this check was corrected: source-file records are now sorted by portable project identity rather than machine-specific absolute path. Source-tree and installed-wheel builds therefore produce the same semantic bundle fingerprint.

## Deliberate current limitations

Phase 6 does not yet propagate:

- physical-feature coordinate uncertainty;
- centreline geometry uncertainty;
- GPX elevation uncertainty;
- road-grade force;
- full constrained multivariate relationships;
- Phase 7 uncertainty attribution;
- Sobol or Shapley indices;
- parallel execution, progress estimates, or resume support;
- production-scale 30–100 scenario results.

Those omissions are explicit in run manifests and documentation rather than silently treated as zero uncertainty.
