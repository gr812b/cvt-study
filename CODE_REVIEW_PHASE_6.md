# Phase 6 code review

## Scope

The review covered uncertainty contracts, distribution transforms, semantic sampling modes, paired gate sampling, correlation groups, scenario planning, study execution, infinite-reference reuse, statistical summaries, output serialization, CLI integration, package data, and installed-wheel behavior.

The review was performed against deliberately short studies plus decision-quality single-scenario checks. Phase 7 attribution is outside this checkpoint.

## Findings corrected before release

### 1. Obstacle uncertainty was being classified by location rather than meaning

**Finding:** Broad estimated obstacle coefficients live in the track bundle, so an early implementation treated them as measured-track variability.

**Risk:** A track-robustness interval would have mixed observed lap variation with epistemic uncertainty in the obstacle model, making the result impossible to interpret.

**Correction:** Added explicit semantic roles: `structural`, `measured_track`, and `initial_condition`. Stochastic obstacle quantities and model choices must explicitly state their role. Broad priors now default to structural and remain nominal in a measured-track robustness study.

### 2. Invalid physical tails could enter a simulation silently

**Finding:** Generic normal distributions can assign nonzero probability to negative mass, negative coefficients, or invalid efficiency.

**Risk:** Rejection or clipping during sampling would alter the declared distribution invisibly and could create nondeterministic failure rates.

**Correction:** Validate distribution support before execution. Material invalid support is an error requiring a truncated or otherwise bounded declaration. The sampler does not silently clip, reject, or redraw impossible values.

### 3. Pairing could be broken at the gate level

**Finding:** Independent empirical gate draws are statistically convenient but can combine unrelated fast and slow laps into one artificial scenario.

**Risk:** The resulting track realization can be physically inconsistent and can overstate output spread.

**Correction:** The default policy samples one complete run/lap/vehicle/driver identity across active gates. Maximum-coverage fallback is explicit and the independently sampled gate IDs are recorded.

### 4. “Probability of best” overstated what finite Monte Carlo samples establish

**Finding:** A raw winning fraction was initially named as a probability of being best.

**Risk:** Seven or thirty deterministic Monte Carlo draws could be interpreted as a calibrated posterior probability.

**Correction:** Renamed the quantity to paired win fraction, split exact ties equally, and added row-bootstrap intervals. Paired regret is reported alongside it so a frequent but negligible win is distinguishable from a material advantage.

### 5. Threshold estimates lacked finite-sample intervals

**Finding:** Threshold exceedance outputs were point fractions only.

**Risk:** Small studies could display deceptively precise percentages.

**Correction:** Added Wilson 95% intervals and retained the scenario count.

### 6. Physical spread and estimator uncertainty were mixed

**Finding:** Percentile bands and uncertainty in the estimated percentile could be read as the same object.

**Risk:** Users could add or compare them incorrectly.

**Correction:** Physical p10/median/p90 bands and bootstrap estimation intervals are distinct fields and are explained separately in reports and documentation.

### 7. Correlation declarations were under-validated

**Finding:** Early correlation handling accepted member mismatches and matrices without complete structural checks.

**Risk:** A typo could create a misleading dependence structure or fail deep inside sampling.

**Correction:** Validate known stochastic paths, selected sampling role, group-name agreement, no overlapping groups, finite/symmetric/unit-diagonal matrices, and positive semidefiniteness before a study starts.

### 8. Correlation semantics were too casually described

**Finding:** The copula matrix could be read as the final Pearson correlation for any marginals.

**Risk:** Bounded, empirical, and discrete distributions do not preserve that direct interpretation.

**Correction:** Documentation now calls it latent Gaussian dependence and limits the exact Pearson interpretation to normal marginals.

### 9. Design sweeps could sample their own design variable

**Finding:** The registry and the design planner initially operated independently.

**Risk:** A declared sweep value could be overwritten by a stochastic sample of the same path.

**Correction:** The design variable is excluded from scenario sampling and override paths are validated before execution.

### 10. Design-domain checks were incomplete

**Finding:** Finite numeric values were accepted without always checking physical domains or CVT-bound ordering.

**Risk:** A sweep could intentionally or accidentally request negative dimensions, invalid efficiency, or crossed ratio bounds.

**Correction:** Reused physical-support validation for design values and added post-override cross-field checks.

### 11. Reference reuse was initially too broad

**Finding:** A generic “one reference per scenario” optimization would have reused the infinite reference across final-drive values.

**Risk:** The Phase 5 infinite reference still shares finite launch-torque capacity, which depends on final drive and maximum CVT reduction. Reuse would have changed the comparison contract.

**Correction:** Reference caching is path-specific and permitted only after proving invariance. Minimum CVT reduction is currently safe; final drive and maximum CVT reduction are not. Counts and cache hits are recorded in the manifest.

### 12. Structural sensitivity treated numeric and categorical assumptions alike

**Finding:** The first structural implementation assumed every parameter had numeric quantiles.

**Risk:** Model-form alternatives such as `none` versus `distributed_resistance` would be forced into meaningless numeric ordering.

**Correction:** Numeric quantities use nominal plus selected quantiles. Discrete choices use nominal plus each declared alternative, preserving categorical values and omitting probability-style confidence intervals.

### 13. Nominal structural cases were inferred from a median quantile

**Finding:** For skewed distributions, the median is not necessarily the declared nominal engineering estimate.

**Risk:** Changes “from nominal” could use the wrong baseline.

**Correction:** The exact nominal declaration is always simulated as its own level. Quantiles are additional sensitivity points.

### 14. JSON could contain non-standard NaN or infinity

**Finding:** Standard Python JSON serialization accepts non-finite floats by default.

**Risk:** Output would not be strict JSON and downstream tools could disagree about parsing it.

**Correction:** All public JSON uses strict serialization with `allow_nan=False`. Undefined one-sample standard errors are represented as `null`.

### 15. Result directories could be partially published

**Finding:** Direct writes exposed incomplete outputs after failures.

**Risk:** A partial directory could be mistaken for a valid study.

**Correction:** Studies write to a staging directory and publish atomically only after all required outputs succeed.

### 16. Track-bundle schema did not advertise semantic roles

**Finding:** Existing `1.0.x` bundle readers had no contract for role-separated uncertainty.

**Risk:** A newer simulator could reinterpret older stochastic obstacle metadata.

**Correction:** Bumped the bundle schema to `1.2.0`, added `uncertainty_roles_ready`, updated the JSON schema, and reject other minor families rather than providing accidental backward compatibility.

### 17. Source and installed-wheel builds had different semantic fingerprints

**Finding:** Portable provenance records were sorted by machine-specific absolute source path before those paths were removed.

**Risk:** Identical projects produced different semantic fingerprints depending on installation location.

**Correction:** Sort source records by portable scope/path/hash and GPX records by portable run identity. The source checkout and installed wheel now produce the same semantic bundle fingerprint.

### 18. Coarse smoke tests could appear decision-ready

**Finding:** Complete execution alone could be mistaken for validated output.

**Risk:** A 10 ms integration step produced visible energy residual and, in one design sweep, a reference-dominance failure.

**Correction:** Every study aggregates completion, dominance, gate, vehicle-energy, and powertrain-energy checks into `valid_for_decision`. Coarse checks remain intentionally visible as invalid integration tests.

## Architecture review

The Phase 6 code is divided into focused modules:

```text
uncertainty/distributions.py   distribution transforms and support
uncertainty/registry.py        discover declared uncertain inputs
uncertainty/sampling.py        paired scenarios, gates, copulas
uncertainty/statistics.py      bands, bootstrap, regret, convergence
studies/planning.py            design and sensitivity case planning
studies/service.py             execution and reference caching
studies/analysis.py            paired result aggregation
studies/reporting.py           strict outputs and plots
```

No study module imports GPX parsing or track-reconstruction internals. Simulation receives the published bundle view and a resolved scenario.

The largest current module is `studies/service.py` at roughly 400 lines. Its responsibilities remain coherent at this checkpoint—study orchestration, scenario/design loops, and safe reference reuse—but it should be watched during Phase 7. Attribution execution should go into a separate service rather than expanding this module further.

## Verification reviewed

- 140 clean-package tests passed in grouped suites;
- 16 preserved prototype regressions passed;
- source compilation and runtime annotation checks passed;
- all four study paths completed short adversarial smoke runs;
- numerical quality correctly rejected coarse-step results;
- a 2 ms minimum-ratio sweep demonstrated safe reference reuse;
- a 2 ms real Arizona full-uncertainty scenario sampled 43 inputs and 13 paired gates and passed all numerical checks;
- numeric and categorical structural sensitivities both completed;
- a fresh wheel installed with explicit SciPy dependency and executed outside the source tree.

## Known limitations accepted for this checkpoint

- Feature-coordinate, centreline, and elevation uncertainty are not yet sampled.
- Grade force remains disabled.
- Gaussian-copula groups express dependence but not hard multivariate constraints.
- Ranking bootstrap uses scenario-row resampling; it is not a Bayesian posterior.
- Production-scale convergence has not been claimed or tested here.
- No Phase 7 uncertainty attribution has been implemented.
- No multiprocessing, progress estimate, resume, or checkpointing exists yet.

## Review conclusion

Phase 6 is suitable as the uncertainty and study-runner foundation for Phase 7. Its most important guarantees are semantic separation of uncertainty sources, complete pairing across designs and references, explicit numerical-quality gating, strict output traceability, and refusal to optimize by changing the physical comparison contract.
