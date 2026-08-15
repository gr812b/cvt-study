# Maryland endurance-traffic model — root overlay

This overlay installs a reviewed empirical traffic generator for the Maryland drivetrain study. It is intended to be extracted directly over the latest production `cvt-study` repository root and then wired into the current source tree with `apply_traffic_model.ps1`.

## What the model is trying to reproduce

Traffic is represented as an exogenous stochastic process, not as a lap-time penalty. A generated world contains encounter times plus whole observed event marks `(type, duration, retained-speed fraction, confidence)`. The same world is paired across bounded/infinite CVT cases, design candidates, and crossed track cases.

The target is Maryland. Maryland's reviewed one-hour stream (26 events) sets the Maryland event-rate level and supplies Maryland's duration/severity marks. Arizona contributes two independent traffic-exposure streams (CWRU 15 min, ETS 75 min) and a 59-car four-hour field-survival reduction. Arizona is used only to help describe how traffic exposure might change as the field thins through the endurance race.

## Arrival model

The production process is a non-homogeneous Poisson process

`lambda_M(t) = lambda0_M * S_AZ(t)^alpha`

where `S_AZ(t)` is the empirical Arizona field-survival curve. The fit gives every Arizona exposure stream its own nuisance initial rate. This is important because CWRU and ETS have very different average observed rates and CWRU was observed only early; forcing one common Arizona rate would confound between-stream exposure with race-time decay.

The shared exponent `alpha` is fitted from within-stream event timing in CWRU, ETS, and Maryland. Maryland gets its own production rate scale. A joint parametric/bootstrap calibration resamples the field-population records, re-simulates all observation streams, and refits the model. Each Monte-Carlo traffic world samples one correlated `(lambda0_M, alpha)` bootstrap pair.

Current deterministic calibration from the included evidence is approximately:

- Maryland initial rate: 32.41 events/h.
- CWRU nuisance initial rate: 86.56 events/h.
- ETS nuisance initial rate: 38.84 events/h.
- shared `alpha`: 2.835.
- nominal four-hour time-average Maryland rate: 15.91 events/h.
- 95% bootstrap interval for Maryland initial rate: 18.83–48.61 events/h.
- 95% bootstrap interval for `alpha`: approximately 0–9.45.
- 95% bootstrap interval for the four-hour average rate: 5.77–29.85 events/h.

The near-zero lower tail on `alpha` is deliberate. Once CWRU/ETS are allowed separate source rate scales, homogeneous, clock-exponential, and field-survival source models are all statistically competitive over the observed source window. The data do not justify pretending that a strong decay is certain. Field survival is retained as the four-hour extrapolation structure because it is tied to measured cars still circulating; the bootstrap keeps nearly-flat alternatives alive.

## Event marks

Maryland event rows are resampled jointly. Duration, event type, retained-speed fraction, and confidence therefore keep their empirical dependence. Confidence labels remain audit metadata; they are not converted into arbitrary Gaussian noise. Arizona marks are not substituted for Maryland marks.

The Arizona source spreadsheet heading says "speed reduction", but ordinary slowdown/yellow entries behave as retained-speed fractions. Rows explicitly labelled `full stop` are normalized to retained fraction 0. The Maryland normalized evidence also includes two stop-like yellow restrictions with retained fraction 0.

## Simulator coupling

For a traffic-enabled uncertainty world:

1. Run the traffic-free infinite-CVT reference.
2. Convert that trace into a compact `v_ref(s)` profile.
3. Run the traffic-aware infinite-CVT reference.
4. Run the traffic-aware bounded/design candidate.
5. At an active encounter, apply `v_cap(s) = retained_fraction * v_ref(s)` inside the normal driver/dynamics loop.

The cap is based on a shared traffic-free reference rather than the candidate's already-slowed current speed. This prevents feedback where a worse drivetrain would make the same external traffic obstruction artificially slower. Overlapping encounters use the most restrictive retained fraction.

The ordinary nominal baseline remains traffic-free. Traffic is automatically active in `full_uncertainty` and design studies. Uncertainty-informed design replay reuses the exact traffic world saved by the source full-uncertainty draw rather than drawing a new one.

## Apply

From a clean latest production checkout, extract this archive over the repository root, then run:

```powershell
.\apply_traffic_model.ps1
python -m pytest .\tests\test_traffic_model.py -q
```

The installer edits narrow known blocks and refuses ambiguous source shapes rather than silently overwriting newer whole files.

## Review the calibration before a long study

```powershell
drivetrain-study calibrate-traffic .\projects\maryland
```

Open the reported `traffic_calibration_report.html`. It contains the fitted rates, model comparison, field-survival curve, normalized marks, 15-minute checks, and a 3,000-replicate posterior/predictive-style validation of the actual Maryland observation window.

Then run the normal uncertainty study:

```powershell
drivetrain-study run full-uncertainty .\projects\maryland --workers 8
```

The resulting full-uncertainty HTML is augmented with a traffic audit section and exports the generated traffic worlds/events and traffic-impact summary. When those completed worlds are later reduced for uncertainty-informed design screening, traffic phase, calibration draw, event count, total duration, minimum retained fraction, and a duration-weighted restriction burden participate in the representative-world diversity selection; the exact selected traffic realization is then replayed.

## Evidence limitations kept explicit

- Maryland has one reviewed observer-hour, so its absolute rate and mark distribution have finite-sample uncertainty.
- Maryland does not currently have a four-hour field-population history; the Arizona field-survival curve is a cross-race proxy.
- The 59-car Arizona survival CSV is a derived artifact recovered from the earlier lap-timing analysis. Re-importing the raw lap archive is preferable if it becomes available again.
- Traffic locations are not measured consistently enough to justify a spatial hotspot distribution, so encounters are time-based.
- CWRU/ETS show real between-stream rate heterogeneity. With only two Arizona exposure streams, the model does not invent a population random-effect law; separate nuisance rates remove that heterogeneity from the decay estimate instead.
