# Measured-Gate CVT Track Study

A compact engineering workflow for comparing CVT ratio bounds and final-drive choices using measured endurance-track GPS data.

The project deliberately separates two different uncertainty questions:

| Study | What changes? | What stays fixed? | Engineering question |
|---|---|---|---|
| **Measured Track Robustness Study** | Measured gate speeds and GPS-seeded event losses | Vehicle model and drivetrain assumptions | Does the drivetrain ranking survive plausible versions of the measured course? |
| **Structural Sensitivity Study** | Drag, power, efficiency, rolling resistance, tire behaviour, and obstacle resistance | One drivetrain design and one selected track | Which model assumptions most strongly affect the predicted finite-ratio penalty? |

These studies are complementary, but they are not interchangeable. A percentile band from the measured-track study is not a complete model-confidence interval, and a structural-sensitivity curve is not a probability distribution.

## Standard vehicle case

All public runners use the centralized defaults in `standard_case.py`:

- CVT ratio range: **3.5 to 0.9**
- Final-drive reduction ratio: **7.556**
- Tire diameter: **22 in**
- Wheel radius supplied to the model: **11 in**
- Integration step: **0.001 s**

Command-line arguments named `wheel-radius` always expect a radius, not a diameter.

## What the simulator compares

Every bounded design is compared with the same vehicle using an unbounded-ratio reference CVT.

The reference still obeys:

- the same engine power;
- the same tire and traction limits;
- the same aerodynamic and rolling resistance;
- the same obstacles and measured gates;
- the same driver controller.

Only the available CVT ratio range is idealized. The main decision outputs are:

1. **Finite-ratio opportunity loss** — propulsion opportunity lost because the bounded CVT cannot maintain the desired engine operating point.
2. **Time penalty versus the unbounded reference** — the practical track-time consequence of that finite ratio range.

A valid comparison should also pass the energy-balance, gate-compliance, and reference-dominance checks.

# 1. Measured Track Robustness Study

## Purpose

This is the uncertainty-aware drivetrain design sweep.

It asks:

> If the measured course is replayed with realistic lap-to-lap variation, does the preferred gearing or CVT ratio bound remain preferred?

The vehicle model is held fixed. Each replicate creates one plausible realization of the measured track by sampling:

- accepted gate target speeds from their measured distributions;
- effective event losses from the GPS-seeded low/nominal/high ranges.

The same realization is then used for every drivetrain candidate. This paired comparison prevents one design from receiving an easier random track than another.

## What a replicate means

One replicate is **one plausible measured-track scenario**, not another numerical integration of the exact same inputs.

For replicate `r`:

1. gate speeds are sampled;
2. event losses are sampled;
3. one unbounded reference is simulated;
4. every bounded design value is simulated against that same reference and scenario.

The unbounded reference is reused across all design values within the replicate because its wheel-power behaviour does not depend on the finite ratio bounds being swept.

For `R` replicates and `D` design values, the simulation count is:

\[
N_{\rm simulations}=R(D+1).
\]

For seven final-drive values and seven replicates:

\[
7(7+1)=56\text{ full-track simulations}.
\]

## What varies and what does not

### Varied in this study

- measured gate speed;
- effective obstacle/event loss;
- the selected drivetrain design parameter.

### Not varied in this study

- aerodynamic drag;
- engine power;
- drivetrain efficiency;
- rolling resistance;
- peak tire traction;
- tire slip stiffness;
- vehicle mass or wheel radius, unless explicitly changed as fixed command-line inputs.

Therefore, its percentile bars describe robustness to the **measured-track uncertainty represented in the code**, not total real-world uncertainty.

## Full GPS rebuild and final-drive sweep — PowerShell

From the repository root:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

py -m pip install --upgrade pip
pip install -r .\requirements.txt
pip install -e .\baja_track_validation

py .\run_full_measured_study.py `
  --gps .\baja_track_validation\examples\reference_run_gps.csv `
  --vehicle-ids vehicle_A `
  --definitions .\baja_track_validation\examples\obstacle_event_definitions_CLEANED.csv `
  --config .\baja_track_validation\examples\config.example.toml `
  --output-dir .\outputs\full_measured_study `
  --minimum-cvt-ratio 0.9 `
  --maximum-cvt-ratio 3.5 `
  --final-drive-ratio 7.556 `
  --wheel-radius-in 11 `
  --minimum-gate-confidence 60 `
  --integration-step-s 0.001 `
  --sweep-parameter final_drive_ratio `
  --sweep-values 5.5 6.2 6.9 7.556 8.2 8.9 9.6 `
  --sweep-replicates 7 `
  --random-seed 20260714
```

Use `--sweep-replicates 7` for a quick directional study and approximately `30` for more stable percentile and win-fraction estimates.

Available design parameters are:

```text
final_drive_ratio
minimum_speed_ratio
maximum_speed_ratio
```

## Multiple GPS recordings or vehicles

Provide one vehicle ID for every GPS file:

```powershell
--gps `
  .\data\vehicle_A_run1.csv `
  .\data\vehicle_A_run2.csv `
  .\data\vehicle_B_run1.csv `
--vehicle-ids `
  vehicle_A `
  vehicle_A `
  vehicle_B
```

The first recording establishes the reference centreline. Every later recording is projected onto the same along-track coordinate. Independent vehicle data can add cross-vehicle agreement evidence to gate confidence.

## Main outputs

```text
outputs\full_measured_study\
├── 01_track_validation\
│   ├── GPS cleaning and lap QA
│   ├── event/pass metrics
│   ├── gate-confidence tables
│   ├── top-down measurement and gate maps
│   └── simulator_track_bundle.json
├── 02_bounded_vs_unbounded\
│   ├── bounded/reference decision plot
│   ├── full simulation traces
│   ├── gate-compliance table
│   └── summary metrics
└── 03_design_sweep\
    ├── sweep_replicates.csv
    ├── scenario_gate_draws.csv
    ├── scenario_loss_draws.csv
    ├── sweep_confidence_summary.csv
    ├── sweep_ranking.csv
    ├── 01_sweep_decision_with_confidence.png
    └── SWEEP_REPORT.md
```

## How to read the design-sweep plot

The three panels show:

- median finite-ratio opportunity loss;
- median time penalty versus the unbounded CVT;
- positive-demand time spent inside the variable-ratio range.

The bars are the 10th–90th percentiles across paired measured-track scenarios.

Look for:

- a broad low-loss region rather than one artificially precise optimum;
- opportunity loss and time penalty pointing in the same general direction;
- low regret and a high win fraction across replicates;
- reduced positive-demand time at the harmful ratio bound;
- a result that is not merely the edge of the tested sweep.

If the best value sits at the end of the tested range, the study has identified a **direction**, not an optimum. Extend the sweep until performance begins to flatten or reverse.

## What seven replicates can establish

Seven replicates are useful for answering:

> Is the ranking obviously fragile to ordinary measured gate-speed and event-loss variation?

They are not enough to claim a precise 90% confidence interval or rule out rare scenarios. A `7/7` win fraction means only that the design won all seven sampled scenarios.

Thirty or more replicates make the percentile bands, win fractions, and paired-regret summaries steadier. They do not fix missing structural uncertainty.

# 2. Structural Sensitivity Study

## Purpose

This study asks a different question:

> How strongly does the standard design's bounded-versus-unbounded penalty depend on uncertain vehicle and model assumptions?

It changes one structural assumption at a time over seven ordered levels. This is an **engineering sensitivity study**, not Monte Carlo sampling.

The current parameters are:

- aerodynamic `C_DA` scale;
- engine-power scale;
- drivetrain efficiency;
- rolling-resistance scale;
- peak tire-traction scale;
- tire slip-build-up stiffness scale;
- obstacle-resistance scale.

## What the seven levels mean

The levels reveal:

- direction — whether increasing a parameter raises or lowers the predicted penalty;
- magnitude — whether the effect is negligible or decision-relevant;
- curvature — whether the response is approximately linear;
- threshold behaviour — whether the result changes rapidly in one region.

The seven levels are not seven equally likely outcomes, and the resulting lines are not confidence bands.

## Run the complete structural study — PowerShell

```powershell
py .\run_structural_sensitivity.py `
  --output-dir .\outputs\full_structural_sensitivity `
  --track .\tracks\lot_m.json `
  --minimum-cvt-ratio 0.9 `
  --maximum-cvt-ratio 3.5 `
  --final-drive-ratio 7.556 `
  --wheel-radius-in 11 `
  --integration-step-s 0.001 `
  --workers 6 `
  --no-show
```

Run only selected assumptions during development:

```powershell
py .\run_structural_sensitivity.py `
  --output-dir .\outputs\drag_and_efficiency_sensitivity `
  --parameters drag_area_scale transmission_efficiency `
  --workers 6 `
  --no-show
```

## Important scope boundary

The current structural runner evaluates the sensitivity of **one fixed drivetrain design** on the selected JSON track. It does not automatically rerun a complete final-drive or CVT-bound sweep at every structural level.

Therefore, it can tell you:

- which assumptions substantially change the standard design's finite-ratio penalty;
- which uncertain inputs deserve measurement or calibration;
- whether the standard result is directionally fragile.

By itself, it cannot prove that a design ranking remains unchanged under every structural assumption. A ranking-reversal study would require a nested design sweep for the structurally important parameters.

## Structural outputs

```text
outputs\full_structural_sensitivity\
├── structural_sensitivity.csv
├── structural_sensitivity.json
├── 01_time_penalty_sensitivity.png
├── 02_opportunity_loss_sensitivity.png
├── 03_directional_tornado_time_penalty.png
└── README.md
```

The tornado plot ranks endpoint effects relative to each parameter's nominal case. Large bars identify assumptions that most strongly influence the bounded-versus-unbounded result.

# Combining the two studies

Use the studies in this order:

1. **Measured Track Robustness Study:** identify designs that perform well across plausible versions of the measured course.
2. **Structural Sensitivity Study:** identify uncertain model assumptions capable of materially changing the predicted penalty.
3. **Targeted follow-up:** measure the influential assumptions or rerun the design sweep under their low and high cases.

A drivetrain recommendation is strongest when:

- it wins most paired measured-track realizations;
- its advantage is larger than the measured-track percentile spread;
- the result survives high-confidence gates alone and high-plus-medium gates;
- structurally plausible drag, power, efficiency, rolling resistance, and tire cases do not erase the advantage;
- it does not create a major clutch, low-ratio, traction, or recovery penalty;
- the chosen value lies inside a tested performance plateau rather than at a sweep boundary.

If the measured-track study is stable but structural sensitivity is high, the track evidence is consistent but the vehicle model needs calibration. If structural sensitivity is low but the measured-track ranking switches frequently, the model is stable but the measured course does not distinguish the designs. If both are stable, the engineering recommendation is comparatively strong.

# Tire model

The host evolves vehicle speed and wheel speed separately. Their difference at the tire surface is the longitudinal slip speed.

The reduced-order force law is:

\[
F_x=F_{\max}\tanh\!\left(\frac{K_s v_{\rm slip}}{F_{\max}}\right).
\]

It exposes two independent uncertainty axes:

- **Peak traction:** maximum longitudinal force the terrain can provide.
- **Slip stiffness:** how quickly that force develops as wheelspin begins.

Additional wheelspin eventually produces little extra force, so tire force saturates instead of growing without limit. Rolling resistance is modeled separately.

The low/medium/high tire levels are engineering scenarios. GPS alone does not identify these parameters.

# What the project can and cannot support

## Defensible uses

- identify whether the existing drivetrain spends useful propulsion time at a ratio bound;
- compare final-drive and ratio-bound design directions;
- find performance plateaus and break-even regions;
- quantify paired regret and ranking stability across measured-track scenarios;
- identify structural assumptions that deserve better measurement;
- support adjustable-gearing or testing decisions when the ranking is fragile.

## Claims that remain too strong

- exact future lap time;
- exact obstacle energy dissipation;
- exact tire force or wheel slip from GPS alone;
- a universally optimal final drive;
- precise probabilities from seven replicates;
- physical efficiency, durability, or shift-quality penalties of a wider real CVT range unless separately modeled or measured.

The strongest output is a robust engineering direction with clearly stated uncertainty, not an exact simulated optimum.

# Numerical checks

Before interpreting a result, confirm:

- `reference_dominance_pass` is true;
- gate-speed errors remain within tolerance;
- energy-balance residuals are small;
- the result is stable at the chosen integration step;
- all compared designs use paired scenario draws.

# Repository guide

```text
baja_track_validation/       GPS processing, event metrics, gate confidence and bundle export
track_builder/               Reduced-order physical/effective track sections
measured_track.py            Converts the exported gate bundle into simulator sections
decision_study.py            Bounded/reference comparison and measured-track sweep
simulation.py                Coupled vehicle, wheel, tire and ideal-CVT time integration
models.py                    Engine, vehicle, tire, driver and CVT model definitions
standard_case.py             Canonical standard-vehicle defaults
run_full_measured_study.py   Complete GPS-to-design-sweep workflow
run_measured.py              One bounded/reference measured-track comparison
run_measured_sweep.py        Sweep using an existing simulator bundle
run_structural_sensitivity.py One-factor-at-a-time model sensitivity
```

Further details:

- `docs/METHOD_AND_CONFIDENCE.md`
- `docs/REPOSITORY_GUIDE.md`
- `baja_track_validation/docs/`
- `track_builder/README.md`

# Tests

```powershell
py -m unittest discover -s .\tests -v
$env:PYTHONPATH = ".\baja_track_validation\src"
py -m unittest discover -s .\baja_track_validation\tests -v
Remove-Item Env:PYTHONPATH
```
