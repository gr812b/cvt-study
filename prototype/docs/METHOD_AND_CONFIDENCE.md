# Measured speed-gate method and confidence

## Decision question

The workflow does not try to predict an exact future lap. It asks whether a CVT ratio range or final-drive choice loses less propulsion opportunity than another choice across paired, physically plausible track realizations.

The primary reference is the same vehicle and track with an unbounded ideal CVT that can hold peak-power engine speed. Reported finite-ratio opportunity loss is:

\[
E_{\mathrm{opp}} = E_{\mathrm{clutch}} + \int\max(P_{\mathrm{peak,requested}}-P_{\mathrm{engine}},0)\,dt.
\]

The reference still obeys tire traction, drag, rolling resistance, event losses, and the same measured gates. It is an ideal ratio-range reference, not a zero-loss vehicle.

## What a measured gate means

A gate contains a course position, a target-speed distribution, and an evidence score. Upstream of a gate at \(s_g\), the allowed speed follows

\[
v_{\mathrm{limit}}(s)=\sqrt{v_g^2+2a_b(s_g-s)}.
\]

This produces a finite braking approach. It never raises speed. After the vehicle passes the gate, the constraint disappears, so designs remain free to differ on the following straight.

Turns use the repeatable control/minimum-speed location when the data supports it. Track events use the physical event start and immediate entry speed. This distinction prevents an upstream turn-braking state from being mistaken for the entry state of a log or bump.

## Confidence score

Every candidate gate remains visible. Default acceptance requires both a sufficient total score and braking/state-convergence evidence. The score combines:

| Evidence | Why it matters |
|---|---|
| Valid pass count | A gate seen only a few times is unstable. |
| Target-speed spread | A narrow 10th–90th range is stronger evidence of convergence. |
| Repeatable braking or speed reduction | Distinguishes a controlled state from ordinary acceleration. |
| Pace independence | A target strongly correlated with lap pace may be drivetrain-limited, not geometry/driver-limited. |
| Position and interval quality | Penalizes uncertain anchors or provisional extents. |
| Cross-vehicle agreement | Tests whether the state generalizes beyond one vehicle. |

Classes are `HIGH`, `MEDIUM`, `LOW`, and `INSUFFICIENT`. A one-vehicle study cannot earn cross-vehicle evidence; that limitation is written into every report and bundle.

## How uncertainty reaches the design result

Gate targets retain observed p10, median, and p90 speeds. GPS-derived event response retains low, nominal, and high effective-loss seeds. A sweep draws from both distributions. The same draw is reused for each design value in a replicate, so the comparison is paired and differences are not polluted by different random tracks.

The plotted confidence bars are scenario-sensitivity bands, not statistical proof about next year's course. A robust choice has a useful advantage across most paired draws; overlapping bands or frequent ranking switches mean the data does not distinguish the designs strongly.

## Numerical safeguards

- The simulation uses a 1 ms default integration step.
- The longitudinal tire-slip update is solved with a bracketed implicit root, preventing the old Newton iteration from jumping between saturated branches and injecting energy.
- Bounded and unbounded cases use identical track, gates, loss realization, tire model, and driver controller.
- Energy accounting reports a residual; a large relative residual invalidates the result.
- `simulated_gate_check.csv` reports target and achieved speed for both cases.
- `reference_dominance_pass` must be true before interpreting a bounded-versus-reference comparison.

## What is and is not calibrated

Measured entry/control speeds are empirical. The current effective event losses are seeded from observed kinetic-energy change, but GPS cannot separate driver braking, grade, tire deformation, soil, or suspension work. Those values are therefore uncertainty inputs, not identified obstacle coefficients.

Use the workflow to compare ratio decisions, locate break-even regions, and plan testing. Do not present its absolute lap time, obstacle energy, or next-course optimum as a measured fact. The strongest next validation step is a second vehicle on the same course; the strongest calibration step is synchronized GPS, throttle/brake, engine/CVT speed, wheel speed, and video.

## Reduced-order tire model

The host evolves vehicle speed and wheel speed separately. Their difference at the tire surface is the slip speed. Longitudinal tire force is

\[
F_x = F_{\max}\tanh\!\left(K_s v_{\rm slip}/F_{\max}\right).
\]

Only two tire-shape inputs are exposed:

- **Peak traction** sets the force ceiling, through the track friction coefficient multiplied by `peak_traction_scale`.
- **Slip stiffness** sets how quickly force approaches that ceiling as wheelspin begins.

Low/medium/high levels are provided for each axis independently. Rolling resistance remains a separate loss. These are engineering uncertainty scenarios, not claims that GPS identifies tire parameters.
