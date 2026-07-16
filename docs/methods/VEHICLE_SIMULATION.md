# Phase 5 longitudinal simulation

## State

The reduced host integrates distance \(s\), vehicle speed \(v\), and driven-wheel
speed \(\omega_w\). Tire slip speed is

\[
v_{\mathrm{slip}}=R_w\omega_w-v.
\]

## Tire force

The longitudinal tire law is

\[
F_x=F_{\max}\tanh\left(K_s v_{\mathrm{slip}}/F_{\max}\right),
\]

with

\[
F_{\max}=\mu\,\gamma_{\mathrm{peak}}\,f_d N.
\]

Here \(f_d\) is the driven normal-load fraction. `peak_traction_scale` controls
the force ceiling and `slip_stiffness` controls how quickly force builds with
slip. The stiff scalar slip equation is advanced by backward Euler and solved by
deterministic bracketed bisection. SciPy is permitted in the project, but this
monotone scalar solve does not benefit from a more general solver.

Centreline curvature and lateral demand are reported as diagnostics. They do not
consume tire capacity because Phase 5 has no validated lateral/yaw model; measured
speed gates represent repeatable driver-limited corner entry.

## Vehicle and wheel equations

\[
m\dot v=F_x-F_{\mathrm{grade}}-F_{rr}-F_a-F_{obs},
\]

\[
J_w\dot\omega_w=T_w-T_b-R_wF_x.
\]

Aerodynamic resistance is \(F_a=\tfrac12\rho C_DA v^2\). Rolling resistance is
regularized smoothly near rest. Grade force is available for explicit smooth
feature profiles but remains disabled for raw GPX/FIT altitude. A track-only
materiality screen decides whether a paired grade sensitivity is worth running;
it does not silently activate a new force.

## CVT models

The bounded ideal CVT attempts to hold the declared target engine speed. Its
required ratio is clipped to the declared maximum and minimum ratios. At launch,
an idealized slipping clutch allows the engine to hold target speed while wheel
speed is too low for synchronous operation. Clutch dissipation is accounted for.

The infinite reference uses the same engine curve, target speed, vehicle, tire,
driver, track, obstacles, and gates. It removes only the finite ratio window. To
avoid infinite launch torque, one scenario-level wheel-torque cap is frozen before
design values are applied, then shared by every candidate. Any power that cannot
pass through that cap is recorded as launch-clutch loss; it is not credited as a
finite-ratio advantage.

## Gates

An accepted entry gate is a one-way speed ceiling. The code backward-propagates its
target through a finite braking envelope. A vehicle below the envelope continues
normally; it is never reset upward. Exact gate-crossing samples are inserted into
the public trace so compliance is not distorted by the coarser reporting grid.
When response-minimum evidence passes significance, spatial-repeatability, and
leave-one-lap-out checks, a second sustained-response ceiling is paired with the
entry gate. Failed checks leave the original entry gate unchanged.

## Energy accounting

Work is integrated at the solver step, not reconstructed from the downsampled
trace. Outputs separate engine energy, transmitted energy, drivetrain-efficiency
loss, launch-clutch loss, engine operating shortfall, tire slip, braking, rolling
resistance, aerodynamic loss, obstacle loss, and net grade work.

For one simulated case, total opportunity loss is

\[
E_{opp,case}=E_{clutch}+E_{off\text{-}peak}.
\]

The design-comparison metric removes the launch loss shared by the otherwise
identical infinite-ratio reference:

\[
E_{opp,finite}=
\max\!\left(0,
E_{opp,bounded}-E_{opp,infinite}
\right).
\]

This leaves loss attributable to the bounded ratio window rather than the common
launch-torque contract. Drivetrain-efficiency loss is physical energy dissipation,
not finite-ratio opportunity loss, because both designs use the same efficiency.
Vehicle-level and engine-to-wheel energy-balance residuals are exported as
numerical verification metrics.
