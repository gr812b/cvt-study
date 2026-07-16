# Explicit obstacle models

Physical features own obstacle models. Response groups only describe which
features share one measurable GPS response. No measured speed drop is silently
converted into terrain energy.

For a feature interval of length \(L\), the normalized raised-cosine density is

\[
q(x)=\frac{1-\cos(2\pi x/L)}{L}, \qquad \int_0^L q(x)\,dx=1.
\]

It distributes lumped energy smoothly without changing the requested line
integral.

## `none`

No obstacle force is applied. Geometry and gate evidence remain available.
Choosing `none` is explicit; omission is invalid.

## `fixed_specific_energy`

\[
E=m e_s, \qquad F(x)=E q(x),
\]

where \(e_s\) is in J/kg. The spatial integral of force equals \(E\).

## `speed_quadratic_energy`

\[
E=m e_0+k_{\mathrm{impact}}v_{\mathrm{entry}}^2,
\qquad F(x)=E q(x).
\]

The entry speed is captured once when the vehicle first enters the physical
feature. It is not recomputed from a vehicle that has already slowed inside the
feature. `k_impact` has units of kg. It is an effective reduced-order coefficient,
not a claim that all suspension/soil/contact physics have been identified.

## `distributed_resistance`

\[
F(x)=F_0.
\]

This is appropriate when the mechanism is best represented as approximately
constant resistance over a known interval.

## `roughness_energy_density`

\[
F(x)=m e'_s,
\]

where \(e'_s\) is in J/(kg m). Energy therefore scales with feature length.

## `smooth_profile`

The geometric profile is

\[
z(x)=\frac{h}{2}\left[1-\cos(2\pi x/L)\right].
\]

Its slope contributes conservative grade work. Its vertical curvature modifies
normal load using the rigid-following approximation

\[
N/N_0=1+v^2z''/g,
\]

clipped to declared minimum and maximum scales. A traction multiplier can modify
local friction. Optional unresolved dissipation uses the same entry-speed energy
law as `speed_quadratic_energy`.

This profile is for user-supplied feature geometry. Raw GPX/FIT elevation does not
activate grade force in Phase 5.

## Defaults and uncertainty

Built-in profiles are intentionally broad, uncertainty-aware engineering priors.
They make the workflow usable without asking a new user to invent a coefficient,
but they do not turn an unknown coefficient into a measurement. Phase 5 uses the
nominal value; Phase 6 samples the declared spread and model alternatives.

Each baseline output includes `obstacle_energy_by_feature.csv`, including model
type, captured entry speed, and energy for both bounded and infinite cases.
