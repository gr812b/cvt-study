# Vehicle and nominal-study input contract

Phase 5 resolves one vehicle and one baseline study. Every physical parameter is
an uncertainty-aware quantity even though the baseline runner uses only its
nominal value.

## Required vehicle quantities

The vehicle contract contains mass, conventional gravity, tire diameter, driven
wheel/axle rotational inertia, driven normal-load fraction, drag area, air
density, rolling-resistance coefficient, tire peak-traction scale, and tire slip
stiffness. The drivetrain contract contains final drive, efficiency, CVT ratio
bounds, engine model, engine target speed, engine power scale, and launch-clutch
model.

`gravity = 9.80665 m/s^2` is still declared in the resolved configuration. It is
a fixed derived constant, not an invisible numerical literal.

The built-in Baja profile provides broad priors. These are starting assumptions,
not calibration. Projects should override them when measurements, manufacturer
data, coast-down tests, dyno data, or better engineering estimates exist.

## Baseline study

A baseline study declares:

- vehicle identity;
- nominal sampling mode;
- maximum braking deceleration;
- maximum brake force;
- braking trigger margin;
- initial vehicle and wheel speeds;
- maximum simulated time, integration step, and report step;
- gate target statistic (`p10`, `median`, or `p90`).

The braking quantities are separate because a driver/vehicle may be limited by a
behavioral deceleration target, brake hardware force, or tire force. The smallest
applicable limit defines the backward braking envelope.

Initial conditions are physical inputs and therefore carry uncertainty metadata.
Numerical integration and reporting steps are algorithm settings rather than
sampled physical quantities.

## Configuration resolution

The final value follows:

```text
built-in profile → shared/team profile → vehicle/project file → study override → CLI override
```

Every run exports the complete resolved configuration and provenance chain.
Changing only a nominal value while accidentally retaining inherited uncertainty
produces a warning.
