# Vehicle profiles

Vehicle profiles can provide reusable mass, tire, aero, resistance, engine, or
drivetrain assumptions for several vehicles and tracks. They are broad priors or
team estimates, not silent truths. Every physical number remains a complete
uncertainty-aware quantity.

A project vehicle selects profiles in `vehicles/<id>/vehicle.toml` and then locally
overrides measurements that differ for that vehicle.
