# Track reconstruction method

This document follows the Phase 3 implementation in data-flow order.

## 1. Local frame

All valid GPX points are mapped to a local tangent approximation centered on the
median latitude and longitude:

\[
x = R_E\cos\phi_0\,(\lambda-\lambda_0),\qquad
 y = R_E(\phi-\phi_0).
\]

This is appropriate for a compact closed course and avoids mixing degree units
with metre-based map thresholds.

## 2. Analysis speed cleaning

The canonical `analysis_speed_mps` is bounded by the configured reasonable-speed
limit. A centered five-sample median identifies isolated spikes. Only samples
whose difference from that local median exceeds the configured spike threshold
are replaced. Normal samples are retained; the track is not globally speed
smoothed.

## 3. Lap-gate visits

The configured lap-gate event defines the start/finish coordinate and therefore
\(s=0\). Points within `lap_gate_radius_m` are grouped into one visit while their
time separation remains below three seconds. The closest point in each visit is
the crossing representative. Visits closer than `minimum_lap_time_s` are merged.
Every consecutive pair of retained visits forms one complete lap.

A lap is initially valid when:

- all timing needed for duration exists;
- path distance is within 85–115% of the median complete-lap distance;
- stationary time fraction is at most 15%;
- usable analysis speed covers at least the configured fraction of the lap;
- there are no excessive sampling gaps;
- there are no timestamp regressions.

The fastest valid lap enabled by `use_for_centreline` becomes the reference lap.

## 4. Reference centreline

The reference-lap coordinates receive a short five-sample median followed by a
three-sample mean. This removes metre-scale GNSS zig-zag while never averaging
spatially adjacent but temporally distant branches. The two lap endpoints are
replaced by their midpoint, duplicate nodes are removed, and the path is
resampled at `centreline_spacing_m`.

Elevation is interpolated onto the same nodes and retained as reference metadata.

## 5. Progress-aware map matching

For every point, all line-segment projections are available. A point is selected
using

\[
J = d_\perp^2 + (0.08\,\Delta s)^2,
\]

where \(d_\perp\) is lateral map error and \(\Delta s\) is error relative to the
speed-integrated progress estimate for that lap. The progress term prevents a
point from jumping to a nearby but non-consecutive branch at a crossover.

After matching, a lap is rejected when its 95th-percentile map error exceeds the
configured maximum or it contains a large backward jump in \(s\).

## 6. Spatial track profile

Each valid gate-evidence lap is deduplicated in \(s\), wrapped across start/finish,
and interpolated onto the common profile grid. The output reports p10, median,
and p90 speed and elevation at every grid point, together with the contributing
lap count.

## 7. Ordered event projection

Each event anchor generates several distinct centreline candidates. A dynamic
program chooses the sequence of candidates that minimizes squared projection
error while heavily penalizing backward course order. The first declared event
is expected to be the lap gate and is allowed to project exactly to \(s=0\).

Explicit start/end coordinates are projected near the selected anchor. Otherwise,
configured before/after extents define the feature interval. Anchor, endpoint,
and extent uncertainty are preserved independently. For an extent-derived start,
the effective physical-start error combines anchor projection error, declared
anchor uncertainty, and declared before-anchor extent uncertainty in quadrature.
An explicit endpoint instead combines its own projection error and coordinate
uncertainty. The same treatment is applied to the physical end.

The output exposes anchor and endpoint projection errors, declared uncertainty,
physical-boundary effective errors, provenance, the nearest alternative branch,
geometry source, and review flags. Gate coordinate quality uses the physical-start
effective error because the entry window is defined relative to that boundary.

## 8. Response groups and pass metrics

Physical events sharing one `response_group_id` are unioned into a single
analysis interval, while their map identities remain intact. For each valid lap,
the following are measured:

- median approach speed;
- median entry speed immediately before physical start;
- minimum speed in the physical interval and its location;
- median exit speed;
- entry elevation;
- approach-to-entry braking drop;
- distance after physical end required to recover 98% of entry speed.

The exact approach, entry, exit, and recovery windows are project settings in
`track.toml`, not hidden constants.
