# Physical-feature and map-input contract

`track/events.toml` defines physical features in course order. Physical features
stay separate even when several features share one measured GPS response.

## Required event fields

```toml
[[events]]
id = "logs_01"
name = "Logs"
sequence = 14
kind = "interval"
analysis_role = "feature"
response_group_id = "logs_01"
gate_candidate = true
notes = "First log to last log."

[events.anchor]
latitude_deg = 43.000100
longitude_deg = -79.000200
horizontal_uncertainty_m = 4.0
source = "video frame matched to satellite map"

[events.extent]
before_anchor_m = 3.0
after_anchor_m = 8.0
before_anchor_uncertainty_m = 1.5
after_anchor_uncertainty_m = 2.0
source = "estimated from video and known vehicle length"
```

Meanings:

- `id`: permanent physical-feature identifier;
- `sequence`: unique positive course-order index;
- `kind`: `point`, `turn`, `interval`, or `obstacle`;
- `analysis_role`: normally `feature`; one event may be the `lap_gate`;
- `response_group_id`: features with inseparable GPS response share this value;
- `gate_candidate`: whether this response may constrain simulated entry speed;
- `anchor`: recognizable map/video reference, not necessarily physical contact;
- `extent`: physical start/end offsets around the anchor when explicit endpoint
  coordinates are unavailable.

## Geometry uncertainty is mandatory

The anchor coordinate, physical-start location, and physical-end location can each
be uncertain. That uncertainty matters because pass metrics are measured relative
to the physical interval, not merely relative to a convenient map marker.

`anchor.horizontal_uncertainty_m` describes uncertainty in where the intended
anchor lies relative to the supplied coordinate. When start/end are constructed
from offsets, each offset also requires a declared uncertainty and source. A value
of zero is a conscious fixed-value declaration and requires `fixed_reason`:

```toml
[events.anchor]
latitude_deg = 43.0
longitude_deg = -79.0
horizontal_uncertainty_m = 0.0
fixed_reason = "surveyed control point"
source = "total-station survey"

[events.extent]
before_anchor_m = 3.0
after_anchor_m = 8.0
before_anchor_uncertainty_m = 0.0
after_anchor_uncertainty_m = 0.0
fixed_reason = "boundaries surveyed from the same control network"
source = "total-station survey"
```

The current implementation carries these declarations into map-quality evidence.
Random coordinate and extent sampling belongs to the later joint-uncertainty
phase.

## Explicit physical start and end

Explicit endpoint coordinates take precedence over extent offsets:

```toml
[events.start]
latitude_deg = 43.000080
longitude_deg = -79.000180
horizontal_uncertainty_m = 2.0
source = "first-contact location identified from synchronized video"

[events.end]
latitude_deg = 43.000130
longitude_deg = -79.000240
horizontal_uncertainty_m = 3.0
source = "last-contact location identified from synchronized video"
```

Explicit endpoints require their own uncertainty and source because an accurate
anchor does not imply equally accurate physical boundaries. An extent value is
required only for a side whose explicit endpoint is absent. Therefore, an event
with both explicit endpoints needs no extent table; an event with only an explicit
start still needs `after_anchor_m`, its uncertainty, and an extent source.

Start and end are physical feature boundaries. Entry and exit measurement windows
are calculated by the pipeline from project-wide settings and are not entered
manually for each event.

## Effective physical-boundary error

For an extent-derived start, the declared uncertainty is

\[
u_{\mathrm{start}} =
\sqrt{u_{\mathrm{anchor}}^2 + u_{\mathrm{before}}^2}.
\]

The map-review effective error also includes the anchor-to-centreline projection
error:

\[
e_{\mathrm{start}} =
\sqrt{e_{\mathrm{anchor,projection}}^2 +
      u_{\mathrm{anchor}}^2 + u_{\mathrm{before}}^2}.
\]

For an explicit start coordinate,

\[
e_{\mathrm{start}} =
\sqrt{e_{\mathrm{start,projection}}^2 + u_{\mathrm{start}}^2}.
\]

Equivalent expressions apply to the physical end. These values are exported so
the user can see whether map projection, anchor uncertainty, or extent uncertainty
is controlling the review result.

## Response groups

A response group is a measurement statement, not a geometry merge. For example,
three adjacent physical features can remain three map objects while sharing one
entry-speed distribution:

```toml
response_group_id = "turn_slalom_turn_12_14"
```

The review output retains every physical feature in `event_projection.csv` and
creates one union interval in `response_features.csv`. Non-adjacent or very long
response groups receive review flags. The group inherits the start uncertainty
from the physical member defining its earliest start and the end uncertainty from
the member defining its latest end.
