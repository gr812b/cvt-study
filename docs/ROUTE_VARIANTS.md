# Shared course with alternate route branches

Route handling has three distinct levels.

1. **Strict line clusters** conservatively separate noticeably different whole-lap
   driving lines before consensus.
2. **Traversal families** merge strict clusters that represent the same complete
   long/short traversal. This absorbs ordinary line choice.
3. **Branch-network detection** compares supported traversals locally and identifies
   only sustained divergence/re-merge corridors as genuinely different course
   geometry.

The result is one physical course network, not several independent tracks.

## Shared-course contract

Outside a detected branch corridor, all supported traversal laps belong to the same
course section. Their locally compatible evidence is pooled regardless of which
long/short traversal the lap used elsewhere.

Inside a branch corridor, evidence is branch-specific. A lap can support the short
branch or the long branch only if it actually traversed that local geometry.

For example, with 21 short-route laps and 14 long-route laps:

```text
shared backbone                 -> 35 laps of available support
short divergence branch         -> 21 laps
long divergence branch          -> 14 laps
shared backbone after re-merge  -> 35 laps again
```

This is the intended interpretation of an endurance course where one section has
an alternate long/short route.

## Genuine branch detection

A branch must be a **large and sustained** path separation. The default separation
threshold is `max(25 m, 2.5 × same_variant_p95_distance_m)`. The separation must be
sustained for at least 60 m, and short close-approach gaps may be bridged inside the
same branch corridor.

The detector therefore does not classify ordinary inside/outside turn choice as a
branch merely because two lap centrelines are several metres apart around a corner.

The branch is bounded by a divergence point and a later re-merge point. A small
padding distance is included around those boundaries so events immediately at the
split/rejoin are treated conservatively as branch-local.

## Event and gate evidence

Every response feature is explicitly classified per traversal as either:

- `shared`: outside all genuine divergence corridors;
- `branch`: inside a detected branch corridor.

For a shared event, all data-quality-valid laps belonging to any supported branch
may contribute if the local event window has adequate projection coverage.

For a branch event, local projection compatibility is still required, and the
source lap must belong to that target branch. This prevents a long-route lap from
being interpolated across a short-route obstacle or vice versa.

## Complete traversal bundles

The simulator currently consumes one continuous distance coordinate `s`, so a full
bundle is still exported for each complete branch choice:

```text
route_001 -> shared course + long branch + shared course
route_002 -> shared course + short branch + shared course
```

These are two traversals through one course network. The route-family manifest and
branch audit make that relationship explicit.

Full uncertainty and design studies continue to treat the complete branch choices
as equal course cases rather than weighting them by observed lap count.

## Configuration

Whole-lap line clustering and nominal selection remain explicit:

```toml
[track.route_variants]
enabled = true
minimum_supported_laps = 2
sample_spacing_m = 5.0
same_variant_p95_distance_m = 10.0
divergence_distance_m = 15.0
maximum_divergent_fraction = 0.03
maximum_length_relative_difference = 0.08
maximum_within_variant_length_deviation_fraction = 0.15
selection = "largest_supported"
```

Branch-network settings normally need no project-specific override:

```toml
[track.route_variants.family]
enabled = true
merge_line_clusters = true
pool_shared_gate_evidence = true
minimum_event_window_coverage_fraction = 0.60
event_projection_bin_size_m = 5.0
maximum_event_interpolation_gap_m = 25.0
maximum_local_backward_matches = 1

# Optional branch overrides:
# branch_divergence_distance_m = 25.0
# branch_minimum_sustained_length_m = 60.0
# branch_gap_tolerance_m = 25.0
# branch_boundary_padding_m = 15.0

# Optional topology-cluster merge overrides:
# topology_same_route_p95_distance_m = 15.0
# topology_maximum_divergent_fraction = 0.03
# topology_maximum_length_relative_difference = 0.08
# topology_attachment_ambiguity_ratio = 0.80
```

Do not loosen the strict line-cluster threshold just to reduce cluster count. Strict
clusters are an audit/seed mechanism. The topology and branch stages are responsible
for determining physical course equivalence.

### Local event-evidence compatibility

Shared-course evidence uses the same interpolation idea as the event extractor. A
short analysis window does **not** need a raw telemetry sample to land inside it.
Instead, consecutive acceptable projected samples may bracket the window, provided
the spatial gap is no larger than `maximum_event_interpolation_gap_m`. Invalid
projection samples and large gaps break the bracket, so the method cannot bridge a
missing branch or telemetry dropout.

Once a sustained branch corridor is resolved, shared-course event projection gets a
modestly relaxed tolerance so ordinary racing-line variation is not mistaken for a
missing event. The default shared-course limit is the larger of the strict event
limit and `min(1.5 × strict_limit, 0.60 × branch_divergence_threshold)`. Branch-local
events keep the strict route-specific limit.

This distinction is visible in `event_route_applicability.csv` and the per-pass
`route_projection_*` audit columns.

## Audit outputs

A route-family build writes:

```text
route_family_manifest.json
track/
├── route_variant_summary.csv
├── route_variant_pairwise.csv
├── route_branch_summary.csv
├── event_route_sections.csv
├── shared_gate_evidence.csv
├── event_route_applicability.csv
├── route_course_cases.csv
├── route_001/
│   └── ...
└── route_002/
    └── ...
```

`route_branch_summary.csv` contains the divergence/re-merge intervals on each
complete traversal and the shared/branch lap support counts.

`event_route_sections.csv` labels each response feature as shared or branch-local.

`shared_gate_evidence.csv` reports the unique lap count after branch-aware pooling.

## Shared-course gate contracts and route-family robustness

When supported traversals diverge only through a sustained branch corridor, events
outside that corridor are physical shared-course events. Their gate evidence is
therefore deduplicated by physical event and lap, scored once, and reused on every
complete traversal. The route bundles retain their own `s` positions/windows, but
the empirical target distribution and confidence/recommendation are identical for
the same shared physical event. Events overlapping the divergence corridor retain
branch-specific evidence.

Track robustness follows the same model. Fine/coarse centreline-spacing cases hold
the physical smoothing width approximately fixed, while explicit less/more
smoothing cases change that width. The normal command
`drivetrain-study run track-robustness <project>` automatically expands over the
supported route family and writes a top-level family robustness report and balanced
track-ensemble manifest for the subsequent full-uncertainty study.
