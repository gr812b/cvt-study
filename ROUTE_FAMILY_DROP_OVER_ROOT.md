# Route-family / branch-aware overlay — revision 11

Extract this archive directly at the root of `cvt-study` and allow `src/`,
`tests/` and `docs/` to merge and overwrite.

## Revision 11: final track-robustness cleanup

Revision 11 keeps the long/short course model from revisions 6–10 and makes the
remaining robustness/evidence semantics production-facing.

### Shared-course gates are now one contract

A physical event outside a sustained divergence/re-merge corridor is no longer
scored independently on the long and short complete traversals. The family layer:

1. maps every supported lap locally against every route;
2. keeps one best locally compatible observation per `(event_id, lap_id)`;
3. scores that deduplicated shared pass set once;
4. reuses the same target-speed distribution, confidence score and recommendation
   on every complete traversal bundle;
5. restores each route's own `s` position/window geometry when the gate is written
   into that route bundle.

Events whose analysis window overlaps the genuine branch corridor remain
route-specific.

This removes cases where the same shared physical event could be `accepted` on one
traversal and `recommended_review` on the other merely because its declared point
sat a few metres closer to one route centreline.

### Centreline spacing no longer changes smoothing by accident

The nominal smoother is specified in nodes. Previously, changing centreline node
spacing while leaving the smoothing-node count fixed also changed the *physical*
smoothing width. Fine/coarse spacing therefore partly duplicated the explicit
less/more-smoothing tests.

The fine/coarse spacing robustness cases now change the odd smoothing-node count
inversely so physical smoothing stays near nominal. Less/more smoothing remain the
only cases intended to alter smoothing strength.

### Normal CLI now runs route-family robustness

When `[track.route_variants] enabled = true` and route-family handling is enabled,
this ordinary framework command automatically runs the family-aware study:

```powershell
drivetrain-study run track-robustness .\projects\maryland
```

It produces one top-level:

```text
track_robustness_report.html
track_robustness_manifest.json
track_ensemble_manifest.json
route_family_robustness_case_summary.csv
route_family_course_cases.csv
```

with full per-route reports under `route_001/`, `route_002/`, etc. The old
`run_route_family_track_robustness.py` launcher remains only as a compatibility
helper; it is no longer required.

### Full uncertainty uses a balanced route-family ensemble

The normal uncertainty command is also the intended path:

```powershell
drivetrain-study run full-uncertainty .\projects\maryland
```

The latest family robustness result is exposed through the standard
`track_ensemble_manifest.json`. Only robustness case IDs that are downstream-
eligible on *every* supported route are propagated. If
`track_ensemble.maximum_cases` limits the number of cases, the loader selects the
same interpretation groups on every route so a long/short route cannot receive
more weight simply because of flat list truncation.

The selection order prefers physical reconstruction diversity (centreline,
telemetry cleanup, event windows) before policy-only gate-weighting alternatives.
Each route remains an equal course case; observed lap count does not become a
probability weight.

## Build and recommended sequence

```powershell
drivetrain-study build-track .\projects\maryland

drivetrain-study run track-robustness .\projects\maryland

drivetrain-study run full-uncertainty .\projects\maryland
```

`build-track` should show the shared backbone, local branch corridor, shared/branch
event classification and event evidence counts directly in
`review/track_evidence_report.html`.

## Key audit artifacts

Track build:

```text
track/route_branch_summary.csv
track/event_route_sections.csv
track/event_route_applicability.csv
track/shared_gate_evidence.csv
track/route_001/event_passes.csv
track/route_002/event_passes.csv
```

Track robustness:

```text
track_robustness_report.html
route_family_course_cases.csv
route_family_robustness_case_summary.csv
track_ensemble_manifest.json
route_001/track_robustness_report.html
route_002/track_robustness_report.html
```

## Retained semantics

- strict whole-lap clusters are only seeds/audit evidence;
- ordinary driving-line variation is absorbed into topology families;
- a route branch requires a large sustained divergence followed by a re-merge;
- shared sections pool evidence across all supported traversals;
- branch sections use only laps that actually traversed that branch;
- complete route bundles remain separate because the simulator requires one
  continuous `s` coordinate per simulated course case;
- routes are equal/unweighted course cases in uncertainty/design aggregation.
