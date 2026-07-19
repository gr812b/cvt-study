# Telemetry cleanup contract

Telemetry cleanup protects lap reconstruction and review plots from isolated GPS
excursions while preserving the original FIT/GPX source and a complete audit
trail.

## What is removed automatically

A short group of at most `maximum_excursion_points` is removed before lap
detection only when:

1. the jump into the group is physically impossible;
2. the jump back is physically impossible;
3. connecting the valid points on either side is physically plausible;
4. the bridge occurs within `maximum_bridge_gap_s`; and
5. the total number of proposed removals stays below both configured safety
   limits.

After the centreline is built, a second pass can remove a short map-error burst
only when both neighbouring points are already close to the centreline.

## What is never repaired automatically

- sustained off-course travel;
- long GPS outages;
- a whole lap displaced from the centreline;
- an excursion at the start or end of a segment;
- a large number of suspect points;
- ordinary speed noise without a coordinate excursion.

No replacement coordinates are interpolated and no point is snapped onto the
track.

## Configuration

Add this under `[track]` in `track/track.toml`:

```toml
[track.telemetry_cleanup]
enabled = true
maximum_excursion_points = 3
minimum_excursion_leg_m = 35.0
impossible_speed_multiplier = 1.5
maximum_bridge_speed_multiplier = 1.0
maximum_bridge_gap_s = 8.0
maximum_auto_removed_fraction = 0.005
maximum_auto_removed_points = 25
isolated_map_error_m = 40.0
maximum_isolated_map_outlier_points = 3
```

The speed multipliers use
`track.reconstruction.maximum_reasonable_speed_mps`. The post-map threshold
must exceed `track.reconstruction.maximum_map_error_m`.

## Outputs

Ingestion results contain:

- `canonical_points.csv` — cleaned points used downstream;
- `rejected_telemetry_points.csv` — every pre-lap exclusion and its metrics;
- `telemetry_cleanup_map.png` — retained points plus explicit exclusions;
- run summaries with raw, cleaned, candidate, and removed point counts.

Track builds additionally contain:

- `track/rejected_map_points.csv`;
- `review/telemetry_cleanup_map.png`;
- cleanup counts in `track_build_manifest.json`.

The source FIT/GPX files and their SHA-256 provenance remain unchanged.
