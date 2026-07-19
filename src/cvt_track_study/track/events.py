"""Physical-event normalization, ordered map projection, and response grouping."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .geo import Centreline
from .settings import ReconstructionSettings


@dataclass(frozen=True)
class _EndpointProjection:
    """One plausible projection of an explicit event endpoint."""

    s_m: float
    error_m: float
    declared_uncertainty_m: float
    provenance: str
    original_endpoint: str


def normalize_events(
    raw_events: list[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(raw_events):
        anchor = event.get("anchor", {})
        start = event.get("start", {})
        end = event.get("end", {})
        extent = event.get("extent", {})
        rows.append(
            {
                "id": str(event.get("id", "")),
                "name": str(
                    event.get("name", event.get("id", ""))
                ),
                "sequence": int(
                    event.get("sequence", index + 1)
                ),
                "kind": str(event.get("kind", "point")),
                "analysis_role": str(
                    event.get("analysis_role", "feature")
                ),
                "response_group_id": str(
                    event.get(
                        "response_group_id",
                        event.get("id", ""),
                    )
                ),
                "gate_candidate": bool(
                    event.get("gate_candidate", True)
                ),
                # Closed-course wrapping is normally inferred only when
                # the resolved interval is physically near s=0/L. This
                # explicit opt-in exists for unusual genuine wraps.
                "allow_start_finish_wrap": bool(
                    event.get(
                        "allow_start_finish_wrap", False
                    )
                ),
                "anchor_latitude_deg": float(
                    anchor.get("latitude_deg")
                ),
                "anchor_longitude_deg": float(
                    anchor.get("longitude_deg")
                ),
                "anchor_horizontal_uncertainty_m": float(
                    anchor.get(
                        "horizontal_uncertainty_m", 10.0
                    )
                ),
                "anchor_source": str(
                    anchor.get("source", "unspecified")
                ),
                "start_latitude_deg": _optional_float(
                    start.get("latitude_deg")
                ),
                "start_longitude_deg": _optional_float(
                    start.get("longitude_deg")
                ),
                "start_horizontal_uncertainty_m": (
                    _optional_float(
                        start.get(
                            "horizontal_uncertainty_m"
                        )
                    )
                ),
                "start_source": str(
                    start.get("source", "")
                ),
                "end_latitude_deg": _optional_float(
                    end.get("latitude_deg")
                ),
                "end_longitude_deg": _optional_float(
                    end.get("longitude_deg")
                ),
                "end_horizontal_uncertainty_m": (
                    _optional_float(
                        end.get(
                            "horizontal_uncertainty_m"
                        )
                    )
                ),
                "end_source": str(end.get("source", "")),
                "feature_before_m": _optional_float(
                    extent.get("before_anchor_m")
                ),
                "feature_after_m": _optional_float(
                    extent.get("after_anchor_m")
                ),
                "feature_before_uncertainty_m": (
                    _optional_float(
                        extent.get(
                            "before_anchor_uncertainty_m"
                        )
                    )
                ),
                "feature_after_uncertainty_m": (
                    _optional_float(
                        extent.get(
                            "after_anchor_uncertainty_m"
                        )
                    )
                ),
                "feature_extent_source": str(
                    extent.get("source", "")
                ),
                "notes": str(event.get("notes", "")),
                "obstacle_model": event.get(
                    "obstacle_model", {}
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("sequence")
        .reset_index(drop=True)
    )


def _find_lap_gate(
    events: pd.DataFrame,
    configured_id: str,
) -> pd.Series:
    if configured_id:
        match = events[events["id"] == configured_id]
    else:
        match = events[
            events["analysis_role"] == "lap_gate"
        ]
    if len(match) != 1:
        raise ValueError(
            "Exactly one lap-gate event must be selected by "
            "track.reconstruction.lap_gate_event_id or "
            "analysis_role='lap_gate'."
        )
    return match.iloc[0]


def project_events(
    events: pd.DataFrame,
    centreline: Centreline,
    settings: ReconstructionSettings,
) -> pd.DataFrame:
    """Project event evidence onto the already-built consensus track.

    Event coordinates never move or reshape the centreline. Anchor
    sequence resolves which track branch each event belongs to; explicit
    interval endpoints are then resolved jointly using geometric error
    and forward course direction.
    """

    x, y = centreline.frame.to_xy(
        events["anchor_latitude_deg"],
        events["anchor_longitude_deg"],
    )
    candidate_lists = [
        centreline.distinct_candidates(
            float(px), float(py)
        )
        for px, py in zip(x, y)
    ]
    if candidate_lists:
        near_start = [
            candidate
            for candidate in candidate_lists[0]
            if min(
                candidate["s_m"],
                centreline.length_m
                - candidate["s_m"],
            )
            <= 50.0
        ]
        if near_start:
            best = min(
                near_start,
                key=lambda item: item["error_m"],
            )
            synthetic = dict(best)
            synthetic["s_m"] = 0.0
            candidate_lists[0] = [synthetic] + [
                item
                for item in candidate_lists[0]
                if min(
                    abs(item["s_m"] - best["s_m"]),
                    centreline.length_m
                    - abs(
                        item["s_m"] - best["s_m"]
                    ),
                )
                >= 18.0
            ]

    chosen_indices = _choose_ordered_candidates(
        candidate_lists
    )
    rows: list[dict[str, Any]] = []

    for (_, event), candidates, chosen_index in zip(
        events.iterrows(),
        candidate_lists,
        chosen_indices,
    ):
        chosen = candidates[chosen_index]
        alternatives = [
            item
            for index, item in enumerate(candidates)
            if index != chosen_index
        ]
        alternative = (
            min(
                alternatives,
                key=lambda item: item["error_m"],
            )
            if alternatives
            else None
        )
        latitude, longitude = (
            centreline.frame.to_latlon(
                [chosen["x_m"]],
                [chosen["y_m"]],
            )
        )

        endpoint_flags: list[str] = []
        if (
            str(event["kind"]) == "interval"
            and _has_explicit_endpoint(event, "start")
            and _has_explicit_endpoint(event, "end")
        ):
            (
                start_values,
                end_values,
                endpoint_flags,
            ) = _resolve_explicit_interval_endpoints(
                event,
                float(chosen["s_m"]),
                centreline,
                settings,
            )
        else:
            start_values = _event_endpoint_relative_s(
                event,
                "start",
                float(chosen["s_m"]),
                centreline,
            )
            end_values = _event_endpoint_relative_s(
                event,
                "end",
                float(chosen["s_m"]),
                centreline,
            )

        (
            start_rel,
            start_source,
            start_projection_error,
            start_declared_uncertainty,
            start_provenance,
        ) = start_values
        (
            end_rel,
            end_source,
            end_projection_error,
            end_declared_uncertainty,
            end_provenance,
        ) = end_values

        if start_rel is None:
            start_rel = -float(
                event["feature_before_m"]
            )
            start_source = "configured_extent"
            start_projection_error = float(
                chosen["error_m"]
            )
            start_declared_uncertainty = math.hypot(
                float(
                    event[
                        "anchor_horizontal_uncertainty_m"
                    ]
                ),
                float(
                    event[
                        "feature_before_uncertainty_m"
                    ]
                ),
            )
            start_provenance = str(
                event["feature_extent_source"]
            )

        if end_rel is None:
            end_rel = float(event["feature_after_m"])
            end_source = "configured_extent"
            end_projection_error = float(
                chosen["error_m"]
            )
            end_declared_uncertainty = math.hypot(
                float(
                    event[
                        "anchor_horizontal_uncertainty_m"
                    ]
                ),
                float(
                    event[
                        "feature_after_uncertainty_m"
                    ]
                ),
            )
            end_provenance = str(
                event["feature_extent_source"]
            )

        # Jointly resolved explicit intervals already satisfy this.
        # Retain the old fallback only for configured extents and
        # partially explicit legacy events.
        if end_rel <= start_rel:
            end_rel += centreline.length_m

        flags: list[str] = list(endpoint_flags)
        if (
            chosen["error_m"]
            > settings.maximum_map_error_m
        ):
            flags.append(
                "anchor_far_from_centreline"
            )
        if (
            alternative
            and alternative["error_m"]
            - chosen["error_m"]
            < 3.0
        ):
            flags.append(
                "multiple_nearby_track_branches"
            )
        if (
            str(event["kind"]) == "interval"
            and start_source
            == end_source
            == "configured_extent"
        ):
            flags.append(
                "interval_extent_estimated"
            )
        if (
            np.isfinite(start_projection_error)
            and start_projection_error
            > settings.maximum_map_error_m
        ):
            flags.append(
                "start_endpoint_far_from_centreline"
            )
        if (
            np.isfinite(end_projection_error)
            and end_projection_error
            > settings.maximum_map_error_m
        ):
            flags.append(
                "end_endpoint_far_from_centreline"
            )

        rows.append(
            {
                **event.to_dict(),
                "anchor_s_m": float(chosen["s_m"]),
                "anchor_projection_error_m": float(
                    chosen["error_m"]
                ),
                "projected_latitude_deg": float(
                    latitude[0]
                ),
                "projected_longitude_deg": float(
                    longitude[0]
                ),
                "alternative_s_m": (
                    float(alternative["s_m"])
                    if alternative
                    else math.nan
                ),
                "alternative_projection_error_m": (
                    float(alternative["error_m"])
                    if alternative
                    else math.nan
                ),
                "feature_start_rel_m": float(
                    start_rel
                ),
                "feature_end_rel_m": float(end_rel),
                "feature_start_source": start_source,
                "feature_start_provenance": (
                    start_provenance
                ),
                "feature_start_projection_error_m": (
                    float(start_projection_error)
                ),
                "feature_start_horizontal_uncertainty_m": (
                    float(start_declared_uncertainty)
                ),
                "feature_start_effective_error_m": (
                    math.hypot(
                        float(start_projection_error),
                        float(
                            start_declared_uncertainty
                        ),
                    )
                ),
                "feature_end_source": end_source,
                "feature_end_provenance": (
                    end_provenance
                ),
                "feature_end_projection_error_m": (
                    float(end_projection_error)
                ),
                "feature_end_horizontal_uncertainty_m": (
                    float(end_declared_uncertainty)
                ),
                "feature_end_effective_error_m": (
                    math.hypot(
                        float(end_projection_error),
                        float(
                            end_declared_uncertainty
                        ),
                    )
                ),
                "approach_start_rel_m": float(
                    start_rel
                    - settings.approach_before_m
                ),
                "approach_end_rel_m": float(
                    start_rel
                    - settings.approach_gap_m
                ),
                "entry_start_rel_m": float(
                    start_rel
                    - settings.entry_before_m
                ),
                "entry_end_rel_m": float(
                    start_rel
                    - settings.entry_gap_m
                ),
                "exit_start_rel_m": float(
                    end_rel + settings.exit_gap_m
                ),
                "exit_end_rel_m": float(
                    end_rel
                    + settings.exit_gap_m
                    + settings.exit_length_m
                ),
                "recovery_limit_m": (
                    settings.recovery_limit_m
                ),
                "review_flags": ";".join(
                    dict.fromkeys(flags)
                ),
            }
        )

    resolved = (
        pd.DataFrame(rows)
        .sort_values("sequence")
        .reset_index(drop=True)
    )
    adjacency_pairs = [
        (index, index + 1, 0.0)
        for index in range(len(resolved) - 1)
    ]
    if len(resolved) > 1:
        adjacency_pairs.append(
            (
                len(resolved) - 1,
                0,
                centreline.length_m,
            )
        )

    for (
        current_index,
        next_index,
        next_lap_offset,
    ) in adjacency_pairs:
        current_end = (
            float(
                resolved.loc[
                    current_index, "anchor_s_m"
                ]
            )
            + float(
                resolved.loc[
                    current_index,
                    "feature_end_rel_m",
                ]
            )
        )
        next_start = (
            float(
                resolved.loc[
                    next_index, "anchor_s_m"
                ]
            )
            + float(
                resolved.loc[
                    next_index,
                    "feature_start_rel_m",
                ]
            )
            + next_lap_offset
        )
        same_group = (
            str(
                resolved.loc[
                    current_index,
                    "response_group_id",
                ]
            )
            == str(
                resolved.loc[
                    next_index,
                    "response_group_id",
                ]
            )
        )
        if next_start < current_end and not same_group:
            for row_index in (
                current_index,
                next_index,
            ):
                _append_review_flag(
                    resolved,
                    row_index,
                    "overlaps_adjacent_feature",
                )
    return resolved


def _append_review_flag(
    frame: pd.DataFrame,
    row_index: int,
    flag: str,
) -> None:
    flags = [
        item
        for item in str(
            frame.loc[row_index, "review_flags"]
        ).split(";")
        if item
    ]
    if flag not in flags:
        flags.append(flag)
        frame.loc[
            row_index, "review_flags"
        ] = ";".join(flags)


def _choose_ordered_candidates(
    candidate_lists: list[
        list[dict[str, float]]
    ],
) -> list[int]:
    if not candidate_lists or any(
        not items for items in candidate_lists
    ):
        raise ValueError(
            "At least one event cannot be projected "
            "onto the centreline."
        )

    costs: list[np.ndarray] = []
    back: list[np.ndarray] = []
    first = candidate_lists[0]
    first_s = np.array(
        [item["s_m"] for item in first]
    )
    first_error = np.array(
        [item["error_m"] for item in first]
    )
    costs.append(
        first_error**2 + (first_s / 5.0) ** 2
    )
    back.append(
        np.full(len(first), -1, dtype=int)
    )

    for index in range(
        1, len(candidate_lists)
    ):
        previous_s = np.array(
            [
                item["s_m"]
                for item in candidate_lists[
                    index - 1
                ]
            ]
        )
        current = candidate_lists[index]
        current_cost = np.full(
            len(current), np.inf
        )
        current_back = np.full(
            len(current), -1, dtype=int
        )
        for candidate_index, item in enumerate(
            current
        ):
            ds = item["s_m"] - previous_s
            penalty = np.where(
                ds >= -3.0,
                0.0,
                1_000_000.0
                + (-ds) * 10_000.0,
            )
            trial = costs[-1] + penalty
            prior = int(np.argmin(trial))
            current_cost[candidate_index] = (
                trial[prior]
                + item["error_m"] ** 2
            )
            current_back[candidate_index] = prior
        costs.append(current_cost)
        back.append(current_back)

    chosen = [
        int(np.argmin(costs[-1]))
    ]
    for index in range(
        len(candidate_lists) - 1, 0, -1
    ):
        chosen.append(
            int(back[index][chosen[-1]])
        )
    return list(reversed(chosen))


def _resolve_explicit_interval_endpoints(
    event: pd.Series,
    anchor_s_m: float,
    centreline: Centreline,
    settings: ReconstructionSettings,
) -> tuple[
    tuple[float, str, float, float, str],
    tuple[float, str, float, float, str],
    list[str],
]:
    """Resolve both explicit endpoint coordinates as one local interval.

    Candidate filtering is driven first by geometric projection error.
    Course direction and interval length then disambiguate nearby track
    branches. This prevents a geographically poor endpoint candidate
    near the anchor's s coordinate from becoming a nearly full-lap
    interval.
    """

    start_candidates = _endpoint_candidates(
        event, "start", centreline
    )
    end_candidates = _endpoint_candidates(
        event, "end", centreline
    )
    if not start_candidates or not end_candidates:
        raise ValueError(
            f"Explicit interval {event.get('id', '')!r} "
            "has no centreline projection candidates."
        )

    declared_distance_m = (
        _declared_endpoint_distance_m(
            event, centreline
        )
    )
    allow_wrap = bool(
        event.get(
            "allow_start_finish_wrap", False
        )
    )
    track_length_m = centreline.length_m

    evaluations: list[
        tuple[
            float,
            _EndpointProjection,
            _EndpointProjection,
            bool,
            bool,
            float,
        ]
    ] = []
    for reordered, starts, ends in (
        (
            False,
            start_candidates,
            end_candidates,
        ),
        (
            True,
            end_candidates,
            start_candidates,
        ),
    ):
        for start in starts:
            for end in ends:
                forward_length_m = (
                    _forward_distance_m(
                        start.s_m,
                        end.s_m,
                        track_length_m,
                    )
                )
                wraps = (
                    end.s_m < start.s_m - 1e-9
                )
                near_boundary = (
                    _near_start_finish(
                        start.s_m,
                        track_length_m,
                    )
                    or _near_start_finish(
                        end.s_m,
                        track_length_m,
                    )
                    or _near_start_finish(
                        anchor_s_m,
                        track_length_m,
                    )
                )
                wrap_is_plausible = (
                    not wraps
                    or allow_wrap
                    or near_boundary
                )
                score = _interval_candidate_score(
                    start,
                    end,
                    forward_length_m,
                    declared_distance_m,
                    track_length_m,
                    reordered=reordered,
                    wrap_is_plausible=(
                        wrap_is_plausible
                    ),
                )
                evaluations.append(
                    (
                        score,
                        start,
                        end,
                        reordered,
                        wraps,
                        forward_length_m,
                    )
                )

    evaluations.sort(key=lambda item: item[0])
    (
        best_score,
        start,
        end,
        reordered,
        wraps,
        forward_length_m,
    ) = evaluations[0]

    start_rel_m = _signed_circular_delta_m(
        start.s_m - anchor_s_m,
        track_length_m,
    )
    end_rel_m = (
        start_rel_m + forward_length_m
    )
    source = (
        "explicit_coordinate_reordered"
        if reordered
        else "explicit_coordinate"
    )

    flags: list[str] = []
    if reordered:
        flags.append(
            "explicit_interval_endpoints_reordered"
        )
    if wraps:
        flags.append(
            "interval_wraps_start_finish"
        )
        if not allow_wrap:
            flags.append(
                "interval_wrap_inferred_near_start_finish"
            )
    if forward_length_m > 150.0:
        flags.append("interval_extent_very_long")
    if (
        forward_length_m
        > 0.25 * track_length_m
    ):
        flags.append(
            "interval_extent_implausibly_long"
        )
    if (
        start.error_m
        > settings.maximum_map_error_m
        or end.error_m
        > settings.maximum_map_error_m
    ):
        flags.append(
            "interval_endpoint_projection_far_from_centreline"
        )

    if len(evaluations) > 1:
        next_best = evaluations[1]
        materially_different = (
            abs(
                next_best[5]
                - forward_length_m
            )
            > 5.0
            or next_best[3] != reordered
        )
        if (
            materially_different
            and next_best[0] - best_score < 3.0
        ):
            flags.append(
                "interval_projection_ambiguous"
            )

    start_values = (
        float(start_rel_m),
        source,
        float(start.error_m),
        float(
            start.declared_uncertainty_m
        ),
        str(start.provenance),
    )
    end_values = (
        float(end_rel_m),
        source,
        float(end.error_m),
        float(end.declared_uncertainty_m),
        str(end.provenance),
    )
    return start_values, end_values, flags


def _interval_candidate_score(
    start: _EndpointProjection,
    end: _EndpointProjection,
    forward_length_m: float,
    declared_distance_m: float,
    track_length_m: float,
    *,
    reordered: bool,
    wrap_is_plausible: bool,
) -> float:
    # Geometric fit is primary.
    score = (
        start.error_m**2
        + end.error_m**2
    )

    # Among similarly good geometric projections, prefer the local
    # course interval rather than a route around most of the lap.
    length_scale_m = max(
        100.0, 0.10 * track_length_m
    )
    score += (
        forward_length_m / length_scale_m
    ) ** 2

    local_limit_m = min(
        300.0, 0.25 * track_length_m
    )
    if forward_length_m > local_limit_m:
        score += (
            (
                forward_length_m
                - local_limit_m
            )
            / 10.0
        ) ** 2

    # Do not collapse two visibly separated coordinates onto nearly
    # the same point of s.
    if (
        np.isfinite(declared_distance_m)
        and declared_distance_m > 1.0
        and forward_length_m
        < 0.25 * declared_distance_m
    ):
        score += (
            (
                0.25 * declared_distance_m
                - forward_length_m
            )
            / 2.0
        ) ** 2

    if not wrap_is_plausible:
        score += 1_000_000.0

    # Preserve declared order when both interpretations are otherwise
    # equally good, while still allowing a clearly reversed pair to
    # be repaired.
    if reordered:
        score += 0.25
    return float(score)


def _endpoint_candidates(
    event: pd.Series,
    endpoint: str,
    centreline: Centreline,
) -> list[_EndpointProjection]:
    lat = event[
        f"{endpoint}_latitude_deg"
    ]
    lon = event[
        f"{endpoint}_longitude_deg"
    ]
    if not np.isfinite(lat) or not np.isfinite(lon):
        return []

    x, y = centreline.frame.to_xy(
        [lat], [lon]
    )
    raw = centreline.distinct_candidates(
        float(x[0]), float(y[0])
    )
    if not raw:
        return []

    uncertainty_m = _finite_or_default(
        event[
            f"{endpoint}_horizontal_uncertainty_m"
        ],
        3.0,
    )
    minimum_error_m = min(
        float(item["error_m"])
        for item in raw
    )
    error_allowance_m = max(
        3.0, uncertainty_m
    )
    plausible = [
        item
        for item in raw
        if float(item["error_m"])
        <= minimum_error_m
        + error_allowance_m
    ]
    if not plausible:
        plausible = [
            min(
                raw,
                key=lambda item: item["error_m"],
            )
        ]

    return [
        _EndpointProjection(
            s_m=float(item["s_m"]),
            error_m=float(item["error_m"]),
            declared_uncertainty_m=float(
                event[
                    f"{endpoint}_horizontal_uncertainty_m"
                ]
            ),
            provenance=str(
                event[f"{endpoint}_source"]
            ),
            original_endpoint=endpoint,
        )
        for item in plausible
    ]


def _event_endpoint_relative_s(
    event: pd.Series,
    endpoint: str,
    anchor_s_m: float,
    centreline: Centreline,
) -> tuple[
    float | None,
    str,
    float,
    float,
    str,
]:
    """Project one legacy/partial endpoint by geometric fit first."""

    candidates = _endpoint_candidates(
        event, endpoint, centreline
    )
    if not candidates:
        return (
            None,
            "",
            math.nan,
            math.nan,
            "",
        )

    chosen = min(
        candidates,
        key=lambda item: (
            item.error_m,
            _circular_distance_m(
                item.s_m,
                anchor_s_m,
                centreline.length_m,
            ),
        ),
    )
    relative = (
        chosen.s_m - anchor_s_m
    )
    if (
        endpoint == "start"
        and relative
        > centreline.length_m / 2
    ):
        relative -= centreline.length_m
    if (
        endpoint == "end"
        and relative
        < -centreline.length_m / 2
    ):
        relative += centreline.length_m
    return (
        float(relative),
        "explicit_coordinate",
        float(chosen.error_m),
        float(
            chosen.declared_uncertainty_m
        ),
        str(chosen.provenance),
    )


def _has_explicit_endpoint(
    event: pd.Series,
    endpoint: str,
) -> bool:
    return bool(
        np.isfinite(
            event[
                f"{endpoint}_latitude_deg"
            ]
        )
        and np.isfinite(
            event[
                f"{endpoint}_longitude_deg"
            ]
        )
    )


def _declared_endpoint_distance_m(
    event: pd.Series,
    centreline: Centreline,
) -> float:
    x, y = centreline.frame.to_xy(
        [
            event["start_latitude_deg"],
            event["end_latitude_deg"],
        ],
        [
            event["start_longitude_deg"],
            event["end_longitude_deg"],
        ],
    )
    return float(
        math.hypot(
            float(x[1] - x[0]),
            float(y[1] - y[0]),
        )
    )


def _forward_distance_m(
    start_s_m: float,
    end_s_m: float,
    track_length_m: float,
) -> float:
    raw = end_s_m - start_s_m
    if abs(raw) <= 1e-9:
        return 0.0
    return float(raw % track_length_m)


def _signed_circular_delta_m(
    delta_m: float,
    track_length_m: float,
) -> float:
    return float(
        (
            delta_m
            + track_length_m / 2.0
        )
        % track_length_m
        - track_length_m / 2.0
    )


def _circular_distance_m(
    left_s_m: float,
    right_s_m: float,
    track_length_m: float,
) -> float:
    difference = abs(left_s_m - right_s_m)
    return float(
        min(
            difference,
            track_length_m - difference,
        )
    )


def _near_start_finish(
    s_m: float,
    track_length_m: float,
    *,
    threshold_m: float = 50.0,
) -> bool:
    wrapped = s_m % track_length_m
    return bool(
        min(
            wrapped,
            track_length_m - wrapped,
        )
        <= threshold_m
    )


def _finite_or_default(
    value: Any,
    default: float,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def build_response_features(
    events: pd.DataFrame,
    track_length_m: float,
    settings: ReconstructionSettings,
) -> pd.DataFrame:
    """Collapse physical events into declared GPS-response groups.

    Physical events remain visible in ``event_projection.csv`` and on
    the review map. Gate evidence is extracted once for each response
    group because GPS cannot defensibly separate overlapping members
    of the same measured response.
    """

    rows: list[dict[str, Any]] = []
    for group_id, members in events.groupby(
        "response_group_id", sort=False
    ):
        members = (
            members.sort_values("sequence")
            .copy()
        )
        unwrapped_anchor: list[float] = []
        previous: float | None = None
        for raw_anchor in members[
            "anchor_s_m"
        ].to_numpy(float):
            current = float(raw_anchor)
            if previous is not None:
                while current < previous - 3.0:
                    current += track_length_m
            unwrapped_anchor.append(current)
            previous = current

        members["_anchor_unwrapped_m"] = (
            unwrapped_anchor
        )
        starts = (
            members["_anchor_unwrapped_m"]
            + members["feature_start_rel_m"]
        )
        ends = (
            members["_anchor_unwrapped_m"]
            + members["feature_end_rel_m"]
        )
        start_member = members.loc[
            starts.idxmin()
        ]
        end_member = members.loc[ends.idxmax()]
        start_abs = float(starts.min())
        end_abs = float(ends.max())
        anchor_s = float(unwrapped_anchor[0])
        while end_abs < start_abs:
            end_abs += track_length_m

        flags = {
            flag
            for text in members[
                "review_flags"
            ].fillna("")
            for flag in str(text).split(";")
            if flag
            and flag
            != "overlaps_adjacent_feature"
        }
        sequence_values = (
            members["sequence"]
            .astype(int)
            .tolist()
        )
        if (
            len(sequence_values) > 1
            and (
                max(sequence_values)
                - min(sequence_values)
                + 1
                != len(sequence_values)
            )
        ):
            flags.add(
                "response_group_members_not_adjacent"
            )
        if end_abs - start_abs > 150.0:
            flags.add(
                "response_group_extent_very_long"
            )

        names = (
            members["name"]
            .astype(str)
            .tolist()
        )
        event_ids = (
            members["id"]
            .astype(str)
            .tolist()
        )
        rows.append(
            {
                "id": str(group_id),
                "name": (
                    names[0]
                    if len(names) == 1
                    else "Compound: "
                    + " + ".join(names)
                ),
                "sequence": int(
                    members["sequence"].min()
                ),
                "response_group_id": str(
                    group_id
                ),
                "source_event_ids": ";".join(
                    event_ids
                ),
                "source_event_names": ";".join(
                    names
                ),
                "analysis_feature_type": (
                    "individual"
                    if len(members) == 1
                    else "response_group"
                ),
                "gate_candidate": bool(
                    members[
                        "gate_candidate"
                    ].any()
                ),
                "analysis_role": (
                    "lap_gate"
                    if (
                        members["analysis_role"]
                        == "lap_gate"
                    ).any()
                    else "feature"
                ),
                "anchor_s_m": (
                    anchor_s % track_length_m
                ),
                "anchor_projection_error_m": float(
                    members[
                        "anchor_projection_error_m"
                    ].max()
                ),
                "anchor_horizontal_uncertainty_m": float(
                    members[
                        "anchor_horizontal_uncertainty_m"
                    ].max()
                ),
                "anchor_source": "; ".join(
                    sorted(
                        set(
                            members[
                                "anchor_source"
                            ].astype(str)
                        )
                    )
                ),
                "feature_start_rel_m": (
                    start_abs - anchor_s
                ),
                "feature_start_source": (
                    start_member[
                        "feature_start_source"
                    ]
                ),
                "feature_start_provenance": (
                    start_member[
                        "feature_start_provenance"
                    ]
                ),
                "feature_start_projection_error_m": float(
                    start_member[
                        "feature_start_projection_error_m"
                    ]
                ),
                "feature_start_horizontal_uncertainty_m": float(
                    start_member[
                        "feature_start_horizontal_uncertainty_m"
                    ]
                ),
                "feature_start_effective_error_m": float(
                    start_member[
                        "feature_start_effective_error_m"
                    ]
                ),
                "feature_end_rel_m": (
                    end_abs - anchor_s
                ),
                "feature_end_source": (
                    end_member[
                        "feature_end_source"
                    ]
                ),
                "feature_end_provenance": (
                    end_member[
                        "feature_end_provenance"
                    ]
                ),
                "feature_end_projection_error_m": float(
                    end_member[
                        "feature_end_projection_error_m"
                    ]
                ),
                "feature_end_horizontal_uncertainty_m": float(
                    end_member[
                        "feature_end_horizontal_uncertainty_m"
                    ]
                ),
                "feature_end_effective_error_m": float(
                    end_member[
                        "feature_end_effective_error_m"
                    ]
                ),
                "approach_start_rel_m": float(
                    (
                        start_abs - anchor_s
                    )
                    - settings.approach_before_m
                ),
                "approach_end_rel_m": float(
                    (
                        start_abs - anchor_s
                    )
                    - settings.approach_gap_m
                ),
                "entry_start_rel_m": float(
                    (
                        start_abs - anchor_s
                    )
                    - settings.entry_before_m
                ),
                "entry_end_rel_m": float(
                    (
                        start_abs - anchor_s
                    )
                    - settings.entry_gap_m
                ),
                "exit_start_rel_m": float(
                    (
                        end_abs - anchor_s
                    )
                    + settings.exit_gap_m
                ),
                "exit_end_rel_m": float(
                    (
                        end_abs - anchor_s
                    )
                    + settings.exit_gap_m
                    + settings.exit_length_m
                ),
                "recovery_limit_m": float(
                    members[
                        "recovery_limit_m"
                    ].max()
                ),
                "review_flags": ";".join(
                    sorted(flags)
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("sequence")
        .reset_index(drop=True)
    )


def _optional_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
