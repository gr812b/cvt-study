"""Evidence-readiness assessment shared by baseline and study reporting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def assess_evidence(
    *, diagnostics: Iterable[Any], bundle: Mapping[str, Any] | Any
) -> dict[str, Any]:
    """Summarize project warnings and unresolved track-evidence review state.

    Simulation is allowed to proceed when configuration errors are absent.  This
    assessment is deliberately stricter: unresolved warnings remain visible and
    prevent a result from being labelled ready for an engineering decision.
    """

    data = bundle.data if hasattr(bundle, "data") else bundle
    if not isinstance(data, Mapping):
        raise TypeError("Evidence assessment requires a track-bundle mapping.")

    warnings: list[str] = []
    blockers: list[str] = []
    project_records: list[dict[str, str]] = []
    for diagnostic in diagnostics:
        record = _diagnostic_record(diagnostic)
        if record.get("severity") != "warning":
            continue
        project_records.append(record)
        message = (
            f"Project validation warning [{record.get('code', 'UNSPECIFIED')}]: "
            f"{record.get('message', 'review the resolved-input validation report')}"
        )
        warnings.append(message)
        blockers.append(message)

    evidence = data.get("evidence", {})
    review_records = evidence.get("review_records", []) if isinstance(evidence, Mapping) else []
    status_counts = Counter(
        str(record.get("recommendation", "unknown"))
        for record in review_records
        if isinstance(record, Mapping)
    )
    review_count = int(status_counts.get("recommended_review", 0))
    must_fix_count = int(status_counts.get("must_fix", 0))
    if must_fix_count:
        message = (
            f"Track review contains {must_fix_count} must-fix event(s); resolve them "
            "before treating simulation results as decision-ready."
        )
        warnings.append(message)
        blockers.append(message)
    if review_count:
        message = (
            f"Track review contains {review_count} event(s) still recommended for review."
        )
        warnings.append(message)
        blockers.append(message)

    simulation_contract = data.get("simulation_contract", {})
    gates = (
        simulation_contract.get("speed_gates", [])
        if isinstance(simulation_contract, Mapping)
        else []
    )
    active_gates = [
        gate
        for gate in gates
        if isinstance(gate, Mapping) and bool(gate.get("active_by_default"))
    ]
    if not active_gates:
        message = "The track bundle contains no accepted speed gate active by default."
        warnings.append(message)
        blockers.append(message)

    lap_summary = evidence.get("lap_summary", {}) if isinstance(evidence, Mapping) else {}
    lap_records = lap_summary.get("records", []) if isinstance(lap_summary, Mapping) else []
    identity_counts = {
        key: len(
            {
                str(record.get(key))
                for record in lap_records
                if isinstance(record, Mapping) and record.get(key) not in (None, "")
            }
        )
        for key in ("run_id", "vehicle_id", "driver_id")
    }
    if identity_counts["run_id"] < 2:
        warnings.append(
            "Track evidence contains fewer than two run identities; between-session repeatability is not measured."
        )
    if identity_counts["vehicle_id"] < 2:
        warnings.append(
            "Track evidence contains one vehicle identity; cross-vehicle agreement is not measured."
        )
    if identity_counts["driver_id"] < 2:
        warnings.append(
            "Track evidence contains one driver identity; cross-driver agreement is not measured."
        )

    cross_vehicle_counts = Counter(
        str(gate.get("confidence", {}).get("cross_vehicle_status", "unknown"))
        for gate in active_gates
        if isinstance(gate.get("confidence"), Mapping)
    )
    return {
        "schema_version": 1,
        "ready": not blockers,
        "project_warning_count": len(project_records),
        "project_warnings": project_records,
        "track_review_status_counts": dict(sorted(status_counts.items())),
        "active_gate_count": len(active_gates),
        "evidence_identity_counts": identity_counts,
        "active_gate_cross_vehicle_status_counts": dict(sorted(cross_vehicle_counts.items())),
        "blocking_reasons": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
    }


def unavailable_evidence_assessment() -> dict[str, Any]:
    """Return a conservative marker for legacy artifacts without an assessment."""

    message = (
        "Evidence readiness was not recorded in this artifact; regenerate from a project run "
        "before treating it as decision-ready."
    )
    return {
        "schema_version": 1,
        "ready": False,
        "project_warning_count": 0,
        "project_warnings": [],
        "track_review_status_counts": {},
        "active_gate_count": 0,
        "evidence_identity_counts": {},
        "active_gate_cross_vehicle_status_counts": {},
        "blocking_reasons": [message],
        "warnings": [message],
    }


def numerical_valid(quality: Mapping[str, Any]) -> bool:
    """Read the current field while supporting pre-v0.8 regenerated artifacts."""

    return bool(quality.get("numerically_valid", quality.get("valid_for_decision", False)))


def _diagnostic_record(diagnostic: Any) -> dict[str, str]:
    if hasattr(diagnostic, "to_dict"):
        raw = diagnostic.to_dict()
    elif isinstance(diagnostic, Mapping):
        raw = diagnostic
    else:
        raise TypeError(f"Unsupported diagnostic record: {type(diagnostic).__name__}")
    record = {str(key): str(value) for key, value in raw.items() if value not in (None, "")}
    severity = getattr(getattr(diagnostic, "severity", None), "value", None)
    if severity is not None:
        record["severity"] = str(severity)
    return record
