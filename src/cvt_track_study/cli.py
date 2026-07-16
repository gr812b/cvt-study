"""Command line for the measured track-based drivetrain design framework."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .bundle import TrackBundleError, load_track_bundle
from . import __version__
from .config import ProjectError, ProjectLoader, initialize_project, parse_override
from .gpx import ingest_project
from .track import build_project_track
from .simulation import SimulationError, run_baseline_project
from .simulation.reporting_v8 import regenerate_baseline_reports
from .studies import run_study_project
from .studies.reporting_v8 import regenerate_study_reports
from .runtime.cache import SimulationCache
from .runtime.doctor import run_doctor
from .runtime.migration import migrate_prototype_events
from .runtime.results import discover_results, write_results_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drivetrain-study",
        description="Measured track-based drivetrain design framework.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Create a clean project workspace from the bundled template."
    )
    init_parser.add_argument("destination", type=Path)
    init_parser.add_argument("--name", help="Project name written to project.toml.")

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check installation and optional project readiness."
    )
    doctor_parser.add_argument("project", type=Path, nargs="?")

    results_parser = subparsers.add_parser(
        "results", help="List completed project results and refresh results/INDEX.md."
    )
    results_parser.add_argument("project", type=Path)
    results_parser.add_argument("--json", action="store_true", dest="as_json")

    report_parser = subparsers.add_parser(
        "report", help="Regenerate human reports from an existing result's machine artifacts."
    )
    report_parser.add_argument("result", type=Path)

    cache_parser = subparsers.add_parser("cache", help="Inspect or clear project simulation cache.")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", required=True)
    for cache_command in ("status", "clear"):
        item = cache_subparsers.add_parser(cache_command)
        item.add_argument("project", type=Path)

    migrate_parser = subparsers.add_parser(
        "migrate", help="Conservatively migrate prototype inputs without inventing physics."
    )
    migrate_subparsers = migrate_parser.add_subparsers(dest="migrate_command", required=True)
    migrate_events = migrate_subparsers.add_parser("prototype-events")
    migrate_events.add_argument("source", type=Path)
    migrate_events.add_argument("destination", type=Path)

    validate_parser = subparsers.add_parser(
        "validate", help="Resolve and validate a project without running analysis."
    )
    validate_parser.add_argument("project", type=Path)
    validate_parser.add_argument(
        "--study",
        help="Resolve one study's configuration overrides and validate its design path.",
    )
    validate_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Override an existing resolved leaf. Repeat for multiple values. "
            "TOML scalar syntax is accepted."
        ),
    )
    validate_parser.add_argument(
        "--output",
        type=Path,
        help="Artifact directory. Defaults to project/results/validation/<UTC timestamp>.",
    )
    validate_parser.add_argument(
        "--no-export",
        action="store_true",
        help="Print diagnostics without writing resolved-input artifacts.",
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a failing exit code when warnings are present.",
    )

    ingest_parser = subparsers.add_parser(
        "ingest", help="Parse declared GPX tracks and export canonical telemetry."
    )
    ingest_parser.add_argument("project", type=Path)
    ingest_parser.add_argument(
        "--run",
        dest="run_ids",
        action="append",
        default=[],
        help="Ingest only this run id. Repeat to select multiple runs.",
    )
    ingest_parser.add_argument(
        "--output",
        type=Path,
        help="Artifact directory. Defaults to project/results/ingestion/<UTC timestamp>.",
    )

    build_track_parser = subparsers.add_parser(
        "build-track",
        help="Ingest GPX, reconstruct laps/centreline, score speed gates, and create review artifacts.",
    )
    build_track_parser.add_argument("project", type=Path)
    build_track_parser.add_argument(
        "--output",
        type=Path,
        help="Artifact directory. Defaults to project/results/track_build/<UTC timestamp>.",
    )

    review_parser = subparsers.add_parser(
        "review",
        help="Run the Phase 3 track build and emphasize the generated review package.",
    )
    review_parser.add_argument("project", type=Path)
    review_parser.add_argument(
        "--output",
        type=Path,
        help="Artifact directory. Defaults to project/results/track_review/<UTC timestamp>.",
    )


    run_parser = subparsers.add_parser(
        "run", help="Run a simulation or design study from a project workspace."
    )
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)
    baseline_parser = run_subparsers.add_parser(
        "baseline", help="Run the nominal bounded CVT and infinite-ratio reference."
    )
    baseline_parser.add_argument("project", type=Path)
    baseline_parser.add_argument(
        "--study", default="baseline", help="Baseline study name. Defaults to baseline."
    )
    baseline_parser.add_argument(
        "--bundle",
        type=Path,
        help="Use this validated track bundle. Without it, the track is rebuilt first.",
    )
    baseline_parser.add_argument(
        "--output",
        type=Path,
        help="Result directory. Defaults to project/results/baseline/<UTC timestamp>.",
    )

    for command, default_study, help_text in (
        ("sweep", "final_drive_sweep", "Run a paired design sweep."),
        ("track-robustness", "track_robustness", "Propagate measured gate and obstacle uncertainty."),
        ("structural-sensitivity", "structural_sensitivity", "Run one-at-a-time structural sensitivity."),
        ("uncertainty", "full_uncertainty", "Jointly propagate all declared uncertainty."),
    ):
        study_parser = run_subparsers.add_parser(command, help=help_text)
        study_parser.add_argument("project", type=Path)
        study_parser.add_argument("--study", default=default_study)
        study_parser.add_argument("--bundle", type=Path, help="Use an existing validated track bundle.")
        study_parser.add_argument("--output", type=Path, help="Explicit result directory.")
        study_parser.add_argument(
            "--replicates",
            type=int,
            help="Temporary replicate override for short checks; the resolved study file remains unchanged.",
        )
        study_parser.add_argument("--workers", type=int, default=1, help="Parallel scenario workers.")
        study_parser.add_argument("--resume", action="store_true", help="Resume a matching incomplete run.")
        study_parser.add_argument("--restart", action="store_true", help="Discard a matching incomplete run and restart.")
        study_parser.add_argument("--no-cache", action="store_true", help="Disable persistent simulation-summary caching.")
        study_parser.add_argument("--no-progress", action="store_true", help="Disable progress and ETA output.")
        study_parser.add_argument("--run-name", help="Stable human-readable result name prefix.")

    bundle_parser = subparsers.add_parser(
        "validate-bundle",
        help="Validate a versioned track bundle and verify its checksum when present.",
    )
    bundle_parser.add_argument("bundle", type=Path)
    bundle_parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Validate bundle content without checking the adjacent checksum file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            destination = initialize_project(args.destination, name=args.name)
            print(f"Created project: {destination}")
            print(f"Next: drivetrain-study validate {destination}")
            return 0
        if args.command == "doctor":
            checks = run_doctor(args.project)
            icons = {"pass": "✓", "warning": "!", "fail": "✗"}
            for check in checks:
                print(f"{icons[check.status]} {check.name}: {check.detail}")
            failed = any(check.status == "fail" for check in checks)
            if failed:
                print("Not ready: correct failed checks.")
            elif any(check.status == "warning" for check in checks):
                print("Ready for exploratory work; review warnings before decision runs.")
            else:
                print("Ready.")
            return 1 if failed else 0
        if args.command == "results":
            resolution = ProjectLoader().resolve(args.project)
            root = resolution.paths.results_directory
            index = write_results_index(root)
            records = discover_results(root)
            if args.as_json:
                print(json.dumps(records, indent=2, sort_keys=True, allow_nan=False))
            else:
                for record in records:
                    print(
                        f"{record['created_utc']}  {record['type']}  "
                        f"{record['study']}  {record['path']}"
                    )
                print(f"Result index: {index}")
            return 0
        if args.command == "report":
            result = args.result.resolve()
            if (result / "comparison_summary.json").is_file():
                regenerate_baseline_reports(result)
            else:
                regenerate_study_reports(result)
            print(f"Regenerated: {args.result.resolve() / 'SUMMARY.md'}")
            print(f"Regenerated: {args.result.resolve() / 'REPORT.md'}")
            return 0
        if args.command == "cache":
            resolution = ProjectLoader().resolve(args.project)
            cache = SimulationCache(
                resolution.paths.root / ".drivetrain-study-cache" / "simulations"
            )
            if args.cache_command == "clear":
                count = cache.clear()
                print(f"Removed {count} cached simulation entr{'y' if count == 1 else 'ies'}.")
            else:
                print(json.dumps(cache.status(), indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "migrate" and args.migrate_command == "prototype-events":
            count = migrate_prototype_events(args.source, args.destination)
            print(f"Migrated {count} geometry anchor(s) to {args.destination}.")
            print("Review every row and assign explicit obstacle physics before use.")
            return 0
        if args.command == "validate":
            return _run_validate(args)
        if args.command == "ingest":
            _, results, output = ingest_project(
                args.project,
                run_ids=args.run_ids,
                output_directory=args.output,
            )
            for result in results:
                for diagnostic in result.diagnostics:
                    print(diagnostic.format())
                print(
                    f"Ingested {result.metadata.run_id}: {len(result.points)} valid point(s), "
                    f"{len(result.segments)} segment(s)."
                )
            print(f"Ingestion artifacts: {output}")
            return 1 if any(result.error_count for result in results) else 0
        if args.command == "run" and args.run_command in {
            "sweep", "track-robustness", "structural-sensitivity", "uncertainty"
        }:
            print("Resolving uncertainty study and track bundle...")
            output = run_study_project(
                args.project,
                study=args.study,
                bundle_path=args.bundle,
                output_directory=args.output,
                replicates_override=args.replicates,
                workers=args.workers,
                resume=args.resume,
                restart=args.restart,
                use_cache=not args.no_cache,
                progress=not args.no_progress,
                run_name=args.run_name,
                command=tuple(sys.argv if argv is None else ["drivetrain-study", *argv]),
            )
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            print(f"Study artifacts: {output}")
            print(
                f"{manifest['scenario_count']} scenario(s), "
                f"{manifest['bounded_simulation_count']} bounded run(s), "
                f"{manifest['reference_simulation_count']} reference run(s), "
                f"{manifest['reference_cache_hits']} safe reference reuse(s), "
                f"{manifest.get('simulation_cache_hits', 0)} persistent cache hit(s)."
            )
            print(f"Start here: {output / 'SUMMARY.md'}")
            return 0
        if args.command == "run" and args.run_command == "baseline":
            print("Resolving project and track bundle...")
            output = run_baseline_project(
                args.project,
                study=args.study,
                bundle_path=args.bundle,
                output_directory=args.output,
                command=tuple(sys.argv if argv is None else ["drivetrain-study", *argv]),
            )
            comparison = __import__("json").loads(
                (output / "comparison_summary.json").read_text(encoding="utf-8")
            )
            print(f"Baseline artifacts: {output}")
            print(
                "time penalty="
                f"{float(comparison['lap_time_penalty_vs_infinite_s']):.3f} s, "
                "opportunity loss="
                f"{float(comparison['finite_ratio_opportunity_loss_energy_kj']):.3f} kJ, "
                "reference dominance="
                f"{comparison['reference_dominance_pass']}"
            )
            print(f"Start here: {output / 'SUMMARY.md'}")
            return 0
        if args.command == "validate-bundle":
            bundle = load_track_bundle(
                args.bundle, verify_checksum=not args.no_checksum
            )
            print(
                f"Valid track bundle {bundle.schema_version}: "
                f"{bundle.track_length_m:.1f} m, "
                f"{len(bundle.physical_features)} physical feature(s), "
                f"{len(bundle.response_groups)} response group(s), "
                f"{len(bundle.active_speed_gates)} active speed gate(s)."
            )
            if bundle.sha256:
                print(f"SHA-256: {bundle.sha256}")
            return 0
        if args.command in {"build-track", "review"}:
            output = args.output
            if args.command == "review" and output is None:
                resolution = ProjectLoader().resolve(args.project)
                output = resolution.paths.results_directory / "track_review" / _timestamp()
            result = build_project_track(args.project, output_directory=output)
            for diagnostic in result.diagnostics:
                print(diagnostic.format())
            print(
                f"Track build: {result.metadata['complete_lap_count']} complete lap(s), "
                f"{result.metadata['valid_lap_count']} valid lap(s), "
                f"{result.metadata['accepted_gate_count']} accepted gate(s)."
            )
            print(f"Track length: {result.centreline.length_m:.1f} m")
            print(f"Review package: {result.output_directory / 'review'}")
            print(f"Track bundle: {result.output_directory / 'track_bundle.json'}")
            return 1 if result.error_count else 0
    except (ProjectError, TrackBundleError, SimulationError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 2


def _run_validate(args: argparse.Namespace) -> int:
    overrides = [parse_override(item) for item in args.overrides]
    result = ProjectLoader().resolve(
        args.project,
        study=args.study,
        cli_overrides=overrides,
    )
    for diagnostic in result.diagnostics:
        print(diagnostic.format())
    print(
        f"Validation summary: {result.error_count} error(s), "
        f"{result.warning_count} warning(s)."
    )
    if not args.no_export:
        output = args.output or _default_output(result.paths.results_directory)
        if not output.is_absolute():
            output = (Path.cwd() / output).resolve()
        result.export(output)
        print(f"Resolved artifacts: {output}")
    if result.error_count:
        return 1
    if args.strict and result.warning_count:
        return 1
    return 0


def _default_output(results_directory: Path) -> Path:
    return results_directory / "validation" / _timestamp()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
