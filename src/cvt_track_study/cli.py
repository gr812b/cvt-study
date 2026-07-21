"""Command line for the measured-track drivetrain design framework."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .bundle import TrackBundleError, load_track_bundle
from .config import ProjectError, ProjectLoader, initialize_project, parse_override
from .gpx import ingest_project
from .reports import primary_report_path, regenerate_framework_report
from .runtime.cache import SimulationCache
from .runtime.doctor import run_doctor
from .runtime.migration import migrate_prototype_events
from .runtime.results import discover_results, write_results_index
from .simulation import SimulationError, run_baseline_project
from .studies import run_study_project
from .track import build_project_track


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drivetrain-study",
        description="Measured track-based drivetrain design framework with six canonical reports.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a project workspace.")
    init_parser.add_argument("destination", type=Path)
    init_parser.add_argument("--name")

    doctor_parser = subparsers.add_parser("doctor", help="Check installation and project readiness.")
    doctor_parser.add_argument("project", type=Path, nargs="?")

    results_parser = subparsers.add_parser("results", help="List results and refresh results/INDEX.md.")
    results_parser.add_argument("project", type=Path)
    results_parser.add_argument("--json", action="store_true", dest="as_json")

    report_parser = subparsers.add_parser(
        "report",
        help="Regenerate the canonical HTML report from an existing result's machine artifacts.",
    )
    report_parser.add_argument("result", type=Path)

    cache_parser = subparsers.add_parser("cache", help="Inspect or clear simulation cache.")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", required=True)
    for cache_command in ("status", "clear"):
        item = cache_subparsers.add_parser(cache_command)
        item.add_argument("project", type=Path)

    migrate_parser = subparsers.add_parser("migrate", help="Conservatively migrate prototype inputs.")
    migrate_subparsers = migrate_parser.add_subparsers(dest="migrate_command", required=True)
    migrate_events = migrate_subparsers.add_parser("prototype-events")
    migrate_events.add_argument("source", type=Path)
    migrate_events.add_argument("destination", type=Path)

    validate_parser = subparsers.add_parser("validate", help="Resolve and validate without analysis.")
    validate_parser.add_argument("project", type=Path)
    validate_parser.add_argument("--study")
    validate_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    validate_parser.add_argument("--output", type=Path)
    validate_parser.add_argument("--no-export", action="store_true")
    validate_parser.add_argument("--strict", action="store_true")

    ingest_parser = subparsers.add_parser("ingest", help="Parse GPX/FIT and export canonical telemetry.")
    ingest_parser.add_argument("project", type=Path)
    ingest_parser.add_argument("--run", dest="run_ids", action="append", default=[])
    ingest_parser.add_argument("--output", type=Path)

    build_track_parser = subparsers.add_parser(
        "build-track",
        help="Report 1: build the nominal track evidence and reconstruction report.",
    )
    build_track_parser.add_argument("project", type=Path)
    build_track_parser.add_argument("--output", type=Path)

    review_parser = subparsers.add_parser(
        "review",
        help="Alias for build-track with a track_review result directory.",
    )
    review_parser.add_argument("project", type=Path)
    review_parser.add_argument("--output", type=Path)

    run_parser = subparsers.add_parser("run", help="Run one of the remaining five canonical reports.")
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)

    for command, hidden_alias in (("nominal", False), ("baseline", True)):
        item = run_subparsers.add_parser(
            command,
            help=("Report 2: nominal bounded and infinite-CVT simulation." if not hidden_alias else "Legacy alias for run nominal."),
        )
        item.add_argument("project", type=Path)
        item.add_argument("--study", default="baseline")
        item.add_argument("--bundle", type=Path)
        item.add_argument("--output", type=Path)

    study_specs = (
        ("track-robustness", "track_robustness", "Report 3: test defensibility of the inferred track from telemetry only."),
        ("structural-sensitivity", "structural_sensitivity", "Report 4: vary one structural assumption at a time."),
        ("full-uncertainty", "full_uncertainty", "Report 5: jointly propagate defensible uncertainty."),
        ("uncertainty", "full_uncertainty", "Legacy alias for run full-uncertainty."),
        ("design-comparison", "final_drive_sweep", "Report 6: paired drivetrain design comparison."),
        ("sweep", "final_drive_sweep", "Legacy alias for run design-comparison."),
    )
    for command, default_study, help_text in study_specs:
        item = run_subparsers.add_parser(command, help=help_text)
        item.add_argument("project", type=Path)
        item.add_argument("--study", default=default_study)
        if command != "track-robustness":
            item.add_argument("--bundle", type=Path, help="Use an existing validated track bundle.")
            item.add_argument(
                "--replicates",
                type=int,
                help=(
                    "Override the configured draw count. For full uncertainty with "
                    "sampling.layout='cross_track_cases', this is the number of common "
                    "structural/traversal draws replayed on every track case."
                ),
            )
            item.add_argument("--no-cache", action="store_true")
        else:
            item.set_defaults(bundle=None, replicates=None, no_cache=False)
        item.add_argument("--output", type=Path)
        item.add_argument("--workers", type=int, default=1)
        item.add_argument("--resume", action="store_true")
        item.add_argument("--restart", action="store_true")
        item.add_argument("--no-progress", action="store_true")
        item.add_argument("--run-name")

    bundle_parser = subparsers.add_parser("validate-bundle", help="Validate a versioned track bundle.")
    bundle_parser.add_argument("bundle", type=Path)
    bundle_parser.add_argument("--no-checksum", action="store_true")
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
                    print(f"{record['created_utc']}  {record['type']}  {record['study']}  {record['path']}")
                print(f"Result index: {index}")
            return 0
        if args.command == "report":
            report = regenerate_framework_report(args.result.resolve())
            print(f"Regenerated: {report}")
            return 0
        if args.command == "cache":
            resolution = ProjectLoader().resolve(args.project)
            cache = SimulationCache(resolution.paths.root / ".drivetrain-study-cache" / "simulations")
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
            _, results, output = ingest_project(args.project, run_ids=args.run_ids, output_directory=args.output)
            for result in results:
                for diagnostic in result.diagnostics:
                    print(diagnostic.format())
                print(f"Ingested {result.metadata.run_id}: {len(result.points)} valid point(s), {len(result.segments)} segment(s).")
            print(f"Ingestion artifacts: {output}")
            return 1 if any(result.error_count for result in results) else 0
        if args.command == "run" and args.run_command in {"nominal", "baseline"}:
            print("Resolving project and nominal track bundle...")
            output = run_baseline_project(
                args.project,
                study=args.study,
                bundle_path=args.bundle,
                output_directory=args.output,
                command=tuple(sys.argv if argv is None else ["drivetrain-study", *argv]),
            )
            comparison = json.loads((output / "comparison_summary.json").read_text(encoding="utf-8"))
            print(f"Nominal artifacts: {output}")
            print(
                f"time penalty={float(comparison['lap_time_penalty_vs_infinite_s']):.3f} s, "
                f"opportunity loss={float(comparison['finite_ratio_opportunity_loss_energy_kj']):.3f} kJ, "
                f"reference dominance={comparison['reference_dominance_pass']}"
            )
            _print_start(output)
            return 0
        if args.command == "run" and args.run_command in {
            "track-robustness", "structural-sensitivity", "full-uncertainty", "uncertainty", "design-comparison", "sweep"
        }:
            label = {
                "track-robustness": "track defensibility",
                "structural-sensitivity": "structural sensitivity",
                "full-uncertainty": "full uncertainty",
                "uncertainty": "full uncertainty",
                "design-comparison": "design comparison",
                "sweep": "design comparison",
            }[args.run_command]
            print(f"Resolving {label} report...")
            output = run_study_project(
                args.project,
                study=args.study,
                bundle_path=getattr(args, "bundle", None),
                output_directory=args.output,
                replicates_override=getattr(args, "replicates", None),
                workers=args.workers,
                resume=args.resume,
                restart=args.restart,
                use_cache=not getattr(args, "no_cache", False),
                progress=not args.no_progress,
                run_name=args.run_name,
                command=tuple(sys.argv if argv is None else ["drivetrain-study", *argv]),
            )
            print(f"Report artifacts: {output}")
            _print_start(output)
            return 0
        if args.command == "validate-bundle":
            bundle = load_track_bundle(args.bundle, verify_checksum=not args.no_checksum)
            print(
                f"Valid track bundle {bundle.schema_version}: {bundle.track_length_m:.1f} m, "
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
            print(f"Track bundle: {result.output_directory / 'track_bundle.json'}")
            _print_start(result.output_directory)
            return 1 if result.error_count else 0
    except (ProjectError, TrackBundleError, SimulationError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 2


def _print_start(output: Path) -> None:
    report = primary_report_path(output)
    print(f"Start here: {report or output}")


def _run_validate(args: argparse.Namespace) -> int:
    overrides = [parse_override(item) for item in args.overrides]
    result = ProjectLoader().resolve(args.project, study=args.study, cli_overrides=overrides)
    for diagnostic in result.diagnostics:
        print(diagnostic.format())
    print(f"Validation summary: {result.error_count} error(s), {result.warning_count} warning(s).")
    if not args.no_export:
        output = args.output or _default_output(result.paths.results_directory)
        if not output.is_absolute():
            output = (Path.cwd() / output).resolve()
        result.export(output)
        print(f"Resolved artifacts: {output}")
    if result.error_count or (args.strict and result.warning_count):
        return 1
    return 0


def _default_output(results_directory: Path) -> Path:
    return results_directory / "validation" / _timestamp()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
