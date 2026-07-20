from cvt_track_study.cli import build_parser


def test_new_cli_names_are_first_class() -> None:
    parser = build_parser()
    assert parser.parse_args(["run", "nominal", "project"]).run_command == "nominal"
    assert parser.parse_args(["run", "track-robustness", "project"]).run_command == "track-robustness"
    assert parser.parse_args(["run", "structural-sensitivity", "project"]).run_command == "structural-sensitivity"
    assert parser.parse_args(["run", "full-uncertainty", "project"]).run_command == "full-uncertainty"
    assert parser.parse_args(["run", "design-comparison", "project"]).run_command == "design-comparison"


def test_track_robustness_has_no_simulation_sampling_switches() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "track-robustness", "project"])
    assert args.bundle is None
    assert args.replicates is None
    assert args.no_cache is False
