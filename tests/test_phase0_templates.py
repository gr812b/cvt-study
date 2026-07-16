from pathlib import Path
import sys

from cvt_track_study.config import UncertainQuantity

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "project"


def load_toml(path: Path):
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_project_template_is_self_contained() -> None:
    project = load_toml(TEMPLATE / "project.toml")
    paths = project["project"]
    for key in ("track", "runs", "events"):
        assert (TEMPLATE / paths[key]).exists(), key
    for key in ("vehicles_directory", "studies_directory", "results_directory"):
        assert (TEMPLATE / paths[key]).is_dir(), key
    for profile_root in project["profiles"]["roots"]:
        assert (TEMPLATE / profile_root).is_dir()


def test_run_manifest_is_empty_until_user_adds_real_gpx() -> None:
    runs = load_toml(TEMPLATE / "track" / "runs.toml")["runs"]
    assert runs == []
    assert not list((TEMPLATE / "track" / "gpx").glob("*.csv"))


def test_elevation_is_preserved_but_grade_is_disabled() -> None:
    track = load_toml(TEMPLATE / "track" / "track.toml")
    elevation = track["track"]["elevation"]
    assert elevation["store_from_gpx"] is True
    assert elevation["use_for_grade_force"] is False


def test_template_numeric_quantities_declare_uncertainty() -> None:
    vehicle = load_toml(TEMPLATE / "vehicles" / "vehicle_A" / "vehicle.toml")
    drivetrain = load_toml(
        TEMPLATE / "vehicles" / "vehicle_A" / "drivetrain.toml"
    )
    quantities = [
        vehicle["vehicle"]["mass"],
        vehicle["vehicle"]["tire_diameter"],
        drivetrain["drivetrain"]["final_drive_ratio"],
        drivetrain["drivetrain"]["cvt"]["maximum_reduction_ratio"],
        drivetrain["drivetrain"]["cvt"]["minimum_reduction_ratio"],
    ]
    for quantity in quantities:
        parsed = UncertainQuantity.from_mapping(quantity)
        assert parsed.unit
