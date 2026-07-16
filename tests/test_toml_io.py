import sys

from cvt_track_study.config.toml_io import dumps_toml

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def test_writer_round_trips_nested_tables_and_arrays_of_tables() -> None:
    source = {
        "project": {"name": "test", "enabled": True},
        "runs": [
            {
                "file": "gpx/a.gpx",
                "vehicle_id": "A",
                "metadata": {"driver": "Kai"},
            },
            {
                "file": "gpx/b.gpx",
                "vehicle_id": "B",
                "metadata": {"driver": "Other"},
            },
        ],
        "empty": [],
    }
    text = dumps_toml(source)
    assert tomllib.loads(text) == source
