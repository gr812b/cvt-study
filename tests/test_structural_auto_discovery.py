from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from cvt_track_study.studies.planning import (
    selected_structural_paths,
)


@dataclass
class _Item:
    path: str
    category: str
    value: object


class _Registry:
    def __init__(self):
        variable = SimpleNamespace(
            uncertainty=SimpleNamespace(
                distribution=SimpleNamespace(
                    value="uniform"
                )
            )
        )
        fixed = SimpleNamespace(
            uncertainty=SimpleNamespace(
                distribution=SimpleNamespace(
                    value="fixed"
                )
            )
        )
        self.inputs = (
            _Item(
                "vehicle.mass",
                "structural",
                variable,
            ),
            _Item(
                "driver.reaction_time",
                "structural",
                variable,
            ),
            _Item(
                "initial_conditions.speed",
                "initial_condition",
                variable,
            ),
            _Item(
                "vehicle.gravity",
                "structural",
                fixed,
            ),
        )

    def stochastic(self):
        return tuple(
            item
            for item in self.inputs
            if item.value.uncertainty.distribution.value
            != "fixed"
        )


def test_wildcard_discovers_all_nonfixed_structural_inputs():
    raw = {
        "sensitivity": {
            "parameters": ["*"],
            "exclude_parameters": [
                "driver.reaction_time"
            ],
        }
    }
    assert selected_structural_paths(
        raw, _Registry()
    ) == ("vehicle.mass",)


def test_explicit_followup_list_remains_supported():
    raw = {
        "sensitivity": {
            "parameters": [
                "driver.reaction_time"
            ]
        }
    }
    assert selected_structural_paths(
        raw, _Registry()
    ) == ("driver.reaction_time",)
