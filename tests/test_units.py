from math import isclose

import pytest

from cvt_track_study.config import UnitValidationError, convert_to_si, require_dimension


def test_inches_convert_to_metres() -> None:
    value, unit = convert_to_si(22.0, "in")
    assert isclose(value, 0.5588)
    assert unit == "m"


def test_unit_alias_is_normalized() -> None:
    value, unit = convert_to_si(0.67, "m²")
    assert value == 0.67
    assert unit == "m^2"


def test_wrong_dimension_is_rejected() -> None:
    with pytest.raises(UnitValidationError, match="expected 'mass'"):
        require_dimension("m", "mass")


def test_unknown_unit_is_rejected() -> None:
    with pytest.raises(UnitValidationError, match="Unknown unit"):
        convert_to_si(1.0, "furlong_per_fortnight")
