from __future__ import annotations

import ast
from pathlib import Path


def test_all_report_errorbars_are_explicitly_black() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "cvt_track_study"
    failures: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "errorbar":
                continue
            keyword = next((item for item in node.keywords if item.arg == "ecolor"), None)
            if keyword is None:
                failures.append(f"{path.relative_to(source_root)}:{node.lineno}: missing ecolor")
            elif not isinstance(keyword.value, ast.Constant) or keyword.value.value != "black":
                failures.append(f"{path.relative_to(source_root)}:{node.lineno}: ecolor must be 'black'")
    assert not failures, "\n".join(failures)


def test_paired_winner_card_is_not_mislabeled_as_raw_median() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "cvt_track_study" / "reports" / "postprocess.py").read_text(encoding="utf-8")
    assert "Best median lap time" not in source
    assert "Top paired design" in source
