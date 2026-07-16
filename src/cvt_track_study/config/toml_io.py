"""TOML loading and deterministic writing for project artifacts."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - selected by interpreter version
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


class TomlError(ValueError):
    pass


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise
    except tomllib.TOMLDecodeError as exc:
        raise TomlError(f"Could not parse TOML: {exc}") from exc


def dump_toml(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dumps_toml(data)
    path.write_text(text, encoding="utf-8")


def dumps_toml(data: Mapping[str, Any]) -> str:
    lines: list[str] = []
    _emit_table(data, (), lines, emit_header=False)
    return "\n".join(lines).rstrip() + "\n"


def _emit_table(
    table: Mapping[str, Any],
    path: Sequence[str],
    lines: list[str],
    *,
    emit_header: bool,
) -> None:
    scalar_items: list[tuple[str, Any]] = []
    child_tables: list[tuple[str, Mapping[str, Any]]] = []
    array_tables: list[tuple[str, list[Mapping[str, Any]]]] = []

    for key, value in table.items():
        if isinstance(value, Mapping):
            child_tables.append((str(key), value))
        elif isinstance(value, list) and value and all(
            isinstance(item, Mapping) for item in value
        ):
            array_tables.append((str(key), value))  # type: ignore[arg-type]
        else:
            scalar_items.append((str(key), value))

    # Parent tables containing only subtables are implicit in TOML. Omitting
    # their empty headers keeps resolved_inputs.toml compact and readable.
    if emit_header and (scalar_items or not child_tables and not array_tables):
        _blank_before_header(lines)
        lines.append(f"[{_format_path(path)}]")

    for key, value in scalar_items:
        lines.append(f"{_format_key(key)} = {_format_value(value)}")

    for key, child in child_tables:
        _emit_table(child, (*path, key), lines, emit_header=True)

    for key, items in array_tables:
        for item in items:
            _emit_array_table(item, (*path, key), lines)


def _emit_array_table(
    table: Mapping[str, Any], path: Sequence[str], lines: list[str]
) -> None:
    _blank_before_header(lines)
    lines.append(f"[[{_format_path(path)}]]")

    scalar_items: list[tuple[str, Any]] = []
    child_tables: list[tuple[str, Mapping[str, Any]]] = []
    array_tables: list[tuple[str, list[Mapping[str, Any]]]] = []
    for key, value in table.items():
        if isinstance(value, Mapping):
            child_tables.append((str(key), value))
        elif isinstance(value, list) and value and all(
            isinstance(item, Mapping) for item in value
        ):
            array_tables.append((str(key), value))  # type: ignore[arg-type]
        else:
            scalar_items.append((str(key), value))

    for key, value in scalar_items:
        lines.append(f"{_format_key(key)} = {_format_value(value)}")
    for key, child in child_tables:
        _emit_table(child, (*path, key), lines, emit_header=True)
    for key, items in array_tables:
        for item in items:
            _emit_array_table(item, (*path, key), lines)


def _format_path(path: Sequence[str]) -> str:
    return ".".join(_format_key(part) for part in path)


def _format_key(key: str) -> str:
    if key and all(character.isalnum() or character in "_-" for character in key):
        return key
    return json.dumps(key, ensure_ascii=False)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not isfinite(value):
            raise TomlError("TOML export does not support non-finite floats.")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        if any(isinstance(item, Mapping) for item in value):
            raise TomlError("Inline arrays of tables are not supported by this writer.")
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    raise TomlError(f"Unsupported TOML value type: {type(value).__name__}.")


def _blank_before_header(lines: list[str]) -> None:
    if lines and lines[-1] != "":
        lines.append("")
