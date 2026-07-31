from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import shutil
import tokenize
from io import StringIO

IGNORED_TOKENS = {
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
}


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    replacement: str


def _offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _absolute(offsets: list[int], position: tuple[int, int]) -> int:
    row, column = position
    return offsets[row - 1] + column


def _previous_significant(tokens: list[tokenize.TokenInfo], index: int) -> int | None:
    index -= 1
    while index >= 0:
        if tokens[index].type not in IGNORED_TOKENS:
            return index
        index -= 1
    return None


def _next_significant(tokens: list[tokenize.TokenInfo], index: int) -> int | None:
    index += 1
    while index < len(tokens):
        if tokens[index].type not in IGNORED_TOKENS:
            return index
        index += 1
    return None


def _matching_close(tokens: list[tokenize.TokenInfo], open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(tokens)):
        token = tokens[index]
        if token.type != tokenize.OP:
            continue
        if token.string == "(":
            depth += 1
        elif token.string == ")":
            depth -= 1
            if depth == 0:
                return index
    raise SyntaxError("Unmatched parenthesis in errorbar call")


def _errorbar_calls(source: str) -> list[tuple[int, int, int]]:
    tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    calls: list[tuple[int, int, int]] = []
    for index, token in enumerate(tokens):
        if token.type != tokenize.NAME or token.string != "errorbar":
            continue
        previous = _previous_significant(tokens, index)
        following = _next_significant(tokens, index)
        if previous is None or following is None:
            continue
        if tokens[previous].type != tokenize.OP or tokens[previous].string != ".":
            continue
        if tokens[following].type != tokenize.OP or tokens[following].string != "(":
            continue
        calls.append((following, _matching_close(tokens, following), index))
    return calls


def patch_source(source: str) -> tuple[str, int]:
    tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    offsets = _offsets(source)
    edits: list[Edit] = []
    count = 0

    for open_index, close_index, _ in _errorbar_calls(source):
        open_token = tokens[open_index]
        close_token = tokens[close_index]
        call_start = _absolute(offsets, open_token.end)
        call_end = _absolute(offsets, close_token.start)
        call_body = source[call_start:call_end]

        literal = re.search(
            r"\becolor\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
            call_body,
            flags=re.DOTALL,
        )
        if literal:
            value_start = call_start + literal.start("quote")
            value_end = call_start + literal.end("quote")
            if source[value_start:value_end] != '"black"':
                edits.append(Edit(value_start, value_end, '"black"'))
                count += 1
            continue

        if re.search(r"\becolor\s*=", call_body):
            raise RuntimeError(
                "Found a non-literal ecolor expression in an errorbar call. "
                "Set it explicitly to 'black' before applying this drop-in."
            )

        previous = _previous_significant(tokens, close_index)
        if previous is None or previous <= open_index:
            insertion_at = call_start
            prefix = ""
        else:
            previous_token = tokens[previous]
            insertion_at = _absolute(offsets, previous_token.end)
            prefix = "" if previous_token.type == tokenize.OP and previous_token.string == "," else ","

        multiline = open_token.start[0] != close_token.start[0]
        if multiline:
            close_line = source.splitlines()[close_token.start[0] - 1]
            close_indent = close_line[: close_token.start[1]]
            inner_indent = close_indent + "    "
            replacement = (
                f"{prefix}\n{inner_indent}ecolor=\"black\", "
                "elinewidth=1.5, capthick=1.5, zorder=5"
            )
        else:
            replacement = (
                f"{prefix} ecolor=\"black\", "
                "elinewidth=1.5, capthick=1.5, zorder=5"
            )
        edits.append(Edit(insertion_at, insertion_at, replacement))
        count += 1

    for edit in sorted(edits, key=lambda item: (item.start, item.end), reverse=True):
        source = source[: edit.start] + edit.replacement + source[edit.end :]

    ast.parse(source)
    return source, count


def validate_tree(source_root: Path) -> list[str]:
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
                failures.append(f"{path}:{node.lineno}: missing ecolor='black'")
                continue
            if not isinstance(keyword.value, ast.Constant) or keyword.value.value != "black":
                failures.append(f"{path}:{node.lineno}: ecolor is not the literal 'black'")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Make every Matplotlib error bar in the report code explicitly black."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    source_root = repo / "src" / "cvt_track_study"
    if not source_root.is_dir():
        raise SystemExit(f"Could not find source tree: {source_root}")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_root = repo / ".dropin-backups" / f"black-errorbars-{stamp}"
    changed: list[tuple[Path, int]] = []

    for path in sorted(source_root.rglob("*.py")):
        original = path.read_text(encoding="utf-8")
        patched, count = patch_source(original)
        if patched == original:
            continue
        relative = path.relative_to(repo)
        backup = backup_root / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        path.write_text(patched, encoding="utf-8")
        changed.append((relative, count))

    # The ranking is paired; the raw absolute median can differ by a few milliseconds.
    # Keep the headline label aligned with the actual sorting rule.
    report_source = source_root / "reports" / "postprocess.py"
    if report_source.is_file():
        original = report_source.read_text(encoding="utf-8")
        corrected = original.replace(
            '("Best median lap time", str(winner)',
            '("Top paired design", str(winner)',
        )
        if corrected != original:
            relative = report_source.relative_to(repo)
            backup = backup_root / relative
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(report_source, backup)
            report_source.write_text(corrected, encoding="utf-8")
            changed.append((relative, 1))

    failures = validate_tree(source_root)
    if failures:
        raise SystemExit("\n".join(failures))

    print(f"Applied {sum(count for _, count in changed)} report-source edit(s) in {len(changed)} file(s).")
    for relative, count in changed:
        print(f"  {relative}: {count}")
    if changed:
        print(f"Backups: {backup_root}")
    else:
        print("No source changes were needed; all error bars are already explicitly black.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
