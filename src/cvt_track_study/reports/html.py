"""Small dependency-free HTML helpers shared by all report types."""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


CSS = r"""
:root { color-scheme: light dark; --ink:#1f2933; --muted:#52606d; --line:#d9e2ec;
  --panel:#f7f9fb; --accent:#245b8f; --good:#176b3a; --warn:#8a5a00; --bad:#9b1c1c;
  --sticky:#f7f9fb; --sticky-alt:#eef3f7; }
@media (prefers-color-scheme: dark) { :root { --ink:#e8eef4; --muted:#b8c4cf; --line:#44515e;
  --panel:#17212b; --accent:#7db8ef; --good:#72d69a; --warn:#ffd27a; --bad:#ff9191;
  --sticky:#17212b; --sticky-alt:#1c2935; } }
* { box-sizing:border-box; }
body { font-family:Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
  margin:0; color:var(--ink); background:Canvas; line-height:1.48; }
main { max-width:1320px; margin:0 auto; padding:2rem 2.2rem 4rem; }
header { border-bottom:1px solid var(--line); margin-bottom:1.6rem; }
h1 { font-size:2rem; margin:.1rem 0 .35rem; }
h2 { margin-top:2.6rem; border-bottom:1px solid var(--line); padding-bottom:.4rem; scroll-margin-top:1rem; }
h3 { margin-top:1.55rem; scroll-margin-top:1rem; }
.subtitle { color:var(--muted); max-width:78rem; }
.report-nav { display:flex; flex-wrap:wrap; gap:.45rem .9rem; padding:.75rem 0 1rem; border-bottom:1px solid var(--line); }
.report-nav a { color:var(--accent); text-decoration:none; font-weight:650; }
.report-nav a:hover { text-decoration:underline; }
.scope { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:.8rem; margin:1.1rem 0; }
.card { border:1px solid var(--line); border-radius:10px; padding:.9rem 1rem; background:var(--panel); }
.card .label { color:var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.04em; }
.card .value { font-size:1.25rem; font-weight:700; margin-top:.2rem; }
.good { border-left:5px solid var(--good); } .warning { border-left:5px solid var(--warn); }
.bad { border-left:5px solid var(--bad); } .note { border-left:5px solid var(--accent); }
.section-intro { max-width:78rem; padding:.85rem 1rem; margin:.75rem 0 1.1rem; border-left:4px solid var(--accent); background:var(--panel); border-radius:0 8px 8px 0; }
.section-intro strong { display:block; margin-bottom:.2rem; }
.finding-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:.75rem; margin:1rem 0 1.5rem; }
.finding { border:1px solid var(--line); border-radius:9px; padding:.8rem .9rem; }
.finding strong { display:block; margin-bottom:.2rem; }
.figure { margin:1.2rem 0 1.9rem; }
.figure img { display:block; max-width:100%; height:auto; border:1px solid var(--line); border-radius:8px; }
.caption { color:var(--muted); font-size:.9rem; margin-top:.4rem; max-width:78rem; }
.table-note { color:var(--muted); font-size:.91rem; margin:.55rem 0 .65rem; max-width:78rem; }
.table-controls { display:flex; flex-wrap:wrap; gap:.55rem; align-items:center; margin:.45rem 0 .55rem; }
.table-search { min-width:18rem; max-width:34rem; width:100%; padding:.48rem .6rem; border:1px solid var(--line); border-radius:7px; background:Canvas; color:var(--ink); }
.table-count { color:var(--muted); font-size:.86rem; }
.table-wrap { overflow:auto; max-height:680px; border:1px solid var(--line); border-radius:8px; position:relative; }
.table-wrap.compact { max-height:520px; }
table { border-collapse:separate; border-spacing:0; width:max-content; min-width:100%; font-size:.86rem; }
th, td { border-bottom:1px solid var(--line); padding:.45rem .55rem; text-align:left; vertical-align:top; white-space:nowrap; }
th { position:sticky; top:0; background:var(--panel); z-index:4; box-shadow:0 1px 0 var(--line); }
th button.sort-button { all:unset; cursor:pointer; display:flex; align-items:center; gap:.35rem; width:100%; font-weight:700; }
th button.sort-button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:3px; }
.sort-indicator { color:var(--muted); min-width:1em; font-size:.78rem; }
tbody tr:nth-child(even) td { background:color-mix(in srgb, var(--panel) 65%, transparent); }
td.sticky-col, th.sticky-col { position:sticky; z-index:3; background:var(--sticky); box-shadow:1px 0 0 var(--line); }
tbody tr:nth-child(even) td.sticky-col { background:var(--sticky-alt); }
th.sticky-col { z-index:6; }
.identity-column { min-width:8.5rem; max-width:18rem; white-space:normal; overflow-wrap:anywhere; }
.numeric-column { font-variant-numeric:tabular-nums; }
.status-core { font-weight:700; color:var(--good); }
.status-conditional, .status-near-miss { font-weight:700; color:var(--warn); }
.status-unsupported { color:var(--muted); }
details { margin:1rem 0; border:1px solid var(--line); border-radius:9px; padding:.2rem .9rem .9rem; }
details > summary { cursor:pointer; font-weight:700; padding:.7rem 0; }
code, pre { font-family:ui-monospace, SFMono-Regular, Consolas, monospace; }
pre { overflow:auto; padding:.8rem; background:var(--panel); border:1px solid var(--line); border-radius:8px; }
footer { color:var(--muted); margin-top:3rem; border-top:1px solid var(--line); padding-top:1rem; font-size:.88rem; }
@media (max-width:760px) { main { padding:1.1rem .8rem 3rem; } h1 { font-size:1.65rem; } }
"""

SCRIPT = r"""
(function () {
  function updateStickyOffsets(table) {
    const stickyHeaders = Array.from(table.querySelectorAll('thead th.sticky-col'));
    let left = 0;
    stickyHeaders.forEach((header) => {
      const columnIndex = header.dataset.columnIndex;
      table.querySelectorAll('[data-column-index="' + columnIndex + '"]').forEach((cell) => {
        cell.style.left = left + 'px';
      });
      left += header.getBoundingClientRect().width;
    });
  }

  function compareValues(a, b, kind) {
    const aBlank = a === '' || a === null || a === undefined;
    const bBlank = b === '' || b === null || b === undefined;
    if (aBlank && bBlank) return 0;
    if (aBlank) return 1;
    if (bBlank) return -1;
    if (kind === 'number') return Number(a) - Number(b);
    return String(a).localeCompare(String(b), undefined, {numeric:true, sensitivity:'base'});
  }

  document.querySelectorAll('input[data-table-search]').forEach((input) => {
    const table = document.getElementById(input.dataset.tableSearch);
    if (!table || !table.tBodies[0]) return;
    const rows = Array.from(table.tBodies[0].rows);
    const count = document.querySelector('[data-table-count="' + input.dataset.tableSearch + '"]');
    function applyFilter() {
      const query = input.value.trim().toLocaleLowerCase();
      let visible = 0;
      rows.forEach((row) => {
        const match = !query || row.textContent.toLocaleLowerCase().includes(query);
        row.hidden = !match;
        if (match) visible += 1;
      });
      if (count) count.textContent = visible + ' of ' + rows.length + ' rows';
    }
    input.addEventListener('input', applyFilter);
    applyFilter();
  });

  document.querySelectorAll('table[data-sortable="true"]').forEach((table) => {
    const body = table.tBodies[0];
    if (!body) return;
    const originalRows = Array.from(body.rows);
    const headers = Array.from(table.querySelectorAll('thead th'));
    headers.forEach((header) => {
      const button = header.querySelector('button.sort-button');
      if (!button) return;
      button.addEventListener('click', () => {
        const previous = header.dataset.sortState || 'none';
        const next = previous === 'none' ? 'asc' : previous === 'asc' ? 'desc' : 'none';
        headers.forEach((item) => {
          item.dataset.sortState = 'none';
          item.setAttribute('aria-sort', 'none');
          const indicator = item.querySelector('.sort-indicator');
          if (indicator) indicator.textContent = '↕';
        });
        header.dataset.sortState = next;
        header.setAttribute('aria-sort', next === 'none' ? 'none' : next === 'asc' ? 'ascending' : 'descending');
        const indicator = header.querySelector('.sort-indicator');
        if (indicator) indicator.textContent = next === 'asc' ? '▲' : next === 'desc' ? '▼' : '↕';

        let rows = originalRows.slice();
        if (next !== 'none') {
          const index = Number(header.dataset.columnIndex);
          const kind = header.dataset.sortKind || 'text';
          rows.sort((leftRow, rightRow) => {
            const left = leftRow.cells[index]?.dataset.sortValue ?? '';
            const right = rightRow.cells[index]?.dataset.sortValue ?? '';
            const compared = compareValues(left, right, kind);
            if (compared !== 0) return next === 'asc' ? compared : -compared;
            return Number(leftRow.dataset.originalIndex) - Number(rightRow.dataset.originalIndex);
          });
        }
        rows.forEach((row) => body.appendChild(row));
      });
    });
    updateStickyOffsets(table);
  });
  window.addEventListener('resize', () => {
    document.querySelectorAll('table').forEach(updateStickyOffsets);
  });
})();
"""

_TABLE_IDS = count(1)


def image_data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "image/svg+xml" if suffix == "svg" else f"image/{suffix}"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(path: Path, caption: str, alt: str | None = None) -> str:
    if not path.is_file():
        return ""
    return (
        '<div class="figure"><img alt="'
        + html.escape(alt or caption)
        + '" src="'
        + image_data_uri(path)
        + '"><div class="caption">'
        + html.escape(caption)
        + "</div></div>"
    )


def dataframe_table(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    max_rows: int = 200,
    float_digits: int = 3,
    sticky_columns: Iterable[str] = (),
    sortable: bool = True,
    table_id: str | None = None,
    compact: bool = False,
    column_labels: Mapping[str, str] | None = None,
    searchable: bool = False,
    search_placeholder: str = "Search rows…",
) -> str:
    """Render a readable table with optional sticky identity columns and sorting.

    Sortable headers cycle through ascending, descending, and the original row
    order. Sticky columns are moved to the left before rendering so their left
    offsets remain deterministic while horizontally scrolling.
    """

    if frame is None or frame.empty:
        return '<p class="subtitle">No rows available.</p>'
    selected = frame.copy()
    if columns is not None:
        available = [column for column in columns if column in selected.columns]
        selected = selected[available]
    sticky = [column for column in sticky_columns if column in selected.columns]
    if sticky:
        selected = selected[sticky + [column for column in selected.columns if column not in sticky]]
    selected = selected.head(max_rows)
    labels = dict(column_labels or {})
    identifier = table_id or f"data-table-{next(_TABLE_IDS)}"

    numeric = {
        column: bool(pd.api.types.is_numeric_dtype(selected[column].dtype))
        and not pd.api.types.is_bool_dtype(selected[column].dtype)
        for column in selected.columns
    }

    def visible(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(float(value)):
                return ""
            return f"{float(value):.{float_digits}f}"
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if bool(value) else "No"
        return str(value)

    def sort_value(value: Any, is_numeric: bool) -> str:
        if value is None:
            return ""
        if is_numeric:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return ""
            return "" if not np.isfinite(number) else repr(number)
        return str(value).strip().lower()

    head_cells = []
    for index, column in enumerate(selected.columns):
        classes = []
        if column in sticky:
            classes.extend(("sticky-col", "identity-column"))
        sort_kind = "number" if numeric[column] else "text"
        label = labels.get(column, column.replace("_", " ").strip().title())
        button = (
            '<button type="button" class="sort-button">'
            + html.escape(label)
            + '<span class="sort-indicator" aria-hidden="true">↕</span></button>'
            if sortable
            else html.escape(label)
        )
        head_cells.append(
            f'<th class="{" ".join(classes)}" data-column-index="{index}" '
            f'data-sort-kind="{sort_kind}" data-sort-state="none" aria-sort="none">{button}</th>'
        )

    rows = []
    for original_index, (_, record) in enumerate(selected.iterrows()):
        cells = []
        for index, column in enumerate(selected.columns):
            value = record[column]
            classes = []
            if column in sticky:
                classes.extend(("sticky-col", "identity-column"))
            if numeric[column]:
                classes.append("numeric-column")
            if column in {"frontier_classification", "status"}:
                status = str(value).strip().lower().replace("_", "-")
                if status in {"core", "conditional", "near-miss", "unsupported"}:
                    classes.append(f"status-{status}")
            cells.append(
                f'<td class="{" ".join(classes)}" data-column-index="{index}" '
                f'data-sort-value="{html.escape(sort_value(value, numeric[column]), quote=True)}">'
                + html.escape(visible(value))
                + "</td>"
            )
        rows.append(f'<tr data-original-index="{original_index}">' + "".join(cells) + "</tr>")

    wrap_class = "table-wrap compact" if compact else "table-wrap"
    controls = ""
    if searchable:
        controls = (
            '<div class="table-controls"><input class="table-search" type="search" '
            f'data-table-search="{html.escape(identifier)}" placeholder="{html.escape(search_placeholder)}" '
            'aria-label="Search table rows">'
            f'<span class="table-count" data-table-count="{html.escape(identifier)}"></span></div>'
        )
    return (
        controls
        + f'<div class="{wrap_class}"><table id="{html.escape(identifier)}" '
        f'data-sortable="{"true" if sortable else "false"}"><thead><tr>'
        + "".join(head_cells)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def metric_cards(records: Iterable[tuple[str, str, str]]) -> str:
    return '<div class="scope">' + "".join(
        f'<div class="card {html.escape(kind)}"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value, kind in records
    ) + "</div>"


def render_page(
    *,
    title: str,
    subtitle: str,
    body: str,
    report_key: str,
    source_note: str = "",
) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>
<header><div class="subtitle">Measured-track drivetrain framework · {html.escape(report_key)}</div>
<h1>{html.escape(title)}</h1><p class="subtitle">{html.escape(subtitle)}</p></header>
{body}
<footer>Generated {html.escape(generated)}. {html.escape(source_note)}</footer>
</main><script>{SCRIPT}</script></body></html>"""


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def nested_get(mapping: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value
