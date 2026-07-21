# Unified six-report HTML style drop-in

Base repository: `gr812b/cvt-study`

Verified base commit: `9db64c18a2e948dd0cc44c45a6c552cc1f0b39e2`

Extract this archive over the repository root after the existing six-report,
track-robustness, structural-sensitivity, and full-uncertainty updates.

## What changes

All six canonical reports now use one presentation contract:

- one shared responsive report shell;
- one quick-link bar directly below the report header;
- stable anchors for every major section;
- consistent typography, cards, warnings, figures, captions, tables, and footer;
- major plots and conclusions before exhaustive supporting tables;
- three-state sortable tables: ascending, descending, original order;
- searchable wide tables and sticky identity columns;
- self-contained embedded images.

`render_page` now automatically builds quick links from `h2` headings when a
report does not provide a curated navigation bar. Existing curated navigation in
the track-robustness and structural-sensitivity reports remains intact.

## Track evidence cleanup

The track-generation report was the visual outlier. It is now rebuilt from the
saved track-build CSV, JSON, and image artifacts through the shared report shell.
It includes:

1. an executive evidence summary;
2. the map and telemetry-cleanup plots;
3. event interpretation and flagged intervals;
4. a compact gate-evidence table;
5. elevation and lap support;
6. provenance and diagnostics;
7. full searchable appendices.

Both of these paths now contain the same unified document:

```text
review/track_evidence_report.html
review/track_review.html
```

The second path is retained as a compatibility alias so existing links do not
break.

## Track robustness navigation cleanup

The curated quick-link bar now includes the previously unlinked
`Interpretation and decision rules` section.

## Apply

```powershell
Expand-Archive .\cvt-study-unified-report-style-dropin.zip `
  -DestinationPath F:\Code\Projects\cvt-study `
  -Force

cd F:\Code\Projects\cvt-study
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"
```

Run the focused checks:

```powershell
pytest -q `
  tests/test_report_style_consistency.py `
  tests/test_sortable_sticky_tables.py `
  tests/test_results_index_reports.py
```

Then regenerate any completed report without rerunning its simulations:

```powershell
drivetrain-study report <completed-result-directory>
```

A new `build-track` run automatically writes the unified track-evidence report
because the track-build router invokes the canonical postprocessor.

## Preserved functionality

This change does not alter:

- telemetry reconstruction;
- gate confidence calculations;
- simulation physics;
- uncertainty sampling;
- scenario checkpoints or resume behavior;
- saved machine-readable artifacts;
- existing report filenames.

It changes report rendering and organization only, apart from adding the missing
track-robustness navigation anchor.
