# Canonical HTML report style contract

All six framework reports use `cvt_track_study.reports.html.render_page` as their
outer document shell.

## Required presentation contract

Every canonical report must provide:

- the same responsive typography, spacing, cards, figures, warnings, and footer;
- exactly one `report-nav` quick-link bar immediately below the report header;
- stable anchors for every major `h2` section;
- major conclusions and plots before exhaustive tables;
- sortable tables whose headers cycle ascending, descending, then original order;
- sticky identity columns wherever horizontal scrolling is necessary;
- searchable supporting tables when row count or width makes scanning difficult;
- self-contained embedded plots so the HTML remains portable.

`render_page` automatically creates section anchors and quick links from `h2`
headings unless the report already supplies a deliberate `report-nav`. This keeps
simple reports consistent while preserving curated navigation for larger reports.

## Track evidence compatibility

The track build still writes the historical `review/track_review.html` path. The
canonical postprocessor now rewrites both that legacy path and
`review/track_evidence_report.html` to the same shared-style document. This avoids
leaving the track-generation output as a visual outlier while preserving old links.

## Scientific-content rule

Presentation cleanup must not remove machine-readable artifacts, plots, warnings,
review flags, or detailed evidence. Wide tables may be moved into collapsible
appendices, but their data and interactive sorting/search remain available.
