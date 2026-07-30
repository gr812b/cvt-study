# Validation

- Base commit inspected: `97cf8e84b44b1411538502552aeb24f193158bf0`.
- `python -m py_compile` on all included Python files: PASS.
- Updated template parsed with Python `tomllib`: PASS.
- `pytest -q tests/test_route_variants.py`: **4 passed**.

Not run:

- complete repository suite;
- real Maryland telemetry build;
- full track robustness study.

Those require the local repository and raw FIT/GPX data. The first Maryland build
should use `selection = "require_explicit"`; it is expected to stop after detection
when more than one supported route exists.
