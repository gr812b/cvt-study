# Consensus cleanup drop-in v4

This version fixes the export failure:

```text
pandas.errors.InvalidIndexError:
Reindexing only valid with uniquely valued Index objects
```

The cause was a duplicate `map_error_m` label in the post-map rejected
point table. The rejection schema now guarantees unique labels, and the
exporter defensively coalesces any legacy duplicate columns before
concatenation.

Extract over the repository root, approve replacement, and run:

```powershell
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"
pytest -q tests/test_rejection_export_schema.py
drivetrain-study build-track .\projects\arizona
```

No FIT/GPX data or completed centreline work is lost. `build-track`
reconstructs and exports a new atomic result package.
