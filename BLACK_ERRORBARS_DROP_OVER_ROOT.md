# Black error bars — root overlay

Copy the contents of this archive directly over the repository root and allow
`src` and `tests` to merge.

The change installs one report-wide Matplotlib policy before any report module
is imported. Every `Axes.errorbar(...)` call is therefore rendered with black,
thicker error intervals, including design, uncertainty, track-robustness, and
structural-sensitivity reports.

No simulation rerun is needed for an existing result. Regenerate its report:

```powershell
drivetrain-study report .\path\to\existing\result
```

Optional verification:

```powershell
python -m pytest tests\test_black_errorbars.py -q
```
