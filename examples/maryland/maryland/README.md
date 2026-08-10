# Maryland project configuration

This folder is ready to copy to:

```text
projects/maryland
```

It includes:

- complete project, track, event, vehicle, and study TOMLs;
- the known Maryland `ETS.gpx`;
- the known `78_Baja_ETS.csv`;
- explicit placeholders for the timed McMaster recording and driver IDs;
- `CONFIG_TODO.md` listing every unresolved item.

The workbook controls all disagreements with the timestamped notes.

## Workflow

After replacing the required placeholders:

```powershell
drivetrain-study validate .\projects\maryland `
  --study track_robustness --strict

drivetrain-study build-track .\projects\maryland
```

Review:

```text
projects\maryland\results\track_build\<result>\review\track_evidence_report.html
projects\maryland\results\track_build\<result>\track\route_variant_summary.csv
projects\maryland\results\track_build\<result>\track\gate_review.csv
projects\maryland\results\track_build\<result>\lap_time_reconstruction.csv
```

Once the nominal track is approved:

```powershell
drivetrain-study run track-robustness .\projects\maryland `
  --study track_robustness `
  --workers 6 `
  --restart `
  --run-name maryland-track-robustness
```

Once the robustness report is approved:

```powershell
drivetrain-study run full-uncertainty .\projects\maryland `
  --study full_uncertainty `
  --workers 6 `
  --restart `
  --run-name maryland-uncertainty
```

Once the uncertainty result looks physically and numerically reasonable:

```powershell
drivetrain-study run design-comparison .\projects\maryland `
  --study final_drive_sweep `
  --workers 6 `
  --restart `
  --run-name maryland-final-drive-screen
```

The final-drive sweep uses the uncertainty-informed scenario-reduction code: it selects representative completed worlds from the latest Maryland uncertainty result and varies only `drivetrain.final_drive_ratio`.
