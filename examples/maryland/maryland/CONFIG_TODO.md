# Maryland configuration TODO

Replace or confirm these items before the first strict validation/build.

## Required placeholders

1. **McMaster recording file**
   - Current placeholder: `track/gpx/REPLACE_WITH_MCMASTER_MARYLAND_RECORDING.gpx`
   - Put the real timed McMaster Maryland GPX/FIT in the project and update `track/runs.toml`.
   - If the source is FIT, change the path and extension accordingly.

2. **Driver IDs**
   - `REPLACE_WITH_MCMASTER_DRIVER_ID`
   - `REPLACE_WITH_ETS_DRIVER_ID`

## Required review

3. **Start/finish**
   - The workbook `Start 1` coordinate is used as `start_finish`.
   - Confirm repeated crossings produce sensible lap segmentation.

4. **Route variant selection**
   - The project selects the supported route containing `ets_78_maryland_reconstructed`.
   - Review `track/route_variant_summary.csv` and confirm this is the intended long course.
   - Do not disable ETS lap-time reconstruction without also changing the route-selection policy.

5. **ETS lap-time reconstruction**
   - `ETS.gpx` and `78_Baja_ETS.csv` are included.
   - Review `lap_time_reconstruction.csv`.
   - Keep `use_for_gate_evidence = false`; reconstructed ETS speed must not set speed gates.

6. **Circuit entry at 3:14**
   - The timestamped notes list “Circuit Entry” but provide no coordinate.
   - It is intentionally omitted from `events.toml`.
   - Add it only after a coordinate is supplied and only if it represents a distinct physical response feature.

7. **Timestamp-only additions**
   - The workbook has no explicit rows for the 7:54–8:26 sequence or the circuit 180° left turn.
   - These were added from `message.txt`; review their extents after the first map projection.

8. **Exact site label**
   - The project name is `Baja SAE Maryland 2025 Endurance`.
   - Replace with the exact venue/site name if a more precise label is wanted.

## Vehicle-model review

9. **McMaster vehicle/drivetrain**
   - Values are copied from the current McMaster reduced-order study configuration.
   - Confirm installed final-drive tooth counts, mass, tire diameter, CVT ratio limits, and engine target.

10. **ETS/Cornell vehicle 78**
    - The supplied ETS vehicle/drivetrain files are generic placeholders used only so the geometry evidence has a valid vehicle ID.
    - They do not affect the supplied McMaster-targeted studies.
    - Replace them before simulating or comparing the ETS/Cornell vehicle.

## Known modeling boundary

- Workbook bump/log dimensions are retained in event notes.
- The current reduced-order obstacle model uses broad uncertain energy-loss profiles rather than directly converting every listed height/radius into contact dynamics.
- GPX elevation is stored and plotted, but grade force remains disabled.
