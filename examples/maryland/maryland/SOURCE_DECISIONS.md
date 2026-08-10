# Maryland source decisions

## Controlling-source rule

Per the user's instruction, the **Maryland CVT Data Entry workbook controls wherever it disagrees with the timestamped event notes**.

Therefore the generated configuration retains the workbook's:

- event identity and count;
- uphill/downhill labels;
- turn directions;
- bump/log/berm classification where a workbook row exists;
- listed dimensions and radii;
- workbook endpoints when the timestamped notes duplicate or omit a coordinate.

The timestamped notes are used only to add events that have no corresponding workbook row:

- four large bumps after workbook `Bump 17`;
- 90° left turn at 8:00;
- centre berm and left turn beginning at 8:02;
- turns at 8:14, 8:19, 8:23, and 8:26;
- circuit 180° left turn before workbook `Turn 19`.

The 3:14 circuit-entry note has no coordinate and is omitted pending review.

## Recording roles

- **McMaster timed recording:** centreline + native speed/gate evidence.
- **ETS untimed GPX with lap-time reconstruction:** centreline/route/elevation support only.
- ETS has `use_for_gate_evidence = false`, so reconstructed speed does not set gates.

## Route policy

The McMaster recording is expected to contain more than one repeated route. The nominal build uses:

```toml
selection = "reference_run"
reference_run_id = "ets_78_maryland_reconstructed"
```

This selects the supported route containing the ETS long-course evidence and keeps other supported routes auditable rather than averaging them together.
