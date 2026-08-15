# Traffic model validation snapshot

Calibration evidence bundled in this overlay:

- Maryland target: 26 reviewed events, 3600 s exposure.
- Arizona CWRU source: 21 reviewed events, 900 s exposure.
- Arizona ETS source: 37 reviewed events, 4500 s exposure.
- Arizona field survival: 59 derived active-duration/censor records; non-finishers use a 0.5-final-lap retirement offset.

Current fit (schema v3):

- Maryland `lambda0`: ~32.413/h.
- CWRU source nuisance `lambda0`: ~86.558/h.
- ETS source nuisance `lambda0`: ~38.845/h.
- shared field-survival exponent: ~2.8350.
- nominal four-hour average Maryland rate: ~15.914/h.
- source NHPP time-rescaling KS p: ~0.435.
- Maryland target NHPP time-rescaling KS p: ~0.263.

Source-only model comparison with equal stream-intercept freedom:

- field-survival: AIC ~634.10.
- clock-exponential: AIC ~633.94.
- no-decay/homogeneous: AIC ~633.10.

These are effectively tied. The production distribution therefore does not interpret the nominal positive decay exponent as certain. The deterministic 512-draw calibration bootstrap includes near-zero decay and propagates that uncertainty into generated traffic worlds.

Maryland 3,000-replicate predictive validation currently gives approximately:

- event count: observed 26; predictive median 25; central 90% about 14–38.
- occupied traffic fraction: observed 0.179; predictive median ~0.155; central 90% ~0.079–0.242.
- mean duration: observed 24.73 s; predictive mean ~24.6 s.
- mean retained-speed fraction: observed 0.585; predictive mean ~0.585.
- yellow-flag fraction: observed 0.346; predictive mean ~0.349.
- stop-like retained=0 fraction: observed 0.077; predictive mean ~0.075.

The target summaries are therefore reproduced on average without fitting an additional parametric severity law.

## Arrival-process complexity check

The point-process diagnostics do not justify adding a self-exciting/Hawkes or burst-state process. The fitted NHPP time-rescaling checks are non-rejecting for both Arizona source streams pooled through their own rate scales and the Maryland target stream. Simple fixed-window dispersion checks are also modest: five-minute count Fano factors are about 1.29 for CWRU, 1.15 for ÉTS, and 0.57 for Maryland. Given the short CWRU exposure and only one Maryland observer stream, a more complicated clustering law would add poorly identified parameters rather than demonstrably improve average emulation.

## Downstream design replay

Traffic is serialized as part of each full-uncertainty source world. Representative-world reduction includes compact traffic descriptors (race phase, calibrated rate/exponent, event count, annotated duration, minimum retained fraction, and duration-weighted restriction burden) so traffic variation contributes to diversity selection. The selected design replay then carries the exact saved traffic event realization rather than drawing a replacement.
