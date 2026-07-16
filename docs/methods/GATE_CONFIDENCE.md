# Speed-gate evidence and confidence score

A speed gate is accepted only when repeated GPX/FIT passes support a repeatable,
driver-limited entry state. The score is a transparent evidence summary; it is
**not** a probability that the gate is correct.

For gate \(g\), the default score is

\[
C_g = 100\left(
0.15C_n + 0.25C_r + 0.20C_b + 0.15C_p + 0.15C_c + 0.10C_v
\right).
\]

Every component and weight is configurable and exported.

## Pass count, \(C_n\)

\[
C_n = \min\left(1,\frac{N_{\mathrm{valid}}}{N_{\mathrm{target}}}\right).
\]

The default target is ten valid passes; fewer than five triggers review even if
other components are strong.

## Entry-speed repeatability, \(C_r\)

Using the empirical entry-speed interquartile range,

\[
C_r = \max\left(0,1-\frac{\operatorname{IQR}(v_{\mathrm{entry}})}
{s_r}\right),
\]

where the default repeatability scale \(s_r\) is 2 m/s.

## Braking evidence, \(C_b\)

\(C_b\) is the fraction of valid laps whose approach-to-entry speed reduction
exceeds the configured threshold. It prevents a naturally slow map location from
being treated as a driver-enforced gate without upstream slowing evidence.

## Pace independence, \(C_p\)

Let \(\rho_s\) be Spearman correlation between event entry speed and each lap's
median speed:

\[
C_p = 1-|\rho_s|.
\]

A gate that merely rises and falls with whole-lap pace is weaker evidence of a
local obstacle constraint. When correlation cannot be estimated, the component
uses a neutral 0.5 rather than claiming independence.

## Coordinate quality, \(C_c\)

Gate entry speed is measured relative to the physical start of the response
interval. Coordinate quality therefore uses the effective **physical-start**
error rather than anchor error alone.

For a start constructed from an anchor and a before-anchor extent,

\[
e_{\mathrm{start}} =
\sqrt{e_{\mathrm{anchor,projection}}^2
+u_{\mathrm{anchor}}^2
+u_{\mathrm{before}}^2}.
\]

For an explicitly supplied start coordinate,

\[
e_{\mathrm{start}} =
\sqrt{e_{\mathrm{start,projection}}^2+u_{\mathrm{start}}^2}.
\]

The component is then

\[
C_c = \max\left(0,1-\frac{e_{\mathrm{start}}}{e_{\max}}\right).
\]

This prevents a precisely projected anchor from receiving high coordinate
confidence when the first physical contact or interval extent remains poorly
known. The physical-end error is also exported for obstacle and exit-window
review, although the current gate score uses the start because it controls the
entry measurement.

## Cross-vehicle agreement, \(C_v\)

When at least two vehicles contribute valid passes, the spread between their
median entry speeds is compared with a configured agreement scale. With one
vehicle, the score is deliberately neutral at 0.5 and the output states
`single_vehicle_neutral`; it is not counted as observed agreement.

## Slowdown signature

Separately from the weighted score, each response group is labelled strong,
moderate, weak, or insufficient using the median approach-to-minimum slowdown
and the fraction of laps exceeding the braking threshold. This is a diagnostic
summary, not another acceptance rule.

## Review status

- `must_fix`: anchor projection or effective physical-start error exceeds the allowed map error;
- `recommended_review`: too few valid passes or intermediate evidence score;
- `accepted`: candidate score meets the acceptance threshold;
- `rejected`: candidate evidence is below the review threshold;
- `not_a_candidate`: the user explicitly did not nominate the response as a gate.

Acceptance preserves the empirical entry-speed distribution (median, mean,
standard deviation, p10, p90, and IQR). It does not collapse it to an exact speed.

## Optional sustained-response gate

An accepted entry gate does not automatically constrain speed deeper into a feature.
A second `sustained_response` gate is emitted only when all of the following pass:

- at least the configured minimum number of complete response passes;
- at least 80% of laps show an entry-to-minimum drop exceeding the braking threshold;
- a one-sided exact binomial test rejects a 50% success rate at p <= 0.05;
- response-minimum speed and spatial location both meet repeatability limits; and
- leave-one-lap-out medians stay within the declared method limits.

When any check fails, the accepted entry gate remains and the bundle records
`entry_only_fallback`; no response gate is added. A qualified response gate carries
its own empirical minimum-speed samples and pairs with the entry gate through the
same one-way braking-envelope logic.
