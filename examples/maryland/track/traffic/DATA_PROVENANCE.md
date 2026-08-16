# Traffic evidence provenance

- `maryland_2025_traffic_observations.csv` contains the 26 reviewed Maryland 2025 traffic events recovered from the prior `maxChanges` traffic dataset (`Maryland 2025 Traffic Data`). The one-hour observer exposure is 0–3600 s. Maryland supplies the **target event-rate level and the duration/severity marks**.
- `cwru_az_2025_cleaned.csv` and `ets_az_2025_cleaned.csv` are the cleaned Arizona 2025 observer tables supplied by the user. CWRU reviewed exposure is 0–900 s; ÉTS is 0–4500 s. Together with field attrition they identify the **race-time decay shape**.
- The Arizona source heading calls its speed field a “reduction”, but slowdown/yellow annotations behave as **retained speed fraction**. The three rows explicitly labelled `full stop` are normalized to retained fraction `0` even though the source cell is `1.0`.
- `arizona_2025_field_survival_derived.csv` contains 59 active-duration, final-lap-duration, and censor records recovered from the prior Arizona lap-timing analysis (`Lap Time Data-20260809T100506Z-1-001.zip`). Non-finishers use a 0.5-final-lap nominal retirement offset, matching the original derivation.
- Maryland has no corresponding four-hour field-attrition dataset. The model therefore transfers the Arizona field-survival **shape** to Maryland while fitting Maryland's own rate scale from its one-hour observer data. This is narrower than transferring the entire Arizona traffic distribution.
- CWRU and ÉTS are not forced to share one absolute source encounter rate. Each receives a nuisance rate scale while the race-time exponent is shared. This prevents the much higher CWRU 15-minute rate from being misread as stronger time decay simply because CWRU was only observed early in the race.

## Evidence hashes

- `cwru_az_2025_cleaned.csv` — SHA-256 `db099143c71e09bea6434b7eb04019404d6a7df19bc182cbc90b063f1cc5684c` (matches the user-supplied cleaned CWRU CSV byte-for-byte).
- `ets_az_2025_cleaned.csv` — SHA-256 `9c5e0d4936ebbbec7cc69a64cc46240147ad8b2c825f10c34bdc00d67223bd2b` (matches the user-supplied cleaned ÉTS CSV byte-for-byte).
- `maryland_2025_traffic_observations.csv` — SHA-256 `8ed49e33610d39f89860b21ee1908481c5f09e52293bef73d0ba9e55ea76e798`.
- `arizona_2025_field_survival_derived.csv` — SHA-256 `6945e909f3a214da577f2893d6b9318d9cc682b43980c5b91c67958d0c124e6e`.
