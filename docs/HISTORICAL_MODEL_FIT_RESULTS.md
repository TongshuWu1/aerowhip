# Historical complete-model fit — September 8 UTC

The user subsequently restricted drone response fitting to the three historical whips. The pooled drone results on this page remain historical; see [the replacement whip-only candidate](WHIP_ONLY_DRONE_FIT.md). The cable results below are unchanged, and this joint review must not be reused as validation of the replacement drone model.

The complete candidate is fitted and saved separately. It is **not applied to the active calibration**. The initial unrestricted cable NN was worse than the previous calibration; a zero-initialized dissipative NN recovered most of that difference, but the historical whip comparison still regresses. Do not describe this candidate as a demonstrated improvement or start a new PPO assuming the active model has changed.

Run: `data/historical_model_runs/20260908-002241-654934`.
Bundle: `candidate_bundle/model.json`, with both NN checkpoints beside it. It enables the complete model when deliberately selected, but it is not an exported PPO checkpoint. All rotated cable and combined-model checks are complete. The final machine-readable review is `review.json`, with status `REVIEW_REQUIRED`; the candidate remains inactive.

## Data used

All eight preliminary and three historical whip recordings contribute usable data, including `fig8vertical_002`. No whole recording was discarded. Of 37,145 frames, 36,995 remain eligible for cable fitting (99.6%) and 34,884 for command/drone fitting (93.9%). The objectives use valid contiguous, motion-stratified windows from these frames; these counts do not mean every frame appears in every loss.

Questionable intervals are masked, not deleted: marker gaps, large per-frame jumps, excess marker-chord length, stale commands, invalid attachment orientation and possible near-floor contact. The 105-frame contact quarantine in `whip1_002` is at the beginning during takeoff, not after the whip. Failed airborne whip motion remains eligible. The conservative thresholds and exact intervals are in `audit.json`; input and processed-source checksums are in `protocol.json` and `integrity_review.json`.

Historical whip commands switch to hold around 0.67 s. The fit uses that actual logged command stream rather than inventing the unexecuted end of the intended CSV. Raw recordings, the original role manifest and historical study snapshots remain preserved.

## Fitted components

| Component | Final candidate |
|---|---|
| Measured masses | Drone 157 g; cable and markers 18 g; total 175 g, fixed |
| Cable lateral attachment offset | `[0.0070848663, -0.0141363152]` m; vertical offset remains −0.055 m |
| Effective EI | `1e-9 N m²`, at the search lower bound |
| Internal bending damping Cb | `1e-4 N m² s` |
| External scalar drag | Exactly zero in all new physical/residual candidates |
| Cable residual | 4,577 NN parameters; bounded, dissipative, initialized at zero |
| Effective attachment response | Nine fitted gains; selected delay 60 ms |
| Drone residual | Small 21→32→32→3 bounded acceleration NN |

The cable network learns state-dependent velocity-opposing corrections, not a separate initialized 0.3 s⁻¹ coefficient. Its final output-layer calibration uses training predictions only. The original acceleration-NN attempt, the dissipative pilot, the calibrated candidate and all development checks remain separate artifacts.

EI is weakly identified: fold estimates vary by many orders of magnitude for very small objective changes. The drone model's vertical feedforward gain reaches its allowed upper bound of 2. These are effective prediction parameters for this setup, not material measurements or firmware/controller gains to copy onto the drone.

The fitted 60 ms delay also depends on logger reception timestamps and the alignment procedure; it is not a direct measurement of radio or firmware latency.

The drone surrogate already includes the recorded cable load. Its predicted attachment drives a separate cable simulation without adding the same cable reaction to the drone twice. The full training branch uses frozen 20 Hz virtual forces → 30 Hz commanded P/V/A → predicted attachment → cable physics and residual → strike reward. It does not score recovery. Existing gentle recovery export and old saved policy configurations remain separate.

## Accuracy

All numbers below are equal-take means of Euclidean position RMSE. They are prediction errors, not target hitting errors.

| Diagnostic | Before | Candidate | Interpretation |
|---|---:|---:|---|
| Drone attachment, 1 s, entire takes excluded from fitting | 2.933 cm | 2.732 cm | Nominal response versus nominal + drone NN; development improvement |
| Cable tip, 2 s, entire takes excluded from fitting | 7.593 cm | 6.251 cm | New zero-drag physics-only fit versus physics + cable NN |
| Complete command-to-tip chain, 2 s, entire takes excluded from fitting | 12.137 cm | 9.512 cm | New zero-drag physics + nominal response versus both NNs |
| Cable markers, 2 s, final all-data comparison | 3.615 cm | 3.577 cm | Previous calibration versus new physics + cable NN |
| Cable tip, same all-data comparison | 6.136 cm | 6.154 cm | Essentially unchanged overall; no improvement established |
| Cable tip, historical whip recordings only | 7.275 cm | 7.686 cm | Worse by about 5.6% |
| Complete command-to-tip chain, final all-data fit | 11.515 cm | 8.584 cm | New zero-drag physics + nominal response versus both NNs; training-data comparison only |

The cross-fold cable/combined rows and the final row are **not** comparisons against the previous cable calibration. They show why comparing a residual only against a weakened zero-drag model would be misleading.

Whip-specific cable-tip comparisons:

| Recording | Previous calibration | Candidate |
|---|---:|---:|
| whip1_001 | 6.580 cm | 7.273 cm |
| whip1_002 | 8.412 cm | 9.399 cm |
| whip1_003 | 6.833 cm | 6.386 cm |

See `component_diagnostics.png` in the run folder. Historical records have all informed development; none is an independent test of future hitting performance.

## Verification and current selection

The 1,024-case full-model numerical probe passed on Windows with an NVIDIA RTX 4080: zero numerical failures, about 25.8 s including setup. It used the existing actor without optimizer updates; its predicted hits are not a new-policy success estimate. Targeted cable/derivative/checkpoint, interpolation/export, UI, source-consistency and CPU/GPU boundary tests passed. Other operating systems and GPUs were not tested here.

The active model remains the previous calibration, including its historical 0.3 s⁻¹ term; that term is absent from the new fitted candidate. The selected PPO remains `20260906-201957-294110-measured-mass-seed653`, checkpoint SHA-256 `18f33e89fea72e683c40359853e40710f52ecb45623d53669ed6e240334078d7`. No PPO, supervisor or flight was started.

Keep this fit as an initial research candidate and retain the useful drone fit. The next cable-model decision should address the whip-specific regression and weak physical-parameter identification, not discard difficult trials or assume a larger network solves them. Fresh complete-maneuver recordings will provide prospective evidence after a model and policy are frozen. The existing controller/logger were not changed.

Reproduction and component semantics: [HISTORICAL_MODEL_FITTING.md](HISTORICAL_MODEL_FITTING.md).
