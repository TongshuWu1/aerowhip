> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Cable-only residual from preliminary recordings

Open **Data & Calibration → 4 Cable residual**. The **Train cable residual** button fits a new candidate using the active physical calibration and the preliminary training takes. No PPO or drone-residual training is launched. The calibrated masses, lengths, attachment geometry, EI and internal bending damping remain fixed. External cable damping is now learned inside the residual.

Current user correction: `preliminary_cable_residual_v3`, `drag_mode=nn_only`. The residual is a bounded NN acceleration correction, with no separate drag coefficient and no initialization from the calibrated 0.3 s⁻¹. The candidate's physical `external_drag_s_inv` is zero from the start of training through runtime. NN hidden layers have random initialization; the output layer starts at zero, so the initial correction is zero. The old calibrated drag is used only in the historical baseline comparison. It cannot affect NN initialization. Runtime loading rejects an NN-only model with a nonzero separate drag term.

The learned quantities are the NN weights and biases: 4,577 scalars for the current 12-node cable model. They map 75 position/velocity features through two 32-unit hidden layers to 33 acceleration corrections (XYZ on 11 free cable nodes). No mass, EI, internal damping or named drag parameter is optimized. The NN can represent velocity-dependent effects, but does not identify a unique aerodynamic law or guarantee dissipative output. Existing policy runs retain their original saved physics; a new model only takes effect after candidate acceptance/application and future policy training. This NN-only mode has passed 17 targeted preparation/runtime/UI tests but has not yet been fitted to the recordings.

The existing MotionResidual architecture is reused: two 32-unit tanh hidden layers and an XYZ acceleration correction on each free cable node, limited to ±0.5 m/s² per component. The attachment node gets zero direct correction. Features contain predicted relative node positions/velocities and attachment velocity; no future cable measurements or flight commands are inputs.

Each run snapshots only the permitted processed preliminary recordings, roles, model, settings and relevant source files under `data/cable_residual_runs/<timestamp>`. The protected recording is excluded by name before any data access, even if its manifest role is accidentally changed. Whip trials are not accepted as input. Data gaps remain invalid; only contiguous valid windows are used. Initialization uses the past 0.2 seconds with a causal velocity estimate and fixed constraint projection.

Training uses short 0.05-second differentiable rollouts, 24 updates by default, and a small NN correction penalty. A training-only directional finite-difference check must pass first. Checkpoint selection compares complete 2-second training predictions, including the zero-NN, zero-external-drag initialization. Separate preliminary validation takes compare 2-second and 5-second marker/tip errors against the original fixed-damping physical baseline. Both horizons must improve both average errors, and no validation take's marker error may worsen by more than 5%. These takes were already used in previous development checks: this is repeated development validation, not independent final evidence or a real-whip claim. Do not tune hyperparameters to these results. The unchanged ±0.5 m/s² per-axis bound now covers all NN-modeled external effects; whether it is adequate for the recorded velocities needs assessment before interpreting a fit failure.

Results appear in the candidate table. **Apply validated cable residual** is enabled only for a passing candidate; the backend also verifies that the active calibration has not changed. Applying creates a new versioned baseline for future runs. Existing PPO checkpoints and their saved model configurations stay unchanged. Exports of policies using an applied residual include its immutable baseline weights.

## First local run (historical fixed-drag architecture)

`data/cable_residual_runs/20260907-224254-649336` completed 24 updates on Windows / NVIDIA RTX 4080 in 461.5 seconds. Gradient-check relative error: 2.70e-6. The training selection objective improved from 0.0174292 to 0.0170798, but validation did not improve:

| Horizon | Marker RMSE: physical → NN | Tip RMSE: physical → NN |
|---|---|---|
| 2 s | 5.211 → 5.221 cm | 8.915 → 8.950 cm |
| 5 s | 4.749 → 4.759 cm | 8.040 → 8.068 cm |

The trained weights are preserved as `residual_candidate.pt`; this candidate is not accepted or applied. The active model hash remains `51a6349546d255cd9ebdf8913f5ff10fc70e58dc48a70f725518771a97fb3d11`. Its source snapshot records the exact runner used; a subsequent maintenance change removed warnings from empty validation aggregation during training-only checkpoint selection, without changing the selection objective.

## First learned-drag candidate (historical scalar-plus-NN architecture)

`data/cable_residual_runs/20260907-232213-673553` completed 24 updates in 537.1 seconds on Windows / RTX 4080. Training selected update 24; the coefficient changed from 0.3 to 0.3020478401 s⁻¹. Training objective improved from 0.0174292 to 0.0170700. The gradient check passed (relative error 2.63e-6).

| Horizon | Validation marker RMSE: baseline → residual | Validation tip RMSE: baseline → residual |
|---|---|---|
| 2 s | 5.21118 → 5.20426 cm | 8.91527 → 8.91657 cm |
| 5 s | 4.74895 → 4.75053 cm | 8.03977 → 8.05073 cm |

Both acceptance checks failed. The candidate is preserved and not applied. A coefficient moving away from its initialization is not proof of correct identification or improved prediction. No training settings were adjusted against these results. The active baseline and existing PPO remain unchanged.

31 distinct targeted tests passed, including transition equivalence, finite-difference gradients, both CUDA graph paths, CPU compiled physics, checkpoint/export roundtrip, data isolation and UI. Native Windows review is recorded in `ui_verification.json` and `ui.png` inside the run. The immutable training source snapshot precedes a numerically equivalent device zero-fill change required for CUDA graph capture; both graph paths passed after that fix.

## Yesterday's whip trials

`data/adaptation_rounds/adaptation0/ARCHIVED.json` excludes the old whip round from the active recording UI and processing entry point. Its raw logs, reviews, processed versions and references remain untouched. The number is reserved rather than reused. New flight data can be imported later; no drone residual can be trained in this preliminary-only workflow.
