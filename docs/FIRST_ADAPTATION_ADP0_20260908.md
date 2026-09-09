# First current-flight adaptation: M1 candidate

**Subsequent decision:** the user accepts whip-focused M1 planning despite recovery prediction regression. M1 is now the default model in the independent [MPPI Planner](MPPI_PLANNER.md). The PPO workspace and original flight ghosts remain M0; model assets/results below are preserved exactly. New real adp1 evaluation is still pending.

The first adaptation of both drone/cable physics and their residuals is complete. It uses only the five current `whip_adp_0` flights as new fitting data. Old recordings enter through the preserved M0 parameters/weights; no historical data was pooled into this fit.

Published candidate: [20260908-adp0-M1/model.json](../data/model_candidates/20260908-adp0-M1/model.json), SHA256 `ee99478f670ae36cd21ca15095b52251d394a761ac9b044369845c6e0cb02d5a`.

It is **not the active default**. Whip prediction improves, but late hover position and whip attitude prediction regress. The retained PPO remains stopped, and original M0/flight CSV/data are preserved.

## Held-out results

Each of five fits excluded one complete flight from parameters, NN training and checkpoint selection. Both baseline and adapted complete-model forecasts use causal pre-flight initialization and the actual matched command packets. No measured drone or cable states enter those forecasts after initialization. Scores use native 100 Hz observations and 3D Euclidean errors.

| Mean of the five per-flight results | M0 | Adapted |
|---|---:|---:|
| Drone whip RMS | 11.32 cm | 5.79 cm |
| All cable markers whip RMS | 10.64 cm | 7.31 cm |
| Tip whip RMS | 15.24 cm | 10.42 cm |
| Tip prediction error at 0.94 s | 37.55 cm | 15.07 cm |
| Early recovery drone RMS | 28.72 cm | 22.56 cm |
| Late recovery/hold drone RMS | 9.98 cm | 15.53 cm |
| Complete observed tip RMS | 48.78 cm | 42.61 cm |
| Whip attitude RMS | 3.21 degrees | 4.26 degrees |

The five whip-tip predictions all improved. Updating only the drone gives mean tip RMS 11.05 cm; only the cable 14.10 cm; both 10.42 cm. With measured attachment supplied to isolate the cable, its held-out tip RMS improves 8.54 → 7.03 cm.

These numbers describe **prediction of existing flights**, not improved physical hitting performance. Five repetitions of one command provide local development evidence; an adp1 execution is the prospective assessment. The published all-five model has in-sample current-data evaluation, while the table above evaluates the fitting procedure through separate held-out models.

## Method and limits

Drone fitting updates six effective PD/feedforward gains, delay, three attitude mapping parameters, and the existing bounded translation residual. Attitude is refitted with the learned translation residual included in the predicted state evolution. The small drone NN trains on CPU after CUDA loss/gradient parity verification; predictions are checked on RTX 4080 CUDA.

Cable fitting uses measured rotated attachment motion, fixed geometry/masses, zero separate fixed drag, the tested smooth curvature-frame regularization `2e-7`, and the extended damping-plus-acceleration NN. EI remains `1e-7 N m²`; Cb changes `1e-4 → 2.5e-5 N m² s`. The additional head is bounded to ±0.5 m/s² per node axis. Six independent cable fits run in one GPU batch without sharing weights or held-out gradients.

The fixed budget is 80 drone and 24 cable NN updates. Losses still decrease; no convergence/global-identification claim is made. Several parameters reach conservative search bounds. Drone losses assign 80% to the one-second whip and 20% to recovery through 3 s. Cable uses 0.12 s BPTT windows and training-only selection on 1.02 s complete-whip/early-recovery rollouts. Missing history windows are recorded and excluded; no raw data is deleted or filled. Geometry projection changes only the simulator initial state.

Late hover regression remains unresolved. Frozen effective hover compensation and NN behavior near hover are possible causes, not conclusions established by these data. Recorded initial hover is also 4.5–7.6 cm above the command height, with lateral deviations; actual initial state must be distinguished from nominal launch settings. The roughly 10 ms measurement-stream timing uncertainty limits interpretation of the fitted 20–40 ms effective delay.

## Verification and artifacts

27 targeted tests passed on Windows 11 / Python 3.12.10 / Torch 2.11.0+cu128 / RTX 4080. The fitted model's actual-command full-whip gradient check passed: AD `-0.7365890545`, finite difference `-0.7365891985` at a 1 µm PVA-consistent perturbation; inference/gradient forward difference `3.7e-13`. Numerical test commands were never exported. The initial drone fitting helper's redundant time subdivisions differ from the native executor by at most 8.4 µm in position; final attitude caching matches the exact native subdivision rule.

- [Full report and plots](../runs/adaptation/20260908-adp0-first/report/ADAPTATION_REPORT.md).
- [Held-out prediction metrics](../runs/adaptation/20260908-adp0-first/validation/results.json).
- [Assessment, phase tradeoffs and measured hover positions](../runs/adaptation/20260908-adp0-first/assessment.json).
- [Full-whip gradient check](../runs/adaptation/20260908-adp0-first/verification/full_whip_gradient.json).
- Job: `runs/adaptation/20260908-adp0-first`; raw-source hashes, prepared inputs/masks, protocol, fitting history, independent fold models, ablations and source snapshots are retained.

`tools/adapt_current_flights.py` provides `prepare`, `drone`, `cable_batched`, `attitude_refine`, `baseline` and `validate` stages. Completed stages reuse their saved outputs; do not rerun preparation in the existing job. This is the explicit current-adp0 protocol, not a generic automatic importer for future rounds. The abandoned initial serial cable attempt is retained separately for provenance and is not a selected model.

Workspace PPO/task settings and a separate CEM job changed concurrently outside this fit. The adaptation used frozen recorded commands and saved models, not those changing settings. Preservation exceptions are listed in the job verification record; retained policy, rehearsal, raw adp0 files and original model assets were unchanged.
