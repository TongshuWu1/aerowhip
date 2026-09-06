# Offline adaptation preflight — 6 September 2026 UTC

This is a synthetic development check, not a real-flight result.

The new import/replay/parameter-update path was tested on two generated coupled drone–cable flights, each with a 0.6-second command interval and a strictly pre-contact 0.59-second analysis interval. One whole flight trained the fit; a different horizontal-force trajectory was held out. There was no measurement noise, actuator lag, unknown timing, or unmodeled dynamics.

| Quantity | Result |
|---|---:|
| Baseline cable drag | 0.300 /s |
| Deliberately simulated drag | 0.550 /s |
| Fitted drag | 0.5516 /s |
| Held-out all-marker RMSE, measured-attachment replay | 2.137 → 0.0207 mm |
| Held-out all-marker RMSE, coupled replay | 2.167 → 0.0138 mm |
| Fit and before/after replay time | 377 seconds, one CPU thread |
| Objective evaluations | 7 |

This verifies that the implementation can recover a known discrepancy under favorable same-model conditions. The very small final errors are expected for this noiseless synthetic experiment and should not be extrapolated to OptiTrack or real aircraft response. It does not establish identifiability on real whipping data, transfer to another task, or successful physical adaptation.

Artifacts are in `data/adaptation_preflight/mismatch_audit/`, including source snapshots and package versions, normalized and raw synthetic trials, fit history, baseline/candidate models, per-flight replay NPZs, metrics and PNG/PDF figures. No active physical baseline or PPO/SAC checkpoint was changed.

Other checks passed:

- Causal launch initialization does not depend on future flight measurements.
- Contact-interval exclusion, unknown-clock rejection, invalid-marker rejection, immutable reimport protection, and protected-role exclusion.
- Differentiable force rollout agrees with the existing CPU runtime; its force directional derivative agrees with central finite differences over a short trajectory.
- Bounded parameter fitting produces a candidate without falsely claiming validation when no validation flight exists.
- Force correction exports a single fixed-cutoff sequence and performs the existing nominal first-contact/recovery check; output is always marked not ready for hardware release.
- A transport-neutral recorder callback sequence roundtrips into the importer and preserves controller/IMU telemetry without treating it as applied force.
- The new adaptation page and full five-page application were rendered and inspected. The shared differentiable-fit and reward-UI checks passed alongside the new tests.

Open work: actual ROS topics/message formats and clock mapping, verified Lee-controller force semantics/limits and response identification, next-launch state binding, independent robustness acceptance, preliminary-data forgetting checks, full-strike gradient validation, and physical flight evidence. Neural residual fitting remains conditional on repeatable held-out model error.
