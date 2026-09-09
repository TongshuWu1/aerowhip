> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# All-historical physical and residual fitting

**Current drone scope:** following the user's correction, preliminary data are used for cable identification only. New nominal drone response and drone NN fits use the three whip recordings from the virtual-force-policy → PVA export → cmdFullState route. See [whip-only drone results](WHIP_ONLY_DRONE_FIT.md). The pooled drone results described below are preserved historical development work. New prepared protocols set `drone_response_takes` explicitly, and the fitter intersects the development folds with that scope.

This workflow uses the eight preliminary recordings and three `adaptation0` whips. The user confirmed the same drone and settings and authorized the previously protected `fig8vertical_002` for fitting. All historical data are development data; preserve the old study snapshots and use future flights for prospective assessment.

The September 8 UTC run is `data/historical_model_runs/20260908-002241-654934`. Its `protocol.json`, `audit.json` and immutable `inputs/` record exactly what was admitted. Fitting never starts PPO or changes the active model. Final results are reported separately in `HISTORICAL_MODEL_FIT_RESULTS.md`.

## Measurement selection

Keep failed flights. Reject questionable measurement intervals, not trials with high prediction error. Cable and drone losses have separate masks: a missing command does not remove valid cable motion, and a missing cable marker does not remove valid drone motion.

The audit excludes missing or nonfinite observations, invalid orientation for attachment reconstruction, jumps exceeding 10 cm per 100 Hz frame, marker chords exceeding the configured arc length by more than 25 mm, and possible contact near source Z=0.05 m with a 0.2 s margin. Short chords are permitted because the cable can bend. Command-based fitting also requires finite logged P/V/A with age at most 0.1 s. These conservative rules identify suspicious intervals, not proven sensor faults or a measured floor plane. Original logs remain untouched.

Use actual logged full-state commands, including the historical switch to hold near 0.67 s. The unexecuted tail of the intended 0.8 s whip is not a command observation. Logged finite-difference velocity spikes are not used as measured drone velocity. Initialization uses a polynomial fit to the measured past.

Whole-take folds rotate across figure-eight, vertical, oscillation and whip recordings. Fit and select parameters on each fold's training recordings only; evaluate the other recordings without refitting. Then fit a final candidate using all 11 recordings. Architecture decisions informed by these checks make them development diagnostics, not independent publication evidence.

## Components

1. **Cable physics:** fit lateral attachment offset, effective bending stiffness EI and internal bending damping Cb. Retain measured drone mass 157 g and cable/marker mass 18 g, ruler geometry and the vertical attachment offset. Do not interpret weakly identifiable EI/Cb values as measured material constants.
2. **Cable NN:** use 4,577 weights and biases in a 75→32→32→33 network. The first attempted unrestricted acceleration correction is preserved for comparison. The revised dissipative network outputs bounded, nonnegative, state-dependent per-node/per-axis damping, multiplied by negative cable velocity. Its correction starts at zero and applies only to free cable nodes. External scalar drag remains zero. The 2 s⁻¹ bound is a modeling constraint, not a fixed damping value; no 0.3 s⁻¹ initialization is used. Short differentiable rollouts fit the network, with two-second training predictions selecting the saved state and calibrating its output biases. Nonpositive instantaneous correction power does not by itself prove numerical or closed-loop stability.
3. **Effective drone response:** fit nine position/velocity/feedforward gains and a delay, then a small bounded acceleration NN, using logged full-state commands. This identifies effective tracking for this loaded vehicle; it does not identify motor dynamics from nonexistent thrust measurements.
4. **Attachment prediction:** fit response to the measured cable-attachment trajectory. Measured orientation supplies target attachment positions and the known initial offset. Future measured orientation does not enter the predictor. The model already includes the recorded cable load; do not add a second cable reaction to it.

The all-data NN fit is a finite-budget candidate, not a claim that the global optimum was found. Physical fitting uses 0.5 s windows; residual selection and cable diagnostics use 2 s windows. Drone component diagnostics use 1 s predictions, and combined diagnostics use 2 s predictions. Equal-take means and motion-stratified window selection reduce domination by lengthy hover recordings. They cannot replace evaluation of hitting on new whips.

## Future PPO and export path

For a model with `fullstate_execution.enabled`, PPO plans virtual forces at 20 Hz, the virtual simulator produces a 30 Hz P/V/A reference, and the empirical drone response predicts attachment motion. Cable physics plus cable NN then predict the executed cable motion used for strike scoring. No measured execution feedback reaches the frozen force plan. Recovery is excluded from this strike objective; export retains the existing gentle recovery separately.

The command CSV is the virtual reference, not the predicted tracking-error trajectory. The calibrated attachment-to-cf7 offset shifts position only; velocity and acceleration are unchanged. Offline zero-yaw level hover is assumed for rotating the body offset. The real initial pose must be mapped consistently before vehicle use. Existing saved PPO configurations and their old exports retain their original behavior. No flight controller/logger or ROS sender is changed by this workflow.

## Commands and artifacts

Run commands from the repository root using `.venv\Scripts\python.exe` on this computer:

```text
python -m experimental_data.historical_fit --prepare
python -m experimental_data.historical_fit --job JOB --stage cable
python -m experimental_data.historical_fit --job JOB --stage drone
python -m experimental_data.historical_fit --job JOB --stage attachment
```

The initial cable command preserves the unrestricted candidate. For the revised dissipative run, use the archived `dissipative_long_settings.json` with `--settings-override`, a new `--output-name`, and `--reuse-directory` plus `--physics-only-reuse` to reuse identical completed physical fits. Do not overwrite an earlier result. The September run's pooled pilot and rotated checks are stored separately and assembled with provenance before review.

```text
python -m experimental_data.historical_combined_review --job JOB --cable-directory CABLE_RESULT --output-name COMBINED_RESULT
python -m experimental_data.historical_report --job JOB --cable-directory CABLE_RESULT --combined-directory COMBINED_RESULT --output-name candidate_review.json
python -m experimental_data.historical_runtime_check --job JOB --cable-directory CABLE_RESULT --checkpoint SAVED_PPO --batch-size 1024
```

The runtime command uses an old actor only as a numerical probe, with no optimizer updates. It is not new-policy training or a policy-performance estimate. Review candidate accuracy separately from passing numerical checks. `historical_report --apply` requires the default reviewed model and matching runtime evidence; it must refuse an unreviewed or regressing candidate.

The older single-fit UI workflows still require a fixed training/validation split. The new all-historical protocol uses its own rotated splits and audited runner, so do not treat the older buttons as equivalent or silently relabel an all-data refit as independent validation.
