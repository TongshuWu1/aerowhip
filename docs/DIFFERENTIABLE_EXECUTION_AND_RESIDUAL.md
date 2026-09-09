# Differentiable execution and the optional cable residual extension

Follow-up: the [first current-adp0 adaptation is now complete](FIRST_ADAPTATION_ADP0_20260908.md). Fitted M1 improves held-out whip position prediction, with recorded late-hover/attitude regressions, and remains a separate candidate. The implementation history and unfitted candidates below are preserved.

## What is implemented

The model now supports an optional cable correction beyond damping, and a differentiable FullState-to-drone-to-cable prediction. Ordinary rehearsal retains forward-only execution and CUDA graph acceleration. The two modes share the pose equations, command event schedule, and cable integration step; there is no separately invented fitting simulator.

This work prepares the model for adaptation. It does not fit adp0, update the selected PPO, change the executed CSV, or select a new flight model. The original M0 remains active. Both new capabilities are opt-in.

The full-whip gradient audit also exposed a pre-existing numerical problem in M0's damping calculation. A separately versioned smooth-damping candidate addresses it. The exact-M0 zero-extension candidate remains available as a comparison; it should not be treated as a robust full-whip gradient-optimization model merely because it now exposes an autograd graph.

## Cable correction

The existing dissipative network remains available with exactly its original checkpoint format and semantics. The new mode, `dissipative_plus_acceleration`, retains its learned damping branch and adds a zero-initialized output head to the same hidden features:

    correction_i = -gamma_i(q, v) * v_i + acceleration_limit * tanh(extra_head_i(q, v))

The nonnegative damping rates remain bounded by their saved limit. The additional acceleration is bounded independently on each XYZ component and is applied only to free cable nodes. The prescribed root always has zero residual correction. The input still consists of root-relative cable positions/velocities and root world velocity; no target, trial identifier, elapsed time, or measured future motion is introduced.

`with_acceleration_correction()` copies a dissipative network, preserving its weights and outputs, and initializes only the new head to zero. It rejects acceleration-only checkpoints rather than reinterpreting their learned weights as damping. Existing acceleration-only and dissipative loaders continue to work. `MotionResidual.components()` exposes the old and additional corrections separately for magnitude regularization and diagnostics during subsequent fitting.

The prepared candidate uses ±0.5 m/s² per free-node axis for the additional branch. Its possible vector norm is therefore at most sqrt(3) × 0.5 m/s²; this is not a norm limit of 0.5. The chosen bound is a conservative tuning parameter, not an identified physical constant. The new correction may inject energy and is not claimed to conserve momentum or guarantee rollout stability. It represents effective model discrepancy, not a uniquely identified internal force.

Research rationale: broader learned corrections around differentiable rod dynamics are supported by [DEFORM](https://arxiv.org/html/2406.05931), while learned corrective forces are explored in [Learned Residual Physics](https://arxiv.org/html/2402.01086). Our particular small bounded head is a project design choice. It must be compared against damping-only on held-out trajectories before selecting a fitted replacement.

## Numerical issue found by the full-whip check

The legacy corotational damping Jacobian switches abruptly between a straight-segment expression and a bend-plane expression when squared adjacent-tangent difference crosses 2e-7. Near a straight configuration the bend plane is poorly determined. A one-second test on the actual command exposed severe sensitivity: the reverse-mode directional derivative was about 20,729, while finite differences at decreasing perturbation amplitudes varied in sign and magnitude. Forward-mode differentiation independently agreed with reverse-mode differentiation. Thus a short passing gradient check did not establish a usable full-whip derivative; this was not resolved by merely removing no-grad wrappers.

The new optional `cable.curvature_frame_regularization` replaces that switch with a smooth transition. For adjacent unit tangents a and b, write s = ||a × b||². The resolved-bend contribution receives weight s/(s + rho), and the straight-limit contribution receives rho/(s + rho). The weighted bend expression is evaluated algebraically without dividing by a normalized zero bend-plane vector. The candidate uses rho = 2e-7, at the scale of the previous transition. This is a saved numerical modeling choice, not newly measured material physics or an extra external drag coefficient.

The damping matrix is still assembled as Cb × Jᵀ W J, so its unconstrained velocity-damping contribution remains positive semidefinite. Tests check dissipation, translation invariance, rotational equivariance, and derivative continuity around the previous switch. Regularizing the local frame does change the damping model: exact removal of every rigid-spin contribution is not claimed in this regularized transition region. Neither positive damping nor a bounded NN guarantees global stability of the complete discrete simulation.

Omitting the field or setting it to zero retains legacy behavior exactly. The smooth value is propagated through configuration, direct forces, and inference/differentiable implicit damping. Legacy fused damping implementations that do not support the new expression are bypassed or explicitly rejected rather than used with inconsistent equations. The native float64 CUDA-graph execution path supports the smooth model.

The isolated CPU test changed the nominal whip's largest cable coordinate by about 4.0 mm and the final tip position by about 4.1 mm. This is a change between simulated models, not an improvement against real measurements. It is recorded separately from the exactly preserving zero NN extension. Fitting and prospective validation remain necessary before selecting the smooth candidate for deployment. The exploratory failure, forward-mode check, and smooth-transition investigation are retained in the audit's `gradient_probe.json` and `smoothing_probe.json`.

## Execution and differentiation

The new entry point is [ResearchExecutionModel](../simulator/research_execution.py). It composes:

**30 Hz FullState packets → loaded-drone response and its NN → rotated tracking-frame attachment offset → DDER and cable NN.**

The root remains a free pivot with prescribed position during execution prediction. There is no new feedback of cable reaction into the effective loaded-drone model. The existing virtual force-planning stage remains separate.

- `predict(..., gradients=False, graph=True)` uses captured CUDA inference when the tensors are on CUDA.
- `predict(..., gradients=True, graph=False)` retains temporal derivatives. It uses the same pose midpoint equations and cable integration kernel without mutable graph-replay buffers.
- `checkpoint_steps=25` recomputes blocks during backward to save memory. It does not truncate temporal gradients or reset the cable to measured states. Set it to zero for an uncheckpointed comparison.
- `ResearchPoseModel.predict_differentiable()` also exposes the drone-only learning path, using the same internal schedule as ordinary pose prediction.
- `research_physics_step()` is the shared functional cable step used by both captured inference and the differentiable composition.

Gradients propagate to command P/V/A/heading values, initial state, supported continuous drone-response parameters, supplied EI/Cb tensors, and enabled residual weights. Frozen NN weights still allow derivatives through their state/command inputs. Set `trainable_residuals=True` when loading private model instances if subsequent fitting needs weight gradients; that flag performs no optimization.

### Explicit limits

The implementation preserves exact zero-order-held command events. Command timestamps, delay, gravity, and integration scheduling are fixed inputs. Requests to differentiate trainable delay/timestamp tensors raise an error. Delay should be fitted by a separate bounded search using the exact schedule; it is not silently replaced by a smooth command interpolation. Derivatives of continuous parameters are local to the current integration subdivision.

The composed API uses float64, matching the calibrated native rehearsal path. Output timestamps must lie on the saved uniform cable grid, currently 150 Hz with eight internal steps per interval. Evaluate losses at 100 Hz measurement times by interpolating the predictions outside this routine; do not change the integrator clock to the measurement clock without checking the effect.

Both modes require compatible initial drone/cable states, including agreement at the rotated attachment. Missing/invalid initial data, unsupported material-frame clamps, invalid attitude domains, or nonfinite rollouts raise errors. The API does not supply future measured boundary positions, perform target-based alignment, change rewards, differentiate hit termination, or replace the deployment feasibility/flight-envelope checks. Fit over fixed observed horizons; separately apply the existing command feasibility and numerical-domain criteria when optimizing/exporting trajectories.

## Saved candidate and preservation

Two **UNFITTED** candidates are saved. The [exact-M0 zero extension](../data/model_candidates/20260908-differentiable-zero-extension/model.json) changes no initial predictions. The [smooth differentiable candidate](../data/model_candidates/20260908-smooth-differentiable/model.json) additionally enables the versioned damping regularization and is the candidate proposed for subsequent differentiable adaptation. Each directory owns copies of the drone model/residual, original cable damping checkpoint, and extended cable checkpoint. Manifests record provenance and that no training or deployment selection occurred.

The retained selected PPO remains `20260908-195207-486249-seed655 / best_validation.pt`. Rehearsal `20260908-203914-039721`, the flown FullState CSV, all five policy-scoped adp0 flights, active configs, policy exports, and independent CEM assets are preserved. No existing candidate directory can be overwritten by the preparation tool.

To prepare another explicitly separate candidate:

```powershell
.\.venv\Scripts\python.exe tools/prepare_residual_candidate.py `
  --source config/research_30hz/model.json `
  --output data/model_candidates/ANOTHER_NEW_DIRECTORY `
  --acceleration-limit 0.5 --frame-regularization 0.0000002
```

Example API usage after supplying validated initial states, the actual packets, and their timestamps:

```python
import json
from pathlib import Path
from simulator.research_execution import ResearchExecutionModel

path = Path("data/model_candidates/20260908-smooth-differentiable/model.json")
engine = ResearchExecutionModel.from_mapping(
    json.loads(path.read_text()), root=path.parent,
    device="cuda", trainable_residuals=True,
)
prediction = engine.predict(
    initial_pose, initial_cable, packets, packet_times, output_times,
    hover_command=observed_pre_command_hold,
    gradients=True, graph=False, checkpoint_steps=25,
)
# Construct a masked trajectory loss against observed motion outside this API.
# No fit or optimizer is started by loading or predicting.
```

For nominal cable fitting, pass a two-element positive tensor `[EI, Cb]` as `cable_parameters`. Continuous nominal drone parameters can be supplied by replacing fields of `engine.drone.parameters` with scalar tensors. Geometry and measured masses should remain fixed for the first adaptation. A future loss should penalize the additional correction's magnitude and use phase/flight normalization; neither training loss weights nor fitting data are silently chosen by this infrastructure.

## Verification

Final run: **47 tests passed** on Windows 11 / Python 3.12.10 / PyTorch 2.11.0+cu128, with CUDA tests on the RTX 4080. The only pytest warnings were existing TorchScript tracing deprecations. The complete 11.2-second M0 prediction and zero-extension candidate matched the saved prediction exactly. All 1,523 protected policy/rehearsal/CEM/config/export/current-flight files checked were unchanged.

For the smooth candidate, the entire whip's differentiable and captured-inference predictions matched exactly on this GPU run. The directional derivative was 2.41680315; finite differences were 2.41837001 at a 1e-5 m perturbation and 2.41693996 at 1e-6 m. The latter differs by about 0.0057%. Full-whip forward/backward took about 119 s with roughly 166 MB peak allocated CUDA memory in this audit; this is a single-case implementation check, not a throughput benchmark. The largest cable-coordinate change between the smooth candidate and M0 over the complete whip/recovery/hold was 17.3 mm; the isolated whip-only comparison above is smaller. No claim of better physical prediction follows from these numerical differences.

The targeted regression command covers the extension, the complete differentiable model, original dissipative semantics, existing cable BPTT, native research prediction, and rehearsal/export contracts:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/physics/test_extended_cable_residual.py `
  tests/physics/test_smooth_curvature_damping.py `
  tests/calibration/test_differentiable_execution.py `
  tests/physics/test_dissipative_cable_nn.py `
  tests/calibration/test_differentiable_fit.py `
  tests/training/test_research_pipeline.py `
  tests/flight/test_research_export.py tests/physics/test_cable_core.py -q
```

The saved-flight audit uses the full 11.2-second original rehearsal for forward comparison, then the entire one-second whip for BPTT and a numerical derivative check. The perturbation changes P, V, and A consistently through a smooth position bump; those numerical test commands are never exported. Results, platform details, memory use, and preservation checks are recorded in [verification.json](../runs/audits/20260908-differentiable-execution/verification.json).

```powershell
.\.venv\Scripts\python.exe tools/check_differentiable_execution.py `
  --rehearsal runs/rehearsals/20260908-203914-039721 `
  --candidate data/model_candidates/20260908-differentiable-zero-extension/model.json `
  --smooth-candidate data/model_candidates/20260908-smooth-differentiable/model.json `
  --output runs/audits/20260908-differentiable-execution --device cuda
```

These checks establish implementation agreement and local derivative correctness. They do not establish that the new correction improves real-flight prediction. That requires the separate adp0 fitting and held-out comparisons described in [the adaptation research memo](ADAPTATION_LITERATURE_AND_DESIGN.md).
