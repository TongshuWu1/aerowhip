# Preliminary1 fitting audit

9 September 2026. Audit of completed `runs/adaptation/20260909-preliminary1-M0-v2`.
No refitting, model/checkpoint edits, raw-data changes, or MPPI settings changes.
Diagnostics used Windows / RTX 4080, float64, the frozen fit source and published
M0. Evidence: `runs/audits/preliminary1-fit-pipeline-check`.

## Main finding: inconsistent, influential initial cable velocities

`PreliminaryTrial` supplies 41 preceding samples (0.4 s) in `pre_indices`.
Inherited `Trial.cable_state` fits one quadratic through all of them for the
first cable window. For the second window, `cutoff=1` selects only the last 11
samples (0.1 s). Both predict endpoint velocity, but with different smoothing
and bias during curved, changing motion. These estimated states are then held
fixed while fitting cable parameters and residuals. This is an implementation
inconsistency and an important source of error, not evidence of bad raw recordings.

A targeted diagnostic kept the fitted M0, initial cable positions, measured
attachment motion and all physics fixed, changing only the first window's causal
velocity history from 0.4 s to 0.1 s. The three windows were the already selected
representative diagnostics, not selected by this audit's errors.

| Window | Role | 1 s tip RMS, 0.4 s history | 1 s tip RMS, 0.1 s history | 2 s RMS, before / after |
|---|---|---:|---:|---:|
| figure8_001-02441 | Training | 44.03 cm | 10.71 cm | 61.45 / 28.35 cm |
| figure8_002-02041 | Held out | 30.82 cm | 18.69 cm | 56.16 / 44.67 cm |
| osci_001-02441 | Training | 2.63 cm | 2.40 cm | 7.00 / 6.80 cm |

For the first figure-eight, estimated initial tip Z velocity changes from
1.2715 to 0.4959 m/s. For the held-out figure-eight it changes from 1.7029 to
0.9077 m/s. This does not establish the true velocity or a universally optimal
history length. It demonstrates substantial sensitivity to preparation, before
any model training. The held-out case is diagnostic evidence, not authorization
to tune an estimator on validation data. Missing target observations remain
missing; diagnostic RMS excludes nonfinite observations.

## Optimization stopped without establishing a satisfactory cable fit

- Physical search selected EI≈1e-8 and Cb≈1e-4 at the original grid's extremes.
  Local bounded least squares made four evaluations and stopped on `xtol`.
  EI changed only about 0.00075%, and Cb about -0.0085%. The wider optimizer
  bounds were not active. A small parameter step is not evidence of identified
  material constants or a satisfactory fit.
- Cable residual stopped at the earliest allowed plateau check, update 12.
  Selection objective improved from 0.990701 to 0.986315, only about 0.44%.
  Pre-clipping gradient norms reached 1.44e16; clipping threshold was 1.
  The stopping rule behaved as configured, but its `converged` label means a
  training plateau, not model validity. Longer training alone is not an established fix.
- Drone residual reached its 400-update ceiling while its best checkpoint was
  still the final update. It was explicitly not a plateau stop. This stage
  should not be described as fully converged.
- Existing cable gradient smoke testing covered six steps, whereas training
  windows span 150 steps. This audit did not prove a CUDA/autograd defect.
  Full-window directional-gradient and conditioning checks are still needed
  before trusting a retry of the unstable cable optimization.

## Errors are not explained solely by the drone or solver substep count

With the measured attachment trajectory supplied throughout, M0 still gives
61.45 and 56.16 cm two-second tip RMS on the two representative figure-eights
using the original initialization. Thus predicted aircraft tracking error alone
does not account for the large coupled error. Cable/state estimation needs work.

Fixed-parameter checks at 8, 16 and 32 cable substeps did not resolve these
errors. At 32 substeps, the two figure-eight errors were 62.16 and 57.31 cm.
Relative to eight substeps, predicted-tip differences were about 1.28 and
3.22 cm RMS over two seconds. Numerical sensitivity exists, but merely increasing
substeps does not explain or fix the much larger model error in these cases.
Identical-input captured replays were exactly repeatable. Small damping
perturbations gave scale-sensitive local objectives; the derivative/stability
question remains open. No diagnostic substep value was adopted in MPPI or fitting.

## Preparation checks that passed

- All ten source CSV hashes match the frozen fit's records; the code snapshot
  hashes match its saved manifest. No raw recordings were overwritten.
- Native aligned time equals native OptiTrack time plus the recorded offset.
  Measured-to-measured stream RMS is 2.1–4.0 mm; chunk offset ranges are roughly
  3.8–11.9 ms. This does not identify actuator/cache latency independently.
- Median command receipt spacing is about 33.33 ms for all five takes.
- Whole figure8_002 is excluded from nominal fitting and the training-only
  residual paths. There are 133 training and 42 validation cable windows.
- Missing-marker masks persist, including 342 completely missing cable frames
  in the vertical take. No retrospective Z normalization was introduced.
- The published candidate remains provisional. Existing publication checks
  establish loading, finite predictions and execution consistency, not sufficient
  accuracy for a 5 cm strike target.

## Recommended correction order

1. Use one consistent causal state-estimation procedure across cable windows.
   Validate endpoint velocity and sensitivity on training recordings; retain
   state uncertainty/projection diagnostics rather than forcing parameter fits
   to absorb initialization errors.
2. Check full-window objective/gradient conditioning at the fitted physics.
   Treat tiny-step and early-plateau stops as unresolved when residuals remain
   poor. Preserve the current fit and all failed attempts.
3. Refit in a separately authorized job after those checks, maintaining the
   whole-take split. Report representative errors and the distribution across
   windows/takes, not only an aggregate short-window RMS.
4. Assess measured-boundary cable predictions and coupled predictions before
   interpreting an MPPI search failure as a physical limitation of the drone.

Better new recordings can still be useful. This audit does not establish that
the old model was more physically accurate, nor that the new data are worse.

## User-requested one-second history trial

The user subsequently requested 1 s cable history. New preliminary preparation
protocols now record `cable_history_s=1.0`, consistently used at both cable-window
positions. It covers measured time ending strictly before each prediction start;
insufficient history, time gaps and masked data reject a cable window. Drone
history remains 0.4 s. Jobs without the new field retain their old semantics;
the completed M0 and active MPPI do not change. No new fit was started.

On the same three diagnostic windows, with fixed M0 and measured attachment:

| Window | 1 s tip RMS with original 0.4 s history | With 1 s history |
|---|---:|---:|
| figure8_001-02441 | 44.03 cm | 22.58 cm |
| figure8_002-02041 (held out) | 30.82 cm | 77.16 cm |
| osci_001-02441 | 2.63 cm | 5.62 cm |

This is mixed and worse on the held-out example; it does not validate one-second
quadratic velocity estimation as an improvement. On that example estimated
initial tip Z velocity rises from 1.70 to 3.19 m/s. The requested setting is
implemented for future preparation, with these limitations recorded rather than
silently replaced by another choice. Evidence:
`runs/audits/preliminary1-history-1s/diagnostic.json` and `eligibility.json`.
Four focused preparation tests pass on Windows; numerical diagnostics used RTX 4080.
