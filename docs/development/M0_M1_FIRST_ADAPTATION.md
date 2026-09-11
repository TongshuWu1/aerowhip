# First real M0 adaptation: candidate retained, M0 not replaced

Scope correction: this is a **partial, single-gain trial**, not the complete
drone/cable/residual adaptation requested by the user. Read the current
[full-model contract](../methods/ADAPTATION_MODEL_CONTRACT.md) for the intended workflow and
the follow-up regression audit. This document preserves the trial as performed.

10 September 2026. The user authorized careful adaptation from the five measured
M0 whip flights. One small candidate was fitted and evaluated. It did not improve
the usable held-out take, so **M0 remains selected**. The M1 candidate is retained
for transparent comparison, not promoted or used to plan a flight.

## Data and decision

Adaptation takes are 001/002/004. Take 005 is held out of parameter fitting and
update-scope selection. Take 003 remains reserved validation evidence, excluded
from reinitialized cable diagnostics because marker c5 is missing in its causal
initialization and most of the whip. Its raw observations/original-forecast replay
remain available. No missing marker was fabricated. All 309 logged command rows
match the original frozen M0 CSV in every take.

Preparation uses raw global OptiTrack coordinates, cf_3 tracked origin and rotated
attachment offset, one-second past cable history and 0.4-second past drone history.
Clock alignment is estimated from native versus cached measured XYZ. It is not an
independent measurement of actuator delay. The same accepted samples, actual held
commands and initialization were used for both model predictions.

Original M0 forecast errors were saved before fitting. Separately reinitializing
M0 from measured history gives conditional cable tip errors of 5.6–6.8 cm when
measured attachment motion drives the cable, versus 10.4–16.1 cm with the predicted
drone driving it. All three adaptation-only drone probes favored a reduced
horizontal feedforward response. This justified testing one closed-loop response
gain before changing cable damping. It did not justify identifying saturation,
changing controller gains, fitting a new residual, or calling a maximum observed
acceleration a vehicle limit.

## What was fitted

Only `drone.nominal.parameters.feedforward_xy` changed, **0.74668 → 0.61834**.
This is a simulator response coefficient, not a firmware/controller change.
Cable damping stays 0.4/s. Geometry, mass, EI/Cb, drone feedback gains, vertical
feedforward, 20 ms effective delay, attitude parameters and neural weights remain
fixed. No command or generated-acceleration cap was introduced.

The versioned `whip_response_gain_v1` contract is embedded in the new prepared
job, with checksums binding the earlier adaptation-only diagnostics and decision.
The robust recursive native-pose objective gives equal weight to each adaptation
take. Position and rotation pseudo-Huber terms use 2 cm and 0.05 rad scales. The
dimensionless prior is 0.1 times squared fractional gain change. Declared parameter
search bounds are 0.5–1.25 times M0's gain; these are search choices, not measured
vehicle limits. A cached scalar refinement retains M0/incumbent and stops by
0.5% practical improvement with three stale updates, minimum three and a separate
12-update ceiling.

Five updates reached practical plateau in 57.3 seconds including checks and
coupled diagnostics on Windows/RTX 4080, CUDA float64. Captured production pose
rollouts are used for the search; the small scalar candidates are evaluated
sequentially and cached. This does not claim full GPU occupancy. An independent
eager replay reproduced the selected loss exactly. The candidate is inside the
search bounds. Training objective fell 15.1%.

Adaptation-only ±20 ms clock-shift checks still improved the mean pose loss, but
not every take in every scenario. The same past-only state was reconstructed under
M0/M1; its maximum difference was zero for this gain-only update and stationary
prehold. These checks do not establish absolute synchronization or motor latency.

## Results on identical prepared inputs

All values are 3D RMS errors in centimetres during the roughly 1.133-second whip.
These are **reinitialized diagnostics**, not replacement original forecasts.

| Take | Role | Drone M0 | Drone candidate | Tip M0 | Tip candidate |
|---|---|---:|---:|---:|---:|
| 001 | Adaptation | 9.38 | 6.36 | 11.27 | 11.14 |
| 002 | Adaptation | 9.07 | 8.79 | 16.15 | 16.39 |
| 004 | Adaptation | 13.70 | 9.27 | 10.40 | 9.57 |
| Mean, equal take weight | Adaptation | 10.71 | 8.14 | 12.61 | 12.36 |
| 005 | Held out of fitting | 8.93 | 10.41 | 14.89 | 17.93 |

On 005, drone error increases about 16.6% and tip error about 20.4%. The candidate
does not demonstrate a transferable improvement even on a repeat of the same
command. One held-out take does not quantify generalization reliably, but it is
enough reason not to promote this candidate. Repeating identical commands exposes
execution variability; it does not supply new-motion generalization evidence.

`fit/selection_frozen.json` was written before parent/candidate reinitialized 005
diagnostics. The original M0 forecast/005 observations were already available for
replay, so this is not a claim that the entire recording was blindly hidden. No
candidate validation result was used to choose this gain, and no second fit was
launched after seeing the regression. Cable parameters were not subsequently
changed to compensate for it.

## Where to inspect it

- **Flight comparison → Compare generations:** M0/M1 metrics and prediction traces.
  Choose Validation takes for 005 or Adaptation takes for 001/002/004.
- **Fitting progress:** `M1-whip-response-v1` shows gain/loss/plateau history.
- **Flights by model:** M0 contains the actual flights and original ghost. M1 has
  no planned/flown session. Its catalog status says it was not promoted.

Evidence locations:

- Initial diagnoses: `runs/adaptation/M1-whip-first` (diagnostic only).
- Reviewed response fit: `runs/adaptation/M1-whip-response-v1`.
- Same-input comparison: `runs/evaluation/M0-M1-whip-response-v1`.
- Review/math/source/UI audit: `runs/audits/M0-real-adaptation-20260910`.

34 focused tests passed. A synthetic known-gain check on the production CUDA
solver gave zero loss at the true 0.6 gain and positive loss at 0.55/0.65. Registered
model replay agrees with the fitting diagnostics within 1.2e-12 in saved arrays.
All ten raw files and original frozen M0 evidence hashes remain unchanged.
Comparison scope now checks nested drone JSON parameters as well as binary neural
assets; the old cable-only check could miss a changed nominal drone JSON after
discarding its checkpoint path. New regression tests reject that case.

The next decision is further diagnosis of repeat-to-repeat response variation and
remaining vertical/attitude mismatch using adaptation data, followed by a new
predeclared validation opportunity if model design changes. No automatic new fit,
M1 plan, flight, model promotion, or success claim follows from this trial.
