# Drone identification → optimized command → predicted flight audit

10 September 2026. Selected MPPI `20260910-022818-648386`.

**Verdict:** the checked fitting/planning/export software uses a consistent
command-to-response chain. The CSV is the input to the fitted drone model; the
saved drone/cable motion is its output. This establishes internal consistency,
not precise physical tracking. The actual forthcoming flight sender and logger
have not been identified in this audit; that interface remains unverified.

## What the experiment actually learns and optimizes

1. Preliminary takes provide logged P/V/A references and native OptiTrack pose.
   The model learns the effective closed-loop response of the installed loaded
   drone and its existing controller. It does not identify motors, thrust or the
   firmware gains independently.
2. Its nominal translation is
   `a = Kp*(p_command-p) + Kd*(v_command-v) + Ga*a_command + b + residual`.
   Commands are held between their timestamps and evaluated with the fitted
   20 ms effective delay. State is propagated recursively; predicted position
   is not replaced with commanded position after each step.
3. Each planner candidate is a jerk sequence integrated into P/V/A setpoints.
   The fitted drone predicts origin position/orientation. The attachment is
   `p_origin + R*offset`; this boundary drives the cable simulator. The objective
   scores the predicted cable encounter and motion, plus execution costs.
4. The accepted **input setpoints** are exported. The predicted pose is stored
   separately as the forecast. Sending the forecast coordinates back as the
   setpoints would apply the drone response again and change the experiment.
5. Actual onboard feedback tracks these setpoints. The offline planner/model
   does not run as an online flight correction loop in the present experiment.

Thus this is forward-model-based command optimization. It is not a separate
inverse controller that guarantees an independently prescribed drone XYZ curve.
The task specifies the cable target and motion preferences; the optimizer chooses
both the commands and, through the model, the resulting drone path.

## Independent checks performed

Reproducible scripts/results: `runs/audits/drone-command-chain-20260910`.

| Check | Result |
|---|---|
| Actual selected drone nominal parameters and residual versus original preliminary fit | Match; residual bytes identical |
| Re-extract all five saved preliminary command schedules from raw CSVs | Exact array agreement |
| Invalid command rows/stricter coverage affecting accepted window support | None in these takes |
| Whole-take roles | 69 training windows from four takes; 26 validation windows from figure8_002 |
| Original fit source/input hashes | 292 checked, unchanged |
| Exported CSV versus frozen forecast's command array | Exactly identical; 309 rows, 30 Hz, 10.266667 s |
| Integrate selected jerk commands to exported whip P/V/A | Agrees within 1e-12 |
| Streaming planner versus frozen forecast, whip origin | Maximum coordinate difference 2.11e-15 m |
| Streaming planner versus frozen forecast, whip cable | Maximum coordinate difference 7.46e-13 m |
| Independently predict the complete CSV with selected drone model | Exact saved origin-array agreement |
| Predicted rotated attachment versus saved cable root | Agrees within 1e-9 m |
| Saved input/forecast identities | Original selected SHA-256 hashes verified |

These are tests of the same model across independently exercised code paths,
not a second physical model or evidence of low real-flight prediction error.
No original CSV, forecast, controller setting, model, fit or plan was changed.

Targeted regression suites: 73 passed, 2 skipped. These cover pose response,
residuals, preliminary preparation, initialization, differentiable execution,
command contracts and export. Actual numerical hardware: Windows / RTX 4080.

During the selected whip, the predicted origin and held position setpoint differ
by 16.23 cm RMS and 29.28 cm maximum Euclidean separation. **This is not measured
prediction error.** It illustrates the non-ideal command response the optimizer
is already accounting for. See `command-versus-prediction.png` in the audit.

## Material limits found

**Acceleration extrapolation.** Commands represented in the four training takes'
accepted window/history support reached 8.18 m/s² acceleration magnitude. The
selected whip reaches 14.31 m/s², including X braking of -14.26 m/s² versus
training's -8.18 m/s². Selected speed is 2.61 m/s versus training's 3.77 m/s:
being within the speed range does not remove the acceleration extrapolation.
These are command values, not measured achieved accelerations or actuator limits.
No provisional simulation bound establishes accuracy in this unobserved regime.

**Accuracy is not yet precise-strike evidence.** The original held-out figure8_002
two-second drone position RMS is 11.99 cm (cold seed 32.83 cm). That is useful
retrospective improvement, but it does not support a guarantee for a 5 cm tip
target in a new whip. The residual reached its 400-update safety ceiling before
plateau. The take was later inspected during development, so is no longer blind.

**Effective loaded model, one-way coupling.** Cable motion does not feed an
explicit varying tension back into the drone predictor. Preliminary loaded
response can absorb typical loading; whip-specific tension may depart from it.
Adding cable reaction on top without rederiving the loaded model could double
count loading. Do not make that change as a casual correction.

**Initial-state assumption.** Fitting initializes position, velocity, attitude
and fixed compensation from 0.4 s of preceding measured motion. Offline planning
instead assumes zero velocity, identity tracking alignment, zero effective
compensation and a straight hanging cable. The residual's hover gate preserves
that ideal equilibrium. A ten-second real hover does not prove all those states
are identical. Inspect initial tracking pose/cable motion in the recorded flight;
retain the original forecast and distinguish any later measured-state replay.
The compensation is not an observed onboard I-term and does not evolve like it.

**Clock and receipt semantics.** `time_s - cmd_age` reconstructs a command time
under the logger's convention. It is not evidence of onboard reception or motor
execution. The available `experimental_data/source_audit/experiment_logger.py`
keys commands by ROS message-header timestamps. Its column layout differs from
preliminary1, so it is not established as the logger that produced these files.
The cached OptiTrack/native-stream alignment also contains unknown transport
latency. The fitted 20 ms is an effective end-to-end delay, not independently
measured actuator latency. The actual sender/logger sources remain required to
resolve which host event each timestamp denotes.

## Final command interface to verify

Use the selected `fullstate_30hz.csv` as inputs, unchanged. The intended mapping
is CSV position/velocity/kinematic acceleration to full-state position/velocity/
acceleration, saved yaw to yaw, and zero angular-rate feedforward. Preserve the
same numeric world-frame/estimator convention as preliminary1; never silently
reinterpret tracked origin as COM or add the attachment offset to flight commands.
Preserve 30 Hz zero-order-held packet boundaries and actual timestamps. A sender
that interpolates, rescales time, drops/skips rows, adds acceleration compensation
or changes control gains changes the model input/plant and requires review.

Keep the same normal PVA controller configuration used for the preliminary data.
The old force-controller experiment disabled position gains; those instructions
do not apply to this learned closed-loop PVA model. No gain switch is requested.

Current upstream Crazyswarm2 maps `cmdFullState` P/V/A into ROS message fields and
publishes them; that publication is not an onboard acknowledgement.
[Primary Python source](https://github.com/IMRCLab/crazyswarm2/blob/main/crazyflie_py/crazyflie_py/crazyflie.py).
Current upstream Mellinger includes position/velocity/integral feedback and adds
gravity to acceleration when forming thrust. Do not add gravity to this kinematic
CSV. This upstream reference does not establish the flashed lab revision.
[Primary controller source](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/src/controller/controller_mellinger.c).

## What the next measurement should establish

First verify the actual sender/logger and unchanged controller identity. Then
compare recorded command values/timing to the frozen CSV and native drone/cable
motion to the original forecast. Diagnose timing/initial-state mismatch separately
from model error. In particular inspect the stronger braking interval. If drone
prediction is wrong, cable damping should not absorb that discrepancy; follow
the conditional-versus-command-driven comparison in `M0_TO_M1_ADAPTATION.md`.
No new fit or change to the selected flight was performed by this audit.
