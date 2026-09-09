# Initial drone tracking residual from yesterday's whip trials

The user reauthorized the archived whip recordings specifically for initial drone-model fitting. They remain excluded from preliminary cable-residual training. Raw files and their processed versions were not modified.

Open **Real-world Updates → Drone residual** to inspect or repeat the initial fit. It creates a candidate, not a change to PPO or the active simulator. No controller/logger edits or real flight commands are involved.

## Model and data

The effective tracking model predicts acceleration from position error, velocity error and commanded acceleration, with a fitted effective delay. A small two-layer, 32-unit tanh NN adds a correction limited to ±2 m/s² per component. Inputs contain reference-to-predicted position/velocity errors, commanded acceleration, predicted velocity and a 50 ms command history. Position and velocity are integrated recursively from the initial measured state; subsequent measured drone positions are targets only.

Input is the **logged full-state command stream**, including the original early transition to measured-position hold at roughly 0.67 seconds. It is not the intended exported CSV, and no missing reference tail is fabricated. Target is the measured OptiTrack cf_7 origin, without assuming equivalence to the center of mass or cable attachment. The fit covers approximately −0.2 to +1.2 seconds relative to the first maneuver command. Initial velocity uses only the preceding 0.1 seconds of measured positions.

Split fixed before optimization:

- Fit effective gains/delay and NN: `whip1_001`, `whip1_002`.
- Prediction check only: `whip1_003`.

Source: `data/adaptation_rounds/adaptation0/processed/20260907-194802-922485`. A drone-only input snapshot, fixed settings/split, hashes, source snapshot and results are saved per candidate under `data/drone_residual_runs/`. No cable markers are used as training targets or inputs.

## First result

Candidate: `20260907-230802-464261`, 160 NN updates, Windows / NVIDIA RTX 4080, about 24.6 seconds for fitting and reporting.

| Trial | Role | Tracking model position RMSE | With NN | Maximum error with NN |
|---|---|---|---|---|
| whip1_001 | Training | 4.27 cm | 3.32 cm | 6.76 cm |
| whip1_002 | Training | 4.17 cm | 3.36 cm | 7.42 cm |
| whip1_003 | Validation | 4.66 cm | 3.88 cm | 10.36 cm |

Validation RMSE decreased by about 16.7%; maximum validation error decreased from 12.28 to 10.36 cm. These are **drone trajectory prediction** errors conditional on logged commands, not cable-tip errors or improved real-flight performance.

The selected nominal effective delay is 60 ms. It includes uncertain clock alignment and logger/interface latency; it is not a measured firmware delay. Several nominal parameters reached their fitting bounds, and Y motion has little excitation. The fitted gains must not be copied into the real controller. This is a provisional response model for one repeated maneuver and the fixed drone/cable setup, with no verified contact/intervention annotations.

The NN checkpoint is `drone_residual.pt`; `simulator.drone_tracking.load_tracking_model` loads the network, nominal gains, effective delay and history interval. Prediction takes causal-held commands already adjusted for the effective delay. It is not wired into PPO, the force-driven simulator, or CSV generation yet. A subsequent coupled prediction implementation must account for cable loading consistently, since this effective drone response already includes the observed fixed cable's effects. New complete-sequence recordings are needed to check transfer to the current hover-to-hover execution.
