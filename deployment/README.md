# Offline PPO deployment package

This package plans one open-loop cable strike from the **measured initial drone and cable state**. It contains the selected PPO weights, their original model/task/configuration, and the same planning/physics code used by the desktop project. It neither connects to ROS nor sends flight commands. No retraining is needed merely to transfer it to another computer.

The desktop simulator stays on Windows. Copy the `Whip-Deployment.zip` bundle to your colleague's Ubuntu computer and extract it. A USB drive or your normal file transfer is fine; the bundle checksum is in `BUNDLES.json`. The full research data and training results are in the separate development bundle.

## First offline check on Ubuntu

Use Python 3.12 and a fresh environment. Do not reuse or copy a Windows `.venv`. These commands run from the extracted `Whip-Deployment` directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m deployment.planner --verify
.venv/bin/python -m deployment.planner --example-initial synthetic_initial.npz
.venv/bin/python -m deployment.planner --initial synthetic_initial.npz --output offline_plan
```

On Windows, use `py -3.12 -m venv .venv` and `.venv/Scripts/python.exe` instead. The offline planner runs on CPU, including on the Ubuntu 5080 host. GPU training and GPU batched validation stay in the full development project. The CPU planner favors matching existing behavior; planning latency must be measured before flight. This is not a real-time ROS controller.

The synthetic state is for installation testing only. The example may take a while because it predicts cable dynamics before returning. `offline_plan/commands.csv` contains world-force holds in newtons, with explicit start and end times. `plan.npz` preserves the exact 100 Hz physics trace; `plan.json` records timing, initial-state clock, hashes and `flight_ready: false`.

The packaging workstation produced the nominal example in about 5.2 seconds with a 0.81-second sequence. This is an installation measurement on one Windows CPU, not an Ubuntu timing guarantee or proof that the state remains suitable for launch.

## What your colleague connects

1. While the drone remains in position-controlled hover, record 100 Hz tracking, a timestamp, the drone rigid-body orientation and all ten cable markers. Use a common, explicitly verified Z-up world frame for tracking, target and forces. Transform the top-of-drone tracked origin to the cable attachment using the body rotation and the saved body-frame offset.
2. Estimate the launch state causally from past frames, reconstruct the attachment and interpolated node, and project the inextensible geometry/velocity constraints. Write an NPZ with `positions_m` and `velocities_m_s`, each **12 x 3**, plus scalar `state_time_s` in the agreed tracking clock. Node 0 is the attachment, node 1 is the midpoint to C1, nodes 2–11 are C1–C10. See `experimental_data/state_initialization.py` and `docs/INITIAL_STATE_OPEN_LOOP.md`. Do not assume zero velocities for a moving cable or fit a spline using future measurements.
3. Call `create_plan(package, initial_path, output)` or the CLI while position control remains active. Check the output against the latest measured state before launch. Current simulator tolerances are 2 mm maximum node-position drift and 0.01 m/s node-velocity drift; they are provisional and may be difficult to meet after a slow plan. Do not launch a stale plan; measure this issue and improve/prewarm planning if necessary.
4. Switch the verified outer-loop interface to force-derived feedforward and execute the frozen sequence **once**. Keep attitude/rate stabilization. Policy commands nominally hold for 50 ms (20 Hz); a 50 Hz ROS publisher must choose the command by elapsed monotonic time, not advance the policy each publish. Preserve the final partial hold and the exact cutoff. `force_at_elapsed` returns `None` at the cutoff; it never loops or holds the last force forever. Timing faults and emergency intervention need explicit controller recovery.
5. Restore position control at the **precomputed cutoff**, including on a predicted miss. There is no real hit detector in the maneuver loop. Keep tracking/controller data for diagnosis without feeding it back to the strike policy. Abort/emergency handling is a separate hardware function.

The current actor is queried along a private simulated trajectory starting from that measured state. It is not repeatedly queried with real-flight state. A single force CSV from a synthetic state is therefore **not** the general policy deployment artifact.

## Controller conversion and remaining integration

Our force is the **total world thrust vector**, with gravity applied separately by the simulator. The reviewed upstream Mellinger acceleration interface gives the conditional conversion:

```python
from deployment.planner import controller_acceleration
accel = controller_acceleration(force_world_n, verified_controller_mass_kg,
                                verified_firmware_gravity_m_s2)
```

This computes `F / controller_mass - [0, 0, gravity]`. It is valid only after confirming your firmware equations, frame, mass, thrust calibration and enabled feedback terms. It is not an instruction to add/subtract gravity arbitrarily or use an assumed Crazyflie mass. The simulated drone point mass is 0.159 kg; cable plus markers add 0.01609091 kg. The simulator's 3.2 N limit is provisional, not a measured hardware capability.

`deployment/reference/force_controller.py` is the colleague's original script, copied unchanged for review. **It is not the executor for this package.** The review in `docs/CONTROLLER_INTERFACE_REVIEW.md` found setpoint gaps during gain changes, incomplete abort restoration and an insufficient 0.40 m takeoff for the hanging cable. Confirm actual hardware/firmware parameters and implement/test an acknowledged, continuously streamed controller handoff before a learned strike. Keep the Python/ROS environment that runs that bridge separate if its ROS distribution requires another Python version; the planner can communicate through files initially.

Record OptiTrack, the commands actually sent, actual controller mode/gains, attitude/IMU and timestamps. `experimental_data/flight_recorder.py` supplies a transport-neutral logging sink. ROS topic names, transforms, clock synchronization and the hardware sender are still to be implemented with the colleague. Nothing in this offline package arms the drone.

Read `HANDOFF.md` for the research method, exact checkpoint selection, calibration rationale and adaptation boundaries. References to full data/results there apply to the separate development bundle.
