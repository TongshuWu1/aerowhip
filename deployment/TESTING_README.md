# PPO testing package

Current controller convention: compensate the total hanging weight using `ctrlMel.mass = 0.175` kg and match that in export settings. The physical model remains 0.157 kg drone plus 0.018 kg cable assembly. Controller force is `F_total - [0,0,0.175*g]`; acceleration is that residual divided by 0.175. Zero residual then corresponds to ideal settled hanging hover. This supersedes the earlier 0.157 kg controller convention described in historical notes below. Do not reuse older gravity-subtracted CSVs unchanged. Match the firmware gravity constant; 9.80665 m/s² is the saved export setting. No physical controller is configured by exporting.

This is an offline planner exported from an explicitly selected PPO checkpoint. It does not connect to a drone. `policy/` preserves the checkpoint and its saved model/task/PPO configuration. `TRANSFER_MANIFEST.json` hashes the export-time runtime; `training_source_manifest.json`, when present, describes the original training code separately.

GUI exports also include `experiment_setup.json` with the desired attachment hover position and target selected on the Testing page. This is separate from the immutable training configuration. Pass the actual target explicitly with `--target` for each attempt; the CLI otherwise uses the saved training target.

Create a fresh Python 3.12 environment on the receiving computer and install `requirements.txt`. Do not copy a Windows virtual environment to Linux. Run from this directory:

```bash
python -m deployment.planner --package policy --verify
python -m deployment.planner --package policy --example-initial example.npz
python -m deployment.planner --package policy --initial example.npz --target 1 0 1.4 --output example_plan
```

With CUDA-enabled PyTorch, add `--device cuda` to generate the plan with the same GPU physics and actor as the desktop Testing page. The planner reports startup preparation separately from planning latency. There is no automatic CPU fallback for a CUDA request. The default CLI device remains CPU for compatibility.

The example is synthetic. For drone-only initialization, provide an NPZ with finite `(3,)` arrays `attachment_position_m`, `attachment_velocity_m_s`, and scalar `state_time_s`. Coordinates are in the same world frame as the target. Convert the tracked drone reference to the attachment using the calibrated offset and measured rotation (and angular velocity for the velocity offset). The point-mass simulator has no attitude state; do not interpret attachment coordinates as a real vehicle center-of-mass command.

The intended real preparation is 10 seconds of settled hover, then an assumed vertical cable with no relative cable velocity. The actor still expects its complete state; the adapter constructs that state without online cable measurements. Planning runs a private cable simulation and takes time. Continue hover and check drone state drift before launch; the CLI emits a plan but does not perform that launch check.

`commands.csv` contains start/end seconds and total world thrust Fx/Fy/Fz in newtons. The actor updates at 20 Hz; identical holds may be merged. `plan.npz` preserves every 100 Hz physics sample and `plan.json` records the exact cutoff, target, initialization and policy hash. No measurements update the strike sequence after launch. The normal controller resumes at the cutoff. Firmware force conversion and the real controller handoff must be implemented with the colleague; no flight sender is included.

In the desktop Testing page, Start rehearsal approaches the selected hover position from a suspended position 10 cm below it (ground takeoff/contact are not modeled), settles for 10 simulated seconds based only on drone position/velocity, then plans in the background while hover continues. Inspect the CSV and click Execute sequence. Moving the target before execution invalidates the plan. Replan if the measured drone state has drifted. The recorded physical cable is never a planner input. All attempts, including failed/refused plans, remain in the rehearsal output directory.

Testing now requires CUDA and uses ideal timestamped position packets at 100 Hz, causal backward-difference velocity, and a 20 Hz normal controller as well as the 20 Hz strike commands. No real OptiTrack noise or network delay is assumed. The display reports measured wall-clock rates; full timing and controller update timestamps are saved with each rehearsal. Plant and planner use separate GPU buffers/streams, with higher priority for the plant. This models a position-data interface; it does not implement NatNet or replace the real vehicle's attitude/rate stabilization.
## Controller gravity compensation

The current workspace saves controller settings in `config/controller_export.json`: 0.157 kg drone mass and gravity 9.80665 m/s², following the user's measured 175 g total minus 18 g cable assembly. Testing restores saved values at startup. Edits save when editing finishes, or through **Save controller settings**. A workspace without a saved settings file starts unconfigured. The physical controller must use matching values; this UI does not configure firmware. Old plans/packages retain their original settings.

In Testing, enter the mass actually configured in the colleague's controller and match its gravity constant. Mass starts as **Not configured**: it is not inferred from the simulation mass. These settings are saved in `experiment_setup.json` and used by the exported planner. CLI overrides require both `--controller-mass-kg` and `--controller-gravity-m-s2`.

New plans with these settings include:

- `commands.csv`: total PPO world thrust in N, used by the simulation and unchanged.
- `controller_force.csv`: feedforward force in N, `F_policy - [0, 0, m_controller*g]`, for a force interface that adds that weight.
- `controller_acceleration.csv`: feedforward acceleration in m/s², `(F_policy - [0, 0, m_controller*g])/m_controller`, for the supplied `cmdFullState` acceleration interface. Testing provides an **Open controller acceleration CSV** button.

All three files have identical hold boundaries and cutoff. Negative feedforward Fz is allowed; it does not imply negative total thrust. The table and main Open button default to controller force when available. A labeled selector switches between controller force, controller acceleration and simulation total thrust, updating the displayed units. If controller settings were omitted, the UI explicitly identifies total thrust and requests mass/gravity configuration for a new rehearsal. Use only the file matching the receiving interface; do not subtract gravity again. Conversions assume the agreed common world Z-up frame. The controller mass is its configured compensation mass, not automatically the combined drone/cable mass. Conversion parameters are recorded in `plan.json`. The PPO and physics are unchanged, and no ROS sender is implemented by this change. Existing packages and recordings remain immutable; export a new package to obtain the updated runtime.
