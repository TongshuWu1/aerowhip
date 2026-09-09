# Testing page: drone-only PPO rehearsal

Restart `run_simulation.py`, then open **Testing** (page 6). The original pages remain and SAC remains disabled. No training is started by this page.

## Current GPU timing and tracking interface

Testing now requires CUDA and uses the RTX 4080 for both cable physics and actor inference. Simulated tracking supplies **position-only drone attachment and target packets at 100 Hz**. Velocity is estimated causally from successive positions; exact simulated velocity and cable-marker state are not passed into the planner or the normal controller. Packets are ideal: measured OptiTrack noise, delay, dropouts and rigid-body attitude dynamics have not been identified or added. The interface is simulated; it does not open a NatNet connection.

The outer hover/recovery controller updates every **50 ms (20 Hz)** and holds its force for five **10 ms physics steps**. The frozen strike retains the trained 20 Hz holds and exact partial final interval; the mode switch occurs at its exact cutoff. The measured rates and GPU name appear in the status area. `flight.npz` adds position packet timestamps, measured positions, derived velocities and controller-update timestamps. `timing.npz` records wall-clock tick times, physics processing durations and scheduling lateness. The current worker also records controller wall-clock timestamps. No steps are skipped to hide slow processing.

Independent GPU graphs/buffers protect the physical plant from the concurrent private planner. The live stream has higher priority. The optimized path fuses the existing 12-node free-cable float64 damping and projection operators, retaining 32 PCG iterations, four position projections, one velocity projection and 12 internal substeps. It uses the existing analytical isotropic bending gradient. Same-input 10 ms transitions agree with the reference GPU path within test tolerances of 1e-9 m and 1e-7 m/s on a bent 3D trajectory. It is not bitwise identical: roundoff can accumulate through the existing straight/bent damping branch (a separate 1 s force-rollout check differed by up to about 3.5 mm). Training defaults and saved calibration are unchanged. Generated pinned-cable kernel sources were compared with the frozen training source and remain byte-identical for all existing supported topologies/pin counts and tested iteration counts.

**Native Windows / RTX 4080 full rehearsal:** `runs/rehearsals/20260906-192434-555477`, unchanged final PPO, target `[1.02,0.01,1.4]`. Results:

| Measurement | Observed |
|---|---:|
| Tracking packets / wall second | 99.999 Hz |
| Controller updates / wall second | 20.024 Hz (includes exact partial-interval handoff) |
| Physics processing mean / p95 | 4.93 / 7.25 ms |
| Wall-clock tracking interval p95 | 10.96 ms |
| Tick starts more than 10 ms late | 0 |
| Complete sequence planning after warm-up | 0.599 s |
| Planned / measured wall-clock strike duration | 0.810 / 0.810050 s |
| Outcome | Valid simulated hit and settled recovery |

The executed force trace matched the saved plan exactly. Detailed metrics are in `gpu_verification.json`. This is measured desktop timing, not a hard real-time guarantee, Linux validation or a physical-flight result. The earlier failed GPU setup attempt remains preserved; a missing CUDA context on the Qt worker thread was fixed before this successful run. Repeat the full native check with `python tools/check_gpu_rehearsal.py --checkpoint <saved-checkpoint-path>`; it performs simulation only and stores a new rehearsal.

Twenty-seven targeted regressions passed, covering reference/fused physics, concurrent GPU buffer isolation, 100/20 Hz sample-and-hold behavior, position-derived velocity, frozen execution, target invalidation, existing CPU flight behavior, UI controls, and independently exported CPU and GPU planners. The existing CPU JIT tests emitted only their known deprecation warnings. GPU requests do not start training.

Portable packages expose the same GPU planner via `python -m deployment.planner ... --device cuda`. Startup preparation and sequence planning time are reported separately. CPU remains the portable CLI's explicit compatibility default; the Testing page does not silently fall back to CPU.

1. Select the PPO run and checkpoint. The newest run's `latest.pt` is initially listed first; the stopped current run is `20260906-174733-192436-seed652` at 39,936 attempts. `latest.pt` and `terminal.pt` are identical for that stopped run.
2. Set hover attachment XYZ and target XYZ in meters. The simulated drone is a translational attachment point, not a tracked rigid body or full attitude model.
3. **Export policy…** writes a new folder and ZIP with the explicitly selected checkpoint, original model/task/PPO configuration, export-time headless runtime, manifests, README, and separate `experiment_setup.json` containing the desired hover/target. Nothing is sent externally. This does not change the older hardcoded lab-transfer exporter or its historical policy selection.
4. **Start rehearsal** also freezes a package in a new `runs/rehearsals/<timestamp>/` folder. After physics warm-up, the drone approaches hover from a suspended position 10 cm below it. Ground takeoff/contact are not modeled.
5. The normal controller holds the selected position. Settling requires drone position within 2 cm and speed below 0.01 m/s continuously for 10 simulated seconds. Cable shape and velocity do not influence this gate.
6. The planner samples attachment position/velocity and the known target. It constructs a vertical cable using the saved lengths, with zero relative cable velocity. The PPO still receives its expected complete observation internally. Planning runs separately while physical hover continues; no true cable-node state is passed into planning.
7. The CSV appears before execution. Inspect start/end times and Fx/Fy/Fz in newtons, or use **Open CSV**. **Execute sequence** rechecks drone-state drift, executes the finite frozen sequence once, and returns to normal control at its cutoff. Execution never queries the actor. The rehearsal finishes after 10 more seconds of settled drone hover following recovery.
8. Changing the target with **Apply target** before execution invalidates the prepared plan. **Generate again** creates another preserved plan. Target controls are disabled during the strike/recovery. A refused launch stays in hover; generate a new plan from the current state.

The displayed real-time rate is measured; the simulation never skips physics steps to catch up. Sampling is 100 Hz in simulated time, the actor's held commands update at 20 Hz, and the exact cutoff can end a partial final hold. CSV rows merge identical adjacent commands; the NPZ preserves the 100 Hz trace.

Each rehearsal preserves the selected package, edited rehearsal task, generated plans, `flight.npz`, and `flight.json`. Plans record the initial-state timestamp, target, assumed state, policy hash, and planning latency. Simulated true cable motion is recorded for later comparison, including misses or stopped trials. `status.json` is progress telemetry; the final outcome is in `flight.json`.

## Earlier CPU verification on Windows / RTX 4080

Twenty-four targeted tests cover existing flight behavior, assumed-state isolation, the 10-second gate, drone-only launch drift, exact CSV/execution timing, no actor queries during execution, target invalidation, independent exported-package operation and target overrides, native PyVista/VTK page navigation, and UI controls.

The full native rehearsal used the unchanged 39,936-attempt checkpoint (SHA-256 `214fb97211c1b57a04a7352d9629a0369c185a97efc0ff665e0336b5188d303d`), hover `[0,0,1.5]`, and target `[1.02,0.01,1.4]`. Output: `runs/rehearsals/20260906-190320-029827`. After 10 settled simulated seconds, planning took **2.798 seconds** and produced a **0.81-second** sequence. It executed from simulated time 12.56 to 13.37 seconds, registered a valid simulated hit, returned to PID control, and completed settling. The CPU simulation ran at approximately **0.29× real time**; cold physics preparation is additional startup work. These are one local rehearsal's measurements, not real-flight or Linux validation or an independent estimate of policy success rate.

Native renderer image: `native_viewer.png`. Qt's QWidget-only grabs do not capture the native VTK framebuffer, so the standalone renderer image is used to verify the 3D scene. The earlier `20260906-190032-584059` attempt was stopped by a smoke-test callback assertion before execution and remains preserved.

The real tracking reference transform, firmware force conversion, and controller handoff remain separate unimplemented hardware integration. See `deployment/TESTING_README.md` for the portable planner interface.
