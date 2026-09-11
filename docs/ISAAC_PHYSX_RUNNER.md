# Independent Isaac Lab / PhysX drone and cable

This is a separate process in the `Simulator` checkout. It flies a **153 g
drone with an 18 g passive cable**, using PhysX motion and joint constraints.
It does not load the research DDER simulator, M0/M1, residuals, PPO or MPPI.
The full research application and its seven pages are retained separately.

## Launch

Open `C:/Users/wts28/Documents/PHD/particle_filter_cable_simulator` in PyCharm.
Choose **Isaac PhysX figure-eight** or **Isaac PhysX circle** from the saved run
configurations. These use the ordinary project interpreter to start the
dedicated Isaac interpreter in a separate process.

From a terminal in that checkout:

```powershell
python launch_isaac_simulation.py --trajectory figure8
python launch_isaac_simulation.py --trajectory circle
```

The simulator opens its own Isaac window, holds in the air for 10 simulation
seconds, flies one smooth loop, and returns to its starting hover. It saves the
completed take after a 3 s final hold and continues hovering so you can inspect
the scene. Use **Finish and save** to close, or add `--exit-after-trajectory`.
**Pause / resume** freezes physics and logging. Native timeline Stop finishes
the take; start a fresh process for a new run.

Cyan is the analytic command path; orange is the PhysX measured tracking-origin
trail. The yellow cable, white cable markers and red tip move with their actual
physical links. The viewport supports Isaac's normal camera navigation.

More examples:

```powershell
python launch_isaac_simulation.py --trajectory vertical8 --period 10 --radius 0.55
python launch_isaac_simulation.py --trajectory circle --hold 2 --period 6 --cycles 2 --exit-after-trajectory --screenshot
python launch_isaac_simulation.py --trajectory figure8 --headless --exit-after-trajectory
python launch_isaac_simulation.py --trajectory hover --duration 2 --headless --device cuda:0
```

The installed Isaac interpreter is automatically found at
`C:/Users/wts28/env_isaaclab/Scripts/python.exe`. Elsewhere, set
`ISAACLAB_PYTHON` or pass `--isaac-python PATH`. An explicitly invalid path
fails rather than silently selecting a different environment. Isaac must be
installed in that interpreter; do not install it into the Python 3.12 Qt venv.
The first launch may download the official Crazyflie USD and compile shaders.

## Plant and controller

`config/isaac_physx/rig_153g_paracord.json` owns the independent plant parameters.
The original `rig_153g.json` remains an explicitly selectable legacy baseline.
It is deliberately separate from the model to be identified later.

- Drone: free six-degree-of-freedom rigid body; 0.153 kg mass; explicit inertia;
  four bounded rotor thrusts; rotor torque mixing; first-order motor response;
  geometric attitude feedback and position/velocity/acceleration tracking.
  Maximum collective thrust is 3.2 N. Controller hover support includes the
  loaded cable weight. Motion is never prescribed by writing body poses after
  initialization.
- Cable: 0.9525 m long, 32 capsule links and 32 D6 ball joints, with distributed
  total mass 0.018 kg including ten markers. The top is a free pivot. PhysX
  enforces attachment/length constraints and transmits cable reaction to the
  drone. Separate bending and torsional restoring drives and damping are
  integrated implicitly by PhysX (except the free top pivot). Distributed
  crossflow and axial drag include rotational velocity and force moments;
  marker spheres have their own aerodynamic drag.
  Adjacent links do not collide; nonadjacent cable
  self-collision, hull collision and floor collision are enabled. There is no
  terminal payload.
- Appearance: NVIDIA's Crazyflie mesh at 1.8 scale, tracking beads, navigation
  lights, metric floor grid, command and measured trails. The stock mesh's
  original physics is removed; its stock 27 g dynamics are not used.
- References: analytic circle, horizontal figure-eight or vertical figure-eight
  with smooth angular-speed ramps and analytic PVA. `period` is the steady
  angular period; total moving duration is `period * cycles + ramp_s`.

These are **engineering estimates**, not a calibrated hardware twin or a
firmware-equivalent Crazyflie controller. Inertia, hull geometry, motor limits,
lag, aerodynamics and bending parameters need identification or measurement.
The COM-to-tracking/attachment split is assumed; its net tracking-to-attachment
offset matches the 5.5 cm convention. Marker masses, COM and inertia are computed
at their actual longitudinal locations, under a spherical-marker assumption;
marker collisions are enabled. The 8 g bare cord plus ten 1 g markers split,
6 mm marker radius and 3.5 mm cord diameter remain estimates. Axial
extension, braid-level friction/hysteresis, propeller downwash and flexible propeller dynamics
are not modeled. This is an independent test plant, not proof of real-flight
accuracy. Do not fit the research model by importing the plant's hidden values.

NVIDIA documents [D6 joints and spherical-drive limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/joints.html)
and [articulation root selection](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html).
The articulation root is explicitly on the drone body: automatic selection on
the container can choose a cable link and invalidate the reset coordinates.

## Frames, rates and outputs

World XYZ is in metres with Z up. Command and logged drone position refer to the
**top tracked origin**, not the physical COM. Body-to-world quaternions use
`wxyz`. The body-frame tracking offset is `[0, 0, +0.025]` m and cable attachment
is `[0, 0, -0.030]` m relative to COM. Both offsets rotate with the body;
tracking-point velocity includes the angular-velocity lever arm.

| Component | Rate |
| --- | ---: |
| PhysX and motor dynamics | 1,200 Hz |
| Drone controller | 300 Hz |
| FullState command packets | 30 Hz |
| Synthetic measurements | 100 Hz |
| Viewport update | 30 Hz |

The controller holds each PVA packet until the next 30 Hz update. Synthetic
measurements currently have no sensor noise, latency or clock mismatch.
Timestamps are simulation time, independent of rendering speed. The initial
state is airborne, level, with a straight hanging cable and hover rotor thrust;
ground takeoff/landing is not implemented in this runner.

Each process creates a new ignored `runs/isaac_physx/<timestamp>/` directory:

- `fullstate_30hz.csv`: existing twelve-column command contract, including
  PVA, yaw and yaw rate. These are the commands actually supplied to this plant.
- `ground_truth_100hz.csv`: tracked-origin position/velocity, attitude, world
  angular velocity, held commands, ten cable marker positions, tip, motor thrust
  and joint-gap diagnostic. This is a native synthetic format, **not yet a
  drop-in Motive/controller-log adapter** for the research fitting UI.
- `plant.json`, `source_snapshot/`, `provenance.json`: frozen plant and source.
- `summary.json`: tracking RMS, peak error, tilt, saturation and constraint
  consistency. RMS is Euclidean drone position error relative to held commands;
  `motion_tracking_rms_m` excludes the initial/final holds.
- `scene.png` when `--screenshot` is requested.

`--output PATH` selects a fresh directory; existing output directories are
rejected. No old flight data, models or policy checkpoints are copied.

## Verification on this computer

Windows 11, RTX 4080 16 GB, Isaac Sim 5.1.0, installed Isaac Lab, Python 3.11.
Twelve CPU math tests pass, covering analytic PVA, loaded hover thrust, actuator
lag, cable torque balance/energy gradient, drag dissipation, geometry/velocity offsets and invalid
configuration rejection, resolution-scaled material coefficients, marker mass
moments/inertia, and aerodynamic power dissipation with offset COM and rotation.

Historical baseline runs with 153 g + 18 g and 33 bodies (before paracord upgrades):

| Test | Motion RMS | Peak drone error | Result |
| --- | ---: | ---: | --- |
| Figure-eight, 10 s period, three loops, 45 s total | 0.768 cm | 1.818 cm | Completed with visible rendering and screenshot (`figure8_final16`) |
| Circle, 10 s period, three loops, 45 s total | 1.018 cm | 1.803 cm | Completed headless (`circle_d6_15`) |

Maximum observed joint separation was below 0.0003 mm in these flights.
The figure-eight reached 3.68 degrees of tilt. These are controller/plant smoke
checks, not independent evidence that adaptation improves a real drone.

CPU PhysX is the default: for a two-second single-rig hover it took 1.39 s versus
15.83 s using GPU PhysX, with matching results. RTX rendering stays on the GPU.
Those hover timings were measured before the damping fix. Final headless circle
took 29.62 s for 45 s simulated; final visible figure-eight took 82.86 s for
45 s simulated. Visible mode is capped at real time when fast enough, but **does not
guarantee real-time wall speed**. `--device cuda:0` remains available. This says
nothing about future many-environment GPU throughput.

The standalone lifecycle handles stop/close separately from Lab's installed
STOP render-loop callback. Paused rendering and screenshot capture use
`SimulationContext.render()`, which suppresses uncontrolled physics steps.
An earlier screenshot-only BroadPhase warning was fixed through that path.
Native timeline STOP was exercised: it saved the partial take and exited.

Longer testing exposed instability in the initial explicit cable damping
implementation; short-run results from that prototype are superseded. The
baseline D6 ball joints used an isotropic, passive angular damper of
`6.71916e-7 N m s` on each internal joint axis, with zero target velocity and
zero drive stiffness. The top pivot remains undamped. This angular resistance
also damps relative axial spin; it is an approximate joint material parameter,
not identified torsional physics. The new paracord configuration supersedes
that parameterization as described below. PhysX's implicit drives are described in its
[articulation documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html).

## Next research step

The paracord upgrade, numerical checks and rejected native-cable prototype are
documented in [PARACORD_MODEL.md](PARACORD_MODEL.md). It supersedes the baseline
material parameters and 600 Hz default described in the historical checks above.

Add command-CSV playback and a synthetic measurement adapter into the existing
recording pipeline, then collect a new preliminary dataset and identify M0.
Only after that should policy training and paired adaptation experiments run.
No M0, residual fit, PPO training or real flight was started by this task.
