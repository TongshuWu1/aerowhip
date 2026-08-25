# Isaac Lab drone-cable plant

This package is the implementation-checked plant layer for the whipping project. It replaces
the earlier point-mass drone and prescribed cable attachment with one coupled
PhysX articulation:

- a floating, 6-DoF drone with explicit mass and diagonal inertia;
- bounded force and body-torque inputs applied to the drone rigid body;
- a fixed material-frame attachment between the drone and cable root;
- 20 passive rigid cable links (21 material stations) by default;
- two Kelvin-Voigt bending axes at every internal joint; and
- a mechanically free distal end.

The internal twist axis is free: it has no stiffness, damping, friction, or
limit. This is deliberate for the current one-attachment/free-tip task, where
twist is not observable enough to justify transferring the old fitted `GJ`.
Cable self-contact and aerodynamic drag are also outside this first baseline.

## Physical mapping

All inputs use SI units. For an internal hinge with dual material length
`h_i`, the fitted continuum parameters are mapped as

```text
k_i = EI / h_i          [N m / rad]
c_i = Cb / h_i          [N m s / rad]
```

USD/PhysX angular drives are expressed per degree, so the authored values are
`k_i*pi/180` and `c_i*pi/180`. Regridding deposits the identified material-point
masses onto the requested node grid and conserves total mass and cable length.
The resulting chain is an equivalent discrete bending model, not a claim that
PhysX reproduces the DDER integrator exactly at arbitrary curvature. Resolution
and trajectory comparisons are therefore part of validation.

## Current model provenance

The default artifact is
[`optitrack_offline/models/cable_model.json`](../optitrack_offline/models/cable_model.json),
SHA-256
`0eece55cf9c2b07fcd699c1e7d9f5079d01240aabda2a7237a54b8c3c4fbc827`.
The values transferred into this baseline are:

- length: `0.961 m`;
- diameter: `3.5 mm`;
- dynamic mass after free-tip conversion: `0.01609091 kg`;
- `EI = 1.0154787051679222e-4 N m^2`; and
- `Cb = 1.5e-5 N m^2 s`.

This is explicitly **provisional**. The source is the final v5 two-holder fit,
not a one-attachment/free-tip experiment. The loader retains `EI` and `Cb`,
removes `GJ` and the second terminal-frame constraint, and adds one measured
marker mass at the newly free tip. In addition, fitted `Cb` lies on its old
upper search bound. These values are suitable for plant and controller
development, but the cable must be refitted with one-attachment data before a
reported physical result.

The drone assumptions in [`drone_config.json`](drone_config.json) are also
marked provisional. They currently represent a conservative custom
Crazyflie-Bolt/3-inch/2S starting point (`0.12 kg`, explicit inertia and wrench
limits). The applied action is an ideal net collective thrust plus body torque,
equivalent to assuming a fast low-level flight controller. Rotor allocation,
motor lag, wrench slew, and coupled thrust/torque feasibility are not yet
modeled, so a learned policy must not be transferred to hardware from this
plant. Weigh the complete aircraft and identify its inertia and actuator
dynamics first.

## Runtime

The tested software pairing is:

- Python `3.11`;
- Isaac Sim `5.1` (pip installation); and
- Isaac Lab `2.3.2` at commit `37ddf62`.

This pairing follows the official
[Isaac Lab v2.3.2 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.2)
and its
[versioned pip-installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html).

Use the launcher rather than the project's Python 3.12 virtual environment.
By default the batch file selects
`%USERPROFILE%\env_isaaclab\Scripts\python.exe`; set the
`ISAAC_WHIP_PYTHON` environment variable to override that location.

On this Windows/RTX 4080 machine, the Isaac Lab experience files are configured
for D3D12 (`vulkan = false`). `h5py` is pinned to `3.15.1`, whose HDF5 `1.14.6`
ABI matches the RTX sensor extension loaded by the interactive application.
`pip check` is clean. Preserve those settings if Isaac Sim or Isaac Lab is
reinstalled. The local checkout also changes `source/isaaclab/setup.py` from
the incompatible `starlette==0.49.1` declaration to `starlette>=0.40,<0.46`,
matching Isaac Sim 5.1's FastAPI pin. The warning that 0 of 57 joints have Isaac Lab actuators is
expected: the cable is passive and its two bending drives are authored directly
as PhysX D6-joint drives.

One-environment interactive viewer:

```powershell
.\run_isaac_whip.bat --num-envs 1 --mode hover
```

Sixty-four cloned environments in one viewer:

```powershell
.\run_isaac_whip.bat --num-envs 64 --mode excite
```

Useful runtime arguments are:

```text
--num-envs 1|64
--mode hover|excite|free
--duration-s N
--headless
--link-count N
--physics-dt 0.002
--control-hz 50
--render-hz 60
--output-json PATH
--export-usd PATH
```

With no path override, diagnostics are written to
`data/isaac_whip/last_run.json` and the generated plant to
`data/isaac_whip/drone_cable_20link.usda`.

## Implementation-validation workflow

First test all conversions without launching Kit:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_isaac_whip*.py" -v
```

Then exercise the coupled plant headlessly. The three modes isolate different
failure classes: `free` exercises gravity and ground contact while logging
finiteness and constraint gaps; `hover` checks that the drone supports the
complete suspended mass; and `excite` exercises and logs the driven cable
response. These are implementation smoke tests, not experimental validation of
the bending or damping response.

```powershell
.\run_isaac_whip.bat --headless --num-envs 1 --mode free    --duration-s 3 --output-json data\isaac_whip\free.json
.\run_isaac_whip.bat --headless --num-envs 1 --mode hover   --duration-s 3 --output-json data\isaac_whip\hover.json
.\run_isaac_whip.bat --headless --num-envs 1 --mode excite  --duration-s 3 --output-json data\isaac_whip\excite.json
```

Finally repeat a driven trial at 10, 20, and 30 links. Convergence of tip and
material-station trajectories is more informative than agreement at one
resolution:

```powershell
.\run_isaac_whip.bat --headless --mode excite --duration-s 3 --link-count 10 --output-json data\isaac_whip\grid10.json
.\run_isaac_whip.bat --headless --mode excite --duration-s 3 --link-count 20 --output-json data\isaac_whip\grid20.json
.\run_isaac_whip.bat --headless --mode excite --duration-s 3 --link-count 30 --output-json data\isaac_whip\grid30.json
.\run_isaac_whip.bat --headless --mode excite --duration-s 3 --link-count 40 --output-json data\isaac_whip\grid40.json
```

Compare the common 10 Hz trajectories rather than just their terminal states:

```powershell
.\.venv\Scripts\python.exe .\isaac_whip\compare_grid.py `
  data\isaac_whip\grid10.json data\isaac_whip\grid20.json `
  data\isaac_whip\grid30.json data\isaac_whip\grid40.json `
  --until-s 1 --output-json data\isaac_whip\grid_comparison_1s.json
```

### Validation snapshot (2026-08-24)

The current implementation passes:

- all 10 pure configuration/provenance/comparator tests;
- one-plant and 64-plant interactive D3D12 viewer runs;
- exact GUI/headless trajectory parity for the saved 0.5 s, 64-plant
  excitation trace (zero difference at every 10 Hz drone/tip sample);
- 0.5 s uncontrolled free fall/contact, 2 s hover, and 3 s 3-D excitation;
- topology checks (`21` bodies, `57` passive rotational DoFs at 20 links);
- authored and PhysX-resolved mass/COM/inertia checks;
- fixed-root/free-tip topology, locked translations, drive mode/targets,
  `EI`/`Cb` gains, and the deliberately absent twist drive/limit/friction; and
- reported joint gaps of about `1.4 micrometres` or less in the tested runs.

For the driven resolution study, the first-second tip/drone RMS differences are
`4.833/3.131 mm`, `0.963/0.361 mm`, and `3.054/0.754 mm` for 10->20,
20->30, and 30->40 links. Across the full three-second excitation, the same
tip RMS differences become `65.429 mm`, `31.617 mm`, and `97.614 mm`.
The error is non-monotonic, so these runs do **not** establish grid convergence
and the cause has not yet been isolated. Twenty links is only the current
computational baseline. Repeat reported control conclusions at 30 and 40
links; do not call the result resolution independent unless that sensitivity
study supports it.

Passing these checks establishes implementation consistency, not experimental
accuracy. The next scientific validation is a matched root-motion replay
against held-out OptiTrack trajectories, with error reported at common material
coordinates.

## Policy compatibility

Existing SAC checkpoints are intentionally not loaded. They were trained with
a point-mass acceleration action, prescribed attachment motion, and a different
state transition. This plant exposes a 6-DoF force/torque-driven aircraft and
passive cable reaction. Its action space, observations, dynamics, and therefore
learned value function have changed; the nominal policy must be retrained only
after the plant passes the validation workflow above.
