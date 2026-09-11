# Braided nylon paracord: independent PhysX plant

The user confirmed braided nylon paracord on 9 September 2026. We preserve
153 g aircraft mass, 18 g cable-plus-markers mass and 0.9525 m cable length.
Material construction alone does not identify effective rope stiffness.
The current parameters are engineering estimates, not newly measured values.

## Implemented model

The active standalone runner uses `config/isaac_physx/rig_153g_paracord.json`.
It is an independent PhysX articulation, with no DDER or learned residual in
the plant. Thirty-two rigid segments approximate the cable centerline. Locked
translations enforce segment connectivity and an inextensible centerline;
three angular DOFs per joint permit spatial bending and twisting. The top
attachment is a free pivot. Drone and cable exchange forces inside PhysX.

Internal passive joint coefficients scale with segment length `h`:

- Bend stiffness: `EI/h`, with `EI = 1e-7 N m²`.
- Twist stiffness: `GJ/h`, with `GJ = 7e-8 N m²`.
- Damping: the corresponding stiffness times its relaxation time (0.2 s bend,
  0.3 s twist). These are effective joint parameters, not a fitted nylon law.

PhysX integrates these spring/damper drives implicitly. There is no duplicate
explicit elastic torque in this mode. The legacy config retains its original
bending law and angular damping for reproducibility.

The cord is represented by uniform cylindrical mass plus markers at their
actual longitudinal distances. Each link has the resulting COM and inertia,
including parallel-axis terms and spherical marker inertia. The runner checks
PhysX's mass, COM and inertia against those authored values. Estimated mass
distribution is 8 g bare cord and ten 1 g markers, with 6 mm marker radius.

Two-point quadrature includes air velocity from translation and rotation along
each segment. Crossflow drag, axial skin drag and marker sphere drag produce
both forces and moments about COM. Forces are explicitly applied at COM:
the installed tensor API's default is the link origin, which is different
when a marker offsets the mass distribution. Relative wind is configurable.
Capsules and marker spheres have collision geometry; adjacent links are
filtered, with other self-contact and ground/drone contact enabled. Estimated
friction is 0.35 and restitution is zero.

## What this does not claim

This is a discrete inextensible rod approximation. It does not resolve braid
strands, axial stretch, nonlinear load-dependent stiffness, creep, hysteresis,
or propeller downwash. The 3.5 mm cord diameter is still an estimate. The free
top pivot also approximates the attachment rather than identifying its clamp
stiffness. No hardware fidelity or sim-to-real improvement is demonstrated by
a successful synthetic flight. Sensor noise/latency and Motive/controller-log
compatibility are future integration work.

## Why not the native Isaac Lab cable asset yet?

The installed Isaac Lab is v2.3.2 with Isaac Sim 5.1. Newer experimental
[CableObject documentation](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/newton/using-cables.html)
describes Newton VBD cables; that asset is not a PhysX cable implementation.
The installed NVIDIA PhysX rope demo instead uses capsules and D6 joints,
the representation used here.

A separate native Newton 1.5.1 coupled drone/rod prototype was tested, including
stretch, shear, bending and twist. The coupled system showed unacceptable
hover drift even as iterations increased; the free drone alone balanced its
weight correctly. The prototype is rejected as a reference plant. Its source
and debug runs remain locally under ignored `runs/rejected_native_rod` and
`runs/isaac_newton`, outside the active launcher. The viewer prototype was not
validated. Newton was added to the dedicated Isaac Python environment only;
no Isaac or Warp package was upgraded. The working PhysX runner does not
import or require Newton.

Refining to 48 links also exposed instability in the enhanced PhysX prototype
at 600 and 1200 Hz. More segments are not automatically more accurate. The
32-link model is retained, with explicit timestep checks. Spatial convergence
of fast whipping remains unproven; do not silently increase resolution or
present this as a converged high-speed whip benchmark.

## Essential numerical checks on this computer

Windows 11, RTX 4080, Isaac Sim 5.1 / Lab v2.3.2. Physics uses CPU for this
single articulation; viewport rendering uses RTX. Twelve pure-math tests pass.
Authored mass, COM, inertia and implicit damping are checked against the actual
PhysX articulation at startup. Ten seconds of stationary hover retained the
specified height to float precision with no motor saturation (`paracord_hover09`).

The enhanced 600 Hz version completed 45 s / three-loop figure-eight and circle
checks (`paracord32_com06`, `paracord_circle07`): motion drone RMS was 0.796 cm
and 1.045 cm respectively. These are synthetic tracking errors, not errors
against a real cable. The 600 Hz trajectory showed appreciable cable-tip
timestep sensitivity, so the final default is 1,200 Hz.

Matched 25 s one-loop figure-eight runs at 1,200 and 2,400 Hz
(`paracord_dt08`, `paracord_dt10`) used identical 30 Hz commands and 100 Hz
sample times. Over t=10..25 s (motion and final hold), differences were:

| Quantity | RMS difference | Peak difference |
| --- | ---: | ---: |
| Drone tracked origin | 0.283 mm | 0.673 mm |
| Cable tip | 8.610 mm | 23.170 mm |

This is a useful numerical sensitivity bound for this maneuver, not proof of
complete spatial/temporal convergence or physical accuracy. Near a target,
centimetre-scale claims still need tighter numerical checks. The 1,200 Hz
flight had no motor saturation and a 0.716 cm motion tracking RMS. It took
54.1 s wall time for 25 s simulated, versus 106.3 s at 2,400 Hz; do not promise
real-time speed. Faster or larger maneuvers require their own stability check.

Final default source/config also completed a rendered 45 s, three-loop
figure-eight (`paracord_final11`): motion tracking RMS 0.766 cm, peak 1.664 cm,
zero motor saturation, maximum joint gap 1.901e-7 m. Wall time was 147.8 s.
The saved `scene.png` was inspected and the application closed successfully.
All tests are synthetic checks on the machine specified above, not real data.
