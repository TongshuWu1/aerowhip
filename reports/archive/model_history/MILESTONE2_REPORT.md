# Milestone 2 Completion Report

## Outcome

Milestone 2 is complete. The repository now has one differentiable simulator
that maps a continuous Crazyflie-style FullState command through an effective
closed-loop UAV response, converts the simulated UAV pose into a physical
cable boundary, and propagates the existing DDER cable interior.

The late physics correction was incorporated: FullState UAV simulation uses a
centerline-clamped root, while the previous position-only pivot remains a
separate exact legacy mode.

No fitting, MPPI, residual model, noise, drag, downwash, firmware emulation, or
bidirectional cable-force coupling was added.

## Final relevant tree

```text
config/
  default.json
simulator/
  parameters.py
  simulator.py
  state.py
  cable/
    config.py
    cuda_fixed_pcg.py
    dder.py
    initialization.py
  coupling/
    attachment.py
    root_boundary.py
  uav/
    model.py
    quaternion.py
    state.py
  gui/
    app.py
    main_window.py
    viewer_3d.py
tests/
  _common.py
  test_dder_gradients.py
  test_dder_regression.py
  test_dder_rollout.py
  test_dder_static.py
  test_fullstate_uav.py
  test_simulator_api.py
run_simulator.py
```

## Command and state representation

`FullStateCommand` contains batched SI tensors for:

```text
position          [B,3] m
velocity          [B,3] m/s
acceleration      [B,3] m/s^2
orientation       [B,4] xyzw
angular velocity  [B,3] rad/s, world frame
```

`FullStateCommandSequence` is time-major. The Milestone 1 `UAVCommand` and
`UAVCommandSequence` retain their prescribed-position/velocity meaning.

`UAVState` now contains position, velocity, body-to-world orientation, and
world-frame angular velocity. Legacy construction with position and velocity
alone supplies identity orientation and zero angular velocity.

## UAV equations

The effective translational model is:

```text
p_dot = v
v_dot = k_a a_cmd + K_p (p_cmd - p) + K_v (v_cmd - v)
```

There is deliberately no UAV gravity term because the effective response
represents the vehicle plus its hover-compensating onboard controller.

Quaternion storage is `[qx,qy,qz,qw]`. It is an active body-to-world rotation.
Angular velocity is world-frame, so:

```text
q_dot = 0.5 [omega,0] (x) q
q_error = q_cmd (x) inverse(q)
e_R = 2 vector(shortest_sign(q_error))
omega_dot = K_R e_R + K_omega (omega_cmd - omega)
```

Translation and angular velocity use semi-implicit Euler. Quaternion position
uses normalized Euler with the newly updated angular velocity. Normalization
is differentiable and has a safe identity fallback for a degenerate input.

## Parameters and injection

`SimulatorParameters` contains:

```text
UAVResponseParameters: K_p, K_v, k_a, K_R, K_omega
CableParameters:       EI, Cb
```

Every `CoupledSimulator.step/rollout` passes its selected UAV parameter object
into every UAV propagation step and passes its selected `EI,Cb` tensors into
every DDER step. Candidate parameters are not permanently hidden in the model
object and are not converted to Python values in the differentiable path.

The configured debug-only defaults are:

```text
K_p     = 16.0 s^-2
K_v     = 8.0 s^-1
k_a     = 1.0
K_R     = 25.0 s^-2
K_omega = 10.0 s^-1
```

They are stable placeholders, not calibrated Crazyflie gains.

## Attachment kinematics

Configuration stores fixed, currently unmeasured body-frame geometry:

```text
d_attach_body = [0,0,0] m
t_attach_body = [0,0,-1]
```

For simulated UAV pose `(p,q)`:

```text
R = R(q)
r_offset_world = R d_attach_body
p_root = p + r_offset_world
t_root = normalize(R t_attach_body)
v_root_analytic = v + omega x r_offset_world
```

Analytic attachment position and velocity are exposed in the trajectory. They
are diagnostics/future fitting outputs, not an alternative DDER boundary-rate
path.

## Pivot and clamped DDER boundaries

### Legacy pivot

The legacy mode prescribes only node 0:

```text
r_0,k+1 = p_root,k+1
```

Its first-edge tangent remains dynamic/free. The DDER derives root velocity
from its established position difference. Its numerical result is unchanged.

### Centerline clamp

FullState UAV mode prescribes nodes 0 and 1:

```text
r_0,k+1 = p_root,k+1
r_1,k+1 = p_root,k+1 + l_0 t_root,k+1
```

The first DDER material length is derived from the measured connector-to-c1
interval and the configured two-edge subdivision:

```text
l_0 = marker_interval_lengths_m[0] / 2 = 0.030 m
```

The 30-mm edge is the current mesh's discrete approximation to a clamped
tangent; the measured physical connector-to-c1 material interval is 60 mm.
The previously documented 39-mm edge was superseded by the 2026-08-27 measured
geometry update. Node 1's derived bare-cable lumped mass is now
`0.0002226935313 kg`. In the clamped discretization that node is prescribed and
its corresponding reaction is not fed back to the UAV because coupling remains
one-way.

The DDER derives both prescribed-node velocities using its unchanged discrete
position-difference convention. The analytic rigid-body velocity of the first
edge endpoint is available within the boundary representation for verification.

## What changed inside DDER

Only boundary plumbing was generalized:

- the existing one-start-node pivot remains supported;
- a two-start-node/free-tip pin-count mode was added;
- boundary extraction/replacement handles two consecutive start nodes;
- the damping solve uses nodes 2–20 as free coordinates in clamped mode;
- constraint projection assigns zero inverse mass to nodes 0 and 1;
- the explicit bending stability bound is evaluated for the clamped free slice.

The following interior physics was not redesigned:

- DER curvature and bending energy;
- bending force differentiation;
- Kelvin–Voigt curvature damping equation;
- inextensible length and velocity projection equations;
- mass and marker mapping;
- gravity;
- cable node/edge count.

The measured sites remain `[0,2,4,...,20]`. Node 1 remains latent.

No `GJ`, twist degree of freedom, material roll, or torsional damping was
introduced. The new condition is a centerline position+tangent clamp only.

The existing fused CUDA damping/projection kernels remain specific to the
one-node pivot topology. Clamped CUDA rollouts currently use the generic
differentiable PyTorch DDER path; no new custom CUDA kernel was introduced.

## Reset

Pivot reset builds the existing vertical hanging cable from the single root.

Clamped reset first constructs exact nodes 0 and 1 from UAV pose and attachment
geometry. Nodes 2–20 then continue vertically downward with their exact
material lengths. For a tilted tangent this is a feasible curved initial state,
not a static-equilibrium solve. Analytic rigid-body velocities initialize the
two prescribed nodes.

## Trajectory output

`SimulatorTrajectory` now exposes:

```text
UAV position
UAV velocity
UAV quaternion
UAV world angular velocity
attachment position
analytic attachment velocity
prescribed root tangent (clamped mode)
all 21 cable positions
all 21 cable velocities
```

UAV center and cable attachment are never conflated.

## GUI

Launch remains:

```powershell
.\.venv\Scripts\python.exe run_simulator.py
```

The production GUI provides:

- Prescribed Root — Pivot;
- FullState UAV — Clamped;
- Hover, X Sinusoid, Aggressive X Reversal, and Small Attitude Motion;
- editable debug-only UAV gains and `EI,Cb`;
- oriented UAV body axes;
- UAV center and attachment markers;
- the first-edge tangent drawn directly from actual DDER nodes 0–1;
- all nodes, measured marker sites, and the free tip.

GUI construction, FullState single-step, and pivot reset were exercised. No
testing, fitting, gradient, or regression controls appear in the GUI.

## Verification results

The final focused suite contains 17 tests and passed completely:

```text
17 passed
```

### Pivot regression

The 60-frame dynamic pivot comparison against the frozen offline DDER gave:

```text
maximum position difference = 0.0 m
maximum velocity difference = 0.0 m/s
```

The pre/post Milestone 1 15-frame trajectory hashes remain:

```text
positions  da7863e9d017829032596963500d6bf5ebda932aff3b0fe716a733c5418401d9
velocities 40803efb699169712d0ef5f2b34192f356a47933fbd88661f4b0e48bca61e16a
```

### Clamped geometry and stability

For a 100-frame coupled CPU rollout:

```text
finite state                               true
maximum first-edge length error            1.180e-16 m
maximum first-edge tangent error            9.437e-16
maximum all-edge length error               1.943e-16 m
maximum quaternion-norm error               2.220e-16
```

The reset geometry test verified attachment position and analytic root velocity
to `1e-14` absolute tolerance. A 90-degree yaw case produced the expected
rotated attachment and cross-product velocity.

### Gradients

For the informative combined UAV+cable loss (`0.3799689542`):

```text
dL/dK_p      +5.537265434e-3
dL/dK_v      +6.613735454e-2
dL/dk_a      -3.691033991e-1
dL/dK_R      +1.253840093e-3
dL/dK_omega  +8.645934644e-3
dL/dEI       +1.157880349e-1
dL/dCb       -1.232357360e+1
```

All seven gradients were finite and nonzero. The cable-dependent path through
UAV orientation, body-to-world tangent, and the clamped DDER boundary was also
tested with a cable-only attitude-motion loss:

```text
cable-only loss  1.627805598e-6
dL/dK_R          1.374702559e-8
dL/dK_omega      1.146723065e-7
```

Thus the rotational gains influence cable output through the physical clamped
boundary, not only through UAV-only terms in the combined verification loss.

### CPU/CUDA and batching

CPU rollout, deterministic replay, batched shapes, and gradients passed.

On `NVIDIA GeForce RTX 4080`, a 4-trajectory, 50-frame clamped rollout was
finite and gave the same maximum first-edge errors as CPU:

```text
length error   1.180e-16 m
tangent error  9.437e-16
```

The legacy CUDA batched pivot test also passed.

## Cleanup

No one-off production modes, gradient hooks, testing panels, or temporary GUI
diagnostics remain. The lightweight tests were retained because they protect
the two scientifically distinct boundary models and the seven-parameter
differentiable path. No temporary standalone verification scripts were added.

## Current limitations

- Attachment offset and exit direction still require hardware measurement.
- UAV gains still require identification.
- The clamped first edge is a resolution-dependent discrete tangent
  approximation.
- Cable forces do not react back onto UAV motion.
- Full torsional clamping is not modeled.
- Aerodynamic and delay effects are absent.

Before fitting, the hardware campaign must freeze the Crazyflie firmware,
onboard controller, controller settings, estimator settings, and exact
FullState command path.
