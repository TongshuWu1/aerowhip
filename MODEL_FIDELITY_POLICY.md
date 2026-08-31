# Simulator Model-Fidelity Policy

The production aerial-cable simulator is a model-fidelity-first research
instrument for offline identification, planning, and authoritative replay. It
is not a hardware controller and is not an online receding-horizon controller.

## Frozen production model

The selected model is `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI` and uses:

- attitude-coupled FullState UAV physics;
- the frozen causal 100-ms UAV residual and its fixed normalization/FIFO;
- 12 DDER nodes and 11 edges;
- `node0` attachment root and `node1` rigid clamp support;
- `node2...node11` corresponding to measured cable markers `c1...c10`;
- remeasured 0.9525-m connector-to-c10 geometry and distributed masses;
- three DDER substeps per 10-ms frame;
- four position-constraint projections per substep;
- PCG32 bending damping with frozen `EI` and `Cb`;
- fused projection and CUDA float32 production execution.

`config/active_model.json` selects the model explicitly. Runtime code must
never select a fit by newest timestamp.

## Fidelity boundary

CUDA parallelism, batching, preallocation, reduced Python/synchronization
overhead, fused equivalent algebra, and equivalent memory layouts are
implementation optimizations only after trajectory parity is verified.

Changing node count, substeps, projection iterations, timestep, geometry,
masses, attachment, UAV gains, residual weights/history, `EI`, `Cb`, or damping
backend is a scientific model change. Such a change requires a new versioned
comparison and freeze.

## Command and state contract

```text
INPUT
  FullStateCommand:
    p_cmd, v_cmd, a_cmd, q_cmd, omega_cmd

LATENT STATE
  UAV:
    position, velocity, orientation, angular velocity,
    causal residual FIFO
  cable:
    all 12 DDER positions and velocities

OUTPUT / METRICS
  UAV pose and velocity
  c1...c10 positions and velocities
  complete scientific-gate trajectory metrics
```

For the production CEM action, the active command is followed by the fixed
0.30-s analytic settle and stationary hold. Authoritative evaluation continues
to 2.40 s through the unchanged simulator.

## Protected data and hardware

`fig8vertical_002` remains sealed. No production claim may silently use it for
fitting, checkpoint selection, planning, or policy evaluation. Simulation
success is not hardware readiness, and this repository does not authorize
arming or transmitting commands to a Crazyflie.
