# Point-force simulator

The simulator contains one translational point mass and one DDER cable. The
point mass and cable share node 0, so the attachment has no separate boundary,
attitude, or hidden coupling model.

| Force | Where it acts | Implementation |
|---|---|---|
| Commanded force `[Fx,Fy,Fz]` | Shared node 0 only | Direct world-frame external force |
| Point gravity | Point mass at node 0 | `m_point * g` |
| Cable gravity | Every cable vertex, including its endpoint mass at node 0 | `m_i * g` |
| Elastic bending | Cable vertices | Negative gradient of DDER bending energy |
| Bending damping | Cable vertices | Implicit Kelvin–Voigt curvature damping |
| Inextensibility reaction | Between adjacent cable vertices | Mass-weighted length and velocity constraints |

Elastic, damping, and constraint forces are internal to the point–cable system.
They transmit the effective cable reaction to the point through shared node 0.
`PointForceBreakdown` reports that reaction from the point momentum balance:

```text
F_cable_on_point = m_point * a_point - F_command - m_point * g
```

The model has no attitude, angular velocity, motors, thrust lag, aerodynamic
drag, or UAV attitude response. `live_flight.py` adds a position PID that
commands the same external world-frame force during hover and recovery. A
separate model rollout compiles PPO from the initial cable state into a finite
force sequence ending at its first predicted hit. That sequence runs once with
no motion or hit feedback, then PID resumes at the precomputed end time.
Switching controllers does not reset the dynamics. The live GUI
paces all physics steps against a monotonic clock and keeps rendering separate.
On real hardware, a separate controller must realize the requested force.
