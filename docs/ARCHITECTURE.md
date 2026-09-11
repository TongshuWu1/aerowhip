# AeroWhip architecture

The current research application plans a complete aerial-whip maneuver offline,
then exports frozen desired position, velocity and acceleration at 30 Hz. Onboard
UAV tracking feedback remains active in the separate flight program; the cable
task is open loop during execution.

```text
bounded XYZ jerk → desired PVA packets → loaded-UAV pose response
                                            ↓
                              rotated cable attachment → cable dynamics
                                            ↓
                          predicted motion → planning score and rehearsal

recorded command + native tracking → review → staged model update → new plan
```

## Model and optimization

The aircraft model predicts the effective response of the loaded system to held
PVA packets, including fitted delay, orientation response and a bounded neural
correction. The predicted tracked-origin pose and rotated offset determine the
cable attachment. Distributed cable dynamics then predict marker and tip motion.
Explicit cable reaction is not added again to the empirical aircraft model.
This is not an independently identified motor/battery/rigid-body simulator.

Full adaptation fits aircraft response, aircraft residual, cable physics and
cable residual in stages. Combined command-to-tip prediction is evaluated after
selection; it is not a jointly optimized fitting objective. Retained M0 has its
cable residual disabled, while full M1/M2 enable it.

The main study uses MPPI-inspired whole-maneuver trajectory search. Saved jobs
bind model assets, score/ranking, proposals, settings and source. The separate
PPO path generates PVA trajectories using the shared model/environment. Research
support for receding search and historical methods does not change how the
paper's frozen open-loop commands are executed.

## Source map

| Component | Source |
|---|---|
| PVA integration and geometry | `simulator/pva_commands.py`, `simulator/geometry.py` |
| Loaded-aircraft response | `simulator/drone_pose_response.py`, `simulator/research_pose.py`, `simulator/drone_pose_residual.py` |
| Cable physics and correction | `simulator/cable/dder.py`, `simulator/cable/residual.py`, `simulator/research_physics.py` |
| Shared PVA environment/contact | `learning/pva_env.py`, `learning/pva_tick_graph.py`, `learning/pva_success.py` |
| Whole-maneuver MPPI | `planning/mppi_trajectory.py`, `planning/pva_job.py`, `planning/whip_objective.py` |
| PPO learning and generation | `learning/simple_ppo.py`, `planning/ppo_whip.py` |
| Full update preparation/fitting | `experimental_data/whip_adaptation.py`, `experimental_data/whip_full_data.py`, `experimental_data/whip_full_fit.py` |
| Model and flight evaluation | `experimental_data/model_evaluation.py`, `experimental_data/flight_performance.py`, `experimental_data/system_comparison.py` |
| Native recording/original forecast review | `experimental_data/adaptation_check.py` |
| Rehearsal, recovery and CSV | `deployment/pva_rehearsal.py`, `deployment/curved_recovery.py` |
| Research desktop | `run_simulation.py`, `simulator/gui/pva_main_window.py` |

## Interfaces and saved evidence

The research UI has six pages: Models & fitting, Recordings, PPO, MPPI,
Rehearsals and Flight comparison. Inspection does not launch a job. PPO and MPPI
have independent setup and run libraries; model selection and flight selection
are separate operations.

The `deployment` branch wraps the existing numerical backend in the five-page
lab workflow. Its additional `deployment/lab_workflow.py`,
`deployment/lab_gui.py` and `tools/lab.py` manage study slots, reviewed imports,
updates, exports and reports. Runtime data stay in repository-relative folders;
CSV exports use `exports/<study>/<generation>/`. Those branch-specific modules
are not the entry points for this research checkout.

A saved command and its original preflight prediction remain paired. Postflight
predictions with common causal initialization are separate diagnostic artifacts.
Preserve raw coordinates, marker masks, timestamp provenance and source hashes.
Historical force/20 Hz checkpoints retain their original semantics.

See [the experiment protocol](PAPER_EXPERIMENT_PROTOCOL.md),
[frozen fitting method](FROZEN_SYSTEM_IDENTIFICATION.md) and
[geometry conventions](GEOMETRY_COORDINATE_CONVENTIONS.md) for details and limits.
