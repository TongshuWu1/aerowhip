# Current architecture

The active application generates offline aerial-whip trajectories using desired
tracked-origin position, velocity and acceleration (PVA) at 30 Hz. Bounded XYZ
jerk integrates into consistent next P/V/A knots; exact held packets and fitted
delay drive the loaded-drone pose response plus its bounded neural residual.
Rotated attachment geometry supplies the boundary for the discrete cable model.
The selected development M0 has scalar cable damping and no cable neural residual.

The selected MPPI-inspired optimizer searches complete 1.5 s trajectories using
multiple proposal families and interpolated jerk controls. A separate receding
implementation also exists. Target success is feasible modeled tip contact;
forward preparation, backward release and fold shape are preferences/diagnostics.
The final sequence is frozen before physical execution; onboard tracking feedback
remains separate. See the exact method and limitations in
[the paper handoff](PAPER_WRITING_HANDOFF.md).

The current cascade is an effective loaded-system predictor. Explicit cable reaction
is not added again to the empirical aircraft response. It is not an independently
identified motor/battery/rigid-body simulator.

| Component | Main source |
|---|---|
| Jerk and command geometry | `simulator/pva_commands.py`, `simulator/geometry.py` |
| Shared PVA environment, task and contact | `learning/pva_env.py`, `learning/pva_tick_graph.py` |
| MPPI and immutable jobs | `planning/mppi_trajectory.py`, `planning/mppi_receding.py`, `planning/pva_job.py` |
| PPO implementation | `learning/simple_ppo.py` |
| Loaded aircraft and residual | `simulator/drone_pose_response.py`, `research_pose.py`, `drone_pose_residual.py` |
| Cable and residual | `simulator/cable/dder.py`, `residual.py`, `simulator/research_physics.py` |
| Current fitting and adaptation | `experimental_data/preliminary_fit.py`, `whip_adaptation.py`, `whip_adaptation_fit.py` |
| Model comparison | `experimental_data/model_evaluation.py`, `simulator/gui/model_evolution_page.py` |
| Recovery and export | `deployment/pva_rehearsal.py`, `curved_recovery.py` |
| Measurements and exact forecasts | `experimental_data/adaptation_check.py`, `hover_calibration.py` |
| Desktop entry | `run_simulation.py`, `simulator/gui/pva_main_window.py` |

PPO and MPPI own independent settings and run directories. Legacy force/SAC/CEM
artifacts remain compatible with their historical code and are not reinterpreted
as jerk policies. New PVA PPO infrastructure does not establish a matched trained
baseline for current MPPI.

The UI has six pages: Models & fitting, Recordings, PPO, MPPI, Rehearsals,
Flight comparison. Saved runs can be inspected without starting another job.

See [direct PVA details](DIRECT_PVA_WORKFLOW.md), the [paper technical handoff](PAPER_WRITING_HANDOFF.md),
and [current status](../HANDOFF.md). Historical architecture is in the archive.
