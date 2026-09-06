# Architecture

## Online strike contract

The planner starts from estimated drone-attachment and cable-node positions/velocities. PPO or SAC is queried along a private simulator rollout. It produces a finite sequence of world-frame forces and a cutoff before execution. The command sequence runs once; the actual cable and hit measurements do not change its commands or timing. PID hover recovery follows the frozen cutoff.

Policy commands are held at 20 Hz; physics advances at 100 Hz with twelve internal DDER substeps. The exported physics-rate force trace may repeat the same held command. The final hold can end before the next 50 ms boundary. A real controller must honor the agreed timing contract.

## Code boundaries

| Component | Main locations |
|---|---|
| DDER cable and constraints | `simulator/cable/` |
| Coupled point mass and force accounting | `simulator/point_mass.py` |
| Shared observations, rewards and strict contact scoring | `learning/point_force_env.py` |
| PPO / SAC implementations | `learning/simple_ppo.py`, `learning/simple_sac.py` |
| Nominal planning and independent execution | `learning/deployment_rollout.py`, `simulator/strike_plan.py` |
| Live simulation and hover recovery | `simulator/live_flight.py` |
| Preliminary recording processing and fitting | `experimental_data/` |
| Flight import, replay and physical candidates | `experimental_data/flight_trials.py`, `experimental_data/flight_adaptation.py` |
| Transport-neutral ROS/controller logging sink | `experimental_data/flight_recorder.py` |
| Local force correction without actor retraining | `learning/strike_adaptation.py` |
| Desktop application | `simulator/gui/main_window.py` |

The current five-page interface uses the shared PPO/SAC training workspace. The superseded training/replay pages and MPCC artifact reader have been removed. Small process-status and experiment-source helpers live in their own modules.

## Between-trial adaptation

Measured-attachment replay isolates cable prediction from aircraft tracking error. Independent force-driven replay checks the coupled result under the current force-response assumption. The first flight adaptation fitter changes cable drag within bounds while retaining the baseline geometry, masses, stiffness and internal damping. It uses whole-trial validation and does not apply a candidate automatically.

The sequence optimizer refines a recorded force sequence locally and rescores first contact and PID recovery. It does not retrain actor weights. Its exports are marked as simulation candidates: controller-response identification, independent uncertainty validation and actual next-launch state binding are still required.

## Physical and research limits

The aircraft model contains translation and cable reaction, not a verified attitude/motor/Lee-controller implementation. The current force bounds are provisional simulation bounds. Cable positions do not fully determine material twist. Synthetic mismatch recovery and simulation success do not demonstrate real-flight adaptation. The proposed NN residual is not part of the default flight update workflow.
