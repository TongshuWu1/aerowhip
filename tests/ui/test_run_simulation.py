from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy

import torch

from learning import POINT_FORCE_OBSERVATION_DIM, SimplePPOAgent
from run_simulation import simulate_constant_force
from simulator.rollout import simulate_policy


ROOT = Path(__file__).resolve().parents[2]


def test_default_runner_executes_point_force_hover() -> None:
    model = json.loads((ROOT / "config/model.json").read_text(encoding="utf-8"))
    task = json.loads((ROOT / "config/task.json").read_text(encoding="utf-8"))
    arrays, summary = simulate_constant_force(
        model, task, duration_s=0.05, device=torch.device("cpu")
    )
    assert arrays["cable_node_position_world_m"].shape == (6, 12, 3)
    assert arrays["commanded_force_world_n"].shape == (5, 3)
    assert summary["finite"] is True
    assert summary["steps"] == 5
    assert summary["maximum_segment_error_m"] < 1.0e-10
    # Drag operator splitting permits sub-micrometre drift at this short horizon.
    torch.testing.assert_close(torch.tensor(summary['final_root_position_world_m'],dtype=torch.float64),
                               torch.tensor(task['initial_root_position_m'],dtype=torch.float64),atol=1e-6,rtol=0.)


def test_policy_runner_loads_checkpoint_and_records_physics_frames(tmp_path) -> None:
    model = json.loads((ROOT / "config/model.json").read_text(encoding="utf-8"))
    task = json.loads((ROOT / "config/task.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "config/ppo.json").read_text(encoding="utf-8"))
    task = deepcopy(task)
    task["episode_duration_s"] = 0.1
    agent = SimplePPOAgent(
        POINT_FORCE_OBSERVATION_DIM,
        3,
        device=torch.device("cpu"),
        hidden_dim=int(config["ppo"]["hidden_dim"]),
    )
    checkpoint = tmp_path / "latest.pt"
    torch.save(
        {
            **agent.checkpoint(),
            "observation_dim": POINT_FORCE_OBSERVATION_DIM,
            "action_dim": 3,
            "episodes": 12,
        },
        checkpoint,
    )
    arrays, summary = simulate_policy(
        model,
        task,
        config,
        checkpoint_path=checkpoint,
        device=torch.device("cpu"),
    )
    assert arrays["cable_node_position_world_m"].shape == (11, 12, 3)
    assert arrays["commanded_force_world_n"].shape == (10, 3)
    assert arrays["normalized_policy_action"].shape == (10, 3)
    assert summary["checkpoint_episodes"] == 12
    assert summary["source"] == "latest_ppo_policy"
    assert summary["finite"] is True


def test_policy_rollout_ends_on_success_without_recording_the_remaining_horizon(tmp_path):
    from simulator.point_mass import ForceControlledPointCable
    model, task, config = [json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))
                           for name in ("model.json", "task.json", "ppo.json")]
    cable = ForceControlledPointCable.from_mapping(model)
    state = cable.hanging_state(torch.tensor(task["initial_root_position_m"], dtype=torch.float64))
    task["target_position_m"] = state.positions_m[0, -1].tolist()
    task["success"]["minimum_directed_tip_speed_m_s"] = 0.
    task["success"]["maximum_tip_velocity_to_desired_direction_error_deg"] = 180.
    agent = SimplePPOAgent(POINT_FORCE_OBSERVATION_DIM, 3, device=torch.device("cpu"),
                           hidden_dim=int(config["ppo"]["hidden_dim"]))
    with torch.no_grad():
        for parameter in agent.policy.mean_network.parameters():
            parameter.zero_()
    checkpoint = tmp_path / "one_hit.pt"
    torch.save(agent.checkpoint(), checkpoint)
    progress = []
    arrays, summary = simulate_policy(model, task, config, checkpoint_path=checkpoint,
                                     progress_callback=lambda done, total: progress.append((done, total)))
    assert summary["success"]
    assert summary["hit_time_s"] == summary["duration_s"] == .01
    assert summary["maximum_duration_s"] == task["episode_duration_s"]
    assert len(arrays["time_s"]) == 2
    assert len(arrays["commanded_force_world_n"]) == 1
    assert progress[-1] == (1, 1)
