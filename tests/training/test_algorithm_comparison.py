from pathlib import Path
import json
import torch

from learning import POINT_FORCE_OBSERVATION_DIM, SimpleSACAgent
from simulator.rollout import list_method_checkpoints, load_json, simulate_policy

ROOT = Path(__file__).resolve().parents[2]


def test_sac_actor_and_update_respect_symmetric_three_force_contract() -> None:
    agent = SimpleSACAgent(POINT_FORCE_OBSERVATION_DIM, device=torch.device("cpu"), hidden_dim=32)
    observation = torch.randn(16, POINT_FORCE_OBSERVATION_DIM)
    action = agent.act(observation)
    assert action.shape == (16, 3)
    assert torch.equal(action[:, 1], torch.zeros(16))
    batch = (observation, action, torch.randn(16, 1), torch.randn_like(observation), torch.zeros(16, 1))
    metrics = agent.update(batch)
    assert agent.gradient_updates == 1
    assert metrics.temperature > 0.0


def test_algorithm_files_do_not_duplicate_task_or_reward() -> None:
    sac = json.loads((ROOT / "config/sac.json").read_text(encoding="utf-8"))
    for config in (sac,):
        assert "reward" not in config
        assert "action" not in config
        assert "observation" not in config


def test_replay_library_lists_named_artifacts_for_each_method(tmp_path: Path) -> None:
    artifact = tmp_path / "runs" / "sac" / "run-a"
    checkpoint = artifact / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"artifact listing does not deserialize checkpoints")
    (artifact / "run.json").write_text(
        json.dumps({"display_name": "Paper SAC seed 651"}), encoding="utf-8"
    )
    (artifact / "status.json").write_text(
        json.dumps({"episodes": 12345}), encoding="utf-8"
    )
    entries = list_method_checkpoints(tmp_path, "sac")
    assert len(entries) == 1
    assert entries[0]["path"] == checkpoint.resolve()
    assert "Paper SAC seed 651" in entries[0]["label"]
    assert "12,345 ep" in entries[0]["label"]


def test_replay_library_lists_manual_ppo_snapshot_with_payload_episode(tmp_path: Path) -> None:
    artifact = tmp_path / "runs" / "ppo" / "run-a"
    checkpoint = artifact / "checkpoints" / "manual_000000064_123456789_snapshot.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "schema": "force_ppo_checkpoint_v1",
            "episodes": 64,
            "gradient_updates": 2,
        },
        checkpoint,
    )
    (artifact / "run.json").write_text(
        json.dumps({"display_name": "Saved PPO policy"}), encoding="utf-8"
    )
    (artifact / "status.json").write_text(
        json.dumps({"episodes": 999}), encoding="utf-8"
    )

    entries = list_method_checkpoints(tmp_path, "ppo")
    assert len(entries) == 1
    assert entries[0]["path"] == checkpoint.resolve()
    assert entries[0]["kind"] == "Manual snapshot"
    assert entries[0]["episodes"] == 64
    assert "64 ep" in entries[0]["label"]
    assert "999" not in entries[0]["label"]
