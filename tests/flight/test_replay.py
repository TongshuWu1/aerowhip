from __future__ import annotations

import numpy as np

from simulator.replay import list_saved_replays, load_replay, save_replay


def test_saved_replay_is_listed_and_round_trips(tmp_path) -> None:
    arrays = {
        "time_s": np.asarray([0.0, 0.01]),
        "cable_node_position_world_m": np.zeros((2, 12, 3)),
        "cable_node_velocity_world_m_s": np.zeros((2, 12, 3)),
        "commanded_force_world_n": np.zeros((1, 3)),
        "effective_cable_reaction_on_point_world_n": np.zeros((1, 3)),
    }
    directory = save_replay(
        tmp_path,
        arrays,
        {"source": "latest_ppo_policy", "checkpoint_episodes": 42},
        name="Professor demo",
    )
    entries = list_saved_replays(tmp_path)
    assert len(entries) == 1
    assert entries[0]["display_name"] == "Professor demo"
    restored, summary = load_replay(directory)
    assert set(arrays).issubset(restored)
    assert np.array_equal(restored["time_s"], arrays["time_s"])
    assert summary["checkpoint_episodes"] == 42
    assert summary["saved_replay_name"] == "Professor demo"
