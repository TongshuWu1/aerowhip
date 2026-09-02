from __future__ import annotations

import os

import numpy as np
import torch

from run_ppo_simulation import (
    DEFAULT_CHECKPOINT,
    _terminal_pad,
    _truncate_replay_at_physics_step,
)


def test_ppo_replay_command_padding_preserves_the_last_command() -> None:
    source = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    padded = _terminal_pad(source)
    assert padded.shape == (4, 3)
    assert torch.equal(torch.from_numpy(padded[-1]), source[-1, 0])


def test_successful_replay_ends_at_the_hit_physics_step() -> None:
    arrays = {
        "time_s": np.arange(11, dtype=np.float32) * 0.01,
        "uav_position_m": np.zeros((11, 3), dtype=np.float32),
        "authorization": np.asarray("SIMULATION_ONLY"),
    }
    truncated = _truncate_replay_at_physics_step(arrays, 4)
    assert truncated["time_s"].shape == (5,)
    assert truncated["uav_position_m"].shape == (5, 3)
    assert float(truncated["time_s"][-1]) == float(arrays["time_s"][4])
    assert truncated["authorization"].shape == ()


def test_simulator_page_is_a_simple_frozen_ppo_runner() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from simulator.gui.replay_page import (
        CEM_REFERENCE_TASK_ID,
        SimulatorReplayPage,
        latest_ppo_policy_source,
    )
    from planning.results import latest_planning_result, load_replay_arrays

    application = QApplication.instance() or QApplication([])
    page = SimulatorReplayPage()
    assert DEFAULT_CHECKPOINT.is_file()
    assert page.run_ppo_button.text() == "RUN LATEST PPO"
    assert page.run_ppo_button.objectName() == "runPpoSimulationButton"
    assert page.canvas.backend_name == "MATPLOTLIB TEST FALLBACK"
    assert page.show_command.isChecked()
    assert page.show_paths.isChecked()
    assert page.show_vectors.isChecked()
    source = latest_ppo_policy_source()
    assert source is not None
    assert source[2].is_file()
    assert source[3] > 0
    assert "episodes" in page.policy_description.text()
    assert "PPO" in page.loaded_label.text() or page.loaded_label.text() == "No replay loaded"
    cem = latest_planning_result(CEM_REFERENCE_TASK_ID)
    assert cem is not None
    arrays = load_replay_arrays(cem)
    assert np.isclose(float(arrays["time_s"][-1]), 2.4)
    assert cem.success
    page.stop()
    page.close()
    application.processEvents()


def test_replay_command_orientation_uses_saved_yaw_when_quaternion_is_absent() -> None:
    from simulator.gui.replay_page import ProductionReplayView

    arrays = {"yaw_cmd_rad": np.asarray([np.pi / 2.0])}
    quaternion = ProductionReplayView._command_orientation_at(arrays, 0)
    np.testing.assert_allclose(
        quaternion,
        (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)),
        atol=1.0e-12,
    )
