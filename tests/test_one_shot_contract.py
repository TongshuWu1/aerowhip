from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from learning.one_shot_env import evaluate_open_loop_batch
from learning.policy_action import (
    POLICY_ACTION_DIM,
    decode_policy_action,
    encode_physical_action,
)
from learning.policy_context import (
    POLICY_CONTEXT_DIM,
    build_policy_context,
    policy_context_tensor_metadata,
)
from planning.cem_task import load_variable_duration_task
from planning.command_parameterization import FullStateCommandTrajectory
from planning.rollout import clone_state_batch, hover_preroll
from planning.variable_duration import variable_duration_fullstate
from simulator.cable.dder import DderState
from simulator.parameters import SimulatorSettings
from simulator.production import (
    active_model_paths,
    build_production_simulator,
    load_active_model_manifest,
)
from simulator.state import SimulatorState
from simulator.uav.quaternion import quaternion_multiply_xyzw
from simulator.uav.state import UAVState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = (
    PROJECT_ROOT
    / "config"
    / "tasks"
    / "canonical_whip_variable_duration_tuned_reward_v1.json"
)
REFERENCE_RESULT = (
    PROJECT_ROOT
    / "data"
    / "planning_results"
    / "canonical_whip_variable_duration_tuned_reward_v1"
    / "2026-08-29T160423.856049Z"
)


def _task():
    return load_variable_duration_task(TASK_PATH)


def _settings() -> SimulatorSettings:
    active = load_active_model_manifest()
    return SimulatorSettings.load(active_model_paths(active)["configuration"])


def _z_rotation(angle_rad: float, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    rotation = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    quaternion = torch.tensor(
        [[0.0, 0.0, math.sin(0.5 * angle_rad), math.cos(0.5 * angle_rad)]],
        dtype=dtype,
        device=device,
    )
    return rotation, quaternion


def _transform_state(
    state: SimulatorState,
    *,
    translation_m: torch.Tensor,
    yaw_rotation_rad: float,
) -> SimulatorState:
    dtype, device = state.uav.position_m.dtype, state.uav.position_m.device
    rotation, quaternion = _z_rotation(yaw_rotation_rad, dtype=dtype, device=device)

    def rotate(vectors: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,b...j->b...i", rotation, vectors)

    uav = UAVState(
        rotate(state.uav.position_m) + translation_m,
        rotate(state.uav.velocity_m_s),
        quaternion_multiply_xyzw(quaternion, state.uav.orientation_xyzw),
        rotate(state.uav.angular_velocity_world_rad_s),
        state.uav.residual_history,
        None
        if state.uav.residual_acceleration_m_s2 is None
        else rotate(state.uav.residual_acceleration_m_s2),
    )
    cable = DderState(
        rotate(state.cable.positions_m) + translation_m[:, None],
        rotate(state.cable.velocities_m_s),
        state.cable.endpoint_orientations,
        state.cable.endpoint_twist_rad,
    )
    return SimulatorState(state.time_s, uav, cable)


@pytest.fixture(scope="module")
def cpu_context_case():
    task = _task()
    simulator = build_production_simulator(_settings(), device="cpu", dtype=torch.float64)
    yaw = 0.37
    position = torch.tensor([task.initial_uav_position_m], dtype=torch.float64)
    orientation = torch.tensor(
        [[0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]],
        dtype=torch.float64,
    )
    state = simulator.reset(
        position,
        torch.tensor([[0.2, -0.1, 0.05]], dtype=torch.float64),
        orientation,
        torch.tensor([[0.03, -0.02, 0.04]], dtype=torch.float64),
    )
    target = torch.tensor([task.target_position_m], dtype=torch.float64)
    direction = torch.tensor([task.desired_direction], dtype=torch.float64)
    context = build_policy_context(
        simulator,
        state,
        target_position_world_m=target,
        desired_direction_world=direction,
    )
    return task, simulator, state, target, direction, context


def test_policy_context_frame_schema_and_required_cable_velocity(cpu_context_case):
    _, _, _, _, _, context = cpu_context_case
    tensor = context.to_tensor()
    assert tensor.shape == (1, POLICY_CONTEXT_DIM)
    assert POLICY_CONTEXT_DIM == 83
    assert context.cable_positions_local_m.shape == (1, 10, 3)
    assert context.cable_velocities_local_m_s.shape == (1, 10, 3)
    assert torch.isfinite(tensor).all()
    assert torch.allclose(
        context.frame.points_to_world(torch.zeros(1, 3, dtype=tensor.dtype)),
        context.frame.root_position_world_m,
    )
    assert torch.linalg.vector_norm(context.target_direction_local, dim=-1).item() == pytest.approx(1.0)
    metadata = policy_context_tensor_metadata()
    assert metadata["dimension"] == 83
    assert metadata["fields"][-1]["stop"] == 83
    assert metadata["physics_feature_order"][-2:] == ["log_EI", "log_Cb"]
    assert torch.allclose(
        context.physics.learning_tensor()[:, -2:],
        torch.log(context.physics.raw_tensor()[:, -2:]),
    )


def test_policy_context_is_translation_and_global_yaw_invariant(cpu_context_case):
    _, simulator, state, target, direction, reference = cpu_context_case
    dtype, device = state.uav.position_m.dtype, state.uav.position_m.device
    transformations = (
        (torch.tensor([[1.7, -0.8, 0.35]], dtype=dtype, device=device), 0.0),
        (torch.zeros((1, 3), dtype=dtype, device=device), 1.13),
    )
    for translation, yaw in transformations:
        transformed_state = _transform_state(
            state, translation_m=translation, yaw_rotation_rad=yaw
        )
        rotation, _ = _z_rotation(yaw, dtype=dtype, device=device)
        transformed_target = torch.einsum("ij,bj->bi", rotation, target) + translation
        transformed_direction = torch.einsum("ij,bj->bi", rotation, direction)
        transformed = build_policy_context(
            simulator,
            transformed_state,
            target_position_world_m=transformed_target,
            desired_direction_world=transformed_direction,
        )
        assert torch.allclose(reference.to_tensor(), transformed.to_tensor(), atol=1.0e-10, rtol=1.0e-10)


def test_policy_action_decoder_is_49d_bounded_and_reference_roundtrips():
    task = _task()
    normalized = torch.linspace(-1.5, 1.5, POLICY_ACTION_DIM, dtype=torch.float64)
    decoded = decode_policy_action(normalized, task)
    assert decoded.normalized_action.shape == (1, 49)
    norms = torch.linalg.vector_norm(decoded.acceleration_knots_local_m_s2, dim=-1)
    assert float(norms.max()) <= 20.0 + 1.0e-10
    assert task.cem.duration_min_s <= float(decoded.duration_s[0]) <= task.cem.duration_max_initial_s

    knots = torch.tensor(
        json.loads((REFERENCE_RESULT / "best_acceleration_knots.json").read_text())["values"],
        dtype=torch.float64,
    )
    duration = json.loads((REFERENCE_RESULT / "optimized_duration.json").read_text())["duration_s"]
    encoded = encode_physical_action(knots, duration, task)
    roundtrip = decode_policy_action(encoded, task)
    assert torch.allclose(roundtrip.acceleration_knots_local_m_s2[0], knots)
    assert float(roundtrip.duration_s[0]) == pytest.approx(duration)


def test_decoded_action_generates_consistent_fullstate_active_prefix():
    task = _task()
    action = torch.zeros(POLICY_ACTION_DIM, dtype=torch.float64)
    decoded = decode_policy_action(action, task)
    command = variable_duration_fullstate(
        decoded.acceleration_knots_local_m_s2,
        decoded.duration_s,
        initial_position_m=torch.tensor(task.initial_uav_position_m, dtype=torch.float64),
        initial_velocity_m_s=torch.tensor(task.initial_uav_velocity_m_s, dtype=torch.float64),
        yaw_rad=torch.tensor([task.initial_yaw_rad], dtype=torch.float64),
        maximum_time_s=task.cem.duration_max_initial_s,
        dt_s=0.01,
    )
    stop = int(math.floor(float(decoded.duration_s[0]) / 0.01 + 1.0e-9))
    active = FullStateCommandTrajectory(
        command.times_s[: stop + 1],
        command.positions_m[: stop + 1],
        command.velocities_m_s[: stop + 1],
        command.accelerations_m_s2[: stop + 1],
        command.orientations_xyzw[: stop + 1],
        command.angular_velocities_body_rad_s[: stop + 1],
    )
    dt = active.times_s[1:] - active.times_s[:-1]
    expected_dv = 0.5 * (
        active.accelerations_m_s2[:-1] + active.accelerations_m_s2[1:]
    ) * dt[:, None, None]
    expected_dp = active.velocities_m_s[:-1] * dt[:, None, None] + (
        active.accelerations_m_s2[:-1] / 3.0
        + active.accelerations_m_s2[1:] / 6.0
    ) * dt[:, None, None].square()
    assert torch.max(
        torch.abs(active.velocities_m_s[1:] - active.velocities_m_s[:-1] - expected_dv)
    ) < 1.0e-12
    assert torch.max(
        torch.abs(active.positions_m[1:] - active.positions_m[:-1] - expected_dp)
    ) < 1.0e-12


@pytest.fixture(scope="module")
def cuda_reference_case():
    if not torch.cuda.is_available():
        pytest.skip("CUDA production gate")
    task = _task()
    simulator = build_production_simulator(_settings(), device="cuda", dtype=torch.float32)
    simulator.uav_model.set_fixed_evaluation_batch_size(
        task.cem.fixed_uav_evaluation_batch_size
    )
    initial = hover_preroll(simulator, task)
    context = build_policy_context(
        simulator,
        initial,
        target_position_world_m=torch.tensor(task.target_position_m, device="cuda"),
        desired_direction_world=torch.tensor(task.desired_direction, device="cuda"),
        command_initial_position_world_m=torch.tensor(
            task.initial_uav_position_m, device="cuda"
        ),
        command_initial_velocity_world_m_s=torch.tensor(
            task.initial_uav_velocity_m_s, device="cuda"
        ),
        command_yaw_world_rad=task.initial_yaw_rad,
    )
    knots = torch.tensor(
        json.loads((REFERENCE_RESULT / "best_acceleration_knots.json").read_text())["values"],
        dtype=torch.float32,
        device="cuda",
    )
    duration = json.loads((REFERENCE_RESULT / "optimized_duration.json").read_text())["duration_s"]
    knots_local = context.frame.vectors_to_local(knots.double().unsqueeze(0))[0]
    action = encode_physical_action(knots_local, duration, task)
    saved = json.loads((REFERENCE_RESULT / "final_metrics.json").read_text())
    return task, simulator, initial, context, action, saved


def test_successful_cem_reference_replays_through_one_shot_interface(cuda_reference_case):
    task, simulator, _, context, action, saved = cuda_reference_case
    result = evaluate_open_loop_batch(
        simulator, context, action, task, record_trajectory=True
    )
    row = result.row(0)
    assert row["task_success"] is True
    assert row["first_entry_marker"] == 10
    assert row["duration_s"] == pytest.approx(saved["optimized_duration_s"], abs=1.0e-6)
    # Milestone 6A replaced the discontinuous terminal hold with a smooth
    # settle. The historical decimals are therefore no longer authoritative,
    # but the saved maneuver must remain a finite scientific success through
    # the unchanged action codec.
    assert row["tip_min_distance_m"] <= task.success_radius_m
    assert row["directed_tip_speed_m_s"] >= task.minimum_directed_speed_m_s
    assert row["direction_error_deg"] <= task.maximum_direction_error_deg
    assert row["feasible"] is True
    assert result.trajectory is not None


def test_one_shot_small_batch_matches_batch_one(cuda_reference_case):
    task, simulator, initial, context_one, action_one, _ = cuda_reference_case
    reference = evaluate_open_loop_batch(simulator, context_one, action_one, task).row(0)
    batch = 4
    initial_batch = clone_state_batch(initial, batch)
    context_batch = build_policy_context(
        simulator,
        initial_batch,
        target_position_world_m=torch.tensor(task.target_position_m, device="cuda"),
        desired_direction_world=torch.tensor(task.desired_direction, device="cuda"),
        command_initial_position_world_m=torch.tensor(
            task.initial_uav_position_m, device="cuda"
        ),
        command_initial_velocity_world_m_s=torch.tensor(
            task.initial_uav_velocity_m_s, device="cuda"
        ),
        command_yaw_world_rad=task.initial_yaw_rad,
    )
    actions = action_one.expand(batch, -1).clone()
    repeated = evaluate_open_loop_batch(simulator, context_batch, actions, task)
    for index in range(batch):
        row = repeated.row(index)
        assert row["task_success"] == reference["task_success"]
        assert row["first_entry_marker"] == reference["first_entry_marker"]
        assert row["tip_min_distance_m"] == pytest.approx(
            reference["tip_min_distance_m"], abs=0.002
        )
        assert row["directed_tip_speed_m_s"] == pytest.approx(
            reference["directed_tip_speed_m_s"], abs=0.01
        )
        assert row["direction_error_deg"] == pytest.approx(
            reference["direction_error_deg"], abs=0.2
        )
        assert row["reward"] == pytest.approx(reference["reward"], abs=0.05)
