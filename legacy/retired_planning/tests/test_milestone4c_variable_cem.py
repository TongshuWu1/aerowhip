from __future__ import annotations

from dataclasses import asdict, fields
import math

import pytest
import torch

from planning.cem import (
    CemIterationRecord,
    _regularize_covariance,
    _sample_population,
    feasibility_elite_order,
    initial_distribution,
    load_cem_checkpoint,
    save_cem_checkpoint,
)
from planning.cem_task import load_variable_duration_task
from planning.metrics import PopulationRolloutResult
from planning.rollout import hover_preroll
from planning.variable_duration import (
    VariableWhipAccumulator,
    project_decisions,
    run_variable_population_rollout,
    variable_duration_fullstate,
)
from simulator.parameters import SimulatorSettings
from simulator.production import active_model_paths, build_production_simulator, load_active_model_manifest


def _dummy_result(count: int) -> PopulationRolloutResult:
    payload = {}
    bool_names = {"feasible", "success", "finite"}
    vector_names = {"final_uav_position_m", "final_c10_position_m"}
    for item in fields(PopulationRolloutResult):
        if item.name in bool_names:
            payload[item.name] = torch.zeros(count, dtype=torch.bool)
        elif item.name == "first_entry_marker":
            payload[item.name] = torch.zeros(count, dtype=torch.int64)
        elif item.name in vector_names:
            payload[item.name] = torch.zeros(count, 3)
        else:
            payload[item.name] = torch.zeros(count)
    payload["finite"][:] = True
    payload["task_cost"][:] = 10.0
    payload["best_event_tip_distance_m"][:] = 1.0
    payload["best_event_direction_angle_deg"][:] = 90.0
    payload["first_entry_tip_distance_m"][:] = 1.0
    payload["first_entry_direction_angle_deg"][:] = 90.0
    return PopulationRolloutResult(**payload)


def test_variable_duration_command_active_prefix_is_exactly_consistent():
    task = load_variable_duration_task()
    knots = task.nominal_knots(device=torch.device("cpu"), dtype=torch.float64).unsqueeze(0)
    command = variable_duration_fullstate(
        knots,
        torch.tensor([0.60], dtype=torch.float64),
        initial_position_m=torch.tensor(task.initial_uav_position_m),
        initial_velocity_m_s=torch.zeros(3),
        yaw_rad=0.0,
        maximum_time_s=1.20,
        dt_s=0.01,
    )
    stop = 60
    acceleration = command.accelerations_m_s2[: stop + 1]
    velocity = command.velocities_m_s[: stop + 1]
    position = command.positions_m[: stop + 1]
    expected_dv = 0.5 * (acceleration[:-1] + acceleration[1:]) * 0.01
    expected_dp = velocity[:-1] * 0.01 + (
        acceleration[:-1] / 3.0 + acceleration[1:] / 6.0
    ) * 0.01**2
    assert torch.max(torch.abs(velocity[1:] - velocity[:-1] - expected_dv)) < 1e-12
    assert torch.max(torch.abs(position[1:] - position[:-1] - expected_dp)) < 1e-12
    settle_stop = stop + 30
    assert torch.max(torch.abs(command.accelerations_m_s2[settle_stop:])) < 1e-12
    assert torch.max(torch.abs(command.velocities_m_s[settle_stop:])) < 1e-12
    assert torch.max(
        torch.abs(command.positions_m[settle_stop:] - command.positions_m[settle_stop])
    ) < 1e-12


def test_duration_projection_uses_vector_acceleration_norm():
    task = load_variable_duration_task()
    values = torch.full((2, 49), 50.0)
    values[0, -1] = -1.0
    projected = project_decisions(values, task, 1.2)
    norms = torch.linalg.vector_norm(projected[:, :-1].reshape(2, 16, 3), dim=-1)
    assert float(norms.max()) <= 20.0 + 1e-5
    assert projected[0, -1] == pytest.approx(0.45)
    assert projected[1, -1] == pytest.approx(1.2)


def test_legacy_run_online_reward_profile_and_formula_are_frozen():
    task = load_variable_duration_task(
        "config/tasks/canonical_whip_variable_duration_legacy_reward_v1.json"
    )
    legacy = task.legacy_run_online_objective
    assert legacy is not None
    assert legacy.profile == "legacy_run_online_corrected_v5_full"
    assert legacy.position_weight == 40.0
    assert legacy.speed_weight == 25.0
    assert legacy.predictive_speed_weight == 10.0
    assert legacy.directed_speed_shaping_m_s == 4.0
    assert legacy.direction_weight == 20.0
    assert legacy.direction_shaping_error_deg == 30.0
    assert legacy.success_cost == 600.0
    assert legacy.safety_weight == 180.0

    accumulator = VariableWhipAccumulator(
        task,
        torch.tensor([0.8]),
        torch.zeros(1, task.cem.knot_count, 3),
    )
    uav = torch.tensor([[0.0, 0.0, 1.5]])
    cable = torch.full((1, 12, 3), 5.0)
    cable[:, 11] = torch.tensor([1.1, 0.0, 1.4])
    velocity = torch.zeros_like(cable)
    velocity[:, 11] = torch.tensor([2.0, 0.0, 0.0])
    accumulator.observe(
        0.2,
        uav_position_m=uav,
        uav_velocity_m_s=torch.zeros_like(uav),
        cable_positions_m=cable,
        cable_velocities_m_s=velocity,
    )
    distance_squared = 0.1**2
    proximity = math.exp(-distance_squared / (2.0 * 0.12**2))
    expected = (
        40.0 * distance_squared / (distance_squared + 0.18**2)
        + 25.0 * proximity * (4.0 - 2.0) ** 2
    )
    assert float(accumulator.best_event_cost[0]) == pytest.approx(
        expected, rel=1e-6
    )


def test_tuned_reward_changes_only_direction_shaping():
    baseline = load_variable_duration_task(
        "config/tasks/canonical_whip_variable_duration_legacy_reward_v1.json"
    )
    tuned = load_variable_duration_task(
        "config/tasks/canonical_whip_variable_duration_tuned_reward_v1.json"
    )
    assert baseline.legacy_run_online_objective is not None
    assert tuned.legacy_run_online_objective is not None
    baseline_values = asdict(baseline.legacy_run_online_objective)
    tuned_values = asdict(tuned.legacy_run_online_objective)
    assert tuned_values.pop("profile") == "legacy_run_online_strike_margin_tuned_v4"
    assert baseline_values.pop("profile") == "legacy_run_online_corrected_v5_full"
    assert tuned_values.pop("direction_weight") == pytest.approx(600.0)
    assert baseline_values.pop("direction_weight") == pytest.approx(20.0)
    assert tuned_values.pop("direction_shaping_error_deg") == pytest.approx(20.0)
    assert baseline_values.pop("direction_shaping_error_deg") == pytest.approx(30.0)
    assert tuned_values.pop("directed_speed_shaping_m_s") == pytest.approx(4.5)
    assert baseline_values.pop("directed_speed_shaping_m_s") == pytest.approx(4.0)
    assert tuned_values == baseline_values
    assert tuned.minimum_directed_speed_m_s == baseline.minimum_directed_speed_m_s
    assert tuned.maximum_direction_error_deg == baseline.maximum_direction_error_deg
    assert tuned.maximum_uav_displacement_m == baseline.maximum_uav_displacement_m
    assert tuned.maximum_uav_speed_m_s == baseline.maximum_uav_speed_m_s
    warm_knots = tuned.nominal_knots(device=torch.device("cpu"), dtype=torch.float64)
    assert warm_knots.shape == (16, 3)
    assert warm_knots.isfinite().all()


def test_post_duration_and_post_success_states_do_not_enter_metrics():
    task = load_variable_duration_task()
    duration = torch.tensor([0.45])
    knots = torch.zeros(1, 16, 3)
    accumulator = VariableWhipAccumulator(task, duration, knots)
    uav = torch.tensor([[0.0, 0.0, 1.5]])
    zero = torch.zeros_like(uav)
    cable = torch.full((1, 12, 3), 5.0)
    velocity = torch.zeros_like(cable)
    accumulator.observe(
        0.45,
        uav_position_m=uav,
        uav_velocity_m_s=zero,
        cable_positions_m=cable,
        cable_velocities_m_s=velocity,
    )
    accumulator.observe(
        0.46,
        uav_position_m=torch.tensor([[100.0, 0.0, 1.5]]),
        uav_velocity_m_s=torch.tensor([[100.0, 0.0, 0.0]]),
        cable_positions_m=cable * 100,
        cable_velocities_m_s=velocity,
    )
    assert float(accumulator.maximum_uav_displacement[0]) == pytest.approx(0.0)
    assert float(accumulator.maximum_uav_speed[0]) == pytest.approx(0.0)

    success = VariableWhipAccumulator(task, torch.tensor([1.0]), knots)
    cable[:, 11] = torch.tensor(task.target_position_m)
    velocity[:, 11] = torch.tensor([5.0, 0.0, 0.0])
    success.observe(
        0.2,
        uav_position_m=uav,
        uav_velocity_m_s=zero,
        cable_positions_m=cable,
        cable_velocities_m_s=velocity,
    )
    assert bool(success.success[0])
    success.observe(
        0.3,
        uav_position_m=torch.tensor([[100.0, 0.0, 1.5]]),
        uav_velocity_m_s=torch.tensor([[100.0, 0.0, 0.0]]),
        cable_positions_m=cable,
        cable_velocities_m_s=velocity,
    )
    assert float(success.maximum_uav_displacement[0]) == pytest.approx(0.0)


def test_cem_sampling_covariance_and_global_elite_are_finite():
    task = load_variable_duration_task()
    mean, covariance = initial_distribution(task)
    covariance = _regularize_covariance(covariance, task)
    generator = torch.Generator().manual_seed(42)
    values = _sample_population(mean, covariance, generator, 8192)
    assert values.shape == (8192, 49)
    assert torch.isfinite(values).all()

    result = _dummy_result(8192)
    # Last candidate is the only success and must rank ahead of a feasible
    # near-miss in chunk 0 and an infeasible close event in chunk 1.
    result.feasible[3] = True
    result.task_cost[3] = 0.2
    result.success[-1] = True
    result.feasible[-1] = True
    result.first_entry_tip_distance_m[-1] = 0.04
    result.first_entry_direction_angle_deg[-1] = 20.0
    result.task_cost[4097] = 0.001
    result.feasibility_violation[4097] = 0.5
    order = feasibility_elite_order(result)
    assert int(order[0]) == 8191
    assert int(order[1]) == 3


def test_tuned_success_elites_follow_tuned_reward_after_hard_feasibility():
    task = load_variable_duration_task(
        "config/tasks/canonical_whip_variable_duration_tuned_reward_v1.json"
    )
    result = _dummy_result(2)
    result.feasible[:] = True
    result.success[:] = True
    result.first_entry_tip_distance_m[:] = torch.tensor([0.001, 0.006])
    result.first_entry_direction_angle_deg[:] = torch.tensor([29.9, 18.0])
    result.task_cost[:] = torch.tensor([-596.0, -599.0])
    order = feasibility_elite_order(result, task)
    assert int(order[0]) == 1


def test_checkpoint_roundtrip_preserves_distribution_rng_and_best(tmp_path):
    task = load_variable_duration_task()
    mean, covariance = initial_distribution(task)
    generator = torch.Generator().manual_seed(47)
    _ = torch.randn(4, generator=generator)
    record = CemIterationRecord(
        seed=47,
        iteration=1,
        covariance_type="full",
        mean_duration_s=0.85,
        duration_std_s=0.18,
        minimum_sampled_duration_s=0.45,
        maximum_sampled_duration_s=1.2,
        feasible_candidate_count=1,
        successful_candidate_count=0,
        best_feasible_tip_error_m=0.1,
        best_successful_tip_error_m=None,
        best_directed_tip_speed_m_s=4.0,
        best_direction_error_deg=20.0,
        best_event_time_s=0.8,
        best_uav_displacement_m=0.4,
        best_uav_speed_m_s=2.0,
        elite_success_count=0,
        elite_feasible_non_success_count=1,
        elite_infeasible_count=409,
        best_ever_objective=10.0,
        iteration_runtime_s=1.0,
        cumulative_runtime_s=1.0,
    )
    _, metadata = save_cem_checkpoint(
        tmp_path,
        seed=47,
        iteration=1,
        mean=mean,
        covariance=covariance,
        generator=generator,
        global_best_decision=mean,
        global_best_metrics={"success": False},
        duration_max_s=1.2,
        task=task,
        record=record,
    )
    loaded = load_cem_checkpoint(metadata)
    assert torch.equal(loaded["mean"], mean)
    assert torch.equal(loaded["covariance"], covariance)
    assert torch.equal(loaded["rng_state"], generator.get_state())
    assert torch.equal(loaded["global_best_decision"], mean)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA production gate")
def test_variable_full_horizon_batch_replay_contract():
    task = load_variable_duration_task()
    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    simulator.uav_model.set_fixed_evaluation_batch_size(2048)
    initial = hover_preroll(simulator, task)
    knots = task.nominal_knots(device=simulator.device, dtype=simulator.dtype)
    rows = {}
    for batch in (1, 8, 2048):
        result = run_variable_population_rollout(
            simulator,
            initial,
            knots.unsqueeze(0).expand(batch, -1, -1).clone(),
            torch.full((batch,), 0.70, device=simulator.device),
            task,
            maximum_time_s=0.70,
        )
        rows[batch] = result.row(0)
    reference = rows[1]
    for batch in (8, 2048):
        assert max(abs(a - b) for a, b in zip(reference["final_uav_position_m"], rows[batch]["final_uav_position_m"])) <= 0.001
        assert max(abs(a - b) for a, b in zip(reference["final_c10_position_m"], rows[batch]["final_c10_position_m"])) <= 0.002
        assert abs(float(reference["minimum_tip_target_distance_m"]) - float(rows[batch]["minimum_tip_target_distance_m"])) <= 0.002
        assert reference["success"] == rows[batch]["success"]
        assert reference["first_entry_marker"] == rows[batch]["first_entry_marker"]
