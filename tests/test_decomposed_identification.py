from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fitting.config import load_fit_configuration
from fitting.dataset import load_dataset
from fitting.decomposed import (
    METHODOLOGY,
    causal_residual_suffixes,
    differentiable_batched_uav_training_step,
    differentiable_uav_training_step,
    physical_episode_views,
    prepare_measured_boundary_episodes,
    prepare_uav_episodes,
    prepare_uav_training_batch,
    task_horizon_views,
)
from simulator.parameters import SimulatorSettings
from simulator.simulator import CoupledSimulator
from simulator.uav.model import FullStateUAVModel


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _settings_and_config():
    return (
        SimulatorSettings.load(ROOT / "config" / "default.json"),
        load_fit_configuration(),
    )


def test_causal_residual_coverage_uses_episode_rule_without_moving_parents() -> None:
    dataset = load_dataset()
    config = load_fit_configuration()
    views, audit = causal_residual_suffixes(dataset.training, config)
    parents = {
        view.parent.episode_id: view.parent
        for view in physical_episode_views(dataset.training, config)
    }
    assert METHODOLOGY == "decomposed_full_episode_id_v1"
    assert audit["physical_episode_count"] == 21
    assert audit["residual_eligible_suffix_count"] == 16
    assert math.isclose(float(audit["physical_duration_s"]), 188.04, abs_tol=0.02)
    assert math.isclose(
        float(audit["residual_eligible_duration_s"]), 186.36, abs_tol=0.02
    )
    assert math.isclose(float(audit["excluded_duration_s"]), 1.68, abs_tol=0.02)
    for view in views:
        parent = parents[view.parent.episode_id]
        assert view.start_index == parent.residual_eligible_start_index
        assert view.end_index == parent.end_index
        assert view.start_index >= parent.start_index
    assert all(view.view_id.startswith(view.parent.episode_id) for view in views)


def test_validation_suffixes_are_views_and_protected_take_is_not_selected() -> None:
    dataset = load_dataset()
    config = load_fit_configuration()
    views, audit = causal_residual_suffixes(dataset.validation, config)
    assert audit["physical_episode_count"] == 3
    assert audit["residual_eligible_suffix_count"] == 3
    assert math.isclose(float(audit["physical_duration_s"]), 71.91, abs_tol=0.02)
    assert math.isclose(
        float(audit["residual_eligible_duration_s"]), 71.61, abs_tol=0.02
    )
    assert {view.take.take_id for view in views} == {"fig8_003", "osc_003"}
    assert all(view.take.take_id != "fig8vertical_002" for view in views)


def test_task_horizon_views_preserve_start_and_only_truncate_end() -> None:
    dataset = load_dataset()
    config = load_fit_configuration()
    physical = physical_episode_views(dataset.validation, config)
    prefixes = task_horizon_views(physical, dt_s=0.01, maximum_duration_s=1.0)
    assert len(prefixes) == len(physical)
    for original, prefix in zip(physical, prefixes, strict=True):
        assert prefix.parent is original.parent
        assert prefix.start_index == original.start_index
        assert prefix.end_index == min(original.end_index, original.start_index + 100)
        assert prefix.step_count <= 100
        assert prefix.kind == "task_horizon_prefix"

    residual, _ = causal_residual_suffixes(dataset.validation, config)
    residual_prefixes = task_horizon_views(
        residual, dt_s=0.01, maximum_duration_s=1.0
    )
    assert all(
        prefix.start_index == original.start_index
        and prefix.kind == "causal_residual_task_horizon_prefix"
        for original, prefix in zip(residual, residual_prefixes, strict=True)
    )


def test_measured_boundary_uses_the_production_rigid_clamp() -> None:
    settings, config = _settings_and_config()
    simulator = CoupledSimulator(
        settings.cable_configuration,
        settings.parameters,
        dt_s=settings.dt_s,
        device="cpu",
        dtype=torch.float64,
        uav_model=FullStateUAVModel(),
        attachment_offset_body_m=settings.attachment_offset_body_m,
        attachment_tangent_body=settings.attachment_tangent_body,
    )
    take = next(item for item in load_dataset().training if item.take_id == "osc_001")
    view = physical_episode_views((take,), config)[0]
    prepared = prepare_measured_boundary_episodes((view,), config, settings, simulator)[0]
    first = prepared.measured_boundary_positions_m[0].cpu().numpy()
    p = take.arrays["uav_position_m"][view.start_index]
    q = take.arrays["uav_orientation_xyzw"][view.start_index]
    x, y, z, w = q / np.linalg.norm(q)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    root = p + rotation @ np.asarray(settings.attachment_offset_body_m)
    tangent = rotation @ np.asarray(settings.attachment_tangent_body)
    tangent /= np.linalg.norm(tangent)
    expected = np.stack(
        (root, root + settings.cable_configuration.rest_lengths_m[0] * tangent)
    )
    np.testing.assert_allclose(first, expected, atol=1.0e-12, rtol=0.0)


def test_decomposed_uav_path_is_attitude_coupled_production_model() -> None:
    settings, _config = _settings_and_config()
    model = FullStateUAVModel()
    simulator = CoupledSimulator(
        settings.cable_configuration,
        settings.parameters,
        dt_s=settings.dt_s,
        device="cpu",
        dtype=torch.float64,
        uav_model=model,
        attachment_offset_body_m=settings.attachment_offset_body_m,
        attachment_tangent_body=settings.attachment_tangent_body,
    )
    assert simulator.uav_model is model
    assert simulator.uav_model.model_version == "aerial_cable_attitude_coupled_v1"


def test_parallel_episode_execution_preserves_objective_and_gradient() -> None:
    settings, config = _settings_and_config()
    model = FullStateUAVModel()
    simulator = CoupledSimulator(
        settings.cable_configuration,
        settings.parameters,
        dt_s=settings.dt_s,
        device="cpu",
        dtype=torch.float64,
        uav_model=model,
        attachment_offset_body_m=settings.attachment_offset_body_m,
        attachment_tangent_body=settings.attachment_tangent_body,
    )
    views = sorted(
        physical_episode_views(load_dataset().training, config),
        key=lambda view: view.step_count,
    )[:5]
    episodes = prepare_uav_episodes(views, config, simulator)
    initial = torch.tensor(
        [9.8846079409, 3.7325878814, 0.3462507933, 62.7167487741, 3.701],
        dtype=torch.float64,
    )
    serial_values = initial.clone().requires_grad_(True)
    serial, _ = differentiable_uav_training_step(
        episodes, simulator, settings, config, serial_values
    )
    parallel_values = initial.clone().requires_grad_(True)
    parallel, _ = differentiable_batched_uav_training_step(
        prepare_uav_training_batch(episodes),
        simulator,
        settings,
        config,
        parallel_values,
    )
    assert parallel == pytest.approx(serial, abs=1.0e-15, rel=0.0)
    torch.testing.assert_close(
        parallel_values.grad, serial_values.grad, atol=1.0e-15, rtol=0.0
    )
