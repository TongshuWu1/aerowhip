"""Read-only tracing for catastrophic one-shot rollout diagnosis.

This module mirrors controller algebra only to expose intermediate values.  It
always advances state through the production simulator's private propagation
entry point and never changes the production equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from planning.command_parameterization import command_consistency_errors
from planning.variable_duration import variable_duration_fullstate
from simulator.simulator import CoupledSimulator
from simulator.uav.model import FullStateUAVModel
from simulator.uav.quaternion import (
    desired_orientation_error_body,
    desired_rotation_from_acceleration_and_yaw,
    normalize_quaternion_xyzw,
    quaternion_to_rotation_matrix_xyzw,
    yaw_from_quaternion_xyzw,
)
from simulator.uav.residual import residual_feature

from .one_shot_env import evaluate_open_loop_batch
from .policy_action import decode_policy_action
from .policy_context import PolicyContext


class ResidualDisabledDiagnostic(nn.Module):
    """Append the normal FIFO but return Delta_a=0 for a diagnostic replay."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.register_buffer("feature_mean", source.feature_mean.detach().clone())
        self.register_buffer("feature_std", source.feature_std.detach().clone())

    def forward(self, history_features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (history_features.shape[0], 3),
            dtype=history_features.dtype,
            device=history_features.device,
        )


def disable_residual_diagnostic_only(simulator: CoupledSimulator) -> None:
    model = simulator.uav_model
    if not isinstance(model, FullStateUAVModel) or model.residual_model is None:
        raise TypeError("A residual-enabled production UAV model is required.")
    model.residual_model = ResidualDisabledDiagnostic(model.residual_model).to(
        simulator.device, dtype=simulator.dtype
    )


def _host(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def build_action_command(
    simulator: CoupledSimulator,
    context: PolicyContext,
    normalized_action: torch.Tensor,
    task,
):
    action = torch.as_tensor(
        normalized_action, dtype=simulator.dtype, device=simulator.device
    )
    if action.ndim == 1:
        action = action.unsqueeze(0)
    decoded = decode_policy_action(action, task)
    knots_world = context.frame.vectors_to_world(
        decoded.acceleration_knots_local_m_s2
    )
    command = variable_duration_fullstate(
        knots_world,
        decoded.duration_s,
        initial_position_m=context.command_initial_position_world_m,
        initial_velocity_m_s=context.command_initial_velocity_world_m_s,
        yaw_rad=context.command_yaw_world_rad,
        maximum_time_s=1.20,
        dt_s=simulator.dt_s,
    )
    return decoded, command


def command_diagnostics(command, initial_position_m: torch.Tensor) -> dict[str, Any]:
    consistency = command_consistency_errors(command)
    acceleration_norm = torch.linalg.vector_norm(command.accelerations_m_s2, dim=-1)
    velocity_norm = torch.linalg.vector_norm(command.velocities_m_s, dim=-1)
    displacement = torch.linalg.vector_norm(
        command.positions_m - initial_position_m[None], dim=-1
    )
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (
            command.positions_m,
            command.velocities_m_s,
            command.accelerations_m_s2,
        )
    )
    return {
        **consistency,
        "maximum_command_acceleration_m_s2": float(acceleration_norm.max()),
        "maximum_command_velocity_m_s": float(velocity_norm.max()),
        "maximum_command_displacement_m": float(displacement.max()),
        "finite": finite,
        "dt_s": float(command.times_s[1] - command.times_s[0]),
        "sample_count": int(command.times_s.numel()),
    }


def trace_action(
    simulator: CoupledSimulator,
    context: PolicyContext,
    normalized_action: torch.Tensor,
    task,
    reward_config,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Any]:
    """Trace one exact action while advancing only through production physics."""

    if context.batch_size != 1:
        raise ValueError("Diagnostic tracing is batch-one only.")
    decoded, command = build_action_command(
        simulator, context, normalized_action, task
    )
    sequence = command.simulator_sequence()
    state = context.initial_state_world
    model = simulator.uav_model
    if not isinstance(model, FullStateUAVModel) or model.residual_model is None:
        raise TypeError("FullState residual-enabled model required for tracing.")
    residual_model = model.residual_model
    names = (
        "p_cmd",
        "v_cmd",
        "a_cmd",
        "uav_p",
        "uav_v",
        "uav_q_xyzw",
        "uav_omega_world",
        "e_p",
        "e_v",
        "Kp_e_p",
        "Kv_e_v",
        "ka_a_cmd",
        "a_ctrl",
        "f_des",
        "b3_des",
        "R_des",
        "e_R_body",
        "omega_dot_body",
        "a_phys_before_residual",
        "residual_input_90",
        "residual_input_normalized_90",
        "delta_a",
        "a_total",
        "root_position",
        "root_velocity",
        "c10_position",
        "c10_velocity",
    )
    arrays: dict[str, list[np.ndarray | float | bool]] = {name: [] for name in names}
    for scalar_name in (
        "time_s",
        "p_cmd_norm",
        "v_cmd_norm",
        "a_cmd_norm",
        "uav_speed",
        "uav_omega_norm",
        "e_p_norm",
        "e_v_norm",
        "a_ctrl_norm",
        "s_des",
        "residual_normalized_max_abs",
        "residual_normalized_l2",
        "delta_a_norm",
        "a_total_norm",
        "max_cable_node_speed",
        "finite",
    ):
        arrays[scalar_name] = []

    K_p, K_v, k_a, K_R, K_omega = simulator.parameters.uav.tensors(
        state.uav.position_m
    )
    dt = torch.as_tensor(
        simulator.dt_s, dtype=simulator.dtype, device=simulator.device
    )
    for index in range(sequence.step_count):
        item = sequence.command_at(index)
        p_cmd = item.position_m
        v_cmd = item.velocity_m_s
        a_cmd = item.acceleration_m_s2
        e_p = p_cmd - state.uav.position_m
        e_v = v_cmd - state.uav.velocity_m_s
        kp_ep, kv_ev, ka_a = K_p * e_p, K_v * e_v, k_a * a_cmd
        a_ctrl = kp_ep + kv_ev + ka_a
        rotation = quaternion_to_rotation_matrix_xyzw(
            normalize_quaternion_xyzw(state.uav.orientation_xyzw)
        )
        r_des = desired_rotation_from_acceleration_and_yaw(
            a_ctrl,
            yaw_from_quaternion_xyzw(item.orientation_xyzw),
            model.gravity_m_s2,
        )
        e_r = desired_orientation_error_body(r_des, rotation)
        omega_body = torch.matmul(
            rotation.transpose(-1, -2),
            state.uav.angular_velocity_world_rad_s.unsqueeze(-1),
        ).squeeze(-1)
        omega_dot_body = K_R * e_r + K_omega * (
            item.angular_velocity_body_rad_s - omega_body
        )
        gravity = torch.zeros_like(a_ctrl)
        gravity[..., 2] = model.gravity_m_s2
        f_des = a_ctrl + gravity
        s_des = torch.linalg.vector_norm(f_des, dim=-1)
        current_feature = residual_feature(
            p_cmd,
            v_cmd,
            a_cmd,
            state.uav.position_m,
            state.uav.velocity_m_s,
        )
        if state.uav.residual_history is None:
            raise RuntimeError("Trace requires a causal residual FIFO.")
        history = state.uav.residual_history.append(current_feature).features
        normalized_history = (
            history - residual_model.feature_mean.to(history)
        ) / residual_model.feature_std.to(history)
        boundary = simulator.root_boundary.evaluate(
            state.uav, simulator.cable_configuration.rest_lengths_m[0]
        )
        next_state = simulator._propagate(  # noqa: SLF001
            state, item, simulator.parameters, create_graph=False
        )
        a_total = (next_state.uav.velocity_m_s - state.uav.velocity_m_s) / dt
        delta_a = (
            torch.zeros_like(a_total)
            if next_state.uav.residual_acceleration_m_s2 is None
            else next_state.uav.residual_acceleration_m_s2
        )
        a_before = a_total - delta_a
        cable_speed = torch.linalg.vector_norm(
            state.cable.velocities_m_s, dim=-1
        )

        tensor_values = {
            "p_cmd": p_cmd,
            "v_cmd": v_cmd,
            "a_cmd": a_cmd,
            "uav_p": state.uav.position_m,
            "uav_v": state.uav.velocity_m_s,
            "uav_q_xyzw": state.uav.orientation_xyzw,
            "uav_omega_world": state.uav.angular_velocity_world_rad_s,
            "e_p": e_p,
            "e_v": e_v,
            "Kp_e_p": kp_ep,
            "Kv_e_v": kv_ev,
            "ka_a_cmd": ka_a,
            "a_ctrl": a_ctrl,
            "f_des": f_des,
            "b3_des": r_des[..., :, 2],
            "R_des": r_des,
            "e_R_body": e_r,
            "omega_dot_body": omega_dot_body,
            "a_phys_before_residual": a_before,
            "residual_input_90": history.reshape(1, 90),
            "residual_input_normalized_90": normalized_history.reshape(1, 90),
            "delta_a": delta_a,
            "a_total": a_total,
            "root_position": boundary.attachment_position_m,
            "root_velocity": boundary.attachment_velocity_analytic_m_s,
            "c10_position": state.cable.positions_m[:, 11],
            "c10_velocity": state.cable.velocities_m_s[:, 11],
        }
        for name, value in tensor_values.items():
            arrays[name].append(_host(value[0]))
        scalar_values = {
            "time_s": index * simulator.dt_s,
            "p_cmd_norm": torch.linalg.vector_norm(p_cmd),
            "v_cmd_norm": torch.linalg.vector_norm(v_cmd),
            "a_cmd_norm": torch.linalg.vector_norm(a_cmd),
            "uav_speed": torch.linalg.vector_norm(state.uav.velocity_m_s),
            "uav_omega_norm": torch.linalg.vector_norm(
                state.uav.angular_velocity_world_rad_s
            ),
            "e_p_norm": torch.linalg.vector_norm(e_p),
            "e_v_norm": torch.linalg.vector_norm(e_v),
            "a_ctrl_norm": torch.linalg.vector_norm(a_ctrl),
            "s_des": s_des,
            "residual_normalized_max_abs": normalized_history.abs().max(),
            "residual_normalized_l2": torch.linalg.vector_norm(normalized_history),
            "delta_a_norm": torch.linalg.vector_norm(delta_a),
            "a_total_norm": torch.linalg.vector_norm(a_total),
            "max_cable_node_speed": cable_speed.max(),
        }
        for name, value in scalar_values.items():
            arrays[name].append(float(torch.as_tensor(value).detach().cpu()))
        arrays["finite"].append(
            bool(
                torch.isfinite(next_state.uav.position_m).all()
                and torch.isfinite(next_state.uav.velocity_m_s).all()
                and torch.isfinite(next_state.cable.positions_m).all()
                and torch.isfinite(next_state.cable.velocities_m_s).all()
            )
        )
        state = next_state

    packed = {name: np.asarray(values) for name, values in arrays.items()}
    with torch.no_grad():
        result = evaluate_open_loop_batch(
            simulator,
            context,
            torch.as_tensor(normalized_action, device=simulator.device).reshape(1, 49),
            task,
            rl_reward_config=reward_config,
        )
    summary = summarize_trace(packed)
    summary["command"] = command_diagnostics(
        command, context.command_initial_position_world_m
    )
    summary["rollout_metrics"] = result.row(0)
    summary["duration_s"] = float(decoded.duration_s[0])
    return packed, summary, result


def summarize_trace(trace: dict[str, np.ndarray]) -> dict[str, Any]:
    max_abs = trace["residual_normalized_max_abs"]
    runaway_mask = (
        (trace["uav_speed"] > 10.0)
        | (trace["a_total_norm"] > 100.0)
        | (trace["a_ctrl_norm"] > 100.0)
        | (trace["uav_omega_norm"] > 100.0)
    )
    indices = np.flatnonzero(runaway_mask)
    first = None if indices.size == 0 else int(indices[0])
    crossings: dict[str, int | None] = {}
    for name, value, threshold in (
        ("residual_abs_z_gt_3", max_abs, 3.0),
        ("residual_abs_z_gt_5", max_abs, 5.0),
        ("residual_abs_z_gt_10", max_abs, 10.0),
        ("uav_speed_gt_3", trace["uav_speed"], 3.0),
        ("uav_speed_gt_10", trace["uav_speed"], 10.0),
        ("a_ctrl_gt_100", trace["a_ctrl_norm"], 100.0),
        ("delta_a_gt_10", trace["delta_a_norm"], 10.0),
        ("a_total_gt_100", trace["a_total_norm"], 100.0),
        ("omega_gt_100", trace["uav_omega_norm"], 100.0),
    ):
        found = np.flatnonzero(value > threshold)
        crossings[name] = None if found.size == 0 else int(found[0])
    window = []
    if first is not None:
        for index in range(max(0, first - 3), min(len(trace["time_s"]), first + 4)):
            window.append(
                {
                    "index": index,
                    "time_s": float(trace["time_s"][index]),
                    "a_cmd_norm": float(trace["a_cmd_norm"][index]),
                    "uav_speed": float(trace["uav_speed"][index]),
                    "e_p_norm": float(trace["e_p_norm"][index]),
                    "e_v_norm": float(trace["e_v_norm"][index]),
                    "a_ctrl_norm": float(trace["a_ctrl_norm"][index]),
                    "s_des": float(trace["s_des"][index]),
                    "residual_normalized_max_abs": float(max_abs[index]),
                    "delta_a_norm": float(trace["delta_a_norm"][index]),
                    "a_total_norm": float(trace["a_total_norm"][index]),
                }
            )
    return {
        "maximum_uav_speed_m_s": float(np.max(trace["uav_speed"])),
        "maximum_uav_omega_rad_s": float(np.max(trace["uav_omega_norm"])),
        "maximum_controller_acceleration_m_s2": float(np.max(trace["a_ctrl_norm"])),
        "maximum_desired_specific_force_m_s2": float(np.max(trace["s_des"])),
        "maximum_residual_normalized_abs": float(np.max(max_abs)),
        "maximum_residual_normalized_l2": float(
            np.max(trace["residual_normalized_l2"])
        ),
        "maximum_residual_acceleration_m_s2": float(np.max(trace["delta_a_norm"])),
        "maximum_total_acceleration_m_s2": float(np.max(trace["a_total_norm"])),
        "maximum_tip_speed_m_s": float(np.max(trace["max_cable_node_speed"])),
        "ood_fraction_any_abs_gt_3": float(np.mean(max_abs > 3.0)),
        "ood_fraction_any_abs_gt_5": float(np.mean(max_abs > 5.0)),
        "ood_fraction_any_abs_gt_10": float(np.mean(max_abs > 10.0)),
        "first_runaway_index": first,
        "first_runaway_time_s": None if first is None else float(trace["time_s"][first]),
        "threshold_crossings": crossings,
        "first_runaway_window": window,
        "all_steps_finite": bool(np.all(trace["finite"])),
    }
