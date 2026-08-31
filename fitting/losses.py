"""Physical, masked, robust losses and hierarchical aggregation."""

from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.to(dtype=torch.bool, device=values.device)
    if not bool(valid.any()):
        raise ValueError("A loss family has no valid observations.")
    return torch.mean(values[valid])


def quaternion_geodesic_rad(
    predicted_xyzw: torch.Tensor, measured_xyzw: torch.Tensor
) -> torch.Tensor:
    predicted = predicted_xyzw / torch.clamp(
        torch.linalg.vector_norm(predicted_xyzw, dim=-1, keepdim=True), min=1.0e-12
    )
    measured = measured_xyzw / torch.clamp(
        torch.linalg.vector_norm(measured_xyzw, dim=-1, keepdim=True), min=1.0e-12
    )
    dot = torch.abs(torch.sum(predicted * measured, dim=-1))
    return 2.0 * torch.acos(torch.clamp(dot, max=1.0 - 1.0e-12))


def pseudo_huber_distance(distance_m: torch.Tensor, scale_m: float) -> torch.Tensor:
    scale = torch.as_tensor(scale_m, dtype=distance_m.dtype, device=distance_m.device)
    return scale.square() * (torch.sqrt(1.0 + (distance_m / scale).square()) - 1.0)


def physical_window_losses(
    *,
    predicted_uav_position: torch.Tensor,
    predicted_uav_orientation: torch.Tensor,
    predicted_markers: torch.Tensor,
    measured_uav_position: torch.Tensor,
    measured_uav_orientation: torch.Tensor,
    measured_markers: torch.Tensor,
    uav_valid: torch.Tensor,
    marker_valid: torch.Tensor,
    robust_scale_m: float,
) -> dict[str, torch.Tensor]:
    uav_distance2 = torch.sum((predicted_uav_position - measured_uav_position).square(), dim=-1)
    orientation = quaternion_geodesic_rad(predicted_uav_orientation, measured_uav_orientation)
    marker_distance = torch.linalg.vector_norm(predicted_markers - measured_markers, dim=-1)
    return {
        "uav_position_mse": masked_mean(uav_distance2, uav_valid),
        "uav_orientation_mse_rad2": masked_mean(orientation.square(), uav_valid),
        "cable_robust": masked_mean(pseudo_huber_distance(marker_distance, robust_scale_m), marker_valid),
        "cable_mse": masked_mean(marker_distance.square(), marker_valid),
        "tip_mse": masked_mean(marker_distance[..., -1].square(), marker_valid[..., -1]),
    }
