"""Small differentiable quaternion utilities for the effective UAV model.

Quaternions use ``[x, y, z, w]`` storage and represent the active rotation
from UAV body coordinates to world coordinates.  Angular velocity is expressed
in world coordinates, so ``q_dot = 0.5 * [omega, 0] (x) q``.
"""

from __future__ import annotations

import torch


def normalize_quaternion_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Return a safely normalized quaternion without leaving the tensor graph."""

    quaternion = torch.as_tensor(quaternion)
    if quaternion.shape[-1] != 4:
        raise ValueError("Quaternion tensors must end in four xyzw components.")
    minimum_norm = torch.finfo(quaternion.dtype).eps
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    normalized = quaternion / torch.clamp(norm, min=minimum_norm)
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1.0
    return torch.where(norm > minimum_norm, normalized, identity)


def quaternion_conjugate_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)


def quaternion_multiply_xyzw(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Hamilton product for xyzw tensors with matching leading dimensions."""

    left_vector, left_scalar = left[..., :3], left[..., 3:4]
    right_vector, right_scalar = right[..., :3], right[..., 3:4]
    vector = (
        left_scalar * right_vector
        + right_scalar * left_vector
        + torch.linalg.cross(left_vector, right_vector, dim=-1)
    )
    scalar = left_scalar * right_scalar - torch.sum(
        left_vector * right_vector, dim=-1, keepdim=True
    )
    return torch.cat((vector, scalar), dim=-1)


def quaternion_to_rotation_matrix_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the active body-to-world rotation matrix."""

    q = normalize_quaternion_xyzw(quaternion)
    x, y, z, w = q.unbind(dim=-1)
    two = q.new_tensor(2.0)
    row_0 = torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
        ),
        dim=-1,
    )
    row_1 = torch.stack(
        (
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
        ),
        dim=-1,
    )
    row_2 = torch.stack(
        (
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    )
    return torch.stack((row_0, row_1, row_2), dim=-2)


def yaw_from_quaternion_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Extract Z-up yaw from an active body-to-world xyzw quaternion."""

    q = normalize_quaternion_xyzw(quaternion)
    x, y, z, w = q.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def desired_rotation_from_acceleration_and_yaw(
    desired_acceleration_world_m_s2: torch.Tensor,
    yaw_rad: torch.Tensor,
    gravity_m_s2: float = 9.80665,
) -> torch.Tensor:
    """Construct Mellinger-style desired axes from acceleration and heading.

    Gravity is used only to obtain the desired thrust/body-z direction.  The
    supplied acceleration remains an effective world-frame translational
    acceleration and is not gravity-shifted elsewhere.
    """

    acceleration = torch.as_tensor(desired_acceleration_world_m_s2)
    if acceleration.shape[-1] != 3:
        raise ValueError("desired_acceleration_world_m_s2 must end in 3 values.")
    yaw = torch.as_tensor(yaw_rad, dtype=acceleration.dtype, device=acceleration.device)
    gravity = torch.zeros_like(acceleration)
    gravity[..., 2] = gravity_m_s2
    force = acceleration + gravity
    epsilon = torch.finfo(acceleration.dtype).eps
    force_norm = torch.linalg.vector_norm(force, dim=-1, keepdim=True)
    vertical = torch.zeros_like(force)
    vertical[..., 2] = 1.0
    b3 = torch.where(force_norm > epsilon, force / torch.clamp(force_norm, min=epsilon), vertical)
    heading = torch.stack(
        (torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), dim=-1
    )
    b2_raw = torch.linalg.cross(b3, heading, dim=-1)
    b2_norm = torch.linalg.vector_norm(b2_raw, dim=-1, keepdim=True)
    # With positive gravity this degeneracy is outside ordinary flight, but a
    # deterministic orthogonal fallback keeps the effective model total.
    fallback_heading = torch.zeros_like(heading)
    fallback_heading[..., 1] = 1.0
    fallback_b2 = torch.linalg.cross(b3, fallback_heading, dim=-1)
    fallback_b2 = fallback_b2 / torch.clamp(
        torch.linalg.vector_norm(fallback_b2, dim=-1, keepdim=True), min=epsilon
    )
    b2 = torch.where(
        b2_norm > epsilon,
        b2_raw / torch.clamp(b2_norm, min=epsilon),
        fallback_b2,
    )
    b1 = torch.linalg.cross(b2, b3, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def desired_orientation_error_body(
    desired_rotation: torch.Tensor,
    current_rotation: torch.Tensor,
) -> torch.Tensor:
    """Positive current-body-frame error for ``+K_R * e_R`` dynamics."""

    relative = torch.matmul(current_rotation.transpose(-1, -2), desired_rotation)
    skew = 0.5 * (relative - relative.transpose(-1, -2))
    return torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)


def shortest_orientation_error_world(
    commanded_xyzw: torch.Tensor,
    current_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Small-angle world-frame error ``2 * vec(q_cmd * inverse(q))``."""

    commanded = normalize_quaternion_xyzw(commanded_xyzw)
    current = normalize_quaternion_xyzw(current_xyzw)
    error = quaternion_multiply_xyzw(
        commanded,
        quaternion_conjugate_xyzw(current),
    )
    sign = torch.where(error[..., 3:4] < 0.0, -torch.ones_like(error[..., 3:4]),
                       torch.ones_like(error[..., 3:4]))
    return 2.0 * (error * sign)[..., :3]


def integrate_world_angular_velocity(
    quaternion_xyzw: torch.Tensor,
    angular_velocity_world_rad_s: torch.Tensor,
    dt_s: torch.Tensor | float,
) -> torch.Tensor:
    """Normalized Euler step for ``q_dot = 0.5 [omega, 0] (x) q``."""

    dt = torch.as_tensor(
        dt_s,
        dtype=quaternion_xyzw.dtype,
        device=quaternion_xyzw.device,
    )
    omega_quaternion = torch.cat(
        (
            angular_velocity_world_rad_s,
            torch.zeros_like(angular_velocity_world_rad_s[..., :1]),
        ),
        dim=-1,
    )
    derivative = 0.5 * quaternion_multiply_xyzw(
        omega_quaternion,
        quaternion_xyzw,
    )
    return normalize_quaternion_xyzw(quaternion_xyzw + dt * derivative)
