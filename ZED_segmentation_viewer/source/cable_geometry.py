"""Physical cable geometry derived from a piecewise-linear medial centerline."""

from __future__ import annotations

import numpy as np


def _normalized(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if np.isfinite(norm) and norm > 1e-10:
        return vector / norm
    if fallback is None:
        raise ValueError("Cannot normalize a zero-length vector")
    return np.asarray(fallback, dtype=np.float64)


def _centerline_frames(
    centerline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build stable parallel-transport frames along a nondegenerate centerline."""

    segments = np.diff(centerline, axis=0)
    segment_tangents = np.asarray(
        [_normalized(segment) for segment in segments],
        dtype=np.float64,
    )
    tangents = np.empty_like(centerline)
    tangents[0] = segment_tangents[0]
    tangents[-1] = segment_tangents[-1]
    for index in range(1, len(centerline) - 1):
        tangents[index] = _normalized(
            segment_tangents[index - 1] + segment_tangents[index],
            fallback=segment_tangents[index],
        )

    normals = np.empty_like(centerline)
    binormals = np.empty_like(centerline)
    initial_axis = np.eye(3, dtype=np.float64)[
        int(np.argmin(np.abs(tangents[0])))
    ]
    normals[0] = _normalized(np.cross(tangents[0], initial_axis))
    binormals[0] = _normalized(np.cross(tangents[0], normals[0]))

    for index in range(1, len(centerline)):
        previous_tangent = tangents[index - 1]
        tangent = tangents[index]
        rotation_axis = np.cross(previous_tangent, tangent)
        sine = float(np.linalg.norm(rotation_axis))
        cosine = float(np.clip(np.dot(previous_tangent, tangent), -1.0, 1.0))
        normal = normals[index - 1]
        if sine > 1e-8:
            axis = rotation_axis / sine
            normal = (
                normal * cosine
                + np.cross(axis, normal) * sine
                + axis * np.dot(axis, normal) * (1.0 - cosine)
            )
        elif cosine < 0.0:
            normal = -normal
        normal = normal - tangent * np.dot(tangent, normal)
        normals[index] = _normalized(normal, fallback=normals[index - 1])
        binormals[index] = _normalized(np.cross(tangent, normals[index]))
    return tangents, normals, binormals


def swept_capsule_mesh(
    centerline: np.ndarray,
    radius_m: float,
    radial_sides: int = 12,
    cap_rings: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed round tube with hemispherical caps.

    The centerline remains the estimation state. This mesh is its physical
    radius expansion and is intended for the selected estimate, visualization,
    collision queries, or downstream contact inference.
    """

    points = np.asarray(centerline, dtype=np.float64)
    radius = float(radius_m)
    sides = max(6, int(radial_sides))
    cap_count = max(1, int(cap_rings))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"centerline must have shape Nx3; got {points.shape}")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    if len(points) < 2 or not np.all(np.isfinite(points)):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint32),
        )

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-8))
    points = points[keep]
    if len(points) < 2:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint32),
        )

    tangents, normals, binormals = _centerline_frames(points)
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    circle = (
        np.cos(angles)[None, :, None] * normals[:, None, :]
        + np.sin(angles)[None, :, None] * binormals[:, None, :]
    )
    vertices = list((points[:, None, :] + radius * circle).reshape(-1, 3))
    triangles: list[tuple[int, int, int]] = []

    for ring in range(len(points) - 1):
        first = ring * sides
        second = (ring + 1) * sides
        for side in range(sides):
            next_side = (side + 1) % sides
            triangles.append((first + side, second + side, second + next_side))
            triangles.append((first + side, second + next_side, first + next_side))

    def add_cap(
        center: np.ndarray,
        outward: np.ndarray,
        normal: np.ndarray,
        binormal: np.ndarray,
        base_indices: list[int],
    ) -> None:
        previous_ring = base_indices
        for cap_ring in range(cap_count - 1, 0, -1):
            polar = 0.5 * np.pi * cap_ring / cap_count
            ring_center = center + outward * (radius * np.cos(polar))
            ring_radius = radius * np.sin(polar)
            ring_start = len(vertices)
            for angle in angles:
                vertices.append(
                    ring_center
                    + ring_radius
                    * (np.cos(angle) * normal + np.sin(angle) * binormal)
                )
            ring_indices = list(range(ring_start, ring_start + sides))
            for side in range(sides):
                next_side = (side + 1) % sides
                triangles.append(
                    (
                        previous_ring[side],
                        ring_indices[side],
                        ring_indices[next_side],
                    )
                )
                triangles.append(
                    (
                        previous_ring[side],
                        ring_indices[next_side],
                        previous_ring[next_side],
                    )
                )
            previous_ring = ring_indices
        tip_index = len(vertices)
        vertices.append(center + outward * radius)
        for side in range(sides):
            triangles.append(
                (
                    previous_ring[side],
                    tip_index,
                    previous_ring[(side + 1) % sides],
                )
            )

    add_cap(
        points[0],
        -tangents[0],
        normals[0],
        binormals[0],
        list(range(sides)),
    )
    last_start = (len(points) - 1) * sides
    add_cap(
        points[-1],
        tangents[-1],
        normals[-1],
        binormals[-1],
        list(range(last_start, last_start + sides)),
    )

    return (
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(triangles, dtype=np.uint32),
    )
