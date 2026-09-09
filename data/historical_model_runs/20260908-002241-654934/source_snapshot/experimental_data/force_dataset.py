"""Convert synchronized motion capture into the point-force model contract.

The exported point force is an inverse-dynamics estimate of the real vehicle's
external force, not a measured motor signal. Whole-system linear momentum makes
the internal cable forces cancel.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from simulator.cable import CableConfiguration

from .io import atomic_json, canonical_json_hash, deterministic_npz, sha256_file
from .quality import quaternion_to_rotation_matrix_xyzw


FORCE_DATASET_SCHEMA = "point_force_cable_dataset_v1"
FORCE_TAKE_SCHEMA = "point_force_cable_take_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "config" / "model.json"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed_takes"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "force_takes"


@dataclass(frozen=True, slots=True)
class DifferentiationSettings:
    """Centered local-polynomial derivative settings."""

    window_samples: int = 11
    polynomial_order: int = 3

    def __post_init__(self) -> None:
        if self.window_samples < 3 or self.window_samples % 2 != 1:
            raise ValueError("window_samples must be an odd integer of at least 3.")
        if not 2 <= self.polynomial_order < self.window_samples:
            raise ValueError(
                "polynomial_order must support acceleration and be smaller than the window."
            )


def _derivative_coefficients(
    *, dt_s: float, settings: DifferentiationSettings, derivative_order: int
) -> np.ndarray:
    if derivative_order < 0 or derivative_order > settings.polynomial_order:
        raise ValueError("Unsupported derivative order.")
    half = settings.window_samples // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64) * float(dt_s)
    design = np.stack(
        [offsets**power for power in range(settings.polynomial_order + 1)], axis=1
    )
    return math.factorial(derivative_order) * np.linalg.pinv(design)[derivative_order]


def local_polynomial_derivative(
    values: np.ndarray,
    sample_valid: np.ndarray,
    *,
    dt_s: float,
    derivative_order: int,
    settings: DifferentiationSettings = DifferentiationSettings(),
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate vector samples and invalidate every window touching bad data.

    ``values`` has shape ``(time, ..., xyz)`` and ``sample_valid`` has the same
    leading dimensions without ``xyz``.
    """

    signal = np.asarray(values, dtype=np.float64)
    valid = np.asarray(sample_valid, dtype=bool)
    if signal.ndim < 2 or signal.shape[-1] != 3:
        raise ValueError("values must have shape (time, ..., 3).")
    if valid.shape != signal.shape[:-1]:
        raise ValueError("sample_valid must match values without the xyz dimension.")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive.")

    valid = valid & np.isfinite(signal).all(axis=-1)
    output = np.full(signal.shape, np.nan, dtype=np.float64)
    derivative_valid = np.zeros(valid.shape, dtype=bool)
    window = settings.window_samples
    if signal.shape[0] < window:
        return output, derivative_valid

    coefficients = _derivative_coefficients(
        dt_s=dt_s, settings=settings, derivative_order=derivative_order
    )
    half = window // 2
    count = signal.shape[0] - window + 1
    candidate = np.zeros((count,) + signal.shape[1:], dtype=np.float64)
    finite_signal = np.where(np.isfinite(signal), signal, 0.0)
    valid_count = np.zeros((count,) + valid.shape[1:], dtype=np.int16)
    for offset, coefficient in enumerate(coefficients):
        candidate += coefficient * finite_signal[offset : offset + count]
        valid_count += valid[offset : offset + count]
    center = slice(half, signal.shape[0] - half)
    center_valid = valid_count == window
    output[center] = np.where(center_valid[..., None], candidate, np.nan)
    derivative_valid[center] = center_valid
    return output, derivative_valid


def _normalized_rotations(quaternion_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=1)
    valid = np.isfinite(quaternion).all(axis=1) & (norm > 1.0e-12)
    normalized = np.full_like(quaternion, np.nan)
    normalized[valid] = quaternion[valid] / norm[valid, None]
    return quaternion_to_rotation_matrix_xyzw(normalized), valid


def reconstruct_dder_nodes(
    root_position_m: np.ndarray,
    root_valid: np.ndarray,
    marker_positions_m: np.ndarray,
    marker_valid: np.ndarray,
    cable: CableConfiguration,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the attachment and ten measured markers onto the DDER node topology."""

    root = np.asarray(root_position_m, dtype=np.float64)
    markers = np.asarray(marker_positions_m, dtype=np.float64)
    root_ok = np.asarray(root_valid, dtype=bool)
    marker_ok = np.asarray(marker_valid, dtype=bool)
    if root.shape != (len(root), 3):
        raise ValueError("root_position_m must have shape (time, 3).")
    expected_markers = (len(root), cable.moving_marker_count, 3)
    if markers.shape != expected_markers or marker_ok.shape != expected_markers[:-1]:
        raise ValueError("Marker arrays do not match the cable configuration.")
    if root_ok.shape != (len(root),):
        raise ValueError("root_valid must have shape (time,).")

    sites = np.concatenate((root[:, None, :], markers), axis=1)
    site_valid = np.concatenate((root_ok[:, None], marker_ok), axis=1)
    nodes = np.full((len(root), cable.node_count, 3), np.nan, dtype=np.float64)
    node_valid = np.zeros((len(root), cable.node_count), dtype=bool)
    nodes[:, 0] = root
    node_valid[:, 0] = root_ok & np.isfinite(root).all(axis=1)

    node_index = 0
    for interval, subdivision in enumerate(cable.interval_subdivisions):
        start = sites[:, interval]
        end = sites[:, interval + 1]
        for step in range(1, subdivision + 1):
            node_index += 1
            if step == subdivision:
                nodes[:, node_index] = end
                node_valid[:, node_index] = site_valid[:, interval + 1]
            else:
                alpha = step / subdivision
                nodes[:, node_index] = (1.0 - alpha) * start + alpha * end
                node_valid[:, node_index] = (
                    site_valid[:, interval] & site_valid[:, interval + 1]
                )
    node_valid &= np.isfinite(nodes).all(axis=2)
    nodes[~node_valid] = np.nan
    return nodes, node_valid


def estimate_point_force_world_n(
    root_acceleration_m_s2: np.ndarray,
    cable_node_acceleration_m_s2: np.ndarray,
    force_valid: np.ndarray,
    *,
    point_mass_kg: float,
    cable_vertex_masses_kg: np.ndarray,
    gravity_world_m_s2: np.ndarray,
) -> np.ndarray:
    """Estimate the model's external point force from momentum balance."""

    root_acceleration = np.asarray(root_acceleration_m_s2, dtype=np.float64)
    node_acceleration = np.asarray(cable_node_acceleration_m_s2, dtype=np.float64)
    masses = np.asarray(cable_vertex_masses_kg, dtype=np.float64)
    gravity = np.asarray(gravity_world_m_s2, dtype=np.float64)
    if node_acceleration.shape != (len(root_acceleration), len(masses), 3):
        raise ValueError("Cable acceleration and vertex masses have incompatible shapes.")
    if root_acceleration.shape != (len(root_acceleration), 3) or gravity.shape != (3,):
        raise ValueError("Root acceleration and gravity must contain xyz vectors.")
    total_mass = float(point_mass_kg) + float(np.sum(masses))
    momentum_rate = float(point_mass_kg) * root_acceleration + np.einsum(
        "n,tnj->tj", masses, node_acceleration
    )
    force = momentum_rate - total_mass * gravity[None, :]
    valid = np.asarray(force_valid, dtype=bool)
    if valid.shape != (len(root_acceleration),):
        raise ValueError("force_valid must have shape (time,).")
    force[~valid] = np.nan
    return force


def _node_mapping(cable: CableConfiguration) -> list[dict[str, object]]:
    mapping: list[dict[str, object]] = [
        {"node": 0, "source": "optitrack_attachment", "measured": False}
    ]
    node_index = 0
    for interval, subdivision in enumerate(cable.interval_subdivisions):
        start = "attachment" if interval == 0 else f"c{interval}"
        end = f"c{interval + 1}"
        for step in range(1, subdivision + 1):
            node_index += 1
            if step == subdivision:
                mapping.append(
                    {"node": node_index, "source": end, "measured": True}
                )
            else:
                mapping.append(
                    {
                        "node": node_index,
                        "source": f"linear_interpolation_{start}_to_{end}",
                        "fraction": step / subdivision,
                        "measured": False,
                    }
                )
    return mapping


def convert_processed_arrays(
    source: dict[str, np.ndarray],
    model: dict[str, object],
    *,
    settings: DifferentiationSettings = DifferentiationSettings(),
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Convert one already synchronized take without changing its timeline."""

    time_s = np.asarray(source["time_s"], dtype=np.float64)
    if len(time_s) < settings.window_samples or not np.all(np.diff(time_s) > 0.0):
        raise ValueError("Take timeline is too short or is not strictly increasing.")
    dt_s = float(np.median(np.diff(time_s)))
    if not np.allclose(np.diff(time_s), dt_s, rtol=1.0e-5, atol=1.0e-9):
        raise ValueError("Force conversion currently requires a uniform timeline.")

    cable = CableConfiguration.from_mapping(model["cable"])
    rotation, quaternion_valid = _normalized_rotations(source["uav_orientation_xyzw"])
    rigid_body_position = np.asarray(source["uav_position_m"], dtype=np.float64)
    source_quality_valid = np.asarray(source["auto_frame_valid"], dtype=bool)
    root_valid = (
        np.asarray(source["uav_valid"], dtype=bool)
        & quaternion_valid
        & source_quality_valid
        & np.isfinite(rigid_body_position).all(axis=1)
    )
    attachment_offset = np.asarray(
        model["recorded_data"]["optitrack_to_attachment_offset_body_m"],
        dtype=np.float64,
    )
    root_position = rigid_body_position + np.einsum(
        "tij,j->ti", rotation, attachment_offset
    )
    root_position[~root_valid] = np.nan

    marker_valid = (
        np.asarray(source["cable_marker_valid"], dtype=bool)
        & source_quality_valid[:, None]
    )
    nodes, node_valid = reconstruct_dder_nodes(
        root_position,
        root_valid,
        source["cable_marker_positions_m"],
        marker_valid,
        cable,
    )
    root_velocity, root_velocity_valid = local_polynomial_derivative(
        root_position,
        root_valid,
        dt_s=dt_s,
        derivative_order=1,
        settings=settings,
    )
    root_acceleration, root_acceleration_valid = local_polynomial_derivative(
        root_position,
        root_valid,
        dt_s=dt_s,
        derivative_order=2,
        settings=settings,
    )
    node_velocity, node_velocity_valid = local_polynomial_derivative(
        nodes,
        node_valid,
        dt_s=dt_s,
        derivative_order=1,
        settings=settings,
    )
    node_acceleration, node_acceleration_valid = local_polynomial_derivative(
        nodes,
        node_valid,
        dt_s=dt_s,
        derivative_order=2,
        settings=settings,
    )
    state_valid = root_velocity_valid & np.all(node_velocity_valid, axis=1)
    force_valid = root_acceleration_valid & np.all(node_acceleration_valid, axis=1)
    cable_masses = np.asarray(cable.vertex_masses_kg, dtype=np.float64)
    gravity = np.asarray(cable.gravity_m_s2, dtype=np.float64)
    point_mass = float(model["point_mass"]["mass_kg"])
    estimated_force = estimate_point_force_world_n(
        root_acceleration,
        node_acceleration,
        force_valid,
        point_mass_kg=point_mass,
        cable_vertex_masses_kg=cable_masses,
        gravity_world_m_s2=gravity,
    )
    body_z = rotation[:, :, 2]
    body_z[~quaternion_valid] = np.nan

    force_norm = np.linalg.norm(estimated_force, axis=1)
    diagnostic_valid = force_valid & (force_norm > 0.2)
    alignment_deg = np.full(len(time_s), np.nan, dtype=np.float64)
    if np.any(diagnostic_valid):
        cosine = np.einsum(
            "ti,ti->t",
            estimated_force[diagnostic_valid] / force_norm[diagnostic_valid, None],
            body_z[diagnostic_valid],
        )
        alignment_deg[diagnostic_valid] = np.degrees(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )

    arrays = {
        "time_s": time_s,
        "source_motive_frame": np.asarray(source["motive_frame"], dtype=np.int64),
        "source_motive_time_s": np.asarray(
            source["motive_source_time_s"], dtype=np.float64
        ),
        "source_quality_valid": source_quality_valid,
        "root_position_world_m": root_position,
        "root_velocity_world_m_s": root_velocity,
        "root_acceleration_world_m_s2": root_acceleration,
        "root_valid": root_valid,
        "cable_node_position_world_m": nodes,
        "cable_node_velocity_world_m_s": node_velocity,
        "cable_node_acceleration_world_m_s2": node_acceleration,
        "cable_node_valid": node_valid,
        "state_valid": state_valid,
        "estimated_point_force_world_n": estimated_force,
        "force_estimate_valid": force_valid,
        "measured_body_z_world": body_z,
        "force_body_z_alignment_deg": alignment_deg,
    }
    expected_hover_force_n = -(point_mass + cable.total_dynamic_mass_kg) * gravity[2]
    valid_force_norm = force_norm[force_valid]
    valid_alignment = alignment_deg[np.isfinite(alignment_deg)]
    diagnostics: dict[str, object] = {
        "frames": len(time_s),
        "duration_s": float(time_s[-1] - time_s[0]),
        "sample_rate_hz": 1.0 / dt_s,
        "state_valid_frames": int(np.count_nonzero(state_valid)),
        "state_valid_fraction": float(np.mean(state_valid)),
        "force_valid_frames": int(np.count_nonzero(force_valid)),
        "force_valid_fraction": float(np.mean(force_valid)),
        "expected_hanging_hover_force_n": expected_hover_force_n,
        "estimated_force_norm_median_n": float(np.median(valid_force_norm)),
        "estimated_force_norm_p95_n": float(np.percentile(valid_force_norm, 95)),
        "force_body_z_alignment_median_deg": float(np.median(valid_alignment)),
        "force_body_z_alignment_p90_deg": float(np.percentile(valid_alignment, 90)),
    }
    return arrays, diagnostics


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_force_take(
    take_id: str,
    *,
    role: str,
    model: dict[str, object],
    model_hash: str,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    settings: DifferentiationSettings = DifferentiationSettings(),
) -> dict[str, object]:
    """Build one force-model take and return its manifest row."""

    source_path = Path(processed_root) / take_id / "take.npz"
    source_metadata_path = Path(processed_root) / take_id / "metadata.json"
    if not source_path.is_file() or not source_metadata_path.is_file():
        raise FileNotFoundError(f"Missing processed take: {take_id}")
    with np.load(source_path, allow_pickle=False) as loaded:
        source = {name: loaded[name] for name in loaded.files}
    arrays, diagnostics = convert_processed_arrays(source, model, settings=settings)

    take_output = Path(output_root) / take_id
    take_path = take_output / "take.npz"
    deterministic_npz(take_path, arrays)
    source_metadata = _load_json(source_metadata_path)
    cable = CableConfiguration.from_mapping(model["cable"])
    dt_s = float(np.median(np.diff(arrays["time_s"])))
    metadata = {
        "schema": FORCE_TAKE_SCHEMA,
        "take_id": take_id,
        "role": role,
        "source": {
            "schema": source_metadata.get("schema"),
            "processed_take": str(source_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "processed_take_sha256": sha256_file(source_path),
            "raw_source_sha256": source_metadata.get("source_sha256", {}),
        },
        "model": {
            "schema": model["schema"],
            "config_sha256": model_hash,
            "point_mass_kg": model["point_mass"]["mass_kg"],
            "cable_mass_kg": cable.total_dynamic_mass_kg,
            "cable_vertex_masses_kg": list(cable.vertex_masses_kg),
            "gravity_world_m_s2": list(cable.gravity_m_s2),
        },
        "kinematics": {
            "attachment_source": "OptiTrack rigid-body pose plus rotated body-frame attachment offset",
            "attachment_offset_body_m": model["recorded_data"][
                "optitrack_to_attachment_offset_body_m"
            ],
            "node_mapping": _node_mapping(cable),
            "differentiation": {
                "method": "centered_local_polynomial",
                "window_samples": settings.window_samples,
                "sample_support_span_s": (settings.window_samples - 1) * dt_s,
                "polynomial_order": settings.polynomial_order,
                "boundary_policy": "NaN and invalid mask",
                "dropout_policy": "invalidate every derivative window touching invalid data",
            },
        },
        "force_estimation": {
            "field": "estimated_point_force_world_n",
            "status": "estimated_not_measured",
            "method": "whole_system_linear_momentum_balance",
            "equation": "F_point_est = m_point*a_root + sum(m_cable_node*a_node) - (m_point+m_cable)*g",
            "internal_cable_forces": "cancel from the whole-system balance",
            "unmodeled_effects": ["aerodynamic drag", "OptiTrack differentiation noise"],
            "historical_fullstate_command_used": False,
            "intended_use": [
                "model validation",
                "policy initial-state sampling",
                "force-scale checks",
            ],
            "not_ground_truth_for_supervised_force_learning": True,
        },
        "quality": diagnostics,
        "output_take_sha256": sha256_file(take_path),
    }
    atomic_json(take_output / "metadata.json", metadata)
    return {
        "role": role,
        "path": f"{take_id}/take.npz",
        "metadata_path": f"{take_id}/metadata.json",
        "sha256": metadata["output_take_sha256"],
        **diagnostics,
    }


def build_force_dataset(
    *,
    take_id: str | None = None,
    include_untouched_test: bool = False,
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    settings: DifferentiationSettings = DifferentiationSettings(),
) -> dict[str, object]:
    """Build selected training/validation takes; protect the test take by default."""

    model = _load_json(Path(model_path))
    source_manifest = _load_json(Path(manifest_path))
    model_hash = canonical_json_hash(model)
    selected: list[tuple[str, str]] = []
    excluded: dict[str, str] = {}
    for name, raw_row in source_manifest["takes"].items():
        row = dict(raw_row)
        if take_id is not None and name != take_id:
            continue
        if not bool(row.get("enabled", True)):
            excluded[name] = "disabled_in_source_manifest"
            continue
        role = str(row["role"])
        if role == "untouched_test" and not include_untouched_test:
            excluded[name] = "protected_untouched_test"
            continue
        selected.append((name, role))
    if take_id is not None and not selected:
        if take_id in excluded:
            raise ValueError(
                f"{take_id} is {excluded[take_id]}; pass include_untouched_test=True explicitly."
            )
        raise KeyError(f"Unknown or disabled take: {take_id}")

    rows: dict[str, object] = {}
    for name, role in selected:
        rows[name] = build_force_take(
            name,
            role=role,
            model=model,
            model_hash=model_hash,
            processed_root=Path(processed_root),
            output_root=Path(output_root),
            settings=settings,
        )
    manifest = {
        "schema": FORCE_DATASET_SCHEMA,
        "model_config_sha256": model_hash,
        "source_manifest_sha256": sha256_file(manifest_path),
        "force_semantics": "estimated_not_measured",
        "takes": rows,
        "excluded_takes": excluded,
        "summary": {
            "take_count": len(rows),
            "training_take_count": sum(
                row["role"] == "training" for row in rows.values()
            ),
            "validation_take_count": sum(
                row["role"] == "validation" for row in rows.values()
            ),
            "frames": sum(int(row["frames"]) for row in rows.values()),
            "duration_s": sum(float(row["duration_s"]) for row in rows.values()),
            "force_valid_frames": sum(
                int(row["force_valid_frames"]) for row in rows.values()
            ),
        },
    }
    Path(output_root).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(output_root) / "manifest.json", manifest)
    return manifest


def format_build_rows(manifest: dict[str, object]) -> Iterable[str]:
    """Yield compact terminal rows for the command-line builder."""

    for take_id, raw_row in manifest["takes"].items():
        row = dict(raw_row)
        yield (
            f"{take_id:18s} {row['role']:10s} "
            f"valid={100.0 * float(row['force_valid_fraction']):5.1f}% "
            f"force={float(row['estimated_force_norm_median_n']):.3f} N median, "
            f"alignment={float(row['force_body_z_alignment_median_deg']):.2f} deg median"
        )
