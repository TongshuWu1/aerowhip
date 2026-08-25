"""Analyze distributed cable motion in a saved matched-model MPPI replay.

This is deliberately a post-processing tool.  It does not alter the optimizer
or add an energy/shape term to the task objective.  The plots answer the more
basic question of whether the selected boundary motion produces deformation
and velocity that travel through the rod before the free-tip strike.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cable_twin.shared.dder import curvature_binormals
from drone_mpc.model import CableModelSnapshot, load_cable_model
from optitrack_offline.config import DEFAULT_MODEL_PATH


@dataclass(frozen=True, slots=True)
class PropagationDiagnostics:
    """Node-resolved kinematics and compact event diagnostics."""

    time_s: np.ndarray
    material_coordinate_m: np.ndarray
    relative_speed_m_s: np.ndarray
    relative_kinetic_energy_j: np.ndarray
    curvature_m_inv: np.ndarray
    impact_frame: int
    peak_forward_frame: int
    peak_relative_energy_frame: int
    peak_tip_speed_frame: int
    node_peak_relative_speed_frame: np.ndarray
    distal_energy_fraction_at_impact: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


def compute_propagation_diagnostics(
    *,
    time_s: np.ndarray,
    drone_positions_m: np.ndarray,
    drone_velocities_m_s: np.ndarray,
    cable_positions_m: np.ndarray,
    cable_velocities_m_s: np.ndarray,
    impact_frame: int,
    vertex_masses_kg: np.ndarray,
    rest_lengths_m: np.ndarray,
    forward_direction: np.ndarray,
) -> PropagationDiagnostics:
    """Compute node-resolved quantities through the selected impact event."""

    time = np.asarray(time_s, dtype=np.float64)
    drone_position = np.asarray(drone_positions_m, dtype=np.float64)
    drone_velocity = np.asarray(drone_velocities_m_s, dtype=np.float64)
    positions = np.asarray(cable_positions_m, dtype=np.float64)
    velocities = np.asarray(cable_velocities_m_s, dtype=np.float64)
    masses = np.asarray(vertex_masses_kg, dtype=np.float64)
    rest_lengths = np.asarray(rest_lengths_m, dtype=np.float64)
    forward = np.asarray(forward_direction, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("cable_positions_m must have shape TxNx3.")
    if velocities.shape != positions.shape:
        raise ValueError("cable velocities must match cable positions.")
    frame_count, node_count, _ = positions.shape
    if time.shape != (frame_count,):
        raise ValueError("time_s must contain one value per replay frame.")
    if drone_position.shape != (frame_count, 3) or drone_velocity.shape != (
        frame_count,
        3,
    ):
        raise ValueError("drone traces must have shape Tx3.")
    if masses.shape != (node_count,) or rest_lengths.shape != (node_count - 1,):
        raise ValueError("model masses/rest lengths do not match the replay nodes.")
    if not 1 <= int(impact_frame) < frame_count:
        raise ValueError("impact_frame must identify a non-initial replay frame.")
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-12:
        raise ValueError("forward_direction must be non-zero.")
    forward = forward / norm

    relative_velocity = velocities - drone_velocity[:, None, :]
    relative_speed = np.linalg.norm(relative_velocity, axis=2)
    relative_energy = 0.5 * masses[None] * relative_speed**2

    with torch.no_grad():
        curvature_binormal = curvature_binormals(
            torch.as_tensor(positions, dtype=torch.float64)
        ).numpy()
    dual_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    interior_curvature = np.linalg.norm(curvature_binormal, axis=2) / dual_lengths[None]
    curvature = np.full((frame_count, node_count), np.nan, dtype=np.float64)
    curvature[:, 1:-1] = interior_curvature

    event_stop = int(impact_frame) + 1
    forward_displacement = (drone_position - drone_position[0]) @ forward
    peak_forward = int(np.argmax(forward_displacement[:event_stop]))
    total_relative_energy = np.sum(relative_energy, axis=1)
    peak_relative_energy = int(np.argmax(total_relative_energy[:event_stop]))
    tip_speed = np.linalg.norm(velocities[:, -1], axis=1)
    peak_tip_speed = int(np.argmax(tip_speed[:event_stop]))
    node_peak_speed = np.argmax(relative_speed[:event_stop], axis=0).astype(np.int64)

    distal_start = max(1, int(np.floor(0.75 * (node_count - 1))))
    impact_energy = relative_energy[int(impact_frame)]
    energy_total = float(np.sum(impact_energy))
    distal_fraction = (
        float(np.sum(impact_energy[distal_start:])) / energy_total
        if energy_total > 1.0e-15
        else 0.0
    )
    material_coordinate = np.concatenate(([0.0], np.cumsum(rest_lengths)))
    return PropagationDiagnostics(
        time_s=_readonly(time),
        material_coordinate_m=_readonly(material_coordinate),
        relative_speed_m_s=_readonly(relative_speed),
        relative_kinetic_energy_j=_readonly(relative_energy),
        curvature_m_inv=_readonly(curvature),
        impact_frame=int(impact_frame),
        peak_forward_frame=peak_forward,
        peak_relative_energy_frame=peak_relative_energy,
        peak_tip_speed_frame=peak_tip_speed,
        node_peak_relative_speed_frame=_readonly(node_peak_speed),
        distal_energy_fraction_at_impact=distal_fraction,
    )


def analyze_replay(
    replay_path: str | Path,
    model_path: str | Path,
) -> tuple[PropagationDiagnostics, dict[str, object], CableModelSnapshot]:
    replay = Path(replay_path).expanduser().resolve()
    metadata_path = replay.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    snapshot = load_cable_model(model_path)
    expected_hash = str(metadata.get("model_sha256", ""))
    if snapshot.sha256 != expected_hash:
        raise ValueError(
            "The supplied cable model does not match the replay model hash: "
            f"expected {expected_hash}, found {snapshot.sha256}."
        )
    with np.load(replay) as payload:
        target = np.asarray(payload["target_position_m"], dtype=np.float64)
        drone = np.asarray(payload["drone_positions_m"], dtype=np.float64)
        forward = target - drone[0]
        forward[2] = 0.0
        if float(np.linalg.norm(forward)) <= 1.0e-12:
            forward = np.asarray(payload["impact_direction"], dtype=np.float64)
            forward[2] = 0.0
        diagnostics = compute_propagation_diagnostics(
            time_s=payload["time_s"],
            drone_positions_m=drone,
            drone_velocities_m_s=payload["drone_velocities_m_s"],
            cable_positions_m=payload["cable_positions_m"],
            cable_velocities_m_s=payload["cable_velocities_m_s"],
            impact_frame=int(payload["impact_frame"]),
            vertex_masses_kg=np.asarray(
                snapshot.model.parameters.vertex_masses_kg, dtype=np.float64
            ),
            rest_lengths_m=np.asarray(
                snapshot.model.parameters.rest_lengths_m, dtype=np.float64
            ),
            forward_direction=forward,
        )
    return diagnostics, metadata, snapshot


def _event_lines(axis, diagnostics: PropagationDiagnostics) -> None:
    events = (
        (diagnostics.peak_forward_frame, "peak forward / reversal", "#26734d"),
        (diagnostics.impact_frame, "impact", "#b52b2b"),
    )
    for frame, label, color in events:
        axis.axvline(diagnostics.time_s[frame], color=color, linestyle="--", linewidth=1.2)
        if axis is not None:
            axis.text(
                diagnostics.time_s[frame],
                1.01,
                label,
                color=color,
                fontsize=8,
                rotation=90,
                ha="left",
                va="bottom",
                transform=axis.get_xaxis_transform(),
            )


def render_propagation_figure(
    diagnostics: PropagationDiagnostics,
    metadata: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Render publication-oriented node-by-time evidence for one replay."""

    import matplotlib.pyplot as plt

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stop = diagnostics.impact_frame + 1
    time = diagnostics.time_s[:stop]
    s = diagnostics.material_coordinate_m
    extent = (float(time[0]), float(time[-1]), float(s[0]), float(s[-1]))

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    heatmaps = (
        (
            axes[0, 0],
            diagnostics.relative_speed_m_s[:stop].T,
            "Node speed relative to drone (m/s)",
            "magma",
        ),
        (
            axes[0, 1],
            1000.0 * diagnostics.relative_kinetic_energy_j[:stop].T,
            "Node kinetic energy relative to drone (mJ)",
            "viridis",
        ),
        (
            axes[1, 0],
            diagnostics.curvature_m_inv[:stop].T,
            "DER curvature magnitude (1/m)",
            "cividis",
        ),
    )
    for axis, values, title, color_map in heatmaps:
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=color_map,
        )
        _event_lines(axis, diagnostics)
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Material coordinate from drone (m)")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    summary_axis = axes[1, 1]
    peak_time = diagnostics.time_s[diagnostics.node_peak_relative_speed_frame]
    summary_axis.plot(
        peak_time,
        s,
        "o-",
        color="#315b8a",
        linewidth=1.5,
        markersize=4,
        label="time of each node's peak relative speed",
    )
    summary_axis.axvline(
        diagnostics.time_s[diagnostics.peak_forward_frame],
        color="#26734d",
        linestyle="--",
        label="drone reversal",
    )
    summary_axis.axvline(
        diagnostics.time_s[diagnostics.impact_frame],
        color="#b52b2b",
        linestyle="--",
        label="impact",
    )
    summary_axis.set_xlabel("Peak time before impact (s)")
    summary_axis.set_ylabel("Material coordinate from drone (m)")
    summary_axis.set_title("Timing along the cable (descriptive, not a wave-speed fit)")
    summary_axis.grid(True, color="#dddddd", linewidth=0.7)
    summary_axis.legend(fontsize=8, loc="best")

    terms = metadata.get("terms", {})
    directed_speed = float(terms.get("directional_speed_m_s", float("nan")))
    figure.suptitle(
        "Distributed DDER response in the verified MPPI strike\n"
        f"impact={diagnostics.time_s[diagnostics.impact_frame]:.2f}s, "
        f"directed tip speed={directed_speed:.2f}m/s, "
        f"distal-quarter relative KE at impact="
        f"{100.0 * diagnostics.distal_energy_fraction_at_impact:.1f}%",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def save_diagnostics(
    diagnostics: PropagationDiagnostics,
    metadata: dict[str, object],
    replay_path: str | Path,
    output_stem: str | Path,
) -> tuple[Path, Path]:
    stem = Path(output_stem).expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz_path = stem.with_suffix(".npz")
    json_path = stem.with_suffix(".json")
    np.savez_compressed(
        npz_path,
        time_s=diagnostics.time_s,
        material_coordinate_m=diagnostics.material_coordinate_m,
        relative_speed_m_s=diagnostics.relative_speed_m_s,
        relative_kinetic_energy_j=diagnostics.relative_kinetic_energy_j,
        curvature_m_inv=diagnostics.curvature_m_inv,
        node_peak_relative_speed_frame=diagnostics.node_peak_relative_speed_frame,
    )
    event = {
        "schema": "mppi_distributed_propagation_v1",
        "source_replay": str(Path(replay_path).expanduser().resolve()),
        "source_replay_sha256": _sha256(Path(replay_path).expanduser().resolve()),
        "source_model_sha256": metadata.get("model_sha256"),
        "impact_frame": diagnostics.impact_frame,
        "impact_time_s": float(diagnostics.time_s[diagnostics.impact_frame]),
        "peak_forward_frame": diagnostics.peak_forward_frame,
        "peak_forward_time_s": float(
            diagnostics.time_s[diagnostics.peak_forward_frame]
        ),
        "peak_relative_energy_frame": diagnostics.peak_relative_energy_frame,
        "peak_relative_energy_time_s": float(
            diagnostics.time_s[diagnostics.peak_relative_energy_frame]
        ),
        "peak_tip_speed_frame": diagnostics.peak_tip_speed_frame,
        "peak_tip_speed_time_s": float(
            diagnostics.time_s[diagnostics.peak_tip_speed_frame]
        ),
        "distal_quarter_relative_kinetic_energy_fraction_at_impact": (
            diagnostics.distal_energy_fraction_at_impact
        ),
        "node_peak_relative_speed_time_s": diagnostics.time_s[
            diagnostics.node_peak_relative_speed_frame
        ].tolist(),
        "interpretation_note": (
            "Peak timing is descriptive evidence. It is not identified as a "
            "material wave speed and it is not part of the MPPI objective."
        ),
    }
    json_path.write_text(json.dumps(event, indent=2, sort_keys=True), encoding="utf-8")
    return npz_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("data/drone_mpc/perfect_model_mppi_farther_faster.npz"),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("data/drone_mpc/diagnostics/mppi_farther_faster_propagation"),
    )
    arguments = parser.parse_args()
    diagnostics, metadata, _snapshot = analyze_replay(
        arguments.replay, arguments.model
    )
    arrays, summary = save_diagnostics(
        diagnostics, metadata, arguments.replay, arguments.output_stem
    )
    figure = render_propagation_figure(
        diagnostics, metadata, arguments.output_stem.with_suffix(".png")
    )
    print(f"Figure: {figure}")
    print(f"Arrays: {arrays}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
