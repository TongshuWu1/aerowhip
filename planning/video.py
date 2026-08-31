"""Deterministic, phone-compatible rendering of saved MPPI replays."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import imageio_ffmpeg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from .results import PlanningResult, load_planning_result, load_replay_arrays


VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 30
PLAYBACK_SLOWDOWN = 4.0
BEGIN_HOLD_S = 0.5
END_HOLD_S = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _axis_limits(arrays: dict[str, np.ndarray], target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(
        (arrays["uav_position_m"], arrays["cable_position_m"].reshape(-1, 3), target[None]),
        axis=0,
    )
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 1.2)
    half = 0.58 * span
    return center - half, center + half


def _render_frames(result: PlanningResult):
    arrays = load_replay_arrays(result)
    times = arrays["time_s"].astype(float)
    target = np.asarray(result.task_config["target"]["position_m"], dtype=float)
    direction = np.asarray(
        result.task_config["target"]["desired_impact_direction"], dtype=float
    )
    radius = float(result.task_config["target"]["success_radius_m"])
    lower, upper = _axis_limits(arrays, target)

    figure = Figure(figsize=(VIDEO_WIDTH / 100, VIDEO_HEIGHT / 100), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(left=0.02, right=0.78, bottom=0.04, top=0.96)
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=19, azim=-62)
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    optimizer = str(result.metrics.get("optimizer", "MPPI"))
    axis.set_title(f"Final deterministic {optimizer} replay", fontsize=14, weight="bold")
    axis.grid(True, alpha=0.25)

    phi = np.linspace(0.0, 2.0 * math.pi, 24)
    theta = np.linspace(0.0, math.pi, 12)
    sphere_x = target[0] + radius * np.outer(np.cos(phi), np.sin(theta))
    sphere_y = target[1] + radius * np.outer(np.sin(phi), np.sin(theta))
    sphere_z = target[2] + radius * np.outer(np.ones_like(phi), np.cos(theta))
    axis.plot_wireframe(sphere_x, sphere_y, sphere_z, color="#22c55e", alpha=0.35, linewidth=0.5)
    axis.quiver(*target, *(0.18 * direction), color="#16a34a", linewidth=2.5)

    cable_line, = axis.plot([], [], [], color="#22c7d8", linewidth=3.0, label="Cable")
    node_points = axis.scatter([], [], [], s=22, color="#38d5e5", depthshade=False)
    tip_point = axis.scatter([], [], [], s=90, color="#f97316", edgecolor="white", depthshade=False, label="c10 tip")
    uav_point = axis.scatter([], [], [], s=150, marker="D", color="#f59e0b", edgecolor="#7c2d12", depthshade=False, label="UAV")
    uav_trail, = axis.plot([], [], [], color="#f59e0b", linewidth=1.5, alpha=0.75)
    tip_trail, = axis.plot([], [], [], color="#f97316", linewidth=1.5, alpha=0.8)
    axis.legend(loc="lower left", fontsize=9)
    overlay = figure.text(
        0.79,
        0.82,
        "",
        va="top",
        ha="left",
        fontsize=12,
        family="DejaVu Sans",
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.96},
    )
    figure.text(
        0.79,
        0.18,
        "SIMULATION ONLY\nNo real flight performed",
        va="bottom",
        ha="left",
        fontsize=10,
        color="#475569",
    )

    physical_duration = float(times[-1] - times[0])
    active_frames = max(1, int(round(physical_duration * PLAYBACK_SLOWDOWN * VIDEO_FPS)))
    begin_frames = int(round(BEGIN_HOLD_S * VIDEO_FPS))
    end_frames = int(round(END_HOLD_S * VIDEO_FPS))
    total_frames = begin_frames + active_frames + end_frames
    frame_indices = []
    for frame in range(total_frames):
        active = min(max(frame - begin_frames, 0), active_frames - 1)
        fraction = 0.0 if active_frames <= 1 else active / (active_frames - 1)
        source_time = times[0] + fraction * physical_duration
        frame_indices.append(int(np.argmin(np.abs(times - source_time))))

    for output_index, source_index in enumerate(frame_indices):
        cable = arrays["cable_position_m"][source_index]
        uav = arrays["uav_position_m"][source_index]
        tip = cable[-1]
        cable_line.set_data(cable[:, 0], cable[:, 1])
        cable_line.set_3d_properties(cable[:, 2])
        observed = cable[2:]
        node_points._offsets3d = (observed[:, 0], observed[:, 1], observed[:, 2])
        tip_point._offsets3d = ([tip[0]], [tip[1]], [tip[2]])
        uav_point._offsets3d = ([uav[0]], [uav[1]], [uav[2]])
        uav_path = arrays["uav_position_m"][: source_index + 1]
        tip_path = arrays["cable_position_m"][: source_index + 1, -1]
        uav_trail.set_data(uav_path[:, 0], uav_path[:, 1])
        uav_trail.set_3d_properties(uav_path[:, 2])
        tip_trail.set_data(tip_path[:, 0], tip_path[:, 1])
        tip_trail.set_3d_properties(tip_path[:, 2])
        is_final_hold = output_index >= begin_frames + active_frames
        status_line = f"FINAL RESULT: {result.status}" if is_final_hold else "Planning replay"
        overlay.set_text(
            f"Task\n{result.task_label}\n\n"
            f"Simulation time   {times[source_index]:.2f} s\n"
            f"Tip distance      {1000.0 * arrays['tip_target_distance_m'][source_index]:.1f} mm\n"
            f"Tip speed         {arrays['tip_speed_m_s'][source_index]:.2f} m/s\n"
            f"Directed speed    {arrays['directed_tip_speed_m_s'][source_index]:.2f} m/s\n\n"
            f"{status_line}"
        )
        overlay.set_color("#15803d" if is_final_hold and result.success else "#b91c1c" if is_final_hold else "#0f172a")
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        yield np.ascontiguousarray(rgba[:, :, :3]), total_frames


def render_replay_video(
    result_or_directory: PlanningResult | str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Render saved final_replay.npz to a mobile-compatible H.264 MP4."""

    result = (
        result_or_directory
        if isinstance(result_or_directory, PlanningResult)
        else load_planning_result(result_or_directory)
    )
    output = result.video_path
    metadata_path = result.directory / "video_metadata.json"
    source_hash = _sha256(result.replay_path)
    if output.is_file() and metadata_path.is_file() and not overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_trajectory_sha256") == source_hash:
            return output, metadata

    frame_stream = _render_frames(result)
    first_frame, total_frames = next(frame_stream)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-r", str(VIDEO_FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level:v", "3.1",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "21",
        "-movflags", "+faststart",
        str(output),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        process.stdin.write(first_frame.tobytes())
        for frame, _ in frame_stream:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except BaseException:
        process.kill()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with code {return_code}:\n{stderr[-4000:]}")

    metadata = {
        "schema": "mppi_replay_video_v1",
        "task_id": result.task_id,
        "task_status": result.status,
        "source_replay_artifact": str(result.replay_path),
        "source_trajectory_sha256": source_hash,
        "video_path": str(output),
        "video_sha256": _sha256(output),
        "resolution": [VIDEO_WIDTH, VIDEO_HEIGHT],
        "fps": VIDEO_FPS,
        "frame_count": total_frames,
        "duration_s": total_frames / VIDEO_FPS,
        "physical_duration_s": float(
            load_replay_arrays(result)["time_s"][-1]
            - load_replay_arrays(result)["time_s"][0]
        ),
        "playback_slowdown": PLAYBACK_SLOWDOWN,
        "begin_hold_s": BEGIN_HOLD_S,
        "end_hold_s": END_HOLD_S,
        "container": "MP4",
        "codec": "H.264 (libx264)",
        "h264_profile": "Constrained Baseline",
        "pixel_format": "yuv420p",
        "fast_start": True,
        "audio": False,
        "mobile_compatibility": "H.264 Constrained Baseline / Level 3.1 / yuv420p / MP4 fast-start",
        "file_size_bytes": output.stat().st_size,
        "render_reran_mppi": False,
        "real_hardware_execution": "NOT PERFORMED",
    }
    _write_json(metadata_path, metadata)
    return output, metadata
