from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from drone_mpc.model import load_cable_model
from drone_mpc.reduced import stable_controller_model
from drone_mpc.rl_env import TaskDistribution, VectorWhipEnvironment
from drone_mpc.sac import SacSettings, load_policy
from drone_mpc.simulator import SimulationSettings, WhipSimulator
from optitrack_offline.config import DEFAULT_MODEL_PATH


DEFAULT_POLICY = Path("data/drone_mpc/sac_policy_goals_demo12.pt")
DEFAULT_OUTPUT = Path("videos/sac_learned_multi_target_phone.mp4")


@dataclass(frozen=True, slots=True)
class TargetCase:
    name: str
    target_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class RecordedRollout:
    case: TargetCase
    time_s: np.ndarray
    drone_m: np.ndarray
    cable_m: np.ndarray
    cable_velocity_m_s: np.ndarray
    direction: np.ndarray
    hit_frame: int
    near_initial_reach: bool
    minimum_error_m: float
    hit_error_m: float
    hit_speed_m_s: float
    hit_angle_deg: float
    hit_drone_displacement_m: float


# Selected after a deterministic grid sweep of the retained seed-42 policy.
# None are MPC-prior targets. They are illustrative examples, not evaluation data.
TARGETS = (
    TargetCase("near / low", (-0.502295, -0.290000, -0.140000)),
    TargetCase("near / high", (+0.410000, -0.710141, -0.060000)),
    TargetCase("near / long", (+0.845723, +0.307818, -0.140000)),
    TargetCase("far / low", (+0.085413, +0.976271, -0.180000)),
    TargetCase("far / middle", (-0.884684, +0.619463, -0.100000)),
    TargetCase("far / long", (-0.305406, -1.139792, -0.140000)),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_frame(target: np.ndarray) -> np.ndarray:
    azimuth = math.atan2(float(target[1]), float(target[0]))
    forward = np.array((math.cos(azimuth), math.sin(azimuth), 0.0), dtype=np.float32)
    up = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    lateral = np.cross(up, forward)
    return np.stack((forward, lateral, up))


def _configure_target(environment: VectorWhipEnvironment, target: np.ndarray) -> np.ndarray:
    frame = _target_frame(target)
    environment.target_position_m[0].copy_(
        torch.as_tensor(target, dtype=environment.dtype, device=environment.device)
    )
    environment.target_frame[0].copy_(
        torch.as_tensor(frame, dtype=environment.dtype, device=environment.device)
    )
    environment.minimum_impact_speed_m_s[0] = (
        environment.task.minimum_impact_speed_min_m_s
    )
    environment.hit_tolerance_m[0] = environment.task.hit_tolerance_m
    environment.impact_angle_deg[0] = environment.task.impact_angle_deg
    environment.previous_action[0] = 0.0
    environment.peak_relative_cable_energy_j[0] = 0.0
    return frame[0]


def _record_case(
    environment: VectorWhipEnvironment,
    agent,
    case: TargetCase,
) -> RecordedRollout:
    environment.reset_done(
        torch.ones(1, dtype=torch.bool, device=environment.device)
    )
    target = np.asarray(case.target_m, dtype=np.float32)
    direction = _configure_target(environment, target)
    times = [0.0]
    drones = [environment.state.drone_position_m[0].detach().cpu().numpy().copy()]
    cables = [environment.state.cable.positions_m[0].detach().cpu().numpy().copy()]
    velocities = [environment.state.cable.velocities_m_s[0].detach().cpu().numpy().copy()]
    success = False

    for _ in range(environment.maximum_steps):
        observation = environment.observation()
        action = agent.act(observation, deterministic=True)
        world_action = environment._to_world(action)
        acceleration = (
            world_action * environment.simulator.settings.maximum_acceleration_m_s2
        )
        rollout = environment.simulator.rollout(
            environment.state,
            acceleration[:, None],
            create_graph=False,
        )
        start_time = times[-1]
        local_time = rollout.time_s.detach().cpu().numpy()
        local_drone = rollout.drone_positions_m[0].detach().cpu().numpy()
        local_cable = rollout.cable_positions_m[0].detach().cpu().numpy()
        local_velocity = rollout.cable_velocities_m_s[0].detach().cpu().numpy()
        for frame in range(1, rollout.frame_count):
            times.append(start_time + float(local_time[frame]))
            drones.append(local_drone[frame].copy())
            cables.append(local_cable[frame].copy())
            velocities.append(local_velocity[frame].copy())
        result = environment.step(action)
        if bool(result.done[0].detach().cpu()):
            success = bool(result.success[0].detach().cpu())
            break

    time = np.asarray(times, dtype=np.float64)
    drone = np.asarray(drones, dtype=np.float64)
    cable = np.asarray(cables, dtype=np.float64)
    velocity = np.asarray(velocities, dtype=np.float64)
    tip = cable[:, -1]
    tip_velocity = velocity[:, -1]
    error = np.linalg.norm(tip - target[None], axis=1)
    directed = tip_velocity @ direction
    speed = np.linalg.norm(tip_velocity, axis=1)
    cosine = np.divide(
        directed,
        speed,
        out=np.full_like(directed, -1.0),
        where=speed > 1.0e-9,
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    clearance = np.linalg.norm(drone - target[None], axis=1)
    drone_speed = np.linalg.norm(np.gradient(drone, time, axis=0), axis=1)
    valid = (
        (error <= environment.task.hit_tolerance_m)
        & (directed >= environment.task.minimum_impact_speed_min_m_s)
        & (angle <= environment.task.impact_angle_deg)
        & (np.minimum.accumulate(clearance) >= environment.task.drone_keepout_radius_m)
        & (
            np.maximum.accumulate(drone_speed)
            <= environment.simulator.settings.maximum_speed_m_s * (1.0 + 1.0e-3)
        )
    )
    hit_indices = np.flatnonzero(valid)
    if not success or len(hit_indices) == 0:
        raise RuntimeError(f"Selected montage target did not produce a valid hit: {case.name}")
    hit = int(hit_indices[0])
    initial_attachment = cable[0, 0]
    near = bool(
        np.linalg.norm(target - initial_attachment)
        <= environment.simulator.snapshot.cable_length_m
    )
    return RecordedRollout(
        case=case,
        time_s=time[: hit + 1],
        drone_m=drone[: hit + 1],
        cable_m=cable[: hit + 1],
        cable_velocity_m_s=velocity[: hit + 1],
        direction=direction.astype(np.float64),
        hit_frame=hit,
        near_initial_reach=near,
        minimum_error_m=float(np.min(error[: hit + 1])),
        hit_error_m=float(error[hit]),
        hit_speed_m_s=float(directed[hit]),
        hit_angle_deg=float(angle[hit]),
        hit_drone_displacement_m=float(np.linalg.norm(drone[hit] - drone[0])),
    )


WIDTH = 1080
HEIGHT = 1080
BACKGROUND = (16, 25, 34)
WHITE = (235, 241, 246)
MUTED = (143, 164, 180)
CYAN = (233, 215, 53)
MAGENTA = (216, 91, 239)
ORANGE = (82, 173, 255)
RED = (74, 74, 245)
GREEN = (113, 222, 93)
GRID = (48, 68, 82)


def _project(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    horizontal = 0.75 * values[:, 0] - 0.66 * values[:, 1]
    vertical = 0.34 * values[:, 0] + 0.39 * values[:, 1] - 0.95 * values[:, 2]
    screen = np.empty((len(values), 2), dtype=np.int32)
    screen[:, 0] = np.round(540.0 + 300.0 * horizontal).astype(np.int32)
    screen[:, 1] = np.round(330.0 + 300.0 * vertical).astype(np.int32)
    return screen


def _text(
    image: np.ndarray,
    value: str,
    position: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        value,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _arrow(image: np.ndarray, start: np.ndarray, end: np.ndarray, color) -> None:
    values = _project(np.stack((start, end)))
    cv2.arrowedLine(
        image,
        tuple(values[0]),
        tuple(values[1]),
        color,
        4,
        cv2.LINE_AA,
        tipLength=0.22,
    )


def _draw_floor(image: np.ndarray) -> None:
    floor_z = -1.10
    for coordinate in np.arange(-1.5, 1.51, 0.25):
        first = _project(np.array(((-1.5, coordinate, floor_z), (1.5, coordinate, floor_z))))
        second = _project(np.array(((coordinate, -1.5, floor_z), (coordinate, 1.5, floor_z))))
        cv2.line(image, tuple(first[0]), tuple(first[1]), GRID, 1, cv2.LINE_AA)
        cv2.line(image, tuple(second[0]), tuple(second[1]), GRID, 1, cv2.LINE_AA)


def _draw_drone(image: np.ndarray, center: np.ndarray) -> None:
    offsets = np.array(
        ((0.13, 0.13, 0.0), (0.13, -0.13, 0.0), (-0.13, 0.13, 0.0), (-0.13, -0.13, 0.0))
    )
    points = _project(center[None] + offsets)
    cv2.line(image, tuple(points[0]), tuple(points[3]), ORANGE, 7, cv2.LINE_AA)
    cv2.line(image, tuple(points[1]), tuple(points[2]), ORANGE, 7, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), 12, ORANGE, 3, cv2.LINE_AA)
    cv2.circle(image, tuple(_project(center[None])[0]), 9, WHITE, -1, cv2.LINE_AA)


def _scene_frame(
    rollout: RecordedRollout,
    frame_index: int,
    target_number: int,
    target_count: int,
    *,
    final_hold: bool,
) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    target = np.asarray(rollout.case.target_m, dtype=np.float64)
    drone = rollout.drone_m[frame_index]
    cable = rollout.cable_m[frame_index]
    tip = cable[-1]
    tip_velocity = rollout.cable_velocity_m_s[frame_index, -1]
    error = float(np.linalg.norm(tip - target))
    speed = float(tip_velocity @ rollout.direction)
    displacement = float(np.linalg.norm(drone - rollout.drone_m[0]))
    _draw_floor(image)

    drone_path = _project(rollout.drone_m[: frame_index + 1])
    tip_path = _project(rollout.cable_m[: frame_index + 1, -1])
    if len(drone_path) > 1:
        cv2.polylines(image, [drone_path], False, (58, 103, 152), 2, cv2.LINE_AA)
        cv2.polylines(image, [tip_path], False, (111, 58, 119), 3, cv2.LINE_AA)

    target_screen = _project(target[None])[0]
    cv2.circle(image, tuple(target_screen), 17, RED, 3, cv2.LINE_AA)
    cv2.circle(image, tuple(target_screen), 5, RED, -1, cv2.LINE_AA)
    _arrow(image, target - 0.22 * rollout.direction, target, RED)
    cable_screen = _project(cable)
    cv2.polylines(image, [cable_screen], False, CYAN, 7, cv2.LINE_AA)
    for point in cable_screen[1:-1:2]:
        cv2.circle(image, tuple(point), 4, CYAN, -1, cv2.LINE_AA)
    cv2.circle(image, tuple(cable_screen[-1]), 10, MAGENTA, -1, cv2.LINE_AA)
    _draw_drone(image, drone)

    cv2.rectangle(image, (0, 0), (WIDTH, 104), (9, 15, 22), -1)
    classification = "WITHIN INITIAL REACH" if rollout.near_initial_reach else "BEYOND INITIAL REACH"
    _text(image, "LEARNED SAC CABLE WHIP", (35, 42), 0.92)
    _text(
        image,
        f"target {target_number}/{target_count}  |  {classification}  |  0.5x playback",
        (36, 81),
        0.58,
        MUTED,
        1,
    )
    cv2.rectangle(image, (25, 875), (1055, 1055), (9, 15, 22), -1)
    radius = math.hypot(target[0], target[1])
    azimuth = math.degrees(math.atan2(target[1], target[0]))
    _text(
        image,
        f"TARGET  r={radius:.2f} m   z={target[2]:+.2f} m   azimuth={azimuth:+.0f} deg",
        (48, 920),
        0.62,
    )
    _text(
        image,
        f"t={rollout.time_s[frame_index]:.2f} s   tip error={1000.0 * error:5.1f} mm   "
        f"directed speed={speed:4.2f} m/s   drone move={displacement:.2f} m",
        (48, 970),
        0.53,
        MUTED,
        1,
    )
    _text(
        image,
        "cyan: cable   magenta: free tip   orange: drone   red: target + impact direction",
        (48, 1017),
        0.46,
        MUTED,
        1,
    )
    if final_hold:
        cv2.rectangle(image, (393, 126), (687, 226), (8, 34, 17), -1)
        cv2.rectangle(image, (393, 126), (687, 226), GREEN, 3)
        _text(image, "HIT", (486, 190), 1.55, GREEN, 4)
    return image


def _title_frame() -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    _text(image, "LEARNED SAC", (310, 300), 1.7)
    _text(image, "CABLE WHIP POLICY", (215, 375), 1.35, CYAN, 3)
    _text(image, "six selected unseen target positions", (275, 485), 0.72, MUTED, 2)
    _text(image, "3 within + 3 beyond initial cable reach", (260, 535), 0.67, MUTED, 2)
    _text(image, "deterministic policy seed 42  |  nominal DDER simulation", (190, 665), 0.58, WHITE, 1)
    _text(image, "illustrative montage - quantitative evaluation is reported separately", (160, 715), 0.52, MUTED, 1)
    return image


def _write_video(output: Path, rollouts: list[RecordedRollout]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = 30
    process = subprocess.Popen(
        (
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ),
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("FFmpeg did not open its video input.")

    def write(image: np.ndarray, count: int = 1) -> None:
        for _ in range(count):
            process.stdin.write(image.tobytes())

    write(_title_frame(), int(1.6 * fps))
    for number, rollout in enumerate(rollouts, start=1):
        write(_scene_frame(rollout, 0, number, len(rollouts), final_hold=False), int(0.45 * fps))
        duration = float(rollout.time_s[-1])
        output_count = max(1, int(math.ceil(duration * 2.0 * fps)))
        for output_frame in range(output_count):
            simulation_time = min(duration, output_frame / (2.0 * fps))
            source_frame = int(np.argmin(np.abs(rollout.time_s - simulation_time)))
            write(
                _scene_frame(
                    rollout,
                    source_frame,
                    number,
                    len(rollouts),
                    final_hold=False,
                )
            )
        write(
            _scene_frame(
                rollout,
                rollout.hit_frame,
                number,
                len(rollouts),
                final_hold=True,
            ),
            int(0.75 * fps),
        )
    ending = _title_frame()
    cv2.rectangle(ending, (225, 805), (855, 915), (8, 34, 17), -1)
    _text(ending, "6 / 6 SELECTED TARGETS HIT", (285, 870), 0.78, GREEN, 2)
    write(ending, int(1.4 * fps))
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {return_code}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render learned SAC cable-whip examples.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    policy_path = arguments.policy.resolve()
    payload = torch.load(policy_path, map_location="cpu", weights_only=False)
    setting_names = {field.name for field in fields(SacSettings)}
    settings = SacSettings(
        **{name: payload["settings"][name] for name in setting_names}
    )
    raw_task = dict(payload["task"])
    raw_task.setdefault("minimum_extension_ratio", 0.0)
    task = TaskDistribution(**raw_task)
    source = load_cable_model(arguments.model)
    controller = stable_controller_model(
        source,
        simulation_dt_s=settings.physics_dt_s,
        node_count=settings.controller_node_count,
        constraint_iterations=4,
    )
    simulation = SimulationSettings(
        horizon_s=settings.control_interval_s,
        simulation_dt_s=settings.physics_dt_s,
        control_interval_s=settings.control_interval_s,
        attachment_drop_m=settings.attachment_drop_m,
        maximum_acceleration_m_s2=settings.maximum_acceleration_m_s2,
        maximum_speed_m_s=settings.maximum_speed_m_s,
    )
    simulator = WhipSimulator(controller, simulation, device="cuda")
    environment = VectorWhipEnvironment(
        simulator,
        1,
        settings.episode_horizon_s,
        task,
        seed=settings.seed,
    )
    agent = load_policy(
        policy_path,
        settings,
        task,
        environment.observation_size,
        environment.action_size,
        source,
        controller,
        device=torch.device("cuda"),
    )
    rollouts: list[RecordedRollout] = []
    for case in TARGETS:
        rollout = _record_case(environment, agent, case)
        rollouts.append(rollout)
        print(
            f"{case.name}: hit t={rollout.time_s[-1]:.2f}s "
            f"error={1000.0 * rollout.hit_error_m:.1f}mm "
            f"speed={rollout.hit_speed_m_s:.2f}m/s "
            f"angle={rollout.hit_angle_deg:.1f}deg"
        )
    output = arguments.output.resolve()
    _write_video(output, rollouts)
    metadata = {
        "schema": "learned_sac_multi_target_video_v1",
        "policy_path": str(policy_path),
        "policy_sha256": _sha256(policy_path),
        "source_model_sha256": source.sha256,
        "controller_model_sha256": controller.sha256,
        "selected_examples_not_evaluation": True,
        "playback_speed": 0.5,
        "video": {
            "path": str(output),
            "sha256": _sha256(output),
            "width": WIDTH,
            "height": HEIGHT,
            "fps": 30,
            "codec": "H.264/yuv420p",
        },
        "targets": [
            {
                "name": rollout.case.name,
                "target_m": list(rollout.case.target_m),
                "within_initial_reach": rollout.near_initial_reach,
                "hit_time_s": float(rollout.time_s[-1]),
                "hit_error_m": rollout.hit_error_m,
                "directed_tip_speed_m_s": rollout.hit_speed_m_s,
                "impact_direction_error_deg": rollout.hit_angle_deg,
                "drone_displacement_at_hit_m": rollout.hit_drone_displacement_m,
            }
            for rollout in rollouts
        ],
    }
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Video: {output}")
    print(f"Metadata: {sidecar}")


if __name__ == "__main__":
    main()
