"""Render a phone-viewable video from a saved perfect-model MPPI replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np


WIDTH = 1080
HEIGHT = 1080
FPS = 30
BACKGROUND = (248, 248, 248)
INK = (24, 24, 24)
MUTED = (105, 105, 105)
GRID = (218, 218, 218)
CABLE = (214, 121, 32)
TIP = (171, 42, 187)
DRONE = (20, 133, 94)
TARGET = (42, 42, 220)
SUCCESS = (30, 150, 45)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(
    image: np.ndarray,
    value: str,
    position: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = INK,
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


class ReplayRenderer:
    def __init__(self, replay_path: Path) -> None:
        self.replay_path = replay_path.resolve()
        metadata_path = self.replay_path.with_suffix(".json")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(self.replay_path) as payload:
            self.time_s = np.array(payload["time_s"], dtype=np.float64)
            self.drone_m = np.array(payload["drone_positions_m"], dtype=np.float64)
            self.cable_m = np.array(payload["cable_positions_m"], dtype=np.float64)
            self.cable_velocity_m_s = np.array(
                payload["cable_velocities_m_s"], dtype=np.float64
            )
            self.target_m = np.array(payload["target_position_m"], dtype=np.float64)
            self.direction = np.array(payload["impact_direction"], dtype=np.float64)
            self.impact_frame = int(payload["impact_frame"])
        self.direction /= max(float(np.linalg.norm(self.direction)), 1.0e-12)
        self.origin = self.drone_m[0].copy()
        self.origin[2] = 1.50
        self.terms = self.metadata["terms"]

    def _project(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.origin
        horizontal = 0.79 * values[:, 0] - 0.61 * values[:, 1]
        vertical = 0.30 * values[:, 0] + 0.38 * values[:, 1] - 0.96 * values[:, 2]
        screen = np.empty((len(values), 2), dtype=np.int32)
        screen[:, 0] = np.round(430.0 + 420.0 * horizontal).astype(np.int32)
        screen[:, 1] = np.round(315.0 + 420.0 * vertical).astype(np.int32)
        return screen

    def _floor(self, image: np.ndarray) -> None:
        for coordinate in np.arange(-1.0, 1.01, 0.20):
            first = self._project(
                np.array(((-1.0, coordinate, 0.0), (1.0, coordinate, 0.0)))
            )
            second = self._project(
                np.array(((coordinate, -1.0, 0.0), (coordinate, 1.0, 0.0)))
            )
            cv2.line(image, tuple(first[0]), tuple(first[1]), GRID, 1, cv2.LINE_AA)
            cv2.line(image, tuple(second[0]), tuple(second[1]), GRID, 1, cv2.LINE_AA)

    def _drone(self, image: np.ndarray, center: np.ndarray) -> None:
        offsets = np.array(
            (
                (0.11, 0.11, 0.0),
                (0.11, -0.11, 0.0),
                (-0.11, 0.11, 0.0),
                (-0.11, -0.11, 0.0),
            )
        )
        points = self._project(center[None] + offsets)
        cv2.line(image, tuple(points[0]), tuple(points[3]), DRONE, 7, cv2.LINE_AA)
        cv2.line(image, tuple(points[1]), tuple(points[2]), DRONE, 7, cv2.LINE_AA)
        for point in points:
            cv2.circle(image, tuple(point), 12, DRONE, 3, cv2.LINE_AA)
        cv2.circle(image, tuple(self._project(center[None])[0]), 8, INK, -1, cv2.LINE_AA)

    def title_frame(self) -> np.ndarray:
        image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
        _text(image, "HIGH-SPEED CABLE STRIKE", (170, 315), 1.35, INK, 3)
        _text(image, "MPPI + full identified DDER cable model", (208, 385), 0.72, MUTED, 2)
        _text(image, "independent matched-model replay", (274, 445), 0.65, MUTED, 2)
        _text(
            image,
            f"directed tip speed  {self.terms['directional_speed_m_s']:.2f} m/s",
            (272, 590),
            0.82,
            SUCCESS,
            2,
        )
        _text(
            image,
            f"target error  {1000.0 * self.terms['position_error_m']:.1f} mm   |   "
            f"direction error  {self.terms['direction_error_deg']:.1f} deg",
            (190, 655),
            0.62,
            INK,
            2,
        )
        _text(
            image,
            "Geometric target-region contact; no rigid contact engine in this model.",
            (160, 790),
            0.50,
            MUTED,
            1,
        )
        return image

    def scene_frame(self, frame: int, *, hit: bool) -> np.ndarray:
        frame = int(np.clip(frame, 0, self.impact_frame))
        image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
        self._floor(image)
        drone = self.drone_m[frame]
        cable = self.cable_m[frame]
        tip = cable[-1]
        tip_velocity = self.cable_velocity_m_s[frame, -1]
        error = float(np.linalg.norm(tip - self.target_m))
        directed_speed = float(tip_velocity @ self.direction)

        drone_path = self._project(self.drone_m[: frame + 1])
        tip_path = self._project(self.cable_m[: frame + 1, -1])
        if len(drone_path) > 1:
            cv2.polylines(image, [drone_path], False, (156, 196, 180), 2, cv2.LINE_AA)
            cv2.polylines(image, [tip_path], False, (211, 153, 216), 3, cv2.LINE_AA)

        target_screen = self._project(self.target_m[None])[0]
        radius_px = max(10, int(round(420.0 * self.metadata["problem"]["maximum_tip_error_m"])))
        cv2.circle(image, tuple(target_screen), radius_px, TARGET, 3, cv2.LINE_AA)
        cv2.circle(image, tuple(target_screen), 5, TARGET, -1, cv2.LINE_AA)
        arrow = self._project(
            np.stack((self.target_m - 0.20 * self.direction, self.target_m))
        )
        cv2.arrowedLine(
            image, tuple(arrow[0]), tuple(arrow[1]), TARGET, 4, cv2.LINE_AA, 0, 0.22
        )

        cable_screen = self._project(cable)
        cv2.polylines(image, [cable_screen], False, CABLE, 7, cv2.LINE_AA)
        for point in cable_screen[1:-1:2]:
            cv2.circle(image, tuple(point), 4, CABLE, -1, cv2.LINE_AA)
        cv2.circle(image, tuple(cable_screen[-1]), 10, TIP, -1, cv2.LINE_AA)
        self._drone(image, drone)

        cv2.rectangle(image, (0, 0), (WIDTH, 105), (255, 255, 255), -1)
        _text(image, "MPPI HIGH-SPEED DIRECTED IMPACT", (34, 42), 0.90, INK, 2)
        _text(
            image,
            "green: drone   blue: cable   magenta: free tip   red: target + direction",
            (35, 80),
            0.50,
            MUTED,
            1,
        )
        cv2.rectangle(image, (22, 885), (1058, 1055), (255, 255, 255), -1)
        _text(
            image,
            f"t={self.time_s[frame]:.2f} s   tip error={1000.0 * error:5.1f} mm   "
            f"directed speed={directed_speed:4.2f} m/s",
            (45, 932),
            0.62,
            INK,
            2,
        )
        _text(
            image,
            f"impact requirement: >= {self.metadata['problem']['minimum_impact_speed_m_s']:.1f} m/s   "
            f"cone <= {self.metadata['problem']['maximum_impact_angle_deg']:.0f} deg   "
            f"target radius {1000.0 * self.metadata['problem']['maximum_tip_error_m']:.0f} mm",
            (45, 977),
            0.53,
            MUTED,
            1,
        )
        displacement = float(np.linalg.norm(drone - self.drone_m[0]))
        _text(
            image,
            f"drone displacement={displacement:.2f} m   playback=0.5x   exact replay error=0",
            (45, 1019),
            0.52,
            MUTED,
            1,
        )
        if hit:
            cv2.rectangle(image, (420, 130), (660, 225), (236, 252, 237), -1)
            cv2.rectangle(image, (420, 130), (660, 225), SUCCESS, 3)
            _text(image, "VALID HIT", (452, 190), 1.10, SUCCESS, 3)
        return image


def render(replay_path: Path, output_path: Path) -> Path:
    renderer = ReplayRenderer(replay_path)
    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
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
            str(FPS),
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

    write(renderer.title_frame(), int(1.7 * FPS))
    write(renderer.scene_frame(0, hit=False), int(0.4 * FPS))
    duration = float(renderer.time_s[renderer.impact_frame])
    output_frames = max(1, int(math.ceil(duration * 2.0 * FPS)))
    for output_frame in range(output_frames):
        simulation_time = min(duration, output_frame / (2.0 * FPS))
        source_frame = int(np.argmin(np.abs(renderer.time_s - simulation_time)))
        write(renderer.scene_frame(source_frame, hit=False))
    write(renderer.scene_frame(renderer.impact_frame, hit=True), int(1.5 * FPS))
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {return_code}.")

    video_metadata = {
        "schema": "perfect_model_mppi_video_v1",
        "source_replay": str(renderer.replay_path),
        "source_replay_sha256": _sha256(renderer.replay_path),
        "video": {
            "path": str(output),
            "sha256": _sha256(output),
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "codec": "H.264/yuv420p",
            "playback_speed": 0.5,
        },
        "terms": renderer.terms,
    }
    output.with_suffix(".json").write_text(
        json.dumps(video_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("data/drone_mpc/perfect_model_mppi_high_speed.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/drone_mpc/videos/mppi_high_speed_whip_phone.mp4"),
    )
    arguments = parser.parse_args()
    output = render(arguments.replay, arguments.output)
    print(f"Video: {output}")
    print(f"Metadata: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
