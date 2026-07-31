"""Run the isolated yellow-cube RGB-D tracking experiment on a live ZED."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import shutil
import sys
import time
import tomllib

import cv2
import numpy as np
import pyzed.sl as sl


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MAIN_SOURCE_DIR = PROJECT_DIR / "ZED_segmentation_viewer" / "source"
if str(MAIN_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_SOURCE_DIR))

from cube_tracker import (
    CameraModel,
    CubeTrackerConfig,
    CubeTrackingResult,
    KnownCubeTracker,
    draw_tracking_result,
)


RESOLUTIONS = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA,
}
DEPTH_MODES = {
    "NEURAL": sl.DEPTH_MODE.NEURAL,
    "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
    "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
    "ULTRA": sl.DEPTH_MODE.ULTRA,
    "QUALITY": sl.DEPTH_MODE.QUALITY,
    "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
}
CSV_FIELDS = (
    "frame",
    "zed_timestamp_ns",
    "valid",
    "zed_x_m",
    "zed_y_m",
    "zed_z_m",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "quaternion_w",
    "refinement_valid",
    "refined_zed_x_m",
    "refined_zed_y_m",
    "refined_zed_z_m",
    "refined_quaternion_x",
    "refined_quaternion_y",
    "refined_quaternion_z",
    "refined_quaternion_w",
    "face_count",
    "candidate_plane_count",
    "surface_rms_mm",
    "refined_surface_rms_mm",
    "position_std_x_mm",
    "position_std_y_mm",
    "position_std_z_mm",
    "rotation_std_x_deg",
    "rotation_std_y_deg",
    "rotation_std_z_deg",
    "observed_span_mm",
    "yellow_pixels",
    "valid_depth_points",
    "depth_coverage",
    "refinement_ms",
    "processing_ms",
    "reason",
    "refinement_reason",
)
WINDOW_NAME = "RGB-D cube tracking tester"


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    for section in ("camera", "cube", "yellow", "fit", "output"):
        if not isinstance(values.get(section), dict):
            raise ValueError(f"Configuration is missing [{section}].")
    return values


def tracker_config_from_mapping(values: dict) -> CubeTrackerConfig:
    cube = values["cube"]
    combined = {
        **values["yellow"],
        **values["fit"],
        "cube_side_m": cube["side_m"],
    }
    return CubeTrackerConfig.from_mapping(combined)


def open_zed(camera_config: dict) -> tuple[sl.Camera, sl.RuntimeParameters]:
    resolution_name = str(camera_config["resolution"]).upper()
    depth_name = str(camera_config["depth_mode"]).upper()
    if resolution_name not in RESOLUTIONS:
        raise ValueError(f"Unsupported ZED resolution: {resolution_name}")
    if depth_name not in DEPTH_MODES:
        raise ValueError(f"Unsupported ZED depth mode: {depth_name}")

    initialization = sl.InitParameters()
    initialization.camera_resolution = RESOLUTIONS[resolution_name]
    initialization.camera_fps = int(camera_config["fps"])
    initialization.depth_mode = DEPTH_MODES[depth_name]
    initialization.coordinate_units = sl.UNIT.METER
    initialization.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    zed = sl.Camera()
    status = zed.open(initialization)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"Could not open ZED camera: {status}. Close the cable tracker, "
            "ZED Explorer, and any other application using the camera."
        )

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = int(camera_config["confidence"])
    runtime.texture_confidence_threshold = int(camera_config["texture_confidence"])
    runtime.remove_saturated_areas = False
    if hasattr(runtime, "enable_fill_mode"):
        runtime.enable_fill_mode = bool(camera_config["fill"])
    return zed, runtime


def camera_model_from_zed(zed: sl.Camera) -> CameraModel:
    information = zed.get_camera_information()
    resolution = information.camera_configuration.resolution
    left = information.camera_configuration.calibration_parameters.left_cam
    return CameraModel(
        fx=float(left.fx),
        fy=float(left.fy),
        cx=float(left.cx),
        cy=float(left.cy),
        width=int(resolution.width),
        height=int(resolution.height),
    )


def resolve_output_directory(value: str) -> Path:
    configured = Path(value)
    if not configured.is_absolute():
        configured = PROJECT_DIR / configured
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = configured.resolve() / run_name
    output.mkdir(parents=True, exist_ok=False)
    return output


def finite_or_blank(value: float) -> float | str:
    return float(value) if math.isfinite(float(value)) else ""


def csv_row(
    frame_index: int,
    timestamp_ns: int,
    result: CubeTrackingResult,
) -> dict[str, int | float | str]:
    if result.valid:
        assert result.center_m is not None
        assert result.quaternion_xyzw is not None
        center = result.center_m
        quaternion = result.quaternion_xyzw
    else:
        center = np.full(3, np.nan, dtype=np.float64)
        quaternion = np.full(4, np.nan, dtype=np.float64)
    if result.refinement_valid:
        assert result.refined_center_m is not None
        assert result.refined_quaternion_xyzw is not None
        assert result.pose_covariance is not None
        refined_center = result.refined_center_m
        refined_quaternion = result.refined_quaternion_xyzw
        pose_standard_deviation = np.sqrt(
            np.maximum(np.diag(result.pose_covariance), 0.0)
        )
    else:
        refined_center = np.full(3, np.nan, dtype=np.float64)
        refined_quaternion = np.full(4, np.nan, dtype=np.float64)
        pose_standard_deviation = np.full(6, np.nan, dtype=np.float64)
    return {
        "frame": frame_index,
        "zed_timestamp_ns": timestamp_ns,
        "valid": int(result.valid),
        "zed_x_m": finite_or_blank(center[0]),
        "zed_y_m": finite_or_blank(center[1]),
        "zed_z_m": finite_or_blank(center[2]),
        "quaternion_x": finite_or_blank(quaternion[0]),
        "quaternion_y": finite_or_blank(quaternion[1]),
        "quaternion_z": finite_or_blank(quaternion[2]),
        "quaternion_w": finite_or_blank(quaternion[3]),
        "refinement_valid": int(result.refinement_valid),
        "refined_zed_x_m": finite_or_blank(refined_center[0]),
        "refined_zed_y_m": finite_or_blank(refined_center[1]),
        "refined_zed_z_m": finite_or_blank(refined_center[2]),
        "refined_quaternion_x": finite_or_blank(refined_quaternion[0]),
        "refined_quaternion_y": finite_or_blank(refined_quaternion[1]),
        "refined_quaternion_z": finite_or_blank(refined_quaternion[2]),
        "refined_quaternion_w": finite_or_blank(refined_quaternion[3]),
        "face_count": result.face_count,
        "candidate_plane_count": result.candidate_plane_count,
        "surface_rms_mm": finite_or_blank(result.surface_rms_m * 1000.0),
        "refined_surface_rms_mm": finite_or_blank(
            result.refined_surface_rms_m * 1000.0
        ),
        "position_std_x_mm": finite_or_blank(pose_standard_deviation[0] * 1000.0),
        "position_std_y_mm": finite_or_blank(pose_standard_deviation[1] * 1000.0),
        "position_std_z_mm": finite_or_blank(pose_standard_deviation[2] * 1000.0),
        "rotation_std_x_deg": finite_or_blank(
            np.rad2deg(pose_standard_deviation[3])
        ),
        "rotation_std_y_deg": finite_or_blank(
            np.rad2deg(pose_standard_deviation[4])
        ),
        "rotation_std_z_deg": finite_or_blank(
            np.rad2deg(pose_standard_deviation[5])
        ),
        "observed_span_mm": finite_or_blank(result.observed_span_m * 1000.0),
        "yellow_pixels": result.yellow_pixels,
        "valid_depth_points": result.valid_depth_points,
        "depth_coverage": result.depth_coverage,
        "refinement_ms": result.refinement_ms,
        "processing_ms": result.processing_ms,
        "reason": result.reason,
        "refinement_reason": result.refinement_reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "config.toml",
        help="Tester TOML configuration.",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="Exit automatically after this many seconds; zero runs until Q/Esc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_seconds < 0.0:
        raise ValueError("--run-seconds cannot be negative.")
    config_path = args.config.resolve()
    values = load_config(config_path)
    tracker_config = tracker_config_from_mapping(values)
    tracker_config.validate()
    output_directory = resolve_output_directory(str(values["output"]["directory"]))
    shutil.copy2(config_path, output_directory / "config.toml")
    csv_path = output_directory / "poses.csv"
    last_frame_path = output_directory / "last_frame.png"

    zed: sl.Camera | None = None
    image = sl.Mat()
    depth = sl.Mat()
    last_display: np.ndarray | None = None
    valid_frames = 0
    total_frames = 0
    try:
        print("Opening ZED camera...")
        zed, runtime = open_zed(values["camera"])
        camera = camera_model_from_zed(zed)
        tracker = KnownCubeTracker(camera, tracker_config)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(camera.width, 1280), min(camera.height, 720))
        deadline = (
            time.perf_counter() + args.run_seconds
            if args.run_seconds > 0.0
            else float("inf")
        )

        with csv_path.open("w", newline="", encoding="utf-8", buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            while time.perf_counter() < deadline:
                status = zed.grab(runtime)
                if status != sl.ERROR_CODE.SUCCESS:
                    if status == sl.ERROR_CODE.CAMERA_REBOOTING:
                        continue
                    raise RuntimeError(f"ZED grab failed: {status}")

                image_status = zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
                depth_status = zed.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU)
                if (
                    image_status != sl.ERROR_CODE.SUCCESS
                    or depth_status != sl.ERROR_CODE.SUCCESS
                ):
                    raise RuntimeError(
                        f"ZED retrieval failed: image={image_status}, depth={depth_status}"
                    )

                bgra = np.asarray(image.get_data())
                bgr = np.ascontiguousarray(bgra[:, :, :3])
                depth_array = np.asarray(depth.get_data(), dtype=np.float32).squeeze()
                timestamp_ns = int(
                    zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                )
                result = tracker.track(
                    bgr,
                    depth_array,
                    include_face_pixels=True,
                )
                last_display = draw_tracking_result(
                    bgr,
                    result,
                    camera,
                    tracker_config.cube_side_m,
                )
                total_frames += 1
                valid_frames += int(result.valid)
                writer.writerow(csv_row(total_frames, timestamp_ns, result))

                cv2.imshow(WINDOW_NAME, last_display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    snapshot_path = output_directory / f"snapshot_{total_frames:06d}.png"
                    cv2.imwrite(str(snapshot_path), last_display)
                    print(f"Saved snapshot: {snapshot_path}")
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

        if last_display is not None:
            cv2.imwrite(str(last_frame_path), last_display)
    finally:
        try:
            image.free()
            depth.free()
        finally:
            if zed is not None:
                zed.close()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    valid_percentage = 100.0 * valid_frames / total_frames if total_frames else 0.0
    print(
        f"CUBE_TRACKING_SAVED frames={total_frames} valid={valid_percentage:.1f}% "
        f"csv={csv_path} last_frame={last_frame_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
