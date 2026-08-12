"""Offline PIDNet extraction under a fixed image-plane motion assumption."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_CONFIG_PATH, load_settings
from .planar_data import (
    PLANAR_OBSERVATION_METHOD,
    PlanarSequenceWriter,
)
from ..shared.config import load_settings as load_observation_settings
from ..shared.diagnostics import diagnostic_panels
from ..shared.pidnet_runtime import PidnetRuntime
from ..shared.planar_observer import PlanarCableObserver
from ..shared.zed_source import ZedStereoSource


def _source_metadata(source: ZedStereoSource) -> dict[str, Any]:
    calibration = source.descriptor.calibration
    return {
        "kind": source.descriptor.kind,
        "path": None if source.descriptor.source_path is None else str(source.descriptor.source_path),
        "width_px": calibration.width_px,
        "height_px": calibration.height_px,
        "fps": calibration.fps,
        "serial_number": calibration.serial_number,
        "camera_model": calibration.camera_model,
    }


def _show(frame_bgr: np.ndarray, observed, frame_index: int, total: int | None) -> bool:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - interactive runtime
        raise RuntimeError("OpenCV is required for the 2D extraction view.") from error
    segmentation_rgb, skeleton_rgb = diagnostic_panels(
        frame_bgr,
        observed.view,
        observed.skeleton,
    )
    left = np.ascontiguousarray(segmentation_rgb[:, :, ::-1])
    right = np.ascontiguousarray(skeleton_rgb[:, :, ::-1])
    panel = np.hstack((left, right))
    state = "COMPLETE" if observed.complete else str(observed.view.failure or "INCOMPLETE")
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 35), (10, 18, 28), -1)
    cv2.putText(
        panel,
        f"frame {frame_index}/{total or '?'} | {state} | Q or ESC closes display",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (90, 235, 130) if observed.complete else (50, 180, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("2D PIDNet centerline extraction", panel)
    key = cv2.waitKey(1) & 0xFF
    return key in (27, ord("q"), ord("Q"))


def analyze_svo(
    svo_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    cable_identity: int | None = None,
    replace_existing: bool = False,
    headless: bool = True,
    max_frames: int | None = None,
) -> Path:
    arguments = argparse.Namespace(
        svo=Path(svo_path),
        output=Path(output_path),
        config=Path(config_path),
        cable=cable_identity,
        replace_output=replace_existing,
        headless=headless,
        max_frames=max_frames,
    )
    return run(arguments)


def run(arguments: argparse.Namespace) -> Path:
    offline = load_settings(arguments.config)
    observation_settings = load_observation_settings()
    if arguments.cable is not None:
        observation_settings = replace(
            observation_settings,
            cable_identity=int(arguments.cable),
        )
    output = Path(arguments.output).expanduser().resolve()
    source: ZedStereoSource | None = None
    processed = 0
    complete_count = 0
    writer: PlanarSequenceWriter | None = None
    finished = False
    try:
        pidnet = PidnetRuntime(observation_settings.pidnet_runtime_config)
        print(f"PIDNet CUDA warm-up {pidnet.warm_up(1080, 1920):.1f} ms.", flush=True)
        source = ZedStereoSource(arguments.svo, enable_depth=False)
        observer = PlanarCableObserver(
            observation_settings,
            pidnet=pidnet,
        )
        metadata = {
            "method": PLANAR_OBSERVATION_METHOD,
            "cable_identity": observation_settings.cable_identity,
            "source": _source_metadata(source),
            "observation_assumption": "fixed_level_camera_cable_motion_parallel_to_image_plane",
            "pidnet": pidnet.identity,
            "route_samples": observation_settings.route.dense_samples,
        }
        writer = PlanarSequenceWriter(
            output,
            metadata=metadata,
            route_samples=observation_settings.route.dense_samples,
            cable_length_m=offline.cable.length_m,
            image_width_px=source.descriptor.calibration.width_px,
            image_height_px=source.descriptor.calibration.height_px,
            replace_existing=arguments.replace_output,
        )
        total = source.descriptor.total_frames
        print(
            f"2D PIDNet centerline extraction: {source.descriptor.label} | "
            f"cable {observation_settings.cable_identity}",
            flush=True,
        )
        while arguments.max_frames is None or processed < arguments.max_frames:
            frame = source.read()
            if frame is None:
                break
            observed = observer.process(frame)
            writer.append(frame, observed)
            processed += 1
            complete_count += int(observed.complete)
            if not arguments.headless and _show(frame.left_bgr, observed, processed, total):
                arguments.headless = True
                try:
                    import cv2
                    cv2.destroyAllWindows()
                except ImportError:
                    pass
            if processed == 1 or processed % 30 == 0:
                print(f"PROGRESS {processed} {total or 0}", flush=True)
                print(
                    f"planar frame={frame.source_position} "
                    f"pidnet={observed.pidnet.inference_ms:.1f}ms "
                    f"route={observed.timings_ms['route_ms']:.1f}ms "
                    f"complete={int(observed.complete)}",
                    flush=True,
                )
        if arguments.max_frames is None and total is not None and processed != total:
            raise RuntimeError(f"SVO analysis ended early: {processed}/{total} frames.")
        finished = True
    finally:
        if source is not None:
            source.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass
    if not finished or writer is None:
        raise RuntimeError("2D PIDNet centerline extraction did not complete.")
    saved = writer.close()
    print(
        f"Saved {complete_count}/{processed} complete image-plane observations: {saved}",
        flush=True,
    )
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract metric-scaled image-plane PIDNet curves.")
    parser.add_argument("--svo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cable", type=int, choices=(1, 2), default=None)
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
