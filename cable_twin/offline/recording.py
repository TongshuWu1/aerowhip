from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from ..shared.capture_manifest import build_capture_manifest, write_capture_manifest
from ..shared.zed_source import CameraCaptureSettings
from ..shared.zed_source import ZedStereoSource


EVENT_PREFIX = "OFFLINE_DDER_EVENT "


def _emit(event: str, **values: Any) -> None:
    print(
        EVENT_PREFIX + json.dumps({"event": event, **values}, separators=(",", ":")),
        flush=True,
    )


def _capture_values(settings: CameraCaptureSettings) -> dict[str, Any]:
    return {
        "automatic": settings.automatic_exposure_gain,
        "exposure": settings.exposure,
        "gain": settings.gain,
    }


def _next_command(
    path: Path,
    previous_revision: int,
) -> tuple[dict[str, Any] | None, int]:
    if not path.is_file():
        return None, previous_revision
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Offline capture command must be a JSON object.")
    revision = int(payload.get("revision", -1))
    if revision <= previous_revision:
        return None, previous_revision
    return payload, revision


def _publish_completed_recording(partial_path: Path, final_path: Path) -> None:
    """Atomically publish a cleanly finalized SVO without overwriting."""

    if not partial_path.is_file():
        raise FileNotFoundError(f"Finalized SVO was not created: {partial_path}")
    try:
        os.link(partial_path, final_path)
    except FileExistsError:
        raise FileExistsError(f"Refusing to overwrite SVO recording: {final_path}") from None
    partial_path.unlink()
    try:
        partial_path.parent.rmdir()
    except OSError:
        pass


def run(
    control_file: Path,
    *,
    exit_on_stdin_close: bool = False,
    manual_exposure: int | None = None,
    manual_gain: int | None = None,
) -> int:
    """Record raw SVO2 while a display-only RGB view provides guidance."""

    control_file = control_file.expanduser().resolve()
    revision = -1
    observation_path: Path | None = None
    svo_path: Path | None = None
    partial_svo_path: Path | None = None
    cable_identity: int | None = None
    source: ZedStereoSource | None = None
    normal_exit = False
    controller_input: int | None = None
    if exit_on_stdin_close:
        controller_input = sys.stdin.fileno()
        os.set_blocking(controller_input, False)

    def controller_is_closed() -> bool:
        if controller_input is None:
            return False
        try:
            return os.read(controller_input, 1) == b""
        except BlockingIOError:
            return False

    def finalize_recording() -> None:
        nonlocal svo_path, partial_svo_path, observation_path, cable_identity
        if source is None or not source.is_recording:
            return
        source.stop_recording()
        frame_count = source.last_recording_frame_count
        if frame_count < 1:
            raise RuntimeError("The recording contains no synchronized frames.")
        if partial_svo_path is None or svo_path is None or observation_path is None:
            raise RuntimeError("Recording paths were not initialized.")
        if cable_identity is None:
            raise RuntimeError("Recording experiment metadata was not initialized.")
        manifest = build_capture_manifest(
            source,
            svo_path,
            frame_count,
            cable_identity=cable_identity,
        )
        _publish_completed_recording(partial_svo_path, svo_path)
        manifest_path = write_capture_manifest(svo_path, manifest)
        _emit(
            "capture_saved",
            svo_path=str(svo_path),
            observation_path=str(observation_path),
            capture_manifest_path=str(manifest_path),
            frame_count=frame_count,
        )
        svo_path = None
        partial_svo_path = None
        observation_path = None
        cable_identity = None

    try:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required for the planar recorder view.") from error
        source = ZedStereoSource(
            enable_depth=False,
            manual_exposure=manual_exposure,
            manual_gain=manual_gain,
        )
        capture_settings = source.capture_settings
        if capture_settings is None:
            raise RuntimeError("Live ZED capture settings were not available.")
        mode = "manual" if capture_settings.requested_manual else "automatic"
        print(
            "ZED capture settings (SDK readback): "
            f"mode={mode} exposure={capture_settings.exposure} "
            f"gain={capture_settings.gain} "
            f"aec_agc={int(capture_settings.automatic_exposure_gain)}",
            flush=True,
        )
        print(
            "Raw lossless recorder ready: RGB display only; PIDNet and planar "
            "2D extraction run after capture.",
            flush=True,
        )
        _emit("ready", **_capture_values(capture_settings))
        exit_requested = False
        controller_closed = False
        window_name = "Planar cable recording - RGB"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        while not exit_requested and not controller_closed:
            controller_closed = controller_is_closed()
            if controller_closed:
                break
            command, revision = _next_command(control_file, revision)
            if command is not None:
                action = str(command.get("action", ""))
                if action == "start_recording":
                    if source.is_recording:
                        raise RuntimeError("A lossless recording is already active.")
                    svo_path = Path(str(command["svo_path"])).expanduser().resolve()
                    if svo_path.exists():
                        raise FileExistsError(f"Refusing to overwrite SVO recording: {svo_path}")
                    partial_svo_path = svo_path.parent / ".partial" / svo_path.name
                    if partial_svo_path.exists():
                        raise FileExistsError(
                            f"An incomplete recording already exists: {partial_svo_path}"
                        )
                    observation_path = Path(
                        str(command["observation_path"])
                    ).expanduser().resolve()
                    cable_identity = int(command["cable_identity"])
                    if observation_path.exists():
                        raise FileExistsError(
                            f"Refusing to replace existing 2D observations: {observation_path}"
                        )
                    source.start_recording(partial_svo_path)
                    _emit(
                        "recording_started",
                        svo_path=str(svo_path),
                        observation_path=str(observation_path),
                    )
                elif action in ("stop_recording", "exit"):
                    if source.is_recording:
                        finalize_recording()
                    exit_requested = action == "exit"
                elif action == "set_camera":
                    mode = str(command.get("mode", ""))
                    if mode == "manual":
                        applied = source.set_manual_exposure_gain(
                            int(command["exposure"]),
                            int(command["gain"]),
                        )
                    elif mode == "automatic":
                        applied = source.set_automatic_exposure_gain()
                    else:
                        raise ValueError(f"Unknown camera mode {mode!r}.")
                    print(
                        "ZED capture settings applied: "
                        f"mode={'automatic' if applied.automatic_exposure_gain else 'manual'} "
                        f"exposure={applied.exposure} gain={applied.gain}",
                        flush=True,
                    )
                    _emit("capture_settings", **_capture_values(applied))
                else:
                    raise ValueError(f"Unknown offline capture action: {action!r}.")

            if exit_requested:
                break
            # Acquisition/encoding owns its own full-rate thread. This loop sees
            # only the newest frame, so display work cannot pace the SVO encoder.
            frame = source.read()
            if frame is None:
                raise RuntimeError("Live ZED capture ended unexpectedly.")
            display_width = 1280
            display_height = int(round(display_width * frame.left_bgr.shape[0] / frame.left_bgr.shape[1]))
            panel = cv2.resize(
                frame.left_bgr,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA,
            )
            cv2.rectangle(panel, (0, 0), (display_width, 48), (10, 18, 28), -1)
            cv2.putText(
                panel,
                "RECORDING RAW SVO2" if source.is_recording else "LIVE RGB - DISPLAY ONLY",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (50, 80, 255) if source.is_recording else (235, 240, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                exit_requested = True
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                exit_requested = True
        if controller_closed:
            print("Offline DDER controller closed; stopping the raw recorder.", flush=True)
        normal_exit = True
        return 0
    finally:
        finalization_error: BaseException | None = None
        if source is not None:
            try:
                if source.is_recording:
                    if normal_exit:
                        finalize_recording()
                    else:
                        source.stop_recording()
                        print(f"Incomplete recording retained at {partial_svo_path}", flush=True)
            except BaseException as error:
                finalization_error = error
            finally:
                source.close()
        if not normal_exit and partial_svo_path is not None and partial_svo_path.is_file():
            print(f"Incomplete recording retained at {partial_svo_path}", flush=True)
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass
        if finalization_error is not None:
            raise finalization_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-rate lossless ZED recorder with a display-only RGB viewport."
    )
    parser.add_argument("--control-file", type=Path, required=True)
    parser.add_argument("--exit-on-stdin-close", action="store_true")
    parser.add_argument("--manual-exposure", type=int, default=None)
    parser.add_argument("--manual-gain", type=int, default=None)
    arguments = parser.parse_args()
    raise SystemExit(
        run(
            arguments.control_file,
            exit_on_stdin_close=arguments.exit_on_stdin_close,
            manual_exposure=arguments.manual_exposure,
            manual_gain=arguments.manual_gain,
        )
    )


if __name__ == "__main__":
    main()
