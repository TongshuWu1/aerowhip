"""Live/SVO PIDNet metric-scene runtime with lossless source recording."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

from .cable_observation import CableObservationBuilder, CableObservationFrame
from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, RuntimeSettings, load_settings
from .frames import RecordingStatus, RgbdFrame
from .perception import PerceptionFrame, PerceptionRuntime
from .recording import (
    RecordingManifestWriter,
    exact_frame_timestamps_ns,
    load_recording_manifest,
    new_recording_path,
)
from .replay import validate_replay_manifest
from .scene import SceneViewerSnapshot
from .scene_viewer import AsyncOpenGlSceneViewer, ViewerAction
from .zed_source import LiveZedSource, SvoZedSource


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


class RecordingController:
    """Connect UI recording requests to the ZED-owning capture thread."""

    def __init__(
        self,
        source: LiveZedSource,
        settings: RuntimeSettings,
        perception: PerceptionRuntime,
    ) -> None:
        self.source = source
        self.settings = settings
        self.perception = perception
        self.writer: RecordingManifestWriter | None = None
        self.completed_path: Path | None = None

    @property
    def status(self) -> RecordingStatus:
        return self.source.recording_status

    def _metadata(self) -> dict[str, Any]:
        mask = self.perception.mask_config
        return {
            "zed_sdk_version": self.source.sdk_version,
            "camera_request": asdict(self.settings.camera),
            "svo": {
                "compression": self.settings.svo.compression,
                "contains_lossless_stereo": True,
                "contains_computed_depth": False,
            },
            "pidnet": {
                "runtime_config": _portable_path(
                    self.perception.runtime_config_path
                ),
                "runtime_config_sha256": self.perception.runtime_config_sha256,
                "checkpoint": _portable_path(self.perception.checkpoint_path),
                "checkpoint_sha256": self.perception.checkpoint_sha256,
                "thresholds": list(self.perception.thresholds),
                "body_cleanup": {
                    "minimum_area_px": mask.min_area_px,
                    "open_kernel": mask.open_kernel,
                    "close_kernel": mask.close_kernel,
                },
                "checkpoint_schema": self.perception.checkpoint_schema,
            },
            "observation": asdict(self.settings.observation),
            "random_seed": None,
        }

    def start(self, path: Path | None = None) -> Path:
        if self.status.active:
            raise RuntimeError("SVO recording is already active")
        output = (
            Path(path).expanduser().resolve()
            if path is not None
            else new_recording_path(self.settings.svo.recording_directory)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        writer = RecordingManifestWriter(
            output,
            self.source.descriptor,
            self._metadata(),
            PROJECT_ROOT,
        )
        self.source.start_recording(
            output,
            compression=self.settings.svo.compression,
        )
        try:
            status, timestamps = self.source.recording_snapshot
            writer.write(
                status,
                completed=False,
                frame_timestamps_ns=timestamps,
            )
        except Exception:
            self.source.stop_recording()
            raise
        self.writer = writer
        self.completed_path = None
        return output

    def stop(self, *, error: str | None = None) -> Path | None:
        writer = self.writer
        if writer is None:
            return self.completed_path
        if self.status.active:
            self.source.stop_recording()
        status, timestamps = self.source.recording_snapshot
        writer.write(
            status,
            completed=error is None,
            error=error,
            frame_timestamps_ns=timestamps,
        )
        self.completed_path = writer.svo_path
        self.writer = None
        return self.completed_path


def _recording_elapsed_s(status: RecordingStatus) -> float | None:
    if status.first_timestamp_ns is None or status.last_timestamp_ns is None:
        return None
    return max(0.0, (status.last_timestamp_ns - status.first_timestamp_ns) * 1.0e-9)


def _toggle_recording(recording: RecordingController | None) -> str:
    if recording is None:
        return "Recording is available only in live mode"
    if recording.status.active:
        path = recording.stop()
        print(f"Saved lossless SVO and manifest: {path}")
        return f"Saved {path.name}" if path is not None else "Recording stopped"
    path = recording.start()
    print(f"Recording lossless SVO: {path}")
    return f"Recording {path.name}"


def _set_recording_state(
    recording: RecordingController | None,
    *,
    active: bool,
) -> str:
    """Apply an explicit recording command from the offline controller."""

    if recording is None:
        return "Recording is available only in live mode"
    if active:
        if recording.status.active:
            return "Recording is already active"
        path = recording.start()
        print(f"Recording lossless SVO: {path}", flush=True)
        return f"Recording {path.name}"
    if not recording.status.active:
        return "Recording is already stopped"
    path = recording.stop()
    print(f"Saved lossless SVO and manifest: {path}", flush=True)
    return f"Saved {path.name}" if path is not None else "Recording stopped"


def _read_stdin_controls(commands: queue.SimpleQueue[str]) -> None:
    """Read the private offline-controller protocol without blocking tracking."""

    try:
        for line in sys.stdin:
            command = line.strip()
            if command:
                commands.put(command)
    finally:
        # Losing the controller also closes the live application through its
        # normal cleanup path, which finalizes an active SVO and manifest.
        commands.put("quit")


def _poll_stdin_controls(
    commands: queue.SimpleQueue[str] | None,
    recording: RecordingController | None,
) -> tuple[bool, str | None]:
    if commands is None:
        return False, None
    quit_requested = False
    message: str | None = None
    while True:
        try:
            command = commands.get_nowait()
        except queue.Empty:
            break
        if command == "record_start":
            message = _set_recording_state(recording, active=True)
        elif command == "record_stop":
            message = _set_recording_state(recording, active=False)
        elif command == "quit":
            quit_requested = True
        else:
            raise ValueError(f"Unknown stdin control command: {command!r}")
    return quit_requested, message


def _viewer_snapshot(
    perception: PerceptionFrame,
    observation: CableObservationFrame,
    *,
    source: LiveZedSource | SvoZedSource,
    point_cloud_stride: int,
    pipeline_fps: float | None,
    paused: bool,
    message: str,
) -> SceneViewerSnapshot:
    frame = perception.rgbd
    status = source.recording_status
    total_frames = source.descriptor.total_frames
    position = f"{frame.key.source_position}"
    if total_frames is not None:
        position = f"{frame.key.source_position + 1}/{total_frames}"
    pipeline_text = "--" if pipeline_fps is None else f"{pipeline_fps:.1f}"
    latency_ms = max(0.0, (time.perf_counter_ns() - frame.host_received_ns) / 1.0e6)
    if status.active:
        elapsed_s = _recording_elapsed_s(status)
        state = "RECORDING" if elapsed_s is None else f"RECORDING {elapsed_s:.1f}s"
    else:
        state = "PAUSED" if paused else "RUNNING"
    status_lines = (
        f"{source.descriptor.kind.upper()} {source.descriptor.label} | "
        f"frame={frame.key.sequence_index} source={position} | "
        f"timestamp_ns={frame.key.timestamp_ns}",
        f"PIDNet={perception.timing.total_ms:.1f}ms | pipeline={pipeline_text}fps | "
        f"latency={latency_ms:.1f}ms | source_replaced={source.replaced_count}",
        f"observation={observation.timing.total_ms:.1f}ms | "
        f"edges={len(observation.graph_edges)} "
        f"crossings={len(observation.crossings)} "
        f"routes={observation.timing.route_count} | "
        f"ends={len(observation.endpoints)}",
        f"{state} | {message}",
    )
    return SceneViewerSnapshot.from_rgbd(
        frame,
        source.descriptor.calibration,
        sample_stride_px=point_cloud_stride,
        masks_u8=perception.masks_u8,
        observation=observation,
        geometry=None,
        status_lines=status_lines,
    )


def _snapshot_with_playback_state(
    snapshot: SceneViewerSnapshot,
    *,
    paused: bool,
    message: str,
) -> SceneViewerSnapshot:
    """Change only the viewer status while preserving the frozen SVO frame."""

    if not snapshot.status_lines:
        raise RuntimeError("Viewer snapshot is missing its runtime status")
    state = "PAUSED" if paused else "RUNNING"
    return replace(
        snapshot,
        status_lines=(*snapshot.status_lines[:-1], f"{state} | {message}"),
    )


def _pace_svo(
    frame: RgbdFrame,
    first_timestamp_ns: int,
    playback_started_ns: int,
) -> None:
    target_ns = playback_started_ns + frame.key.timestamp_ns - first_timestamp_ns
    remaining_s = (target_ns - time.perf_counter_ns()) * 1.0e-9
    if remaining_s > 0.0:
        time.sleep(remaining_s)


def run(args: argparse.Namespace) -> None:
    settings = load_settings(Path(args.config).expanduser().resolve())
    perception = PerceptionRuntime(settings.pidnet.runtime_config)
    observation_builder = CableObservationBuilder(settings.observation)
    warmup = perception.warmup_1080p()
    print(
        "PIDNet ready "
        f"device={perception.device} "
        f"checkpoint={perception.checkpoint_sha256[:12]} "
        f"thresholds={perception.thresholds} "
        f"warmup={warmup.total_ms:.1f}ms"
    )

    viewer_enabled = bool(settings.viewer.enabled and not args.headless)
    viewer = (
        AsyncOpenGlSceneViewer(
            window_name=settings.viewer.window_name,
            width_px=settings.viewer.width_px,
            height_px=settings.viewer.height_px,
            depth_min_m=settings.viewer.depth_min_m,
            depth_max_m=settings.viewer.depth_max_m,
            overlay_alpha=settings.viewer.overlay_alpha,
            point_cloud_stride=settings.viewer.point_cloud_stride,
            point_size_px=settings.viewer.point_size_px,
            inset_width_px=settings.viewer.inset_width_px,
            render_fps=settings.viewer.render_fps,
        )
        if viewer_enabled
        else None
    )

    source: LiveZedSource | SvoZedSource | None = None
    try:
        if args.svo is None:
            source = LiveZedSource(settings.camera)
        else:
            svo_path = Path(args.svo).expanduser().resolve()
            manifest = load_recording_manifest(svo_path)
            exact_timestamps = exact_frame_timestamps_ns(manifest)
            source = SvoZedSource(
                svo_path,
                settings.camera,
                playback_realtime=False,
                exact_timestamps_ns=exact_timestamps,
            )
            validate_replay_manifest(
                manifest,
                settings=settings,
                perception=perception,
                source=source,
            )
            print(
                "Replay manifest validated "
                f"frames={source.descriptor.total_frames} "
                f"sdk={source.sdk_version} exact_timestamps=yes"
            )
    except BaseException:
        try:
            if source is not None:
                source.close()
        finally:
            if viewer is not None:
                viewer.close()
        raise
    assert source is not None
    recording = (
        RecordingController(source, settings, perception)
        if isinstance(source, LiveZedSource)
        else None
    )
    if args.record_to is not None and recording is None:
        try:
            if viewer is not None:
                viewer.close()
        finally:
            source.close()
        raise ValueError("--record-to is valid only with a live ZED source")

    processed = 0
    fps_value: float | None = None
    fps_count = 0
    fps_started = time.perf_counter()
    next_status = fps_started
    paused = False
    step_once = False
    last_snapshot: SceneViewerSnapshot | None = None
    message = "Q quit | R record" if recording is not None else "Q quit | Space pause | N step"
    first_svo_timestamp_ns: int | None = None
    playback_started_ns: int | None = None
    redraw_snapshot = False
    primary_error: BaseException | None = None
    control_commands: queue.SimpleQueue[str] | None = None
    if args.control_stdin:
        control_commands = queue.SimpleQueue()
        threading.Thread(
            target=_read_stdin_controls,
            args=(control_commands,),
            name="offline-controller-stdin",
            daemon=True,
        ).start()

    try:
        if args.record_to is not None:
            path = recording.start(Path(args.record_to))
            print(f"Recording lossless SVO: {path}")

        while True:
            quit_requested, control_message = _poll_stdin_controls(
                control_commands,
                recording,
            )
            if control_message is not None:
                message = control_message
                if paused and last_snapshot is not None:
                    last_snapshot = _snapshot_with_playback_state(
                        last_snapshot,
                        paused=True,
                        message=message,
                    )
                    redraw_snapshot = True
            if quit_requested:
                break

            if paused and not step_once:
                if viewer is None or last_snapshot is None:
                    raise RuntimeError("SVO pause requires an active viewer snapshot")
                if redraw_snapshot:
                    action = viewer.show(last_snapshot)
                    redraw_snapshot = False
                else:
                    action = viewer.poll()
                if action is ViewerAction.QUIT:
                    break
                if action is ViewerAction.TOGGLE_RECORDING:
                    message = _toggle_recording(recording)
                    last_snapshot = _snapshot_with_playback_state(
                        last_snapshot,
                        paused=True,
                        message=message,
                    )
                    redraw_snapshot = True
                elif action is ViewerAction.TOGGLE_PAUSE:
                    paused = False
                    last_snapshot = _snapshot_with_playback_state(
                        last_snapshot,
                        paused=False,
                        message=message,
                    )
                    if (
                        first_svo_timestamp_ns is not None
                        and playback_started_ns is not None
                    ):
                        playback_started_ns = time.perf_counter_ns() - (
                            last_snapshot.key.timestamp_ns - first_svo_timestamp_ns
                        )
                elif action is ViewerAction.STEP:
                    step_once = True
                time.sleep(0.005)
                continue

            timeout_s = 0.10 if source.descriptor.kind == "live" else None
            stepping = step_once
            step_once = False
            frame = source.read(timeout_s=timeout_s)
            if frame is None:
                if source.descriptor.kind == "svo":
                    print("Reached end of SVO")
                    break
                if viewer is not None and last_snapshot is not None:
                    action = viewer.poll()
                    if action is ViewerAction.QUIT:
                        break
                    if action is ViewerAction.TOGGLE_RECORDING:
                        message = _toggle_recording(recording)
                continue

            if source.descriptor.kind == "svo" and (
                args.realtime or settings.svo.playback_realtime
            ):
                if first_svo_timestamp_ns is None:
                    first_svo_timestamp_ns = frame.key.timestamp_ns
                    playback_started_ns = time.perf_counter_ns()
                _pace_svo(frame, first_svo_timestamp_ns, int(playback_started_ns))

            result = perception.infer(frame)
            observation = observation_builder.build(
                result,
                source.descriptor.calibration,
            )
            processed += 1
            fps_count += 1
            now = time.perf_counter()
            elapsed = now - fps_started
            if elapsed >= 0.5:
                fps_value = fps_count / elapsed
                fps_count = 0
                fps_started = now

            if viewer is not None:
                last_snapshot = _viewer_snapshot(
                    result,
                    observation,
                    source=source,
                    point_cloud_stride=settings.viewer.point_cloud_stride,
                    pipeline_fps=fps_value,
                    paused=paused,
                    message=message,
                )
                action = viewer.show(last_snapshot)
            else:
                action = ViewerAction.NONE
            if action is ViewerAction.QUIT:
                break
            if action is ViewerAction.TOGGLE_RECORDING:
                message = _toggle_recording(recording)
            elif action is ViewerAction.TOGGLE_PAUSE:
                if source.descriptor.kind == "svo":
                    paused = not paused
                    last_snapshot = _snapshot_with_playback_state(
                        last_snapshot,
                        paused=paused,
                        message=message,
                    )
                    redraw_snapshot = paused
                    if (
                        not paused
                        and first_svo_timestamp_ns is not None
                        and playback_started_ns is not None
                    ):
                        playback_started_ns = time.perf_counter_ns() - (
                            last_snapshot.key.timestamp_ns - first_svo_timestamp_ns
                        )
                else:
                    message = "Pause applies only to SVO playback"
            elif action is ViewerAction.STEP and source.descriptor.kind == "svo":
                if paused:
                    step_once = True

            if stepping and action is not ViewerAction.TOGGLE_PAUSE:
                paused = True
                if last_snapshot is not None:
                    last_snapshot = _snapshot_with_playback_state(
                        last_snapshot,
                        paused=True,
                        message=message,
                    )
                    redraw_snapshot = True

            if now >= next_status:
                print(
                    f"frame={frame.key.sequence_index} "
                    f"source={frame.key.source_position} "
                    f"timestamp_ns={frame.key.timestamp_ns} "
                    f"pidnet={result.timing.total_ms:.1f}ms "
                    f"observation={observation.timing.total_ms:.1f}ms "
                    f"edges={len(observation.graph_edges)} "
                    f"routes={observation.timing.route_count} "
                    f"crossings={len(observation.crossings)} "
                    f"fps={(fps_value or 0.0):.1f} "
                    f"replaced={source.replaced_count}"
                )
                next_status = now + settings.status_period_s

            if args.max_frames is not None and processed >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("Stopping on keyboard interrupt")
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        # On a runtime failure, first join the source thread so any active SVO
        # is flushed and its final status is stable before writing the manifest.
        if primary_error is not None:
            try:
                source.close()
            except Exception as error:
                cleanup_errors.append(error)
        if recording is not None and recording.writer is not None:
            try:
                path = recording.stop(
                    error=(
                        None
                        if primary_error is None
                        else f"{type(primary_error).__name__}: {primary_error}"
                    )
                )
                print(f"Saved lossless SVO and manifest: {path}")
            except Exception as error:  # Preserve capture cleanup, then re-raise.
                cleanup_errors.append(error)
        if viewer is not None:
            try:
                viewer.close()
            except Exception as error:
                cleanup_errors.append(error)
        try:
            source.close()
        except Exception as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            if primary_error is not None:
                for error in cleanup_errors:
                    print(
                        f"Cleanup error after {type(primary_error).__name__}: {error}",
                        file=sys.stderr,
                    )
            else:
                raise cleanup_errors[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronized ZED RGB-D and PIDNet metric-scene runtime",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Runtime TOML configuration",
    )
    parser.add_argument(
        "--svo",
        type=Path,
        help="Read every frame from this SVO/SVO2 instead of the live camera",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the identical source and PIDNet path without a window",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace deterministic SVO playback by its camera timestamps",
    )
    parser.add_argument(
        "--record-to",
        type=Path,
        help="Begin live lossless SVO recording immediately at this new path",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many processed frames (diagnostic use)",
    )
    parser.add_argument(
        "--control-stdin",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()
    if arguments.max_frames is not None and arguments.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if arguments.svo is not None and arguments.record_to is not None:
        parser.error("--record-to cannot be combined with --svo")
    return arguments


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
