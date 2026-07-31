"""Live/SVO PIDNet diagnostic runtime with lossless source recording."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, RuntimeSettings, load_settings
from .frames import RecordingStatus, RgbdFrame
from .perception import PerceptionFrame, PerceptionRuntime
from .recording import (
    RecordingManifestWriter,
    exact_frame_timestamps_ns,
    load_recording_manifest,
    new_recording_path,
)
from .viewer import OpenCvViewer, ViewerAction, ViewerSnapshot
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


def _manifest_mapping(root: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = root.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Recording manifest requires {key!r}")
    return value


def _validate_replay_manifest(
    manifest: Mapping[str, Any],
    *,
    settings: RuntimeSettings,
    perception: PerceptionRuntime,
    source: SvoZedSource,
) -> None:
    """Reject silent changes to a canonical observation replay."""

    recording = _manifest_mapping(manifest, "recording")
    if not bool(recording.get("completed")):
        raise ValueError("Refusing canonical replay of an incomplete recording")
    configuration = _manifest_mapping(manifest, "configuration")
    if configuration.get("zed_sdk_version") != source.sdk_version:
        raise ValueError(
            "ZED SDK version differs from the recording: "
            f"recorded={configuration.get('zed_sdk_version')!r}, "
            f"current={source.sdk_version!r}"
        )
    if configuration.get("camera_request") != asdict(settings.camera):
        raise ValueError("Camera/depth configuration differs from the recording")
    recorded_pidnet = _manifest_mapping(configuration, "pidnet")
    identities = {
        "runtime_config_sha256": perception.runtime_config_sha256,
        "checkpoint_sha256": perception.checkpoint_sha256,
    }
    for name, current in identities.items():
        if recorded_pidnet.get(name) != current:
            raise ValueError(
                f"PIDNet {name} differs from the recording: "
                f"recorded={recorded_pidnet.get(name)!r}, current={current!r}"
            )
    recorded_source = _manifest_mapping(manifest, "source")
    recorded_calibration = _manifest_mapping(recorded_source, "calibration")
    if dict(recorded_calibration) != asdict(source.descriptor.calibration):
        raise ValueError("SVO calibration does not match its recording manifest")


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


def _viewer_snapshot(
    perception: PerceptionFrame,
    *,
    source: LiveZedSource | SvoZedSource,
    pipeline_fps: float | None,
    paused: bool,
    message: str,
) -> ViewerSnapshot:
    frame = perception.rgbd
    status = source.recording_status
    return ViewerSnapshot(
        bgr_u8=frame.bgr_u8,
        depth_m_f32=frame.depth_m_f32,
        masks_u8=perception.masks_u8,
        source_kind=source.descriptor.kind,
        source_label=source.descriptor.label,
        frame_index=frame.key.sequence_index,
        source_position=frame.key.source_position,
        timestamp_ns=frame.key.timestamp_ns,
        total_frames=source.descriptor.total_frames,
        pipeline_fps=pipeline_fps,
        inference_ms=perception.timing.total_ms,
        latency_ms=max(0.0, (time.perf_counter_ns() - frame.host_received_ns) / 1.0e6),
        recording=status.active,
        recording_elapsed_s=_recording_elapsed_s(status),
        paused=paused,
        skipped_frames=source.replaced_count,
        message=message,
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
    warmup = perception.warmup_1080p()
    print(
        "PIDNet ready "
        f"device={perception.device} "
        f"checkpoint={perception.checkpoint_sha256[:12]} "
        f"thresholds={perception.thresholds} "
        f"warmup={warmup.total_ms:.1f}ms"
    )

    if args.svo is None:
        source: LiveZedSource | SvoZedSource = LiveZedSource(settings.camera)
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
        try:
            _validate_replay_manifest(
                manifest,
                settings=settings,
                perception=perception,
                source=source,
            )
        except BaseException:
            source.close()
            raise
        print(
            "Replay manifest validated "
            f"frames={source.descriptor.total_frames} "
            f"sdk={source.sdk_version} exact_timestamps=yes"
        )

    viewer_enabled = bool(settings.viewer.enabled and not args.headless)
    viewer = (
        OpenCvViewer(
            window_name=settings.viewer.window_name,
            width_px=settings.viewer.width_px,
            depth_min_m=settings.viewer.depth_min_m,
            depth_max_m=settings.viewer.depth_max_m,
            overlay_alpha=settings.viewer.overlay_alpha,
        )
        if viewer_enabled
        else None
    )
    recording = (
        RecordingController(source, settings, perception)
        if isinstance(source, LiveZedSource)
        else None
    )
    if args.record_to is not None and recording is None:
        source.close()
        raise ValueError("--record-to is valid only with a live ZED source")

    processed = 0
    fps_value: float | None = None
    fps_count = 0
    fps_started = time.perf_counter()
    next_status = fps_started
    paused = False
    step_once = False
    last_snapshot: ViewerSnapshot | None = None
    message = "Q quit | R record" if recording is not None else "Q quit | Space pause | N step"
    first_svo_timestamp_ns: int | None = None
    playback_started_ns: int | None = None
    redraw_snapshot = False
    primary_error: BaseException | None = None

    try:
        if args.record_to is not None:
            path = recording.start(Path(args.record_to))
            print(f"Recording lossless SVO: {path}")

        while True:
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
                if action is ViewerAction.TOGGLE_PAUSE:
                    paused = False
                    last_snapshot = replace(last_snapshot, paused=False)
                    if (
                        first_svo_timestamp_ns is not None
                        and playback_started_ns is not None
                    ):
                        playback_started_ns = time.perf_counter_ns() - (
                            last_snapshot.timestamp_ns - first_svo_timestamp_ns
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
            processed += 1
            fps_count += 1
            now = time.perf_counter()
            elapsed = now - fps_started
            if elapsed >= 0.5:
                fps_value = fps_count / elapsed
                fps_count = 0
                fps_started = now

            last_snapshot = _viewer_snapshot(
                result,
                source=source,
                pipeline_fps=fps_value,
                paused=paused,
                message=message,
            )
            action = viewer.show(last_snapshot) if viewer is not None else ViewerAction.NONE
            if action is ViewerAction.QUIT:
                break
            if action is ViewerAction.TOGGLE_RECORDING:
                message = _toggle_recording(recording)
            elif action is ViewerAction.TOGGLE_PAUSE:
                if source.descriptor.kind == "svo":
                    paused = not paused
                    last_snapshot = replace(last_snapshot, paused=paused)
                    redraw_snapshot = paused
                    if (
                        not paused
                        and first_svo_timestamp_ns is not None
                        and playback_started_ns is not None
                    ):
                        playback_started_ns = time.perf_counter_ns() - (
                            last_snapshot.timestamp_ns - first_svo_timestamp_ns
                        )
                else:
                    message = "Pause applies only to SVO playback"
            elif action is ViewerAction.STEP and source.descriptor.kind == "svo":
                if paused:
                    step_once = True

            if stepping and action is not ViewerAction.TOGGLE_PAUSE:
                paused = True
                if last_snapshot is not None:
                    last_snapshot = replace(last_snapshot, paused=True)
                    redraw_snapshot = True

            if now >= next_status:
                print(
                    f"frame={frame.key.sequence_index} "
                    f"source={frame.key.source_position} "
                    f"timestamp_ns={frame.key.timestamp_ns} "
                    f"pidnet={result.timing.total_ms:.1f}ms "
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
        description="Synchronized ZED RGB-D and PIDNet diagnostic runtime",
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
