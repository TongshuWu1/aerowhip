"""Thread-owned ZED live capture and deterministic SVO replay.

The live source continuously grabs on its camera-owning thread.  Consumers see
only a capacity-one latest-frame slot, so slow inference can replace unread
frames but can never apply back-pressure to lossless SVO recording.  SVO replay
is deliberately different: each ``read`` requests exactly one frame from its
ZED-owning thread and therefore preserves file order without prefetch or drops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import queue
import threading
import time
from typing import Any, Sequence

import numpy as np

from .config import CameraSettings
from .frames import (
    CameraCalibration,
    FrameKey,
    RecordingStatus,
    RgbdFrame,
    SourceDescriptor,
)

try:
    import pyzed.sl as sl
except ModuleNotFoundError as exc:  # Keep non-camera tooling importable.
    sl = None
    _PYZED_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _PYZED_IMPORT_ERROR = None


_NATIVE_RESOLUTIONS = {"HD1080": (1920, 1080)}
_LOSSLESS_COMPRESSION_NAMES = frozenset({"LOSSLESS"})


class ZedSourceError(RuntimeError):
    """Base class for explicit ZED source failures."""


class ZedOpenError(ZedSourceError):
    """The requested live camera or SVO could not be opened exactly."""


class ZedCaptureError(ZedSourceError):
    """A grab, timestamp, image, or depth retrieval failed."""


class ZedRecordingError(ZedSourceError):
    """Lossless SVO recording could not start, continue, or stop cleanly."""


class ZedSourceClosedError(ZedSourceError):
    """A control operation was requested after source shutdown."""


@dataclass(frozen=True, slots=True)
class CaptureCounters:
    """Source-level frame counts.

    ``captured`` counts complete RGB-D frames produced by the SDK.  For live
    capture, ``replaced`` counts unread frames displaced in the latest-frame
    slot.  Deterministic SVO playback never replaces a frame.
    """

    captured: int
    replaced: int


@dataclass(slots=True)
class _LiveCommand:
    action: str
    path: Path | None = None
    compression: str | None = None
    done: threading.Event = field(default_factory=threading.Event)
    result: RecordingStatus | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class _SvoCommand:
    action: str
    done: threading.Event = field(default_factory=threading.Event)
    result: RgbdFrame | None = None
    error: BaseException | None = None


def _require_pyzed() -> Any:
    if sl is None:
        raise ZedOpenError(
            "pyzed.sl is unavailable; run with the repository virtual environment "
            "and an installed ZED SDK"
        ) from _PYZED_IMPORT_ERROR
    return sl


def _positive_timeout(value: float, name: str) -> float:
    timeout = float(value)
    if timeout <= 0.0:
        raise ValueError(f"{name} must be positive")
    return timeout


def _read_timeout(value: float | None) -> float | None:
    if value is None:
        return None
    timeout = float(value)
    if timeout < 0.0:
        raise ValueError("timeout_s must be nonnegative or None")
    return timeout


def _normal_recording_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() != ".svo2":
        raise ValueError("Lossless recordings must use the .svo2 extension")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing SVO2: {path}")
    return path


def _normal_compression(value: str) -> str:
    name = str(value).strip().upper()
    if name not in _LOSSLESS_COMPRESSION_NAMES:
        allowed = ", ".join(sorted(_LOSSLESS_COMPRESSION_NAMES))
        raise ValueError(f"Recording compression must be one of: {allowed}")
    return name


def _runtime_parameters(camera: CameraSettings) -> Any:
    sdk = _require_pyzed()
    runtime = sdk.RuntimeParameters()
    runtime.enable_depth = True
    runtime.confidence_threshold = int(camera.confidence)
    runtime.texture_confidence_threshold = int(camera.texture_confidence)
    runtime.remove_saturated_areas = False
    if hasattr(runtime, "enable_fill_mode"):
        runtime.enable_fill_mode = bool(camera.fill)
    return runtime


def _calibration_from_camera(camera: Any) -> CameraCalibration:
    information = camera.get_camera_information()
    configuration = information.camera_configuration
    # VIEW.LEFT_BGR is rectified, and the default MEASURE.DEPTH is registered
    # to that rectified left image.  Therefore use calibration_parameters, not
    # calibration_parameters_raw.
    left = configuration.calibration_parameters.left_cam
    return CameraCalibration(
        serial_number=int(information.serial_number),
        camera_model=str(information.camera_model),
        width_px=int(configuration.resolution.width),
        height_px=int(configuration.resolution.height),
        fps=float(configuration.fps),
        fx_px=float(left.fx),
        fy_px=float(left.fy),
        cx_px=float(left.cx),
        cy_px=float(left.cy),
    )


def _validate_native_mode(
    calibration: CameraCalibration,
    requested: CameraSettings,
    *,
    source_label: str,
) -> None:
    expected_size = _NATIVE_RESOLUTIONS.get(requested.resolution)
    if expected_size is None:
        raise ZedOpenError(f"Unsupported native resolution: {requested.resolution}")
    actual_size = (calibration.width_px, calibration.height_px)
    if actual_size != expected_size:
        raise ZedOpenError(
            f"{source_label} opened at {actual_size[0]}x{actual_size[1]}, "
            f"expected {expected_size[0]}x{expected_size[1]}"
        )
    if abs(calibration.fps - requested.fps) > 1.0e-6:
        raise ZedOpenError(
            f"{source_label} opened at {calibration.fps:g} FPS, "
            f"expected {requested.fps} FPS"
        )


def _live_descriptor(calibration: CameraCalibration) -> SourceDescriptor:
    return SourceDescriptor(
        source_id=f"zed-live-{calibration.serial_number}",
        kind="live",
        label=(
            f"{calibration.camera_model} S/N {calibration.serial_number} "
            f"{calibration.width_px}x{calibration.height_px}@{calibration.fps:g}"
        ),
        calibration=calibration,
    )


def _svo_descriptor(
    path: Path,
    calibration: CameraCalibration,
    total_frames: int,
) -> SourceDescriptor:
    return SourceDescriptor(
        source_id=f"svo-{path}",
        kind="svo",
        label=str(path),
        calibration=calibration,
        total_frames=total_frames,
    )


def _retrieve_rgbd_after_grab(
    camera: Any,
    image_mat: Any,
    depth_mat: Any,
    *,
    descriptor: SourceDescriptor,
    sequence_index: int,
    source_position: int,
    exact_timestamp_ns: int | None = None,
) -> RgbdFrame:
    sdk = _require_pyzed()
    embedded_timestamp_ns = int(
        camera.get_timestamp(sdk.TIME_REFERENCE.IMAGE).get_nanoseconds()
    )
    if embedded_timestamp_ns <= 0:
        raise ZedCaptureError(
            f"{descriptor.label} returned invalid IMAGE timestamp "
            f"{embedded_timestamp_ns}"
        )

    image_status = camera.retrieve_image(
        image_mat,
        sdk.VIEW.LEFT_BGR,
        sdk.MEM.CPU,
    )
    if image_status != sdk.ERROR_CODE.SUCCESS:
        raise ZedCaptureError(
            f"LEFT_BGR retrieval failed for {descriptor.label}: {image_status}"
        )
    depth_status = camera.retrieve_measure(
        depth_mat,
        sdk.MEASURE.DEPTH,
        sdk.MEM.CPU,
    )
    if depth_status != sdk.ERROR_CODE.SUCCESS:
        raise ZedCaptureError(
            f"registered DEPTH retrieval failed for {descriptor.label}: "
            f"{depth_status}"
        )

    image_timestamp_ns = int(image_mat.timestamp.get_nanoseconds())
    depth_timestamp_ns = int(depth_mat.timestamp.get_nanoseconds())
    if (
        image_timestamp_ns != embedded_timestamp_ns
        or depth_timestamp_ns != embedded_timestamp_ns
    ):
        raise ZedCaptureError(
            "ZED returned unsynchronized RGB-D timestamps: "
            f"camera={embedded_timestamp_ns}, image={image_timestamp_ns}, "
            f"depth={depth_timestamp_ns}"
        )

    timestamp_ns = (
        embedded_timestamp_ns
        if exact_timestamp_ns is None
        else int(exact_timestamp_ns)
    )
    if timestamp_ns <= 0:
        raise ZedCaptureError(f"Invalid exact frame timestamp {timestamp_ns}")
    if (
        exact_timestamp_ns is not None
        and abs(timestamp_ns - embedded_timestamp_ns) > 1_000
    ):
        raise ZedCaptureError(
            "Exact timestamp sidecar does not match the SVO frame: "
            f"sidecar={timestamp_ns}, embedded={embedded_timestamp_ns}"
        )

    # Deep, contiguous copies detach immutable frames from the two reusable
    # sl.Mat allocations owned by the ZED thread.
    bgr = np.array(
        image_mat.get_data(sdk.MEM.CPU),
        dtype=np.uint8,
        order="C",
        copy=True,
    )
    depth = np.array(
        depth_mat.get_data(sdk.MEM.CPU),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    expected_image_shape = (
        descriptor.calibration.height_px,
        descriptor.calibration.width_px,
        3,
    )
    expected_depth_shape = expected_image_shape[:2]
    if bgr.shape != expected_image_shape:
        raise ZedCaptureError(
            f"Unexpected LEFT_BGR shape {bgr.shape}; expected {expected_image_shape}"
        )
    if depth.shape != expected_depth_shape:
        raise ZedCaptureError(
            f"Unexpected registered DEPTH shape {depth.shape}; "
            f"expected {expected_depth_shape}"
        )

    return RgbdFrame(
        key=FrameKey(
            source_id=descriptor.source_id,
            sequence_index=sequence_index,
            source_position=source_position,
            timestamp_ns=timestamp_ns,
        ),
        # Monotonic host time is used only for live pipeline-age diagnostics.
        # The exact camera timestamp above remains the persistent frame identity.
        host_received_ns=time.perf_counter_ns(),
        bgr_u8=bgr,
        depth_m_f32=depth,
    )


def _free_mat(mat: Any | None) -> None:
    if mat is not None:
        mat.free()


class LiveZedSource:
    """Continuously capture live RGB-D while exposing one latest frame.

    The constructor blocks until the background thread has opened the camera,
    validated HD1080/30, and read the post-self-calibration intrinsics.  Every
    ZED SDK call for this object, including recording control and cleanup, runs
    on that same background thread.
    """

    def __init__(
        self,
        camera: CameraSettings,
        *,
        recording_path: str | Path | None = None,
        recording_compression: str = "LOSSLESS",
        startup_timeout_s: float = 15.0,
    ) -> None:
        self._camera_settings = camera
        self._startup_timeout_s = _positive_timeout(
            startup_timeout_s,
            "startup_timeout_s",
        )
        self._condition = threading.Condition()
        self._control_lock = threading.Lock()
        self._commands: queue.Queue[_LiveCommand] = queue.Queue()
        self._startup_event = threading.Event()
        self._descriptor: SourceDescriptor | None = None
        self._sdk_version: str | None = None
        self._latest_frame: RgbdFrame | None = None
        self._terminal_error: BaseException | None = None
        self._closed = False
        self._captured_count = 0
        self._replaced_count = 0
        self._recording_status = RecordingStatus(
            active=False,
            path=None,
            compression=None,
            frame_count=0,
            first_timestamp_ns=None,
            last_timestamp_ns=None,
        )
        self._recorded_timestamps_ns: list[int] = []

        initial_recording: _LiveCommand | None = None
        if recording_path is not None:
            initial_recording = _LiveCommand(
                action="start_recording",
                path=_normal_recording_path(recording_path),
                compression=_normal_compression(recording_compression),
            )
            self._commands.put(initial_recording)

        self._thread = threading.Thread(
            target=self._capture_main,
            name="zed-live-capture",
            daemon=True,
        )
        self._thread.start()
        if not self._startup_event.wait(self._startup_timeout_s):
            command = _LiveCommand(action="close")
            self._commands.put(command)
            command.done.wait(self._startup_timeout_s)
            self._thread.join(self._startup_timeout_s)
            raise TimeoutError(
                f"Timed out opening live ZED after {self._startup_timeout_s:g}s"
            )

        with self._condition:
            startup_error = self._terminal_error
            descriptor = self._descriptor
        if startup_error is not None:
            self._thread.join(timeout=self._startup_timeout_s)
            raise startup_error
        if descriptor is None:
            self.close()
            raise ZedOpenError("Live ZED thread started without a descriptor")
        if initial_recording is not None:
            # Initial commands are drained before the startup event is set.
            initial_recording.done.wait()
            if initial_recording.error is not None:
                try:
                    self.close()
                finally:
                    raise initial_recording.error

    @property
    def descriptor(self) -> SourceDescriptor:
        with self._condition:
            if self._descriptor is None:
                if self._terminal_error is not None:
                    raise self._terminal_error
                raise ZedOpenError("Live ZED calibration is not available")
            return self._descriptor

    @property
    def sdk_version(self) -> str:
        with self._condition:
            if self._sdk_version is None:
                raise ZedOpenError("ZED SDK version is not available")
            return self._sdk_version

    @property
    def recording_status(self) -> RecordingStatus:
        with self._condition:
            return self._recording_status

    @property
    def counters(self) -> CaptureCounters:
        with self._condition:
            return CaptureCounters(
                captured=self._captured_count,
                replaced=self._replaced_count,
            )

    @property
    def captured_count(self) -> int:
        return self.counters.captured

    @property
    def replaced_count(self) -> int:
        return self.counters.replaced

    @property
    def recorded_timestamps_ns(self) -> tuple[int, ...]:
        """Exact live IMAGE timestamps for the most recent recording."""

        with self._condition:
            return tuple(self._recorded_timestamps_ns)

    @property
    def recording_snapshot(self) -> tuple[RecordingStatus, tuple[int, ...]]:
        """Atomically snapshot recording counters and their timestamp timeline."""

        with self._condition:
            return self._recording_status, tuple(self._recorded_timestamps_ns)

    def read(self, timeout_s: float | None = None) -> RgbdFrame | None:
        """Consume the latest unread live frame.

        Returns ``None`` on timeout or after normal shutdown.  A final unread
        valid frame is returned before a later call exposes a terminal capture
        error.
        """

        timeout = _read_timeout(timeout_s)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._latest_frame is not None:
                    frame = self._latest_frame
                    self._latest_frame = None
                    return frame
                if self._terminal_error is not None:
                    raise self._terminal_error
                if self._closed:
                    return None
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)

    def start_recording(
        self,
        path: str | Path,
        compression: str = "LOSSLESS",
        timeout_s: float = 10.0,
    ) -> RecordingStatus:
        """Start a new lossless SVO2 without ever overwriting a file."""

        command = _LiveCommand(
            action="start_recording",
            path=_normal_recording_path(path),
            compression=_normal_compression(compression),
        )
        with self._control_lock:
            return self._submit_control(command, timeout_s)

    def stop_recording(self, timeout_s: float = 10.0) -> RecordingStatus:
        """Flush and close the current recording on the capture thread."""

        with self._control_lock:
            return self._submit_control(
                _LiveCommand(action="stop_recording"),
                timeout_s,
            )

    def close(self, timeout_s: float = 10.0) -> None:
        """Stop capture, flush recording, free Mats, and close the camera."""

        timeout = _positive_timeout(timeout_s, "timeout_s")
        with self._control_lock:
            with self._condition:
                if self._closed:
                    return
            command = _LiveCommand(action="close")
            self._commands.put(command)
            if not command.done.wait(timeout):
                raise TimeoutError("Timed out requesting live ZED shutdown")
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("Timed out waiting for live ZED thread shutdown")
            if command.error is not None:
                raise command.error

    def __enter__(self) -> LiveZedSource:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _submit_control(
        self,
        command: _LiveCommand,
        timeout_s: float,
    ) -> RecordingStatus:
        timeout = _positive_timeout(timeout_s, "timeout_s")
        with self._condition:
            if self._closed or not self._thread.is_alive():
                raise ZedSourceClosedError("Live ZED source is closed")
            if self._terminal_error is not None:
                raise self._terminal_error
        self._commands.put(command)
        if not command.done.wait(timeout):
            raise TimeoutError(
                f"Timed out waiting for ZED command {command.action!r}"
            )
        if command.error is not None:
            raise command.error
        if command.result is None:
            raise ZedSourceError(
                f"ZED command {command.action!r} returned no recording status"
            )
        return command.result

    def _capture_main(self) -> None:
        sdk = None
        zed = None
        image_mat = None
        depth_mat = None
        startup_announced = False
        try:
            sdk = _require_pyzed()
            with self._condition:
                self._sdk_version = str(sdk.Camera.get_sdk_version())
            zed = sdk.Camera()
            initialization = sdk.InitParameters()
            initialization.camera_resolution = getattr(
                sdk.RESOLUTION,
                self._camera_settings.resolution,
            )
            initialization.camera_fps = int(self._camera_settings.fps)
            initialization.camera_disable_self_calib = bool(
                self._camera_settings.disable_self_calibration
            )
            initialization.depth_mode = getattr(
                sdk.DEPTH_MODE,
                self._camera_settings.depth_mode,
            )
            initialization.coordinate_units = sdk.UNIT.METER
            initialization.coordinate_system = (
                sdk.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            )
            initialization.open_timeout_sec = self._startup_timeout_s
            open_status = zed.open(initialization)
            if open_status != sdk.ERROR_CODE.SUCCESS:
                raise ZedOpenError(f"Could not open live ZED: {open_status}")

            calibration = _calibration_from_camera(zed)
            _validate_native_mode(
                calibration,
                self._camera_settings,
                source_label="Live ZED",
            )
            descriptor = _live_descriptor(calibration)
            runtime = _runtime_parameters(self._camera_settings)
            image_mat = sdk.Mat()
            depth_mat = sdk.Mat()
            with self._condition:
                self._descriptor = descriptor

            should_stop = self._drain_live_commands(zed)
            self._startup_event.set()
            startup_announced = True
            while not should_stop:
                should_stop = self._drain_live_commands(zed)
                if should_stop:
                    break
                grab_status = zed.grab(runtime)
                if grab_status == sdk.ERROR_CODE.CAMERA_REBOOTING:
                    continue
                if grab_status != sdk.ERROR_CODE.SUCCESS:
                    raise ZedCaptureError(
                        f"Live ZED grab failed: {grab_status}"
                    )

                with self._condition:
                    sequence_index = self._captured_count
                frame = _retrieve_rgbd_after_grab(
                    zed,
                    image_mat,
                    depth_mat,
                    descriptor=descriptor,
                    sequence_index=sequence_index,
                    source_position=sequence_index,
                )
                self._update_recording_after_grab(zed, frame.key.timestamp_ns)
                self._publish_live_frame(frame)
        except Exception as exc:
            error = exc if isinstance(exc, ZedSourceError) else ZedCaptureError(
                f"Live ZED capture thread failed: {exc}"
            )
            with self._condition:
                if self._terminal_error is None:
                    self._terminal_error = error
                self._condition.notify_all()
        finally:
            if not startup_announced:
                self._startup_event.set()
            cleanup_error: BaseException | None = None
            if zed is not None:
                try:
                    self._stop_recording_on_thread(zed)
                except Exception as exc:
                    cleanup_error = ZedRecordingError(
                        f"Could not finalize SVO recording: {exc}"
                    )
            for mat in (image_mat, depth_mat):
                try:
                    _free_mat(mat)
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = ZedCaptureError(
                            f"Could not free ZED Mat: {exc}"
                        )
            if zed is not None:
                try:
                    zed.close()
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = ZedCaptureError(
                            f"Could not close live ZED: {exc}"
                        )
            with self._condition:
                if cleanup_error is not None and self._terminal_error is None:
                    self._terminal_error = cleanup_error
                self._closed = True
                terminal_error = self._terminal_error
                self._condition.notify_all()
            self._fail_pending_live_commands(terminal_error)

    def _drain_live_commands(self, zed: Any) -> bool:
        should_stop = False
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return should_stop
            try:
                if command.action == "start_recording":
                    if command.path is None or command.compression is None:
                        raise ZedRecordingError(
                            "start_recording command is missing parameters"
                        )
                    command.result = self._start_recording_on_thread(
                        zed,
                        command.path,
                        command.compression,
                    )
                elif command.action == "stop_recording":
                    command.result = self._stop_recording_on_thread(zed)
                elif command.action == "close":
                    command.result = self._stop_recording_on_thread(zed)
                    should_stop = True
                else:
                    raise ZedSourceError(
                        f"Unknown live ZED command: {command.action!r}"
                    )
            except BaseException as exc:
                command.error = exc
                if command.action == "close":
                    should_stop = True
            finally:
                command.done.set()
            if should_stop:
                return True

    def _start_recording_on_thread(
        self,
        zed: Any,
        path: Path,
        compression: str,
    ) -> RecordingStatus:
        sdk = _require_pyzed()
        with self._condition:
            if self._recording_status.active:
                raise ZedRecordingError(
                    f"Already recording to {self._recording_status.path}"
                )
        # Recheck immediately before enable_recording because the SDK erases an
        # existing target path.
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing SVO2: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        parameters = sdk.RecordingParameters(
            str(path),
            getattr(sdk.SVO_COMPRESSION_MODE, compression),
        )
        status = zed.enable_recording(parameters)
        if status != sdk.ERROR_CODE.SUCCESS:
            raise ZedRecordingError(
                f"Could not start {compression} recording at {path}: {status}"
            )
        recording = RecordingStatus(
            active=True,
            path=path,
            compression=compression,
            frame_count=0,
            first_timestamp_ns=None,
            last_timestamp_ns=None,
        )
        with self._condition:
            # Do not let a frame captured before enable_recording cross the
            # recording boundary into the consumer's next result.
            if self._latest_frame is not None:
                self._latest_frame = None
                self._replaced_count += 1
            self._recorded_timestamps_ns = []
            self._recording_status = recording
        return recording

    def _stop_recording_on_thread(self, zed: Any) -> RecordingStatus:
        with self._condition:
            current = self._recording_status
        if not current.active:
            return current
        zed.disable_recording()
        stopped = RecordingStatus(
            active=False,
            path=current.path,
            compression=current.compression,
            frame_count=current.frame_count,
            first_timestamp_ns=current.first_timestamp_ns,
            last_timestamp_ns=current.last_timestamp_ns,
        )
        with self._condition:
            self._recording_status = stopped
        return stopped

    def _update_recording_after_grab(
        self,
        zed: Any,
        timestamp_ns: int,
    ) -> None:
        with self._condition:
            current = self._recording_status
        if not current.active:
            return
        sdk_status = zed.get_recording_status()
        if not bool(sdk_status.is_recording):
            raise ZedRecordingError(
                f"SVO recording stopped unexpectedly: {current.path}"
            )
        if not bool(sdk_status.status):
            raise ZedRecordingError(
                f"ZED could not write the current frame to {current.path}"
            )
        updated = RecordingStatus(
            active=True,
            path=current.path,
            compression=current.compression,
            frame_count=current.frame_count + 1,
            first_timestamp_ns=(
                timestamp_ns
                if current.first_timestamp_ns is None
                else current.first_timestamp_ns
            ),
            last_timestamp_ns=timestamp_ns,
        )
        with self._condition:
            self._recorded_timestamps_ns.append(timestamp_ns)
            self._recording_status = updated

    def _publish_live_frame(self, frame: RgbdFrame) -> None:
        with self._condition:
            if frame.key.sequence_index != self._captured_count:
                raise ZedCaptureError(
                    "Live ZED sequence index changed unexpectedly"
                )
            self._captured_count += 1
            if self._latest_frame is not None:
                self._replaced_count += 1
            self._latest_frame = frame
            self._condition.notify_all()

    def _fail_pending_live_commands(
        self,
        terminal_error: BaseException | None,
    ) -> None:
        error = terminal_error or ZedSourceClosedError("Live ZED source is closed")
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            command.error = error
            command.done.set()


class SvoZedSource:
    """Demand-driven, ordered SVO playback with the live ``read`` contract.

    No frame is prefetched.  One synchronous read command causes one SDK grab,
    so every recorded frame is returned exactly once and ``replaced`` remains
    zero.  The finite timeout only limits waiting for another caller to finish
    a read; once a frame request is submitted it is never abandoned.
    """

    def __init__(
        self,
        path: str | Path,
        camera: CameraSettings,
        *,
        playback_realtime: bool = False,
        exact_timestamps_ns: Sequence[int] | None = None,
        startup_timeout_s: float = 15.0,
    ) -> None:
        self._path = Path(path).expanduser().resolve()
        if self._path.suffix.lower() not in {".svo", ".svo2"}:
            raise ValueError("SVO playback path must end in .svo or .svo2")
        if not self._path.is_file():
            raise FileNotFoundError(f"SVO file not found: {self._path}")
        if playback_realtime:
            raise ValueError(
                "Ordered SVO playback requires playback_realtime=False; "
                "real-time mode can drop frames"
            )
        self._camera_settings = camera
        self._exact_timestamps_ns = (
            None
            if exact_timestamps_ns is None
            else tuple(int(value) for value in exact_timestamps_ns)
        )
        if self._exact_timestamps_ns is not None:
            if any(value <= 0 for value in self._exact_timestamps_ns):
                raise ValueError("exact_timestamps_ns must contain positive integers")
            if any(
                current <= previous
                for previous, current in zip(
                    self._exact_timestamps_ns,
                    self._exact_timestamps_ns[1:],
                )
            ):
                raise ValueError("exact_timestamps_ns must be strictly increasing")
        self._startup_timeout_s = _positive_timeout(
            startup_timeout_s,
            "startup_timeout_s",
        )
        self._condition = threading.Condition()
        self._read_lock = threading.Lock()
        self._commands: queue.Queue[_SvoCommand] = queue.Queue()
        self._startup_event = threading.Event()
        self._descriptor: SourceDescriptor | None = None
        self._sdk_version: str | None = None
        self._terminal_error: BaseException | None = None
        self._closed = False
        self._eof = False
        self._captured_count = 0
        self._recording_status = RecordingStatus(
            active=False,
            path=None,
            compression=None,
            frame_count=0,
            first_timestamp_ns=None,
            last_timestamp_ns=None,
        )
        self._thread = threading.Thread(
            target=self._playback_main,
            name="zed-svo-playback",
            daemon=True,
        )
        self._thread.start()
        if not self._startup_event.wait(self._startup_timeout_s):
            command = _SvoCommand(action="close")
            self._commands.put(command)
            command.done.wait(self._startup_timeout_s)
            self._thread.join(self._startup_timeout_s)
            raise TimeoutError(
                f"Timed out opening SVO after {self._startup_timeout_s:g}s: "
                f"{self._path}"
            )
        with self._condition:
            startup_error = self._terminal_error
            descriptor = self._descriptor
        if startup_error is not None:
            self._thread.join(timeout=self._startup_timeout_s)
            raise startup_error
        if descriptor is None:
            self.close()
            raise ZedOpenError("SVO thread started without a descriptor")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def descriptor(self) -> SourceDescriptor:
        with self._condition:
            if self._descriptor is None:
                if self._terminal_error is not None:
                    raise self._terminal_error
                raise ZedOpenError("SVO calibration is not available")
            return self._descriptor

    @property
    def sdk_version(self) -> str:
        with self._condition:
            if self._sdk_version is None:
                raise ZedOpenError("ZED SDK version is not available")
            return self._sdk_version

    @property
    def recording_status(self) -> RecordingStatus:
        return self._recording_status

    @property
    def counters(self) -> CaptureCounters:
        with self._condition:
            return CaptureCounters(captured=self._captured_count, replaced=0)

    @property
    def captured_count(self) -> int:
        return self.counters.captured

    @property
    def replaced_count(self) -> int:
        return 0

    @property
    def eof(self) -> bool:
        with self._condition:
            return self._eof

    def read(self, timeout_s: float | None = None) -> RgbdFrame | None:
        """Return exactly the next SVO frame, or ``None`` at explicit EOF."""

        timeout = _read_timeout(timeout_s)
        if timeout is None:
            acquired = self._read_lock.acquire()
        else:
            acquired = self._read_lock.acquire(timeout=timeout)
        if not acquired:
            return None
        try:
            with self._condition:
                if self._terminal_error is not None:
                    raise self._terminal_error
                if self._eof or self._closed:
                    return None
            command = _SvoCommand(action="read")
            self._commands.put(command)
            # Once submitted, never time out and abandon a consumed SVO frame.
            command.done.wait()
            if command.error is not None:
                raise command.error
            return command.result
        finally:
            self._read_lock.release()

    def close(self, timeout_s: float = 10.0) -> None:
        timeout = _positive_timeout(timeout_s, "timeout_s")
        if not self._read_lock.acquire(timeout=timeout):
            raise TimeoutError("Timed out waiting for active SVO read")
        try:
            with self._condition:
                if self._closed:
                    return
            command = _SvoCommand(action="close")
            self._commands.put(command)
            if not command.done.wait(timeout):
                raise TimeoutError("Timed out requesting SVO shutdown")
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("Timed out waiting for SVO thread shutdown")
            if command.error is not None:
                raise command.error
        finally:
            self._read_lock.release()

    def __enter__(self) -> SvoZedSource:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _playback_main(self) -> None:
        sdk = None
        zed = None
        image_mat = None
        depth_mat = None
        startup_announced = False
        active_command: _SvoCommand | None = None
        try:
            sdk = _require_pyzed()
            with self._condition:
                self._sdk_version = str(sdk.Camera.get_sdk_version())
            zed = sdk.Camera()
            initialization = sdk.InitParameters()
            initialization.set_from_svo_file(str(self._path))
            initialization.svo_real_time_mode = False
            initialization.camera_disable_self_calib = bool(
                self._camera_settings.disable_self_calibration
            )
            initialization.depth_mode = getattr(
                sdk.DEPTH_MODE,
                self._camera_settings.depth_mode,
            )
            initialization.coordinate_units = sdk.UNIT.METER
            initialization.coordinate_system = (
                sdk.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            )
            initialization.open_timeout_sec = self._startup_timeout_s
            open_status = zed.open(initialization)
            if open_status != sdk.ERROR_CODE.SUCCESS:
                raise ZedOpenError(
                    f"Could not open SVO {self._path}: {open_status}"
                )

            calibration = _calibration_from_camera(zed)
            _validate_native_mode(
                calibration,
                self._camera_settings,
                source_label=f"SVO {self._path}",
            )
            total_frames = int(zed.get_svo_number_of_frames())
            if total_frames < 0:
                raise ZedOpenError(
                    f"ZED did not report a frame count for SVO {self._path}"
                )
            if (
                self._exact_timestamps_ns is not None
                and len(self._exact_timestamps_ns) != total_frames
            ):
                raise ZedOpenError(
                    "Exact timestamp sidecar length does not match the SVO: "
                    f"timestamps={len(self._exact_timestamps_ns)}, "
                    f"frames={total_frames}"
                )
            descriptor = _svo_descriptor(
                self._path,
                calibration,
                total_frames,
            )
            runtime = _runtime_parameters(self._camera_settings)
            image_mat = sdk.Mat()
            depth_mat = sdk.Mat()
            with self._condition:
                self._descriptor = descriptor
            self._startup_event.set()
            startup_announced = True

            previous_position: int | None = None
            while True:
                active_command = self._commands.get()
                if active_command.action == "close":
                    active_command.done.set()
                    active_command = None
                    break
                if active_command.action != "read":
                    raise ZedSourceError(
                        f"Unknown SVO command: {active_command.action!r}"
                    )

                grab_status = zed.grab(runtime)
                if grab_status == sdk.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    with self._condition:
                        self._eof = True
                        self._condition.notify_all()
                    active_command.result = None
                    active_command.done.set()
                    active_command = None
                    continue
                if grab_status != sdk.ERROR_CODE.SUCCESS:
                    raise ZedCaptureError(
                        f"SVO grab failed at position "
                        f"{zed.get_svo_position()}: {grab_status}"
                    )

                source_position = int(zed.get_svo_position())
                if source_position < 0:
                    raise ZedCaptureError(
                        "ZED returned a negative SVO position after a successful grab"
                    )
                if (
                    previous_position is not None
                    and source_position != previous_position + 1
                ):
                    raise ZedCaptureError(
                        "Ordered SVO playback skipped or repeated a position: "
                        f"previous={previous_position}, current={source_position}"
                    )
                with self._condition:
                    sequence_index = self._captured_count
                frame = _retrieve_rgbd_after_grab(
                    zed,
                    image_mat,
                    depth_mat,
                    descriptor=descriptor,
                    sequence_index=sequence_index,
                    source_position=source_position,
                    exact_timestamp_ns=(
                        None
                        if self._exact_timestamps_ns is None
                        else self._exact_timestamps_ns[source_position]
                    ),
                )
                previous_position = source_position
                with self._condition:
                    self._captured_count += 1
                active_command.result = frame
                active_command.done.set()
                active_command = None
        except Exception as exc:
            error = exc if isinstance(exc, ZedSourceError) else ZedCaptureError(
                f"SVO playback thread failed: {exc}"
            )
            if active_command is not None:
                active_command.error = error
                active_command.done.set()
                active_command = None
            with self._condition:
                if self._terminal_error is None:
                    self._terminal_error = error
                self._condition.notify_all()
        finally:
            if not startup_announced:
                self._startup_event.set()
            cleanup_error: BaseException | None = None
            for mat in (image_mat, depth_mat):
                try:
                    _free_mat(mat)
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = ZedCaptureError(
                            f"Could not free SVO Mat: {exc}"
                        )
            if zed is not None:
                try:
                    zed.close()
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = ZedCaptureError(
                            f"Could not close SVO {self._path}: {exc}"
                        )
            with self._condition:
                if cleanup_error is not None and self._terminal_error is None:
                    self._terminal_error = cleanup_error
                self._closed = True
                terminal_error = self._terminal_error
                self._condition.notify_all()
            self._fail_pending_svo_commands(terminal_error)

    def _fail_pending_svo_commands(
        self,
        terminal_error: BaseException | None,
    ) -> None:
        error = terminal_error or ZedSourceClosedError("SVO source is closed")
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            command.error = error
            command.done.set()


__all__ = [
    "CaptureCounters",
    "LiveZedSource",
    "SvoZedSource",
    "ZedCaptureError",
    "ZedOpenError",
    "ZedRecordingError",
    "ZedSourceClosedError",
    "ZedSourceError",
]
