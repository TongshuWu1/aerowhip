from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np

from .contracts import SourceDescriptor, StereoCalibration, StereoFrame


class ZedStereoError(RuntimeError):
    pass


_LIVE_TIMESTAMP_STALL_TIMEOUT_S = 0.5


@dataclass(frozen=True, slots=True)
class CameraCaptureSettings:
    """Immutable live-camera settings read back from the ZED SDK."""

    requested_manual: bool
    automatic_exposure_gain: bool
    exposure: int
    gain: int


def _validate_manual_settings(
    manual_exposure: int | None,
    manual_gain: int | None,
) -> tuple[int | None, int | None]:
    if (manual_exposure is None) != (manual_gain is None):
        raise ValueError("manual_exposure and manual_gain must be provided together.")
    if manual_exposure is None:
        return None, None
    values: list[int] = []
    for name, value in (
        ("manual_exposure", manual_exposure),
        ("manual_gain", manual_gain),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer between 0 and 100.")
        integer = int(value)
        if not 0 <= integer <= 100:
            raise ValueError(f"{name} must be between 0 and 100.")
        values.append(integer)
    return values[0], values[1]


def _read_capture_settings(
    camera: Any,
    sdk: Any,
    *,
    requested_manual: bool,
) -> CameraCaptureSettings:
    readback: dict[str, int] = {}
    for name, setting in (
        ("aec_agc", sdk.VIDEO_SETTINGS.AEC_AGC),
        ("exposure", sdk.VIDEO_SETTINGS.EXPOSURE),
        ("gain", sdk.VIDEO_SETTINGS.GAIN),
    ):
        status, value = camera.get_camera_settings(setting)
        if status != sdk.ERROR_CODE.SUCCESS:
            raise ZedStereoError(f"Could not read ZED {name} setting: {status}")
        readback[name] = int(value)
    return CameraCaptureSettings(
        requested_manual=bool(requested_manual),
        automatic_exposure_gain=bool(readback["aec_agc"]),
        exposure=readback["exposure"],
        gain=readback["gain"],
    )


def _apply_manual_settings(
    camera: Any,
    sdk: Any,
    exposure: int,
    gain: int,
) -> CameraCaptureSettings:
    for name, setting, value in (
        ("exposure", sdk.VIDEO_SETTINGS.EXPOSURE, int(exposure)),
        ("gain", sdk.VIDEO_SETTINGS.GAIN, int(gain)),
    ):
        status = camera.set_camera_settings(setting, value)
        if status != sdk.ERROR_CODE.SUCCESS:
            raise ZedStereoError(f"Could not set manual ZED {name}={value}: {status}")
    applied = _read_capture_settings(camera, sdk, requested_manual=True)
    if applied.automatic_exposure_gain:
        raise ZedStereoError("ZED kept automatic exposure/gain enabled in manual mode.")
    if applied.exposure != int(exposure) or applied.gain != int(gain):
        raise ZedStereoError(
            "ZED manual exposure/gain readback did not match the request: "
            f"requested={int(exposure)}/{int(gain)}, "
            f"readback={applied.exposure}/{applied.gain}."
        )
    return applied


def _apply_automatic_settings(camera: Any, sdk: Any) -> CameraCaptureSettings:
    status = camera.set_camera_settings(sdk.VIDEO_SETTINGS.AEC_AGC, 1)
    if status != sdk.ERROR_CODE.SUCCESS:
        raise ZedStereoError(f"Could not enable automatic ZED exposure/gain: {status}")
    applied = _read_capture_settings(camera, sdk, requested_manual=False)
    if not applied.automatic_exposure_gain:
        raise ZedStereoError("ZED automatic exposure/gain did not remain enabled.")
    return applied


def _restore_capture_settings(
    camera: Any,
    sdk: Any,
    settings: CameraCaptureSettings,
) -> None:
    if settings.automatic_exposure_gain:
        status = camera.set_camera_settings(sdk.VIDEO_SETTINGS.AEC_AGC, 1)
        if status != sdk.ERROR_CODE.SUCCESS:
            raise ZedStereoError(f"Could not restore automatic ZED exposure/gain: {status}")
        return
    _apply_manual_settings(camera, sdk, settings.exposure, settings.gain)


def _require_zed() -> Any:
    try:
        import pyzed.sl as sl
    except ImportError as error:  # pragma: no cover - requires the lab runtime
        raise ZedStereoError(
            "pyzed.sl is unavailable. Run with the Python environment registered "
            "by the installed ZED SDK."
        ) from error
    return sl


def _intrinsic_matrix(camera_parameters: Any) -> np.ndarray:
    return np.asarray(
        (
            (float(camera_parameters.fx), 0.0, float(camera_parameters.cx)),
            (0.0, float(camera_parameters.fy), float(camera_parameters.cy)),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _mat_bgr(mat: Any, sdk: Any) -> np.ndarray:
    data = np.asarray(mat.get_data(sdk.MEM.CPU))
    if data.ndim != 3 or data.shape[2] != 3:
        raise ZedStereoError(f"Expected native BGR image, received shape {data.shape}.")
    return np.array(data, dtype=np.uint8, order="C", copy=True)


def _mat_depth_m(mat: Any, sdk: Any) -> np.ndarray:
    data = np.asarray(mat.get_data(sdk.MEM.CPU))
    if data.ndim != 2:
        raise ZedStereoError(f"Expected registered metric depth, received shape {data.shape}.")
    return np.array(data, dtype=np.float32, order="C", copy=True)


class ZedStereoSource:
    """Registered RGB-D frames from full-rate live capture or ordered SVO replay.

    Live acquisition owns the ZED from one background thread.  That thread keeps
    calling ``grab`` at the configured camera rate, so an enabled SVO recording is
    not paced by PIDNet or reconstruction. Consumers receive the newest
    complete registered frame through a capacity-one buffer. SVO replay remains
    synchronous so every stored frame is processed in order.
    """

    def __init__(
        self,
        svo_path: str | Path | None = None,
        *,
        record_path: str | Path | None = None,
        manual_exposure: int | None = None,
        manual_gain: int | None = None,
        enable_depth: bool = True,
    ) -> None:
        manual_exposure, manual_gain = _validate_manual_settings(
            manual_exposure,
            manual_gain,
        )
        self._svo_path = None if svo_path is None else Path(svo_path).expanduser().resolve()
        if self._svo_path is not None and manual_exposure is not None:
            raise ValueError("Manual camera settings cannot be applied during SVO replay.")
        sdk = _require_zed()
        self._sdk = sdk
        self._depth_enabled = bool(enable_depth)
        self._camera = sdk.Camera()
        self._left_mat = sdk.Mat()
        self._depth_mat = sdk.Mat()
        self._sensors_data = sdk.SensorsData()
        self._sequence_index = 0
        self._previous_timestamp_ns: int | None = None
        self._closed = False
        self._recording_enabled = False
        self._recording_frame_count = 0
        self._last_recorded_timestamp_ns: int | None = None
        self._last_recording_frame_count = 0
        self._recording_start_source_position: int | None = None
        self.capture_settings: CameraCaptureSettings | None = None
        self._original_capture_settings: CameraCaptureSettings | None = None
        self._restore_capture_settings_on_close = False
        requested_record_path = (
            None if record_path is None else Path(record_path).expanduser().resolve()
        )
        self.record_path: Path | None = None
        if self._svo_path is not None and requested_record_path is not None:
            self.close()
            raise ValueError("SVO replay and live SVO recording are mutually exclusive.")
        if self._svo_path is not None and not self._svo_path.is_file():
            self.close()
            raise FileNotFoundError(f"SVO file not found: {self._svo_path}")
        initialization = sdk.InitParameters()
        initialization.coordinate_units = sdk.UNIT.METER
        initialization.coordinate_system = sdk.COORDINATE_SYSTEM.IMAGE
        initialization.camera_resolution = sdk.RESOLUTION.HD1080
        initialization.camera_fps = 30
        # Depth is caller-controlled. Offline planar capture disables it; the
        # online RGB-D tracker enables the highest-quality registered mode.
        initialization.depth_mode = (
            sdk.DEPTH_MODE.NEURAL_PLUS if self._depth_enabled else sdk.DEPTH_MODE.NONE
        )
        if self._depth_enabled:
            initialization.depth_minimum_distance = 0.20
            initialization.depth_maximum_distance = 4.0
        if self._svo_path is not None:
            initialization.set_from_svo_file(str(self._svo_path))
            initialization.svo_real_time_mode = False
        status = self._camera.open(initialization)
        if status != sdk.ERROR_CODE.SUCCESS:
            self.close()
            label = "live ZED" if self._svo_path is None else str(self._svo_path)
            raise ZedStereoError(f"Could not open {label}: {status}")

        if self._svo_path is None:
            try:
                original = _read_capture_settings(
                    self._camera,
                    sdk,
                    requested_manual=False,
                )
            except BaseException:
                self.close()
                raise
            self._original_capture_settings = original
            if manual_exposure is None or manual_gain is None:
                try:
                    self.capture_settings = _apply_automatic_settings(self._camera, sdk)
                except BaseException:
                    self.close()
                    raise
                self._restore_capture_settings_on_close = not original.automatic_exposure_gain
            else:
                try:
                    self.capture_settings = _apply_manual_settings(
                        self._camera,
                        sdk,
                        manual_exposure,
                        manual_gain,
                    )
                except BaseException:
                    try:
                        _restore_capture_settings(self._camera, sdk, original)
                    finally:
                        self.close()
                    raise
                self._restore_capture_settings_on_close = True

        information = self._camera.get_camera_information()
        camera_configuration = information.camera_configuration
        resolution = camera_configuration.resolution
        parameters = camera_configuration.calibration_parameters
        # ZED SDK camera_imu_transform maps IMU-frame vectors into the
        # rectified left-camera frame (see sl::SensorsConfiguration).
        imu_to_camera = np.asarray(
            information.sensors_configuration.camera_imu_transform.m,
            dtype=np.float64,
        )
        baseline = float(parameters.get_camera_baseline())
        if not 0.02 <= baseline <= 0.5:
            self.close()
            raise ZedStereoError(
                f"Implausible metric ZED baseline {baseline:g}m; check coordinate units."
            )
        calibration = StereoCalibration(
            width_px=int(resolution.width),
            height_px=int(resolution.height),
            fps=float(camera_configuration.fps),
            left_intrinsics=_intrinsic_matrix(parameters.left_cam),
            right_intrinsics=_intrinsic_matrix(parameters.right_cam),
            baseline_m=baseline,
            serial_number=int(information.serial_number),
            camera_model=str(information.camera_model),
            imu_to_camera_transform=imu_to_camera,
        )
        if (calibration.width_px, calibration.height_px) != (1920, 1080):
            self.close()
            raise ZedStereoError(
                f"Source opened at {calibration.width_px}x{calibration.height_px}; "
                "Phase 1 requires native HD1080."
            )
        if abs(calibration.fps - 30.0) > 1.0e-6:
            self.close()
            raise ZedStereoError(
                f"Source opened at {calibration.fps:g} FPS; Phase 1 requires 30 FPS."
            )
        kind = "live" if self._svo_path is None else "svo"
        label = (
            f"{calibration.camera_model} S/N {calibration.serial_number}"
            if self._svo_path is None
            else str(self._svo_path)
        )
        total_frames = (
            None if self._svo_path is None else int(self._camera.get_svo_number_of_frames())
        )
        self.descriptor = SourceDescriptor(
            label=label,
            kind=kind,
            calibration=calibration,
            total_frames=total_frames,
            source_path=self._svo_path,
        )
        self._runtime = sdk.RuntimeParameters()
        self._runtime.enable_depth = self._depth_enabled
        if self._depth_enabled:
            self._runtime.confidence_threshold = 50
            self._runtime.texture_confidence_threshold = 100
        self._live_condition = threading.Condition()
        self._latest_live_frame: StereoFrame | None = None
        self._last_delivered_live_index = -1
        self._live_error: BaseException | None = None
        self._live_finished = False
        self._live_stop = threading.Event()
        self._live_commands: queue.Queue[
            tuple[str, Any, threading.Event, dict[str, Any]]
        ] = queue.Queue()
        self._live_thread: threading.Thread | None = None
        if self._svo_path is None:
            self._live_thread = threading.Thread(
                target=self._live_capture_loop,
                name="zed-full-rate-capture",
                daemon=True,
            )
            self._live_thread.start()
        if requested_record_path is not None:
            try:
                self.start_recording(requested_record_path)
            except Exception:
                self.close()
                raise

    @property
    def is_recording(self) -> bool:
        return self._recording_enabled

    @property
    def last_recording_frame_count(self) -> int:
        return int(self._last_recording_frame_count)

    @property
    def recording_start_source_position(self) -> int | None:
        """Live sequence position corresponding to SVO frame zero while recording."""

        return self._recording_start_source_position

    def _start_recording_sdk(self, output: Path) -> None:
        recording = self._sdk.RecordingParameters()
        recording.video_filename = str(output)
        recording.compression_mode = self._sdk.SVO_COMPRESSION_MODE.H265_LOSSLESS
        recording.target_framerate = 30
        status = self._camera.enable_recording(recording)
        if status != self._sdk.ERROR_CODE.SUCCESS:
            raise ZedStereoError(f"Could not start lossless SVO recording: {status}")
        self.record_path = output
        self._last_recording_frame_count = 0
        self._recording_frame_count = 0
        self._last_recorded_timestamp_ns = None
        self._recording_start_source_position = int(self._sequence_index)
        self._recording_enabled = True

    def _stop_recording_sdk(self) -> Path | None:
        if not self._recording_enabled:
            return None
        output = self.record_path
        self._camera.disable_recording()
        self._recording_enabled = False
        self.record_path = None
        self._last_recording_frame_count = int(self._recording_frame_count)
        self._recording_frame_count = 0
        self._last_recorded_timestamp_ns = None
        self._recording_start_source_position = None
        return output

    def _submit_live_command(self, action: str, value: Any = None) -> Any:
        thread = self._live_thread
        if thread is None or not thread.is_alive():
            if self._live_error is not None:
                raise ZedStereoError("Live ZED capture stopped.") from self._live_error
            raise ZedStereoError("Live ZED capture is not running.")
        completed = threading.Event()
        result: dict[str, Any] = {}
        self._live_commands.put((action, value, completed, result))
        if not completed.wait(timeout=3.0):
            raise ZedStereoError(f"Timed out waiting for live ZED command {action!r}.")
        error = result.get("error")
        if error is not None:
            raise error
        return result.get("value")

    def start_recording(self, path: str | Path) -> Path:
        """Begin lossless SVO2 capture without interrupting the live stream."""

        if self._closed:
            raise ZedStereoError("Cannot record from a closed ZED source.")
        if self._svo_path is not None:
            raise ZedStereoError("SVO replay cannot be recorded again.")
        if self._recording_enabled:
            raise ZedStereoError(f"ZED is already recording to {self.record_path}.")
        output = Path(path).expanduser().resolve()
        if output.suffix.lower() != ".svo2":
            raise ValueError("Live recording output must use the .svo2 extension.")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite SVO recording: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

        self._submit_live_command("start_recording", output)
        return output

    def stop_recording(self) -> Path | None:
        """Finish the active SVO2 while leaving camera acquisition running."""

        if self._svo_path is not None:
            return self._stop_recording_sdk()
        return self._submit_live_command("stop_recording")

    def set_manual_exposure_gain(
        self,
        exposure: int,
        gain: int,
    ) -> CameraCaptureSettings:
        """Apply and verify live manual settings on the acquisition thread."""

        exposure_value, gain_value = _validate_manual_settings(exposure, gain)
        assert exposure_value is not None and gain_value is not None
        if self._svo_path is not None:
            raise ZedStereoError("Camera settings cannot be changed during SVO replay.")
        if self._recording_enabled:
            raise ZedStereoError("Stop recording before changing exposure or gain.")
        settings = self._submit_live_command(
            "manual_settings",
            (exposure_value, gain_value),
        )
        if not isinstance(settings, CameraCaptureSettings):
            raise ZedStereoError("Live ZED returned invalid manual-setting readback.")
        self.capture_settings = settings
        self._restore_capture_settings_on_close = True
        return settings

    def set_automatic_exposure_gain(self) -> CameraCaptureSettings:
        """Enable and verify live automatic exposure/gain."""

        if self._svo_path is not None:
            raise ZedStereoError("Camera settings cannot be changed during SVO replay.")
        if self._recording_enabled:
            raise ZedStereoError("Stop recording before changing exposure or gain.")
        settings = self._submit_live_command("automatic_settings")
        if not isinstance(settings, CameraCaptureSettings):
            raise ZedStereoError("Live ZED returned invalid automatic-setting readback.")
        self.capture_settings = settings
        self._restore_capture_settings_on_close = True
        return settings

    def _read_camera_frame(self) -> StereoFrame | None:
        sdk = self._sdk
        duplicate_started_s: float | None = None
        while True:
            status = self._camera.grab(self._runtime)
            if status == sdk.ERROR_CODE.END_OF_SVOFILE_REACHED:
                return None
            if status != sdk.ERROR_CODE.SUCCESS:
                raise ZedStereoError(f"ZED stereo grab failed: {status}")
            timestamp_ns = int(
                self._camera.get_timestamp(sdk.TIME_REFERENCE.IMAGE).get_nanoseconds()
            )
            if timestamp_ns <= 0:
                raise ZedStereoError(f"ZED returned invalid IMAGE timestamp {timestamp_ns}.")
            if self._recording_enabled and (
                self._last_recorded_timestamp_ns is None
                or timestamp_ns > self._last_recorded_timestamp_ns
            ):
                # The SVO encoder stores unique IMAGE times. A repeated SUCCESS
                # grab is not another stored frame and must not inflate the count.
                self._recording_frame_count += 1
                self._last_recorded_timestamp_ns = timestamp_ns
            previous_timestamp_ns = self._previous_timestamp_ns
            if previous_timestamp_ns is None or timestamp_ns > previous_timestamp_ns:
                break
            if timestamp_ns == previous_timestamp_ns and self._svo_path is None:
                now_s = time.monotonic()
                if duplicate_started_s is None:
                    duplicate_started_s = now_s
                elif now_s - duplicate_started_s >= _LIVE_TIMESTAMP_STALL_TIMEOUT_S:
                    raise ZedStereoError(
                        "Live ZED IMAGE timestamp remained unchanged for "
                        f"{_LIVE_TIMESTAMP_STALL_TIMEOUT_S:g}s at {timestamp_ns}."
                    )
                continue
            raise ZedStereoError(
                f"ZED IMAGE timestamp did not increase: {previous_timestamp_ns} -> {timestamp_ns}."
            )
        left_status = self._camera.retrieve_image(self._left_mat, sdk.VIEW.LEFT_BGR, sdk.MEM.CPU)
        if left_status != sdk.ERROR_CODE.SUCCESS:
            raise ZedStereoError(f"Registered left-image retrieval failed: {left_status}.")
        if self._depth_enabled:
            depth_status = self._camera.retrieve_measure(
                self._depth_mat,
                sdk.MEASURE.DEPTH,
                sdk.MEM.CPU,
            )
            if depth_status != sdk.ERROR_CODE.SUCCESS:
                raise ZedStereoError(f"Registered ZED depth retrieval failed: {depth_status}.")
        left_timestamp = int(self._left_mat.timestamp.get_nanoseconds())
        if left_timestamp != timestamp_ns:
            raise ZedStereoError(
                "Registered image synchronization failed: "
                f"camera={timestamp_ns}, left={left_timestamp}."
            )
        sensor_status = self._camera.get_sensors_data(
            self._sensors_data,
            sdk.TIME_REFERENCE.IMAGE,
        )
        imu_timestamp_ns: int | None = None
        gravity_camera: np.ndarray | None = None
        angular_velocity_camera: np.ndarray | None = None
        if sensor_status == sdk.ERROR_CODE.SUCCESS:
            imu = self._sensors_data.get_imu_data()
            if bool(imu.is_available):
                imu_timestamp_ns = int(imu.timestamp.get_nanoseconds())
                if imu_timestamp_ns <= 0:
                    raise ZedStereoError(
                        f"ZED returned invalid IMU timestamp {imu_timestamp_ns}."
                    )
                acceleration_imu = np.asarray(
                    imu.get_linear_acceleration(), dtype=np.float64
                )
                angular_velocity_imu = np.asarray(
                    imu.get_angular_velocity(), dtype=np.float64
                )
                imu_to_camera_rotation = (
                    self.descriptor.calibration.imu_to_camera_transform[:3, :3]
                )
                # An accelerometer at rest measures proper acceleration opposite gravity.
                gravity_camera = -(imu_to_camera_rotation @ acceleration_imu)
                angular_velocity_camera = (
                    imu_to_camera_rotation @ angular_velocity_imu
                )
        source_position = (
            self._sequence_index
            if self._svo_path is None
            else int(self._camera.get_svo_position())
        )
        frame = StereoFrame(
            sequence_index=self._sequence_index,
            source_position=source_position,
            timestamp_ns=timestamp_ns,
            left_bgr=_mat_bgr(self._left_mat, sdk),
            depth_m=(
                _mat_depth_m(self._depth_mat, sdk) if self._depth_enabled else None
            ),
            imu_timestamp_ns=imu_timestamp_ns,
            gravity_camera_m_s2=gravity_camera,
            angular_velocity_camera_deg_s=angular_velocity_camera,
        )
        self._previous_timestamp_ns = timestamp_ns
        self._sequence_index += 1
        return frame

    def _service_live_commands(self) -> None:
        while True:
            try:
                action, value, completed, result = self._live_commands.get_nowait()
            except queue.Empty:
                return
            try:
                if action == "start_recording":
                    if not isinstance(value, Path):
                        raise ValueError("A recording path is required.")
                    self._start_recording_sdk(value)
                    result["value"] = value
                elif action == "stop_recording":
                    result["value"] = self._stop_recording_sdk()
                elif action == "manual_settings":
                    if not isinstance(value, tuple) or len(value) != 2:
                        raise ValueError("Manual exposure and gain are required.")
                    settings = _apply_manual_settings(
                        self._camera,
                        self._sdk,
                        int(value[0]),
                        int(value[1]),
                    )
                    self.capture_settings = settings
                    result["value"] = settings
                elif action == "automatic_settings":
                    settings = _apply_automatic_settings(self._camera, self._sdk)
                    self.capture_settings = settings
                    result["value"] = settings
                else:
                    raise ValueError(f"Unknown live ZED command {action!r}.")
            except BaseException as error:
                result["error"] = error
            finally:
                completed.set()

    def _live_capture_loop(self) -> None:
        try:
            while not self._live_stop.is_set():
                self._service_live_commands()
                if self._live_stop.is_set():
                    break
                frame = self._read_camera_frame()
                if frame is None:
                    break
                with self._live_condition:
                    self._latest_live_frame = frame
                    self._live_condition.notify_all()
        except BaseException as error:
            self._live_error = error
        finally:
            try:
                self._stop_recording_sdk()
            except BaseException as error:
                if self._live_error is None:
                    self._live_error = error
            while True:
                try:
                    _action, _value, completed, result = self._live_commands.get_nowait()
                except queue.Empty:
                    break
                result["error"] = ZedStereoError("Live ZED capture stopped.")
                completed.set()
            with self._live_condition:
                self._live_finished = True
                self._live_condition.notify_all()

    def read(self) -> StereoFrame | None:
        if self._closed:
            return None
        if self._svo_path is not None:
            return self._read_camera_frame()
        with self._live_condition:
            self._live_condition.wait_for(
                lambda: (
                    self._latest_live_frame is not None
                    and self._latest_live_frame.sequence_index
                    > self._last_delivered_live_index
                )
                or self._live_finished
                or self._live_error is not None
            )
            if self._live_error is not None:
                raise ZedStereoError("Live ZED capture failed.") from self._live_error
            frame = self._latest_live_frame
            if frame is None or frame.sequence_index <= self._last_delivered_live_index:
                return None
            self._last_delivered_live_index = frame.sequence_index
            return frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        camera = getattr(self, "_camera", None)
        live_thread = getattr(self, "_live_thread", None)
        if live_thread is not None:
            if self._recording_enabled:
                try:
                    self.stop_recording()
                except ZedStereoError:
                    pass
            self._live_stop.set()
            with self._live_condition:
                self._live_condition.notify_all()
            live_thread.join(timeout=3.0)
            if live_thread.is_alive():
                raise ZedStereoError("Timed out stopping the live ZED capture thread.")
        elif self._recording_enabled and camera is not None:
            self._stop_recording_sdk()
        for mat_name in ("_left_mat", "_depth_mat"):
            mat = getattr(self, mat_name, None)
            if mat is not None:
                mat.free()
        if camera is not None:
            try:
                if self._restore_capture_settings_on_close:
                    original = self._original_capture_settings
                    if original is None:
                        raise ZedStereoError(
                            "Original ZED capture settings were not retained for restoration."
                        )
                    _restore_capture_settings(camera, self._sdk, original)
                    self._restore_capture_settings_on_close = False
            finally:
                camera.close()

    def __enter__(self) -> "ZedStereoSource":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
