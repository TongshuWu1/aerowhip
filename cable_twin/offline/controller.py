"""Small subprocess controller for recording, 2D review, and fitting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import uuid
from typing import Any, Sequence

import numpy as np

from ..shared.capture_manifest import capture_manifest_path
from .config import DEFAULT_CONFIG_PATH, OfflineSettings
from .planar_data import validate_planar_observation
from .planar_data import load_planar_trajectory
from ..shared.observation_data import trajectory_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVENT_PREFIX = "OFFLINE_DDER_EVENT "


@dataclass(frozen=True, slots=True)
class SessionPaths:
    svo: Path
    observations: Path
    trajectory: Path


@dataclass(frozen=True, slots=True)
class ControllerEvent:
    kind: str
    text: str = ""
    task: str = ""
    return_code: int | None = None
    output_path: Path | None = None
    frame_count: int | None = None
    camera_automatic: bool | None = None
    camera_exposure: int | None = None
    camera_gain: int | None = None


@dataclass(slots=True)
class _ActiveProcess:
    task: str
    process: subprocess.Popen[str]
    reader: threading.Thread
    output_path: Path | None
    control_file: Path | None = None


class OfflineDderController:
    def __init__(self, settings: OfflineSettings) -> None:
        self.settings = settings
        self._messages: queue.Queue[str] = queue.Queue()
        self._active: _ActiveProcess | None = None
        self._recording: SessionPaths | None = None
        self._control_revision = 0
        for directory in (
            settings.recording_directory,
            settings.observation_directory,
            settings.model_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def active_task(self) -> str | None:
        if self._recording is not None:
            return "recording"
        return None if self._active is None else self._active.task

    @property
    def preview_running(self) -> bool:
        return self._active is not None and self._active.task == "preview" and self._active.process.poll() is None

    @property
    def is_recording(self) -> bool:
        return self._recording is not None

    def _launch(
        self,
        task: str,
        command: Sequence[str],
        output_path: Path | None,
        *,
        control_file: Path | None = None,
    ) -> None:
        if self._active is not None:
            raise RuntimeError(f"{self._active.task} is already running.")
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            list(command),
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE if task == "preview" else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        def read_output() -> None:
            if process.stdout is not None:
                for line in process.stdout:
                    self._messages.put(line.rstrip("\r\n"))

        reader = threading.Thread(target=read_output, name=f"cable-model-{task}", daemon=True)
        reader.start()
        self._active = _ActiveProcess(task, process, reader, output_path, control_file)

    def start_preview(
        self,
        *,
        manual_exposure: int | None = None,
        manual_gain: int | None = None,
    ) -> None:
        if self._active is not None:
            raise RuntimeError(f"{self._active.task} is already running.")
        control_file = self.settings.observation_directory / f".live_{uuid.uuid4().hex}.json"
        command = [
            sys.executable,
            "-m",
            "cable_twin.offline.recording",
            "--control-file",
            str(control_file),
            "--exit-on-stdin-close",
        ]
        if manual_exposure is not None or manual_gain is not None:
            if manual_exposure is None or manual_gain is None:
                raise ValueError("Manual exposure and gain must be provided together.")
            command.extend(
                (
                    "--manual-exposure",
                    str(int(manual_exposure)),
                    "--manual-gain",
                    str(int(manual_gain)),
                )
            )
        self._control_revision = 0
        self._launch("preview", command, None, control_file=control_file)

    def set_camera_manual(self, exposure: int, gain: int) -> None:
        self._write_preview_command(
            "set_camera",
            mode="manual",
            exposure=int(exposure),
            gain=int(gain),
        )

    def set_camera_automatic(self) -> None:
        self._write_preview_command("set_camera", mode="automatic")

    def _write_preview_command(self, action: str, **values: Any) -> None:
        active = self._active
        if active is None or active.task != "preview" or active.control_file is None:
            raise RuntimeError("The ZED RGB recorder is not running.")
        self._control_revision += 1
        payload = {"revision": self._control_revision, "action": action, **values}
        target = active.control_file
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, target)

    def start_recording(self, cable_identity: int) -> SessionPaths:
        if int(cable_identity) not in (1, 2):
            raise ValueError("Cable identity must be 1 or 2.")
        if self._recording is not None:
            raise RuntimeError("A synchronized recording is already active.")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        stem = f"fit_{timestamp}_{uuid.uuid4().hex[:8]}_cable{int(cable_identity)}"
        svo = self.settings.recording_directory / f"{stem}.svo2"
        observations = self.settings.observation_directory / f"{stem}.npz"
        paths = SessionPaths(svo, observations, trajectory_path(observations))
        self._write_preview_command(
            "start_recording",
            cable_identity=int(cable_identity),
            svo_path=str(svo),
            observation_path=str(observations),
        )
        self._recording = paths
        return paths

    def request_stop_recording(self) -> None:
        if self._recording is not None:
            self._write_preview_command("stop_recording")

    def recording_artifacts(self, selected: str | Path) -> SessionPaths:
        path = Path(selected).expanduser().resolve()
        recording_directory = self.settings.recording_directory.resolve()
        observation_directory = self.settings.observation_directory.resolve()
        if path.suffix.lower() == ".svo2" and path.parent == recording_directory:
            stem = path.stem
        elif path.suffix.lower() == ".npz" and path.parent == observation_directory:
            stem = path.stem.removesuffix(".trajectory")
        else:
            raise ValueError("Selected file is outside the configured recording directories.")
        observation = observation_directory / f"{stem}.npz"
        return SessionPaths(
            recording_directory / f"{stem}.svo2",
            observation,
            trajectory_path(observation),
        )

    @staticmethod
    def cable_identity(session: SessionPaths) -> int:
        match = re.search(r"_cable([12])$", session.svo.stem)
        if match is None:
            raise ValueError("Recording name does not contain _cable1 or _cable2.")
        return int(match.group(1))

    def delete_recordings(self, selected: Sequence[str | Path]) -> tuple[Path, ...]:
        if self.active_task not in (None, "preview"):
            raise RuntimeError("Wait for the active task before deleting recordings.")
        sessions = tuple(self.recording_artifacts(value) for value in selected)
        if not sessions:
            raise ValueError("Select at least one recording.")
        deleted: list[Path] = []
        for session in sessions:
            for target in (
                session.trajectory,
                session.observations,
                capture_manifest_path(session.svo),
                session.svo,
            ):
                if target.is_file():
                    target.unlink()
                    deleted.append(target)
        return tuple(deleted)

    def _stop_preview(self) -> None:
        active = self._active
        if active is None or active.task != "preview":
            return
        if self._recording is not None:
            raise RuntimeError("Stop recording first.")
        try:
            self._write_preview_command("exit")
            active.process.wait(timeout=5.0)
        except (RuntimeError, subprocess.TimeoutExpired):
            active.process.terminate()
            active.process.wait(timeout=2.0)
        active.reader.join(timeout=2.0)
        self._cleanup_control_file(active.control_file)
        self._active = None

    def start_fit(self, sessions: Sequence[SessionPaths], *, iterations: int) -> Path:
        if not sessions:
            raise ValueError("Select at least one recording.")
        resolved = tuple(self.recording_artifacts(item.svo) for item in sessions)
        for item in resolved:
            if not item.svo.is_file():
                raise FileNotFoundError(f"Raw recording not found: {item.svo}")
        identities = {self.cable_identity(item) for item in resolved}
        if len(identities) != 1:
            raise ValueError("Selected recordings belong to different cables.")
        self._stop_preview()
        output = self.settings.model_directory / f"cable{next(iter(identities))}_dder.json"
        command = (
            sys.executable,
            "-m",
            "cable_twin.offline.pipeline",
            "--svo",
            *(str(item.svo) for item in resolved),
            "--observation",
            *(str(item.observations) for item in resolved),
            "--output",
            str(output),
            "--config",
            str(DEFAULT_CONFIG_PATH),
            "--iterations",
            str(int(iterations)),
        )
        self._launch("fit", command, output)
        return output

    def start_2d_view(self, selected: str | Path) -> SessionPaths:
        session = self.recording_artifacts(selected)
        if not session.svo.is_file():
            raise FileNotFoundError(f"Raw recording not found: {session.svo}")
        self._stop_preview()
        current_observations = self.has_current_observations(session)
        if current_observations and self.has_current_trajectory(session):
            command = (
                sys.executable,
                "-m",
                "cable_twin.offline.replay",
                "--svo",
                str(session.svo),
                "--observation",
                str(session.observations),
            )
            self._launch("view_2d", command, session.trajectory)
        else:
            command = (
                sys.executable,
                "-m",
                "cable_twin.offline.reconstruct",
                "--svo",
                str(session.svo),
                "--observation",
                str(session.observations),
                "--cable",
                str(self.cable_identity(session)),
                "--config",
                str(DEFAULT_CONFIG_PATH),
            )
            self._launch("process_2d", command, session.trajectory)
        return session

    def has_current_observations(self, session: SessionPaths) -> bool:
        if not session.observations.is_file():
            return False
        try:
            with np.load(session.observations, allow_pickle=False) as data:
                metadata = validate_planar_observation(data)
            recorded_length = float(
                metadata["image_plane_mapping"]["cable_length_m"]
            )
        except (KeyError, TypeError, ValueError, OSError):
            return False
        return math.isclose(
            recorded_length,
            self.settings.cable.length_m,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        )

    def has_current_trajectory(self, session: SessionPaths) -> bool:
        if not session.trajectory.is_file() or not session.observations.is_file():
            return False
        try:
            trajectory = load_planar_trajectory(
                session.trajectory,
                observation_path=session.observations,
            )
            metadata = trajectory.metadata
            return bool(
                metadata.get("method") == "pidnet_route_nodes_without_position_filter"
                and int(metadata.get("node_count", 0)) == self.settings.cable.node_count
                and int(metadata.get("velocity_window_frames", 0))
                == self.settings.optimization.velocity_window_frames
                and math.isclose(
                    float(metadata.get("cable_length_m", float("nan"))),
                    self.settings.cable.length_m,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-12,
                )
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return False

    @staticmethod
    def _cleanup_control_file(path: Path | None) -> None:
        if path is None:
            return
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".tmp").unlink(missing_ok=True)

    def _protocol_event(self, line: str) -> ControllerEvent | None:
        if not line.startswith(EVENT_PREFIX):
            return ControllerEvent(kind="log", text=line)
        payload = json.loads(line[len(EVENT_PREFIX) :])
        event = str(payload.get("event", ""))
        if event in ("ready", "capture_settings"):
            return ControllerEvent(
                kind="preview_ready" if event == "ready" else "capture_settings",
                camera_automatic=bool(payload["automatic"]),
                camera_exposure=int(payload["exposure"]),
                camera_gain=int(payload["gain"]),
            )
        if event == "recording_started":
            return ControllerEvent(kind="recording_started")
        if event == "capture_saved":
            output = Path(str(payload["svo_path"])).resolve()
            if self._recording is None or output != self._recording.svo.resolve():
                raise RuntimeError("Recorder returned an unexpected SVO path.")
            self._recording = None
            return ControllerEvent(
                kind="recording_finished",
                task="recording",
                return_code=0,
                output_path=output,
                frame_count=int(payload.get("frame_count", 0)),
            )
        return None

    def poll(self) -> list[ControllerEvent]:
        events: list[ControllerEvent] = []
        while True:
            try:
                event = self._protocol_event(self._messages.get_nowait())
                if event is not None:
                    events.append(event)
            except queue.Empty:
                break
            except Exception as error:
                events.append(ControllerEvent(kind="controller_error", text=str(error)))
        active = self._active
        if active is not None and active.process.poll() is not None and not active.reader.is_alive():
            code = int(active.process.returncode)
            self._cleanup_control_file(active.control_file)
            self._active = None
            if active.task == "preview" and self._recording is not None:
                interrupted = self._recording
                self._recording = None
                events.append(
                    ControllerEvent(
                        kind="recording_finished",
                        task="recording",
                        return_code=code if code != 0 else 1,
                        output_path=interrupted.svo,
                    )
                )
            events.append(
                ControllerEvent(
                    kind="finished",
                    task=active.task,
                    return_code=code,
                    output_path=active.output_path,
                )
            )
        return events

    def shutdown(self) -> None:
        active = self._active
        if active is None:
            return
        if active.task == "preview":
            try:
                self._write_preview_command("exit")
                active.process.wait(timeout=4.0)
            except (RuntimeError, subprocess.TimeoutExpired):
                active.process.terminate()
        else:
            active.process.terminate()
        try:
            active.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            active.process.kill()
            active.process.wait(timeout=2.0)
        active.reader.join(timeout=2.0)
        self._cleanup_control_file(active.control_file)
        self._recording = None
        self._active = None
