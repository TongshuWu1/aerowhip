"""Minimal split RGB/OpenGL point-cloud viewer for the segmentation pipeline."""

from __future__ import annotations

import ctypes
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Event, Lock, Thread
import time

import cv2
import numpy as np

from cable_geometry import swept_capsule_mesh
from cube_tracker import CUBE_CORNERS, CUBE_EDGES
from particle_filter import (
    VISIBILITY_MISSING,
    VISIBILITY_SUPPORTED,
    VISIBILITY_UNKNOWN,
)


_FREEGLUT_HANDLE = None


def _preload_freeglut():
    if sys.platform != "win32":
        return None
    candidates = (
        Path(sys.prefix) / "Lib" / "site-packages" / "OpenGL" / "DLLS" / "freeglut.dll",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/extras/demo_suite/freeglut.dll"),
        Path("C:/Program Files (x86)/ZED SDK/dependencies/freeglut_2.8/x64/freeglut.dll"),
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            return ctypes.WinDLL(str(path))
        except OSError:
            continue
    return None


_FREEGLUT_HANDLE = _preload_freeglut()
if _FREEGLUT_HANDLE is not None:
    import OpenGL.platform

    OpenGL.platform.PLATFORM.GLUT = _FREEGLUT_HANDLE

from OpenGL.GL import *  # noqa: E402,F403
from OpenGL.GLU import *  # noqa: E402,F403
from OpenGL.GLUT import *  # noqa: E402,F403


_GLUT_INITIALIZED = False

UI_BG = (0.015, 0.017, 0.020)
UI_PANEL = (0.035, 0.040, 0.046)
UI_PANEL_DARK = (0.022, 0.025, 0.030)
UI_STROKE = (0.110, 0.125, 0.140)
UI_TEXT = (0.910, 0.940, 0.965)
UI_MUTED = (0.580, 0.640, 0.700)
UI_ACCENT = (0.270, 0.620, 0.960)
ORANGE = (1.00, 0.58, 0.08)
BLUE = (0.05, 0.35, 1.00)
GREEN = (0.15, 1.00, 0.30)
SKELETON_MASK = (0.15, 0.17, 0.19)
SKELETON_LINE = (0.93, 0.96, 0.98)
SKELETON_NODE = (0.22, 0.80, 1.00)
SKELETON_BRANCH = (1.00, 0.28, 0.35)
CUBE_TRACKING = (1.00, 0.22, 0.86)


class DiagnosticWindowRecorder:
    """Bounded asynchronous encoder for complete viewer-window recordings."""

    def __init__(
        self,
        output_directory: str | Path,
        fps: float = 30.0,
        codec: str = "avc1",
        queue_frames: int = 8,
    ):
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.fps = float(np.clip(fps, 1.0, 60.0))
        self.preferred_codec = str(codec)
        self.queue_frames = max(2, int(queue_frames))
        self.lock = Lock()
        self.queue: Queue | None = None
        self.stop_event: Event | None = None
        self.thread: Thread | None = None
        self.active = False
        self.path: Path | None = None
        self.codec = ""
        self.frame_size = (0, 0)
        self.started_at = 0.0
        self.stopped_at = 0.0
        self.next_capture_index = 0
        self.pending_capture_index = 0
        self.target_frame_count = 0
        self.written_frames = 0
        self.dropped_frames = 0
        self.error = ""

    def _open_writer(
        self,
        path: Path,
        frame_size: tuple[int, int],
    ) -> tuple[cv2.VideoWriter, str]:
        attempts = []
        preferred = self.preferred_codec
        if sys.platform == "win32" and preferred in ("avc1", "H264"):
            attempts.append((cv2.CAP_MSMF, "H264"))
        elif len(preferred) == 4:
            attempts.append((cv2.CAP_ANY, preferred))
        attempts.append((cv2.CAP_ANY, "mp4v"))
        for backend, codec in attempts:
            writer = cv2.VideoWriter(
                str(path),
                backend,
                cv2.VideoWriter_fourcc(*codec),
                self.fps,
                frame_size,
            )
            if writer.isOpened():
                return writer, codec
            writer.release()
        raise RuntimeError(
            "OpenCV could not open an H.264 or MPEG-4 video encoder."
        )

    def start(self, width: int, height: int) -> Path:
        width = max(2, int(width) - int(width) % 2)
        height = max(2, int(height) - int(height) % 2)
        with self.lock:
            if self.active:
                raise RuntimeError("A diagnostic recording is already active.")
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError("The previous recording is still finalizing.")

        self.output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_directory / f"pf_diagnostic_{timestamp}.mp4"
        writer, codec = self._open_writer(path, (width, height))
        frame_queue: Queue = Queue(maxsize=self.queue_frames)
        stop_event = Event()
        started_at = time.perf_counter()
        thread = Thread(
            target=self._encode,
            args=(writer, frame_queue, stop_event, (width, height)),
            name="diagnostic-window-recorder",
            daemon=True,
        )
        with self.lock:
            self.queue = frame_queue
            self.stop_event = stop_event
            self.thread = thread
            self.active = True
            self.path = path
            self.codec = codec
            self.frame_size = (width, height)
            self.started_at = started_at
            self.stopped_at = 0.0
            self.next_capture_index = 0
            self.pending_capture_index = 0
            self.target_frame_count = 0
            self.written_frames = 0
            self.dropped_frames = 0
            self.error = ""
        thread.start()
        return path

    def should_capture(self, now: float) -> bool:
        with self.lock:
            if not self.active:
                return False
            capture_index = max(
                0,
                int((now - self.started_at) * self.fps),
            )
            if capture_index < self.next_capture_index:
                return False
            self.dropped_frames += max(
                0,
                capture_index - self.next_capture_index,
            )
            self.pending_capture_index = capture_index
            self.next_capture_index = capture_index + 1
            return True

    def submit(self, frame: np.ndarray) -> None:
        with self.lock:
            if not self.active or self.queue is None:
                return
            frame_queue = self.queue
            capture_index = self.pending_capture_index
        try:
            frame_queue.put_nowait((capture_index, frame))
        except Full:
            with self.lock:
                self.dropped_frames += 1

    def _encode(
        self,
        writer: cv2.VideoWriter,
        frame_queue: Queue,
        stop_event: Event,
        frame_size: tuple[int, int],
    ) -> None:
        last_frame = None
        last_capture_index = -1
        written_frames = 0
        try:
            while not stop_event.is_set() or not frame_queue.empty():
                try:
                    capture_index, frame = frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
                    frame = cv2.resize(
                        frame,
                        frame_size,
                        interpolation=cv2.INTER_AREA,
                    )
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                if last_frame is None:
                    repeated_frames = max(0, capture_index)
                    repeated_frame = frame
                else:
                    repeated_frames = max(
                        0,
                        capture_index - last_capture_index - 1,
                    )
                    repeated_frame = last_frame
                for _ in range(repeated_frames):
                    writer.write(repeated_frame)
                writer.write(frame)
                written_frames += repeated_frames + 1
                with self.lock:
                    self.written_frames = written_frames
                last_frame = frame
                last_capture_index = capture_index

            with self.lock:
                target_frame_count = self.target_frame_count
            while last_frame is not None and written_frames < target_frame_count:
                writer.write(last_frame)
                written_frames += 1
            with self.lock:
                self.written_frames = written_frames
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            writer.release()
            with self.lock:
                self.active = False
                if self.stopped_at <= 0.0:
                    self.stopped_at = time.perf_counter()

    def stop(self, timeout_s: float = 10.0) -> dict:
        with self.lock:
            stop_event = self.stop_event
            thread = self.thread
            if self.active:
                self.active = False
                self.stopped_at = time.perf_counter()
                elapsed_s = max(0.0, self.stopped_at - self.started_at)
                self.target_frame_count = max(
                    self.next_capture_index,
                    int(np.ceil(elapsed_s * self.fps)),
                )
            if stop_event is not None:
                stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.1, float(timeout_s)))
        with self.lock:
            if thread is not None and thread.is_alive():
                self.error = "Video encoder did not finish within the timeout."
            if self.stopped_at <= 0.0:
                self.stopped_at = time.perf_counter()
        return self.snapshot()

    def snapshot(self) -> dict:
        with self.lock:
            end_time = (
                time.perf_counter()
                if self.active
                else self.stopped_at
            )
            elapsed = (
                max(0.0, end_time - self.started_at)
                if self.started_at > 0.0
                else 0.0
            )
            return {
                "active": bool(self.active),
                "path": self.path,
                "codec": self.codec,
                "elapsed_s": elapsed,
                "written_frames": int(self.written_frames),
                "dropped_frames": int(self.dropped_frames),
                "error": self.error,
            }

FEATURE_CONTROLS = (
    ("2", "connected_trace", "GRAPH TRACE"),
    ("5", "smoothness", "SMOOTH"),
    ("6", "fixed_length", "FIXED LENGTH"),
    ("7", "temporal_prediction", "TEMPORAL"),
    ("M", "endpoint_motion_transport", "END TRANSPORT"),
    ("E", "ess_resampling", "ESS RESAMPLE"),
    ("9", "global_particles", "RETRACK 10%"),
    ("V", "local_node_motion", "LOCAL NODE MOTION"),
    ("F", "single_endpoint_updates", "ONE ENDPOINT"),
    ("G", "prediction_without_measurement", "PREDICT HIDDEN"),
    ("H", "posterior_uncertainty", "UNCERTAINTY"),
    ("K", "fused_constraint_kernels", "FUSED LINKS"),
    ("J", "cuda_graph_replay", "CUDA GRAPH"),
    ("Y", "graph_edge_attribution", "EDGE ATTR"),
    ("I", "temporal_edge_identity", "EDGE ID MEMORY"),
    ("A", "visible_edge_scoring", "EDGE SCORE"),
    ("D", "visible_edge_transport", "EDGE MOTION"),
    ("S", "visible_edge_exploration", "EDGE RETRACK"),
)

POINT_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_packed_color;
uniform mat4 u_mvp;
uniform float u_point_size;
out vec3 v_color;
void main() {
    uint packed_color = floatBitsToUint(in_packed_color);
    v_color = vec3(
        float(packed_color & 255u),
        float((packed_color >> 8u) & 255u),
        float((packed_color >> 16u) & 255u)
    ) / 255.0;
    gl_Position = u_mvp * vec4(in_position, 1.0);
    gl_PointSize = u_point_size;
}
"""

POINT_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
out vec4 out_color;
void main() { out_color = vec4(v_color, 1.0); }
"""


class SplitPointCloudViewer:
    """Main-thread OpenGL renderer with latest-frame replacement semantics."""

    def __init__(
        self,
        width=1800,
        height=900,
        left_panel_width=620,
        point_size=1.0,
        title="ZED PIDNet Segmentation",
        recording_directory="diagnostics/recordings",
        recording_fps=30.0,
        recording_codec="avc1",
        recording_queue_frames=8,
    ):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.left_panel_width = max(260, int(left_panel_width))
        self.left_panel_ratio = float(np.clip(self.left_panel_width / self.width, 0.25, 0.55))
        self.point_size = float(np.clip(point_size, 1.0, 8.0))
        self.title = str(title)
        self.window_id = None
        self.available = False
        self.redraw_requested = True

        self.lock = Lock()
        self.pending = None
        self.pending_status = "Waiting for camera"
        self.rgb_image = None
        self.rgb_texture_dirty = False
        self.skeleton_image = None
        self.skeleton_texture_dirty = False
        self.vertices = np.empty((0, 4), dtype=np.float32)
        self.vertex_count = 0
        self.cloud_captured_at = None
        self.status = "Waiting for camera"
        self.stats = {
            "total_pixels": 0,
            "valid_points": 0,
            "invalid_points": 0,
            "cable_points": 0,
            "endpoint_1_points": 0,
            "endpoint_2_points": 0,
        }
        self.pipeline_stats = {}
        self.frame_index = -1
        self.inference_ms = 0.0
        self.processing_ms = 0.0
        self.latency_ms = 0.0
        self.source_age_ms = 0.0
        self.observation = None
        self.particle_filter = None
        self.cube_tracking = None
        self.show_top_particles = True
        self.show_cable_volume = True
        self.cable_meshes = (
            (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint32),
            ),
            (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint32),
            ),
        )
        self.feature_states = {}
        self.feature_callback = None
        self.feature_hitboxes = []
        self.record_button_hitbox = None
        self.reported_recording_path = None
        self.recorder = DiagnosticWindowRecorder(
            recording_directory,
            recording_fps,
            recording_codec,
            recording_queue_frames,
        )

        self.vbo = None
        self.vao = None
        self.rgb_texture = None
        self.skeleton_texture = None
        self.shader_program = None
        self.mvp_location = None
        self.point_size_location = None
        self.fov_y_deg = 70.0
        self.scene_center = np.asarray((0.0, 0.0, -2.0), dtype=np.float32)
        self.scene_radius = 2.0
        self.latest_scene_center = self.scene_center.copy()
        self.latest_scene_radius = self.scene_radius
        self.has_scene = False
        self.yaw_deg = -35.0
        self.pitch_deg = 22.0
        self.zoom = 1.0
        self.rotating = False
        self.panning = False
        self.last_mouse = (0, 0)

    def init(self):
        global _GLUT_INITIALIZED
        if not _GLUT_INITIALIZED:
            glutInit()
            _GLUT_INITIALIZED = True
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(30, 30)
        self.window_id = glutCreateWindow(self.title.encode("utf-8"))
        try:
            glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION)
        except Exception:
            pass

        glClearColor(*UI_BG, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDisable(GL_LIGHTING)
        try:
            glEnable(GL_PROGRAM_POINT_SIZE)
        except Exception:
            pass
        self.vbo = glGenBuffers(1)
        try:
            self.vao = glGenVertexArrays(1)
        except Exception:
            self.vao = None
        self.rgb_texture = glGenTextures(1)
        self._configure_image_texture(self.rgb_texture, GL_LINEAR)
        self.skeleton_texture = glGenTextures(1)
        self._configure_image_texture(self.skeleton_texture, GL_NEAREST)
        self.shader_program = self._create_point_shader()
        if self.shader_program:
            self.mvp_location = glGetUniformLocation(self.shader_program, "u_mvp")
            self.point_size_location = glGetUniformLocation(self.shader_program, "u_point_size")

        glutDisplayFunc(self._draw)
        glutReshapeFunc(self._reshape)
        glutKeyboardFunc(self._keyboard)
        glutSpecialFunc(self._special_key)
        glutMouseFunc(self._mouse)
        glutMotionFunc(self._motion)
        try:
            glutCloseFunc(self._window_closed)
        except Exception:
            pass
        self.available = True

    def is_available(self):
        return self.available

    def poll(self):
        if not self.available:
            return False
        try:
            glutSetWindow(self.window_id)
            if self.redraw_requested:
                glutPostRedisplay()
                self.redraw_requested = False
            glutMainLoopEvent()
        except Exception:
            self.available = False
        return self.available

    def close(self):
        recording = self.recorder.stop()
        if recording["error"]:
            print(
                f"DIAGNOSTIC_RECORDING_FAILED error={recording['error']}",
                flush=True,
            )
        elif (
            recording["path"] is not None
            and recording["written_frames"] > 0
            and recording["path"] != self.reported_recording_path
        ):
            print(
                "DIAGNOSTIC_RECORDING_SAVED "
                f"path={recording['path']} "
                f"frames={recording['written_frames']} "
                f"missed_captures={recording['dropped_frames']}",
                flush=True,
            )
            self.reported_recording_path = recording["path"]
        window_id = self.window_id
        self.available = False
        self.window_id = None
        if window_id is None:
            return
        try:
            glutSetWindow(window_id)
            if self.shader_program:
                glDeleteProgram(self.shader_program)
            if self.vao:
                glDeleteVertexArrays(1, [self.vao])
            if self.vbo:
                glDeleteBuffers(1, [self.vbo])
            if self.rgb_texture:
                glDeleteTextures(1, [self.rgb_texture])
            if self.skeleton_texture:
                glDeleteTextures(1, [self.skeleton_texture])
            glutDestroyWindow(window_id)
        except Exception:
            pass

    def set_camera_fov(self, degrees):
        if degrees is not None and np.isfinite(degrees):
            self.fov_y_deg = float(np.clip(degrees, 35.0, 100.0))

    def update_status(self, text):
        with self.lock:
            self.pending_status = str(text)
            self.redraw_requested = True

    def set_feature_controls(self, features, callback):
        self.feature_states = {
            name: bool(getattr(features, name)) for _key, name, _label in FEATURE_CONTROLS
        }
        self.feature_callback = callback

    def update_frame(self, result, pipeline_stats):
        vertices = np.asarray(result.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 4:
            raise ValueError(
                f"Viewer vertices must have packed-color shape Nx4; got {vertices.shape}."
            )
        image = np.asarray(result.rgb_image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Viewer RGB image must have shape HxWx3; got {image.shape}.")
        skeleton_image = np.asarray(result.skeleton_image, dtype=np.uint8)
        if skeleton_image.ndim != 3 or skeleton_image.shape[2] != 3:
            raise ValueError(
                "Viewer skeleton image must have shape HxWx3; "
                f"got {skeleton_image.shape}."
            )
        payload = {
            "frame_index": int(result.frame_index),
            "image": np.ascontiguousarray(image),
            "skeleton_image": np.ascontiguousarray(skeleton_image),
            "vertices": np.ascontiguousarray(vertices),
            "cloud_captured_at": float(result.cloud_captured_at),
            "stats": dict(result.stats),
            "pipeline": dict(pipeline_stats),
            "center": np.asarray(result.scene_center, dtype=np.float32).reshape(3),
            "radius": float(result.scene_radius),
            "inference_ms": float(result.inference_ms),
            "processing_ms": float(result.processing_ms),
            "latency_ms": float(result.latency_ms),
            "source_age_ms": float(result.source_age_ms),
            "observation": result.observation,
            "particle_filter": result.particle_filter,
            "cube_tracking": result.cube_tracking,
        }
        with self.lock:
            self.pending = payload
            self.pending_status = "Live cable tracking"
            self.redraw_requested = True

    def reset_view(self):
        self.yaw_deg = -35.0
        self.pitch_deg = 22.0
        self.zoom = 1.0
        if self.has_scene:
            self.scene_center = self.latest_scene_center.copy()
            self.scene_radius = self.latest_scene_radius
        self.redraw_requested = True

    def _consume_pending(self):
        with self.lock:
            payload = self.pending
            status = self.pending_status
            self.pending = None
        self.status = status
        if payload is None:
            return
        self.rgb_image = payload["image"]
        self.rgb_texture_dirty = True
        self.skeleton_image = payload["skeleton_image"]
        self.skeleton_texture_dirty = True
        cloud_captured_at = payload["cloud_captured_at"]
        cloud_changed = self.cloud_captured_at != cloud_captured_at
        if cloud_changed:
            self.vertices = payload["vertices"]
            if not self.shader_program and self.vertices.shape[1] == 4:
                packed = np.ascontiguousarray(self.vertices[:, 3]).view(np.uint32)
                expanded = np.empty((len(self.vertices), 6), dtype=np.float32)
                expanded[:, :3] = self.vertices[:, :3]
                expanded[:, 3] = (packed & 255).astype(np.float32) / 255.0
                expanded[:, 4] = ((packed >> 8) & 255).astype(np.float32) / 255.0
                expanded[:, 5] = ((packed >> 16) & 255).astype(np.float32) / 255.0
                self.vertices = expanded
            self.vertex_count = len(self.vertices)
            self.cloud_captured_at = cloud_captured_at
        self.stats = payload["stats"]
        self.pipeline_stats = payload["pipeline"]
        self.frame_index = payload["frame_index"]
        self.inference_ms = payload["inference_ms"]
        self.processing_ms = payload["processing_ms"]
        self.latency_ms = payload["latency_ms"]
        self.source_age_ms = payload["source_age_ms"]
        self.observation = payload["observation"]
        self.particle_filter = payload["particle_filter"]
        self.cube_tracking = payload["cube_tracking"]
        cable_radius_m = float(self.particle_filter.cable_radius_m)
        self.cable_meshes = tuple(
            swept_capsule_mesh(
                # The 16 PF nodes are the actual piecewise-linear medial axis.
                # Dense samples add no geometry and would only repeat rings.
                np.asarray(cable.curve, dtype=np.float32),
                cable_radius_m,
                radial_sides=12,
                cap_rings=4,
            )
            for cable in self.particle_filter.cables
        )
        if cloud_changed:
            self._update_scene_bounds(payload["center"], payload["radius"])
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                self.vertices.nbytes,
                self.vertices,
                GL_STREAM_DRAW,
            )
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _draw(self):
        if not self.available:
            return
        self._consume_pending()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        left_width, cloud_width = self._panel_sizes()
        self._draw_rgb(left_width, self.height)

        glViewport(left_width, 0, cloud_width, self.height)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        mvp = self._set_camera(cloud_width, self.height)
        self._draw_grid()
        self._draw_cloud(mvp)
        self._draw_cube_tracking()
        self._draw_particle_filter()
        self._draw_camera_origin()
        self._draw_cloud_overlay(cloud_width, self.height)
        self._draw_divider(left_width)
        now = time.perf_counter()
        if self.recorder.should_capture(now):
            self.recorder.submit(self._capture_window())
        glutSwapBuffers()

    def _capture_window(self) -> np.ndarray:
        """Read the complete back buffer after every viewer element is drawn."""

        glReadBuffer(GL_BACK)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(
            0,
            0,
            self.width,
            self.height,
            GL_BGR,
            GL_UNSIGNED_BYTE,
        )
        frame = np.frombuffer(
            pixels,
            dtype=np.uint8,
            count=self.width * self.height * 3,
        ).reshape(self.height, self.width, 3)
        return np.ascontiguousarray(frame[::-1])

    def _draw_rgb(self, width, height):
        glViewport(0, 0, width, height)
        glUseProgram(0)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        header = 84
        margin = 14
        self._rect(0, 0, width, height, UI_PANEL_DARK)
        self._rect(0, height - header, width, header, UI_PANEL)
        self._rect(0, height - header, 5, header, ORANGE)
        self._text(margin, height - 28, "RGB + PIDNet segmentation", UI_TEXT)
        self._legend(margin, height - 55, ORANGE, "cable")
        self._legend(margin + 112, height - 55, BLUE, "endpoint 1")
        self._legend(margin + 246, height - 55, GREEN, "endpoint 2")
        self._line(0, height - header, width, height - header, UI_STROKE)
        if self.rgb_image is None:
            self._text(margin, height - header - 30, self.status, UI_MUTED)
            return

        section_header = 52
        gap = 8
        content_bottom = margin
        content_top = height - header - margin
        image_height = max(
            2,
            content_top - content_bottom - section_header - 2 * gap,
        )
        skeleton_pane_height = image_height // 2
        skeleton_y0 = content_bottom
        skeleton_y1 = skeleton_y0 + skeleton_pane_height
        skeleton_header_y0 = skeleton_y1 + gap
        skeleton_header_y1 = skeleton_header_y0 + section_header
        rgb_y0 = skeleton_header_y1 + gap
        rgb_y1 = content_top

        self._draw_image(
            self.rgb_image,
            self.rgb_texture,
            "rgb_texture_dirty",
            margin,
            rgb_y0,
            max(1, width - 2 * margin),
            max(1, rgb_y1 - rgb_y0),
        )

        self._rect(0, skeleton_header_y0, width, section_header, UI_PANEL)
        self._rect(0, skeleton_header_y0, 5, section_header, SKELETON_LINE)
        self._text(
            margin,
            skeleton_header_y1 - 18,
            "Live skeleton graph",
            UI_TEXT,
            GLUT_BITMAP_HELVETICA_12,
        )
        legend_y = skeleton_header_y0 + 9
        self._legend(margin, legend_y, SKELETON_MASK, "mask")
        self._legend(margin + 72, legend_y, SKELETON_LINE, "skeleton")
        self._legend(margin + 172, legend_y, SKELETON_NODE, "node")
        self._legend(margin + 242, legend_y, SKELETON_BRANCH, "junction")
        self._line(
            0,
            skeleton_header_y0,
            width,
            skeleton_header_y0,
            UI_STROKE,
        )
        if self.skeleton_image is not None:
            self._draw_image(
                self.skeleton_image,
                self.skeleton_texture,
                "skeleton_texture_dirty",
                margin,
                skeleton_y0,
                max(1, width - 2 * margin),
                max(1, skeleton_y1 - skeleton_y0),
            )

    @staticmethod
    def _configure_image_texture(texture, filtering):
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filtering)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filtering)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _draw_image(
        self,
        image,
        texture,
        dirty_attribute,
        pane_x,
        pane_y,
        pane_width,
        pane_height,
    ):
        image_h, image_w = image.shape[:2]
        scale = min(pane_width / image_w, pane_height / image_h)
        draw_w = max(1, int(round(image_w * scale)))
        draw_h = max(1, int(round(image_h * scale)))
        x0 = pane_x + (pane_width - draw_w) // 2
        y0 = pane_y + (pane_height - draw_h) // 2
        x1, y1 = x0 + draw_w, y0 + draw_h
        glBindTexture(GL_TEXTURE_2D, texture)
        if bool(getattr(self, dirty_attribute)):
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB,
                image_w,
                image_h,
                0,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                image,
            )
            setattr(self, dirty_attribute, False)
        glEnable(GL_TEXTURE_2D)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex2f(x0, y0)
        glTexCoord2f(1.0, 1.0); glVertex2f(x1, y0)
        glTexCoord2f(1.0, 0.0); glVertex2f(x1, y1)
        glTexCoord2f(0.0, 0.0); glVertex2f(x0, y1)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _set_camera(self, width, height):
        aspect = max(1, width) / max(1, height)
        znear = 0.01
        zfar = max(100.0, self.scene_radius * 20.0)
        projection = self._perspective(self.fov_y_deg, aspect, znear, zfar)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov_y_deg, aspect, znear, zfar)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        distance = max(0.5, self.scene_radius * 2.4) * self.zoom
        modelview = (
            self._translation(0.0, 0.0, -distance)
            @ self._rotation_x(self.pitch_deg)
            @ self._rotation_y(self.yaw_deg)
            @ self._translation(*(-self.scene_center))
        )
        glTranslatef(0.0, 0.0, -distance)
        glRotatef(self.pitch_deg, 1.0, 0.0, 0.0)
        glRotatef(self.yaw_deg, 0.0, 1.0, 0.0)
        glTranslatef(*(-self.scene_center))
        return projection @ modelview

    def _draw_cloud(self, mvp):
        if self.vertex_count == 0:
            return
        if self.shader_program:
            glUseProgram(self.shader_program)
            glUniformMatrix4fv(
                self.mvp_location,
                1,
                GL_TRUE,
                np.ascontiguousarray(mvp, dtype=np.float32),
            )
            glUniform1f(self.point_size_location, self.point_size)
            if self.vao:
                glBindVertexArray(self.vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glEnableVertexAttribArray(0)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(12))
            glDrawArrays(GL_POINTS, 0, self.vertex_count)
            glDisableVertexAttribArray(1)
            glDisableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            if self.vao:
                glBindVertexArray(0)
            glUseProgram(0)
            return

        glPointSize(self.point_size)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 24, ctypes.c_void_p(0))
        glColorPointer(3, GL_FLOAT, 24, ctypes.c_void_p(12))
        glDrawArrays(GL_POINTS, 0, self.vertex_count)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _draw_grid(self):
        center = self.scene_center
        radius = max(0.3, self.scene_radius)
        floor_y = float(center[1] - radius * 0.65)
        extent = radius * 1.25
        step = max(0.05, extent / 10.0)
        count = int(np.ceil(extent / step))
        glDisable(GL_DEPTH_TEST)
        glColor3f(0.15, 0.17, 0.19)
        glBegin(GL_LINES)
        for index in range(-count, count + 1):
            offset = index * step
            glVertex3f(center[0] - extent, floor_y, center[2] + offset)
            glVertex3f(center[0] + extent, floor_y, center[2] + offset)
            glVertex3f(center[0] + offset, floor_y, center[2] - extent)
            glVertex3f(center[0] + offset, floor_y, center[2] + extent)
        glEnd()
        glEnable(GL_DEPTH_TEST)

    def _draw_camera_origin(self):
        length = max(0.08, self.scene_radius * 0.12)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.5)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.25, 0.22); glVertex3f(0, 0, 0); glVertex3f(length, 0, 0)
        glColor3f(0.25, 1.0, 0.42); glVertex3f(0, 0, 0); glVertex3f(0, length, 0)
        glColor3f(0.30, 0.55, 1.0); glVertex3f(0, 0, 0); glVertex3f(0, 0, length)
        glEnd()
        glEnable(GL_DEPTH_TEST)

    def _draw_cube_tracking(self):
        result = self.cube_tracking
        if (
            result is None
            or not result.valid
            or result.center_m is None
            or result.rotation is None
        ):
            return
        center = np.asarray(result.center_m, dtype=np.float32)
        rotation = np.asarray(result.rotation, dtype=np.float32)
        half_side = 0.5 * float(result.cube_side_m)
        corners = center + (half_side * CUBE_CORNERS) @ rotation.T

        glUseProgram(0)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glLineWidth(4.0)
        glColor4f(*CUBE_TRACKING, 1.0)
        glBegin(GL_LINES)
        for first, second in CUBE_EDGES:
            glVertex3f(*corners[first])
            glVertex3f(*corners[second])
        glEnd()

        axis_length = 0.42 * float(result.cube_side_m)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.25, 0.22)
        glVertex3f(*center)
        glVertex3f(*(center + axis_length * rotation[:, 0]))
        glColor3f(0.25, 1.0, 0.42)
        glVertex3f(*center)
        glVertex3f(*(center + axis_length * rotation[:, 1]))
        glColor3f(0.30, 0.55, 1.0)
        glVertex3f(*center)
        glVertex3f(*(center + axis_length * rotation[:, 2]))
        glEnd()

    def _draw_particle_filter(self):
        if self.particle_filter is None:
            return
        glUseProgram(0)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        colors = (BLUE, GREEN)
        if self.show_cable_volume:
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_FALSE)
            glDisable(GL_CULL_FACE)
            for cable_index, mesh in enumerate(self.cable_meshes):
                vertices, triangles = mesh
                if len(vertices) == 0 or len(triangles) == 0:
                    continue
                glColor4f(*colors[cable_index], 0.34)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, vertices)
                glDrawElements(
                    GL_TRIANGLES,
                    int(triangles.size),
                    GL_UNSIGNED_INT,
                    triangles,
                )
                glDisableClientState(GL_VERTEX_ARRAY)
            glDepthMask(GL_TRUE)
        glDisable(GL_DEPTH_TEST)
        for cable_index, cable in enumerate(self.particle_filter.cables):
            color = colors[cable_index]
            if self.show_top_particles:
                glLineWidth(1.0)
                glColor4f(*color, 0.20)
                for particle in np.asarray(cable.top_particles):
                    if len(particle) < 2:
                        continue
                    glBegin(GL_LINE_STRIP)
                    for point in particle:
                        glVertex3f(*point)
                    glEnd()
            dense = np.asarray(cable.dense_curve)
            visibility = np.asarray(cable.dense_visibility, dtype=np.uint8)
            if len(dense) >= 2:
                glLineWidth(4.0)
                glBegin(GL_LINES)
                for index in range(len(dense) - 1):
                    state = (
                        int(visibility[index])
                        if index < len(visibility)
                        else VISIBILITY_UNKNOWN
                    )
                    next_state = (
                        int(visibility[index + 1])
                        if index + 1 < len(visibility)
                        else VISIBILITY_UNKNOWN
                    )
                    if state == VISIBILITY_MISSING or next_state == VISIBILITY_MISSING:
                        segment_color, alpha = (1.0, 0.18, 0.15), 1.0
                    elif state == VISIBILITY_SUPPORTED and next_state == VISIBILITY_SUPPORTED:
                        segment_color, alpha = color, 1.0
                    else:
                        segment_color, alpha = color, 0.30
                    glColor4f(*segment_color, alpha)
                    glVertex3f(*dense[index])
                    glVertex3f(*dense[index + 1])
                glEnd()
            curve = np.asarray(cable.curve)
            if len(curve):
                glPointSize(7.0)
                glColor4f(*color, 1.0)
                glBegin(GL_POINTS)
                for point in curve:
                    glVertex3f(*point)
                glEnd()
                glPointSize(12.0)
                glBegin(GL_POINTS)
                glVertex3f(*curve[0])
                glVertex3f(*curve[-1])
                glEnd()
                node_visibility = np.asarray(cable.node_visibility, dtype=np.uint8)
                covariance = np.asarray(cable.node_covariance, dtype=np.float32)
                if covariance.shape == (len(curve), 3, 3):
                    for node_index, point in enumerate(curve):
                        state = (
                            int(node_visibility[node_index])
                            if node_index < len(node_visibility)
                            else VISIBILITY_UNKNOWN
                        )
                        if state == VISIBILITY_SUPPORTED:
                            continue
                        if state == VISIBILITY_MISSING:
                            uncertainty_color = (1.0, 0.18, 0.15)
                        else:
                            uncertainty_color = color
                        self._draw_covariance_ellipsoid(
                            point,
                            covariance[node_index],
                            uncertainty_color,
                        )
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    @staticmethod
    def _draw_covariance_ellipsoid(center, covariance, color):
        if not np.all(np.isfinite(covariance)):
            return
        values, vectors = np.linalg.eigh(
            0.5 * (covariance + covariance.T)
        )
        radii = 2.0 * np.sqrt(np.clip(values, 0.0, None))
        if float(np.max(radii)) < 2e-4:
            return
        radii = np.minimum(radii, 0.15)
        transform = vectors @ np.diag(radii)
        angles = np.linspace(0.0, 2.0 * np.pi, 25, dtype=np.float32)
        rings = (
            np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles))),
            np.column_stack((np.cos(angles), np.zeros_like(angles), np.sin(angles))),
            np.column_stack((np.zeros_like(angles), np.cos(angles), np.sin(angles))),
        )
        glLineWidth(1.0)
        glColor4f(*color, 0.32)
        for ring in rings:
            points = np.asarray(center)[None, :] + ring @ transform.T
            glBegin(GL_LINE_STRIP)
            for point in points:
                glVertex3f(*point)
            glEnd()

    def _draw_cloud_overlay(self, width, height):
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        header = 296
        self._rect(0, height - header, width, header, (0.018, 0.021, 0.025))
        self._rect(0, height - header, 5, header, UI_ACCENT)
        self._rect(0, height - header, width, 1, UI_STROKE)
        stride = int(self.stats.get("point_cloud_stride", 1))
        self._text(
            18,
            height - 29,
            f"3D point cloud (viewer stride {stride})",
            UI_TEXT,
        )
        recording = self.recorder.snapshot()
        button_width = 190
        button_height = 27
        button_x = max(18, width - button_width - 18)
        button_y = height - 43
        if recording["active"]:
            button_text = f"STOP RECORDING  {recording['elapsed_s']:05.1f}s"
            button_color = (1.00, 0.22, 0.18)
        elif recording["error"]:
            button_text = "RECORDING ERROR"
            button_color = (1.00, 0.55, 0.15)
        else:
            button_text = "START RECORDING"
            button_color = UI_ACCENT
        self._rect(
            button_x,
            button_y,
            button_width,
            button_height,
            UI_PANEL,
        )
        self._rect(
            button_x,
            button_y,
            4,
            button_height,
            button_color,
        )
        self._text(
            button_x + 13,
            button_y + 8,
            button_text,
            UI_TEXT,
            GLUT_BITMAP_HELVETICA_12,
        )
        left_width, _cloud_width = self._panel_sizes()
        self.record_button_hitbox = (
            left_width + button_x,
            left_width + button_x + button_width,
            button_y,
            button_y + button_height,
        )
        self._text(
            18,
            height - 53,
            f"Frame {self.frame_index} | NN {self.inference_ms:.1f} ms | "
            f"viewer prep {self.processing_ms:.1f} ms | display latency {self.latency_ms:.1f} ms | "
            f"cloud age {self.source_age_ms:.1f} ms | "
            f"skipped views {self.pipeline_stats.get('visualization_dropped', 0)}",
            UI_MUTED,
            GLUT_BITMAP_HELVETICA_12,
        )
        y = height - 88
        x = 18
        x = self._metric(x, y, "RENDERED", f"{self.vertex_count:,}", UI_ACCENT)
        x = self._metric(x, y, "INVALID XYZ", f"{self.stats.get('invalid_points', 0):,}", (0.95, 0.42, 0.35))
        x = self._metric(x, y, "CABLE", f"{self.stats.get('cable_points', 0):,}", ORANGE)
        x = self._metric(x, y, "ENDPOINT 1", f"{self.stats.get('endpoint_1_points', 0):,}", BLUE)
        self._metric(x, y, "ENDPOINT 2", f"{self.stats.get('endpoint_2_points', 0):,}", GREEN)

        y = height - 119
        x = 18
        x = self._metric(
            x,
            y,
            "CAPTURE",
            f"{self.pipeline_stats.get('capture_fps', 0.0):.1f} fps",
            UI_ACCENT,
        )
        x = self._metric(
            x,
            y,
            "TRACKING",
            f"{self.pipeline_stats.get('tracking_fps', 0.0):.1f} fps",
            (0.75, 0.58, 1.0),
        )
        x = self._metric(
            x,
            y,
            "VIEWER",
            f"{self.pipeline_stats.get('visualization_fps', 0.0):.1f} fps",
            GREEN,
        )
        self._metric(
            x,
            y,
            "SKIPPED VIEWS",
            f"{self.pipeline_stats.get('visualization_dropped', 0):,}",
            (0.95, 0.65, 0.22),
        )

        if self.cube_tracking is not None:
            cube = self.cube_tracking
            if cube.valid and cube.center_m is not None:
                span = (
                    f" span={cube.observed_span_m * 1000.0:.1f}mm"
                    if np.isfinite(cube.observed_span_m)
                    else ""
                )
                cube_text = (
                    f"CUBE valid {cube.face_count}F | "
                    f"surface={cube.surface_rms_m * 1000.0:.1f}mm{span} | "
                    f"xyz=({cube.center_m[0]:+.3f},"
                    f"{cube.center_m[1]:+.3f},{cube.center_m[2]:+.3f})m | "
                    f"{cube.processing_ms:.1f}ms"
                )
                cube_color = CUBE_TRACKING
            else:
                cube_text = (
                    f"CUBE invalid | {cube.reason} | "
                    f"{cube.processing_ms:.1f}ms"
                )
                cube_color = (1.0, 0.38, 0.28)
            self._text(
                18,
                height - 129,
                cube_text,
                cube_color,
                GLUT_BITMAP_HELVETICA_12,
            )

        if self.particle_filter is not None:
            for cable_index, cable in enumerate(self.particle_filter.cables):
                diagnostic = cable.diagnostics
                y = height - 151 - cable_index * 22
                color = (BLUE, GREEN)[cable_index]
                self._text(
                    18,
                    y,
                    f"PF{cable_index + 1} {diagnostic.tracking_state} | "
                    f"ep={diagnostic.endpoint_visible_count}/2 "
                    f"route={diagnostic.selected_route_index + 1}/"
                    f"{diagnostic.route_candidate_count} "
                    f"edge={diagnostic.attributed_edge_count} "
                    f"src={diagnostic.measurement_source} "
                    f"V/M/U={100.0 * diagnostic.visible_fraction:.0f}/"
                    f"{100.0 * diagnostic.missing_fraction:.0f}/"
                    f"{100.0 * diagnostic.unknown_fraction:.0f}% | "
                    f"sigma={diagnostic.maximum_node_uncertainty_mm:.1f} mm "
                    f"age={diagnostic.complete_observation_age_s:.2f}s | "
                    f"trace={diagnostic.trace_mean_mm:.1f} mm "
                    f"ESS={diagnostic.effective_sample_size:.0f} "
                    f"RS={'Y' if diagnostic.resampled else 'N'}",
                    color,
                    GLUT_BITMAP_HELVETICA_12,
                )
            motion_text = []
            for cable_index, cable in enumerate(self.particle_filter.cables):
                diagnostic = cable.diagnostics
                applied = "ON" if diagnostic.local_node_motion_applied else "OFF"
                motion_text.append(
                    f"PF{cable_index + 1} "
                    f"{diagnostic.predicted_node_speed_mps:.2f}/"
                    f"{diagnostic.local_velocity_innovation_speed_mps:.2f}/"
                    f"{diagnostic.corrected_node_speed_mps:.2f}/"
                    f"{diagnostic.maximum_corrected_node_speed_mps:.2f} "
                    f"support={100.0 * diagnostic.locally_supported_node_fraction:.0f}% "
                    f"[{applied}]"
                )
            self._text(
                18,
                height - 195,
                "Motion m/s predicted/local-innovation/corrected/max | "
                + " | ".join(motion_text),
                UI_TEXT,
                GLUT_BITMAP_HELVETICA_12,
            )
            self._text(
                18,
                height - 217,
                f"Observation {self.pipeline_stats.get('observation_ms', 0.0):.1f} ms | "
                f"PF {self.pipeline_stats.get('particle_filter_ms', 0.0):.1f} ms "
                f"(CUDA {self.pipeline_stats.get('particle_filter_gpu_ms', 0.0):.1f} ms) | "
                f"total {self.pipeline_stats.get('tracking_ms', 0.0):.1f} ms | "
                "PF: color=supported red=missing dim=unknown",
                UI_MUTED,
                GLUT_BITMAP_HELVETICA_12,
            )
        if "observation_total_median_ms" in self.pipeline_stats:
            self._text(
                18,
                height - 239,
                "OBS median/p95 ms | "
                f"total {self.pipeline_stats.get('observation_total_median_ms', 0.0):.1f}/"
                f"{self.pipeline_stats.get('observation_total_p95_ms', 0.0):.1f} | "
                f"geom {self.pipeline_stats.get('observation_geometry_median_ms', 0.0):.1f}/"
                f"{self.pipeline_stats.get('observation_geometry_p95_ms', 0.0):.1f} | "
                f"CC {self.pipeline_stats.get('observation_components_median_ms', 0.0):.1f}/"
                f"{self.pipeline_stats.get('observation_components_p95_ms', 0.0):.1f} | "
                f"ends {self.pipeline_stats.get('observation_endpoints_median_ms', 0.0):.1f}/"
                f"{self.pipeline_stats.get('observation_endpoints_p95_ms', 0.0):.1f} | "
                f"skel {self.pipeline_stats.get('observation_skeleton_median_ms', 0.0):.1f}/"
                f"{self.pipeline_stats.get('observation_skeleton_p95_ms', 0.0):.1f} | "
                f"graph {self.pipeline_stats.get('observation_graph_median_ms', 0.0):.1f}/"
                f"{self.pipeline_stats.get('observation_graph_p95_ms', 0.0):.1f} | "
                f"routes {self.pipeline_stats.get('observation_routes_median_ms', 0.0):.1f}/"
                f"{self.pipeline_stats.get('observation_routes_p95_ms', 0.0):.1f}",
                UI_TEXT,
                GLUT_BITMAP_HELVETICA_12,
            )
            self._text(
                18,
                height - 260,
                "OBS latest GPU ms | "
                f"maps {self.pipeline_stats.get('observation_maps_cuda_ms', 0.0):.2f} | "
                f"unproject {self.pipeline_stats.get('observation_unprojection_cuda_ms', 0.0):.2f} | "
                f"readback block {self.pipeline_stats.get('observation_readback_wall_ms', 0.0):.2f} | "
                f"pixels cable/relevant/valid3D="
                f"{self.pipeline_stats.get('observation_cable_pixels', 0):,}/"
                f"{self.pipeline_stats.get('observation_relevant_pixels', 0):,}/"
                f"{self.pipeline_stats.get('observation_valid_3d_points', 0):,}",
                UI_MUTED,
                GLUT_BITMAP_HELVETICA_12,
            )
            self._text(
                18,
                height - 281,
                "OBS topology | "
                f"components {self.pipeline_stats.get('observation_processed_component_count', 0)}/"
                f"{self.pipeline_stats.get('observation_component_count', 0)} | "
                f"endpoints {self.pipeline_stats.get('observation_endpoint_count', 0)}/4 | "
                f"skeleton px {self.pipeline_stats.get('observation_skeleton_pixels', 0):,} | "
                f"nodes/edges/branches="
                f"{self.pipeline_stats.get('observation_graph_nodes', 0)}/"
                f"{self.pipeline_stats.get('observation_graph_edges', 0)}/"
                f"{self.pipeline_stats.get('observation_branch_pixels', 0)} | "
                f"routes {self.pipeline_stats.get('observation_route_candidates', 0)} | "
                f"observed edges "
                f"{self.pipeline_stats.get('observation_graph_edge_count', 0)}",
                UI_MUTED,
                GLUT_BITMAP_HELVETICA_12,
            )
            self._text(
                18,
                height - 302,
                "OBS latest other ms | "
                f"masks {self.pipeline_stats.get('observation_masks_latest_ms', 0.0):.2f} | "
                f"component prep {self.pipeline_stats.get('observation_component_preparation_latest_ms', 0.0):.2f} | "
                f"graph edges "
                f"{self.pipeline_stats.get('observation_graph_edges_latest_ms', 0.0):.2f} | "
                f"route search/build "
                f"{self.pipeline_stats.get('observation_route_search_latest_ms', 0.0):.2f}/"
                f"{self.pipeline_stats.get('observation_route_assembly_latest_ms', 0.0):.2f} | "
                f"final {self.pipeline_stats.get('observation_finalization_latest_ms', 0.0):.2f} | "
                f"unaccounted {self.pipeline_stats.get('observation_unaccounted_latest_ms', 0.0):.2f}",
                UI_MUTED,
                GLUT_BITMAP_HELVETICA_12,
            )

        self._rect(0, 0, width, 107, (0.018, 0.021, 0.025))
        self.feature_hitboxes = []
        x = 18
        chip_y = 81
        for key, name, label in FEATURE_CONTROLS:
            enabled = bool(self.feature_states.get(name, False))
            text = f"{key} {label} {'ON' if enabled else 'OFF'}"
            text_width = self._text_width(text, GLUT_BITMAP_HELVETICA_12)
            chip_width = text_width + 18
            if x + chip_width > width - 18:
                x = 18
                chip_y -= 25
            color = GREEN if enabled else (0.35, 0.39, 0.43)
            self._rect(x, chip_y, chip_width, 21, UI_PANEL)
            self._rect(x, chip_y, 3, 21, color)
            self._text(x + 9, chip_y + 6, text, UI_TEXT, GLUT_BITMAP_HELVETICA_12)
            left_width, _cloud_width = self._panel_sizes()
            self.feature_hitboxes.append(
                (
                    left_width + x,
                    left_width + x + chip_width,
                    chip_y,
                    chip_y + 21,
                    name,
                )
            )
            x += chip_width + 6
        self._text(
            18,
            10,
            f"Orbit: left drag    Pan: right drag    Zoom: wheel    Refit: R    "
            f"Top particles: P ({'on' if self.show_top_particles else 'off'})    "
            f"9 mm body: B ({'on' if self.show_cable_volume else 'off'})    "
            f"Point size: +/- ({self.point_size:.1f})    Record: C    "
            f"Click a feature or use its key    Quit: Q",
            UI_MUTED,
            GLUT_BITMAP_HELVETICA_12,
        )
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

    def _draw_divider(self, x):
        glViewport(0, 0, self.width, self.height)
        glUseProgram(0)
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix(); glLoadIdentity(); glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix(); glLoadIdentity()
        self._rect(x - 2, 0, 4, self.height, UI_STROKE)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

    def _legend(self, x, y, color, label):
        self._rect(x, y, 10, 10, color)
        self._text(x + 16, y, label, UI_MUTED, GLUT_BITMAP_HELVETICA_12)

    def _metric(self, x, y, label, value, color):
        font = GLUT_BITMAP_HELVETICA_12
        label = str(label)
        value = str(value)
        label_width = self._text_width(label, font)
        value_width = self._text_width(value, font)
        width = max(112, 4 + 10 + label_width + 18 + value_width + 10)
        self._rect(x, y, width, 23, UI_PANEL)
        self._rect(x, y, 4, 23, color)
        self._text(x + 12, y + 7, label, UI_MUTED, font)
        self._text(x + width - value_width - 10, y + 7, value, UI_TEXT, font)
        return x + width + 7

    @staticmethod
    def _text_width(text, font):
        return int(sum(glutBitmapWidth(font, ord(character)) for character in str(text)))

    @staticmethod
    def _text(x, y, text, color, font=GLUT_BITMAP_HELVETICA_18):
        glColor3f(*color)
        glRasterPos2f(float(x), float(y))
        for character in str(text):
            glutBitmapCharacter(font, ord(character))

    @staticmethod
    def _rect(x, y, width, height, color):
        glColor3f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y); glVertex2f(x + width, y)
        glVertex2f(x + width, y + height); glVertex2f(x, y + height)
        glEnd()

    @staticmethod
    def _line(x0, y0, x1, y1, color):
        glColor3f(*color)
        glBegin(GL_LINES); glVertex2f(x0, y0); glVertex2f(x1, y1); glEnd()

    def _panel_sizes(self):
        left = int(round(self.width * self.left_panel_ratio))
        left = int(np.clip(left, min(260, self.width // 2), max(260, self.width - 360)))
        return left, max(1, self.width - left)

    def _update_scene_bounds(self, center, radius):
        """Remember cloud bounds without moving an established viewport.

        The first valid cloud establishes the camera target and distance.
        Later cloud bounds are retained for the explicit R refit command.
        """

        center = np.asarray(center, dtype=np.float32).reshape(3)
        radius = float(np.clip(radius, 0.25, 20.0))
        if not np.all(np.isfinite(center)):
            return
        self.latest_scene_center = center.copy()
        self.latest_scene_radius = radius
        if not self.has_scene:
            self.scene_center = center.copy()
            self.scene_radius = radius
            self.has_scene = True

    def _reshape(self, width, height):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.redraw_requested = True

    def _keyboard(self, key, _x, _y):
        if key in (b"q", b"Q", b"\x1b"):
            self.close()
        elif key in (b"r", b"R"):
            self.reset_view()
        elif key in (b"+", b"="):
            self.point_size = min(8.0, self.point_size + 0.5)
        elif key in (b"-", b"_"):
            self.point_size = max(1.0, self.point_size - 0.5)
        elif key in (b"p", b"P"):
            self.show_top_particles = not self.show_top_particles
        elif key in (b"b", b"B"):
            self.show_cable_volume = not self.show_cable_volume
        elif key in (b"c", b"C"):
            self._toggle_recording()
        else:
            try:
                character = key.decode("ascii").upper()
            except (AttributeError, UnicodeDecodeError):
                character = ""
            for control_key, name, _label in FEATURE_CONTROLS:
                if character == control_key:
                    self._toggle_feature(name)
                    break
        self.redraw_requested = True

    def _special_key(self, key, _x, _y):
        if key == GLUT_KEY_LEFT:
            self.yaw_deg -= 4.0
        elif key == GLUT_KEY_RIGHT:
            self.yaw_deg += 4.0
        elif key == GLUT_KEY_UP:
            self.pitch_deg = min(85.0, self.pitch_deg + 4.0)
        elif key == GLUT_KEY_DOWN:
            self.pitch_deg = max(-85.0, self.pitch_deg - 4.0)
        self.redraw_requested = True

    def _mouse_in_cloud(self, x):
        left, _right = self._panel_sizes()
        return x >= left

    def _mouse(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            bottom_y = self.height - y
            if self.record_button_hitbox is not None:
                x0, x1, y0, y1 = self.record_button_hitbox
                if x0 <= x <= x1 and y0 <= bottom_y <= y1:
                    self._toggle_recording()
                    self.rotating = False
                    return
            for x0, x1, y0, y1, name in self.feature_hitboxes:
                if x0 <= x <= x1 and y0 <= bottom_y <= y1:
                    self._toggle_feature(name)
                    self.rotating = False
                    return
        if button == GLUT_LEFT_BUTTON:
            self.rotating = state == GLUT_DOWN and self._mouse_in_cloud(x)
            self.last_mouse = (x, y)
        elif button == GLUT_RIGHT_BUTTON:
            self.panning = state == GLUT_DOWN and self._mouse_in_cloud(x)
            self.last_mouse = (x, y)
        elif state == GLUT_DOWN and button == 3 and self._mouse_in_cloud(x):
            self.zoom = max(0.25, self.zoom * 0.9)
        elif state == GLUT_DOWN and button == 4 and self._mouse_in_cloud(x):
            self.zoom = min(5.0, self.zoom * 1.1)
        self.redraw_requested = True

    def _toggle_feature(self, name):
        if self.feature_callback is None:
            return
        enabled = not bool(self.feature_states.get(name, False))
        try:
            self.feature_callback(name, enabled)
        except Exception as exc:
            self.update_status(f"Feature change failed: {exc}")
            return
        self.feature_states[name] = enabled
        self.update_status(f"{name.replace('_', ' ')}: {'on' if enabled else 'off'}")

    def start_recording(self) -> Path:
        path = self.recorder.start(self.width, self.height)
        self.reported_recording_path = None
        message = (
            f"Recording started: {path.name} "
            f"({self.recorder.codec}, {self.recorder.fps:.0f} fps)"
        )
        print(f"DIAGNOSTIC_RECORDING_STARTED path={path}", flush=True)
        self.update_status(message)
        self.redraw_requested = True
        return path

    def stop_recording(self) -> dict:
        result = self.recorder.stop()
        path = result["path"]
        if result["error"]:
            message = f"Recording failed: {result['error']}"
            print(f"DIAGNOSTIC_RECORDING_FAILED error={result['error']}", flush=True)
        elif path is None or result["written_frames"] <= 0:
            message = "Recording stopped before any frames were captured."
        else:
            message = (
                f"Saved {path.name} | {result['written_frames']} frames | "
                f"repeated {result['dropped_frames']}"
            )
            print(
                "DIAGNOSTIC_RECORDING_SAVED "
                f"path={path} "
                f"frames={result['written_frames']} "
                f"missed_captures={result['dropped_frames']} "
                f"codec={result['codec']}",
                flush=True,
            )
            self.reported_recording_path = path
        self.update_status(message)
        self.redraw_requested = True
        return result

    def _toggle_recording(self) -> None:
        if self.recorder.snapshot()["active"]:
            self.stop_recording()
            return
        try:
            self.start_recording()
        except Exception as exc:
            self.update_status(f"Recording could not start: {exc}")
            print(f"DIAGNOSTIC_RECORDING_FAILED error={exc}", flush=True)

    def _motion(self, x, y):
        previous_x, previous_y = self.last_mouse
        dx, dy = x - previous_x, y - previous_y
        if self.rotating:
            self.yaw_deg += dx * 0.45
            self.pitch_deg = float(np.clip(self.pitch_deg + dy * 0.35, -85.0, 85.0))
        elif self.panning:
            scale = max(0.0001, self.scene_radius * 0.0015 * self.zoom)
            yaw = np.deg2rad(self.yaw_deg)
            right = np.asarray((np.cos(yaw), 0.0, -np.sin(yaw)), dtype=np.float32)
            up = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
            self.scene_center = self.scene_center - dx * scale * right + dy * scale * up
        self.last_mouse = (x, y)
        self.redraw_requested = True

    def _window_closed(self):
        self.available = False

    @staticmethod
    def _compile_shader(shader_type, source):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            message = glGetShaderInfoLog(shader)
            glDeleteShader(shader)
            raise RuntimeError(message.decode("utf-8", errors="replace"))
        return shader

    def _create_point_shader(self):
        try:
            vertex = self._compile_shader(GL_VERTEX_SHADER, POINT_VERTEX_SHADER)
            fragment = self._compile_shader(GL_FRAGMENT_SHADER, POINT_FRAGMENT_SHADER)
            program = glCreateProgram()
            glAttachShader(program, vertex)
            glAttachShader(program, fragment)
            glBindAttribLocation(program, 0, "in_position")
            glBindAttribLocation(program, 1, "in_color")
            glLinkProgram(program)
            glDeleteShader(vertex)
            glDeleteShader(fragment)
            if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
                message = glGetProgramInfoLog(program)
                glDeleteProgram(program)
                raise RuntimeError(message.decode("utf-8", errors="replace"))
            return program
        except Exception as exc:
            print(f"Point shader unavailable; using fixed-function rendering: {exc}")
            return None

    @staticmethod
    def _perspective(fov_y, aspect, near, far):
        f = 1.0 / np.tan(np.deg2rad(fov_y) * 0.5)
        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = f / aspect
        matrix[1, 1] = f
        matrix[2, 2] = (far + near) / (near - far)
        matrix[2, 3] = (2.0 * far * near) / (near - far)
        matrix[3, 2] = -1.0
        return matrix

    @staticmethod
    def _translation(x, y, z):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = (x, y, z)
        return matrix

    @staticmethod
    def _rotation_x(degrees):
        radians = np.deg2rad(degrees)
        c, s = np.cos(radians), np.sin(radians)
        matrix = np.eye(4, dtype=np.float32)
        matrix[1, 1], matrix[1, 2] = c, -s
        matrix[2, 1], matrix[2, 2] = s, c
        return matrix

    @staticmethod
    def _rotation_y(degrees):
        radians = np.deg2rad(degrees)
        c, s = np.cos(radians), np.sin(radians)
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0], matrix[0, 2] = c, s
        matrix[2, 0], matrix[2, 2] = -s, c
        return matrix
