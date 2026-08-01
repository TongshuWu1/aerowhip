"""Asynchronous metric OpenGL scene viewer for the cable twin.

The application publishes immutable, already-decimated RGB-D snapshots.  One
dedicated thread owns every FreeGLUT/OpenGL call, keeps only the newest pending
snapshot, and returns user actions through a lossless FIFO queue.  No camera,
CUDA, inference, recording, tracking, or reconstruction work lives here.
"""

from __future__ import annotations

from collections import deque
import ctypes
from enum import Enum
import math
from pathlib import Path
import sys
from threading import Event, Lock, Thread
import time
from typing import Any

import numpy as np

from .cable_observation import CableObservationFrame
from .scene import CableGeometry, CubeGeometry, SceneViewerSnapshot


_GLUT_INIT_LOCK = Lock()
_GLUT_INITIALIZED = False
_FREEGLUT_HANDLE: Any | None = None

_BACKGROUND_RGB = (0.012, 0.015, 0.019)
_BODY_RGB_U8 = np.asarray((255, 148, 20), dtype=np.float32)
_ENDPOINT_1_RGB_U8 = np.asarray((13, 89, 255), dtype=np.float32)
_ENDPOINT_2_RGB_U8 = np.asarray((38, 255, 77), dtype=np.float32)
_MASK_COLORS_RGB_U8 = (
    _BODY_RGB_U8,
    _ENDPOINT_1_RGB_U8,
    _ENDPOINT_2_RGB_U8,
)
_CABLE_COLORS = (
    (1.00, 0.48, 0.06, 0.88),
    (0.06, 0.55, 1.00, 0.88),
    (0.18, 0.95, 0.34, 0.88),
    (0.88, 0.30, 0.95, 0.88),
)

_POINT_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
uniform mat4 u_mvp;
uniform float u_point_size;
out vec3 vertex_color;
void main() {
    vertex_color = in_color;
    gl_Position = u_mvp * vec4(in_position, 1.0);
    gl_PointSize = u_point_size;
}
"""

_POINT_FRAGMENT_SHADER = """
#version 330 core
in vec3 vertex_color;
out vec4 out_color;
void main() {
    out_color = vec4(vertex_color, 1.0);
}
"""


class ViewerAction(Enum):
    """A request for the application; the viewer never performs it."""

    NONE = "none"
    QUIT = "quit"
    TOGGLE_RECORDING = "toggle_recording"
    TOGGLE_PAUSE = "toggle_pause"
    STEP = "step"


def deproject_sampled_rgbd(
    snapshot: SceneViewerSnapshot,
    *,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    """Return interleaved XYZRGB vertices for one sampled registered frame.

    Coordinates follow the ZED ``RIGHT_HANDED_Y_UP`` camera convention used
    by OpenGL: +X right, +Y up, and valid scene depth along -Z.  RGB values are
    normalized to [0, 1].
    """

    if not isinstance(snapshot, SceneViewerSnapshot):
        raise TypeError("snapshot must be a SceneViewerSnapshot")
    if not 0.0 < depth_min_m < depth_max_m:
        raise ValueError("depth range must satisfy 0 < min < max")

    depth = snapshot.sampled_depth_m_f32
    valid = (
        np.isfinite(depth)
        & (depth >= float(depth_min_m))
        & (depth <= float(depth_max_m))
    )
    flat_indices = np.flatnonzero(valid.ravel())
    vertices = np.empty((flat_indices.size, 6), dtype=np.float32)
    if flat_indices.size == 0:
        vertices.setflags(write=False)
        return vertices

    width = depth.shape[1]
    rows = flat_indices // width
    columns = flat_indices - rows * width
    distance = depth.ravel()[flat_indices]
    stride = snapshot.sample_stride_px
    calibration = snapshot.calibration

    vertices[:, 0] = (
        (columns.astype(np.float32) * stride - calibration.cx_px)
        * distance
        / calibration.fx_px
    )
    vertices[:, 1] = (
        (calibration.cy_px - rows.astype(np.float32) * stride)
        * distance
        / calibration.fy_px
    )
    vertices[:, 2] = -distance

    bgr = snapshot.sampled_bgr_u8.reshape(-1, 3)[flat_indices]
    vertices[:, 3] = bgr[:, 2].astype(np.float32) / 255.0
    vertices[:, 4] = bgr[:, 1].astype(np.float32) / 255.0
    vertices[:, 5] = bgr[:, 0].astype(np.float32) / 255.0
    vertices.setflags(write=False)
    return vertices


def _segmentation_inset(
    snapshot: SceneViewerSnapshot,
    overlay_alpha: float,
) -> np.ndarray:
    rgb = snapshot.sampled_bgr_u8[:, :, ::-1].astype(np.float32)
    if snapshot.masks_u8 is not None:
        for mask, color in zip(snapshot.masks_u8, _MASK_COLORS_RGB_U8, strict=True):
            foreground = mask != 0
            rgb[foreground] = (
                (1.0 - overlay_alpha) * rgb[foreground]
                + overlay_alpha * color
            )
    return np.ascontiguousarray(np.rint(rgb).clip(0, 255), dtype=np.uint8)


def _depth_inset(
    snapshot: SceneViewerSnapshot,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    depth = snapshot.sampled_depth_m_f32
    valid = (
        np.isfinite(depth)
        & (depth >= depth_min_m)
        & (depth <= depth_max_m)
    )
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = (
        (depth[valid] - depth_min_m) / (depth_max_m - depth_min_m)
    )
    # Compact blue-cyan-yellow-red map; invalid depth remains black.
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=2)
    rgb[~valid] = 0.0
    return np.ascontiguousarray(np.rint(rgb * 255.0), dtype=np.uint8)


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    scale = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = scale / max(aspect, 1.0e-6)
    matrix[1, 1] = scale
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def _translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = (x, y, z)
    return matrix


def _rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = np.eye(4, dtype=np.float32)
    matrix[1, 1] = cosine
    matrix[1, 2] = -sine
    matrix[2, 1] = sine
    matrix[2, 2] = cosine
    return matrix


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = cosine
    matrix[0, 2] = sine
    matrix[2, 0] = -sine
    matrix[2, 2] = cosine
    return matrix


def _freeglut_candidates() -> tuple[Path, ...]:
    pyopengl_dlls = Path(sys.prefix) / "Lib" / "site-packages" / "OpenGL" / "DLLS"
    architecture = "64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "32"
    return (
        pyopengl_dlls / f"freeglut{architecture}.vc14.dll",
        pyopengl_dlls / f"freeglut{architecture}.vc10.dll",
        pyopengl_dlls / f"freeglut{architecture}.vc9.dll",
    )


def _load_opengl() -> tuple[Any, Any]:
    """Load PyOpenGL lazily on the thread that will own the GL context."""

    global _FREEGLUT_HANDLE
    if sys.platform == "win32" and _FREEGLUT_HANDLE is None:
        load_errors: list[str] = []
        for path in _freeglut_candidates():
            if not path.is_file():
                continue
            try:
                _FREEGLUT_HANDLE = ctypes.WinDLL(str(path))
                break
            except OSError as error:
                load_errors.append(f"{path}: {error}")
        if _FREEGLUT_HANDLE is None:
            detail = "; ".join(load_errors) if load_errors else "no candidate DLL exists"
            raise RuntimeError(f"FreeGLUT could not be loaded: {detail}")
        import OpenGL.platform

        OpenGL.platform.PLATFORM.GLUT = _FREEGLUT_HANDLE

    from OpenGL import GL, GLUT

    return GL, GLUT


class AsyncOpenGlSceneViewer:
    """Latest-frame asynchronous OpenGL viewer with explicit lifecycle."""

    def __init__(
        self,
        *,
        window_name: str = "Cable Twin 3D Scene",
        width_px: int = 1600,
        height_px: int = 900,
        depth_min_m: float = 0.20,
        depth_max_m: float = 3.0,
        overlay_alpha: float = 0.58,
        point_cloud_stride: int = 4,
        point_size_px: float = 2.0,
        inset_width_px: int = 420,
        render_fps: float = 15.0,
    ) -> None:
        if not str(window_name).strip():
            raise ValueError("window_name must be non-empty")
        if width_px < 640 or height_px < 480:
            raise ValueError("viewer dimensions must be at least 640x480")
        if not 0.0 < depth_min_m < depth_max_m:
            raise ValueError("depth range must satisfy 0 < min < max")
        if not 0.0 <= overlay_alpha <= 1.0:
            raise ValueError("overlay_alpha must be in [0, 1]")
        if not 1 <= point_cloud_stride <= 16:
            raise ValueError("point_cloud_stride must be in [1, 16]")
        if not 1.0 <= point_size_px <= 10.0:
            raise ValueError("point_size_px must be in [1, 10]")
        if not 160 <= inset_width_px < width_px:
            raise ValueError("inset_width_px must be at least 160 and less than width_px")
        if not 1.0 <= render_fps <= 120.0:
            raise ValueError("render_fps must be in [1, 120]")

        self.window_name = str(window_name)
        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.overlay_alpha = float(overlay_alpha)
        self.point_cloud_stride = int(point_cloud_stride)
        self.point_size_px = float(point_size_px)
        self.inset_width_px = int(inset_width_px)
        self.render_fps = float(render_fps)

        self._snapshot_lock = Lock()
        self._pending_snapshot: SceneViewerSnapshot | None = None
        self._dropped_snapshots = 0
        self._action_lock = Lock()
        self._actions: deque[ViewerAction] = deque()
        self._quit_requested = Event()
        self._close_requested = Event()
        self._wake = Event()
        self._ready = Event()
        self._closed = Event()
        self._error_lock = Lock()
        self._worker_error: BaseException | None = None

        self._thread = Thread(
            target=self._thread_main,
            name="cable-twin-opengl-viewer",
            daemon=True,
        )
        self._thread.start()
        # Native window creation cannot be cancelled safely.  Wait for the
        # owning thread to either finish initialization or publish its error;
        # raising on a timeout would orphan a context the caller cannot close.
        self._ready.wait()
        self._raise_worker_error()
        if self._closed.is_set():
            raise RuntimeError("OpenGL viewer closed during startup")

    @property
    def is_open(self) -> bool:
        return (
            self._ready.is_set()
            and not self._closed.is_set()
            and not self._close_requested.is_set()
            and not self._quit_requested.is_set()
        )

    @property
    def startup_error(self) -> BaseException | None:
        with self._error_lock:
            return self._worker_error

    @property
    def dropped_snapshots(self) -> int:
        with self._snapshot_lock:
            return self._dropped_snapshots

    def show(self, snapshot: SceneViewerSnapshot) -> ViewerAction:
        """Publish the newest snapshot without waiting for rendering."""

        self._raise_worker_error()
        if not isinstance(snapshot, SceneViewerSnapshot):
            raise TypeError("snapshot must be a SceneViewerSnapshot")
        if snapshot.sample_stride_px != self.point_cloud_stride:
            raise ValueError(
                "snapshot sample stride differs from viewer.point_cloud_stride"
            )
        if self._quit_requested.is_set() or self._closed.is_set():
            return ViewerAction.QUIT
        with self._snapshot_lock:
            if self._pending_snapshot is not None:
                self._dropped_snapshots += 1
            self._pending_snapshot = snapshot
        self._wake.set()
        return self.poll()

    def poll(self) -> ViewerAction:
        """Return one queued application action and surface worker failures."""

        self._raise_worker_error()
        if self._quit_requested.is_set() or self._closed.is_set():
            return ViewerAction.QUIT
        with self._action_lock:
            if self._actions:
                return self._actions.popleft()
        return ViewerAction.NONE

    def close(self) -> None:
        """Stop the render thread and synchronously release its GL context."""

        self._close_requested.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("OpenGL viewer thread did not stop within 5 seconds")
        self._raise_worker_error()

    def __enter__(self) -> AsyncOpenGlSceneViewer:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _raise_worker_error(self) -> None:
        with self._error_lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError("OpenGL viewer worker failed") from error

    def _store_worker_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self._worker_error is None:
                self._worker_error = error

    def _request_action(self, action: ViewerAction) -> None:
        if action is ViewerAction.QUIT:
            self._quit_requested.set()
        else:
            with self._action_lock:
                self._actions.append(action)
        self._wake.set()

    def _take_pending_snapshot(self) -> SceneViewerSnapshot | None:
        with self._snapshot_lock:
            snapshot = self._pending_snapshot
            self._pending_snapshot = None
        return snapshot

    def _has_pending_snapshot(self) -> bool:
        with self._snapshot_lock:
            return self._pending_snapshot is not None

    def _thread_main(self) -> None:
        renderer: _OpenGlRenderer | None = None
        try:
            gl, glut = _load_opengl()
            renderer = _OpenGlRenderer(self, gl, glut)
            renderer.initialize()
            self._ready.set()
            renderer.run()
        except BaseException as error:
            self._store_worker_error(error)
            self._quit_requested.set()
        finally:
            if renderer is not None:
                try:
                    renderer.destroy()
                except BaseException as error:
                    self._store_worker_error(error)
            self._ready.set()
            self._closed.set()
            self._wake.set()


class _OpenGlRenderer:
    """Render-thread implementation; all methods run on its owning thread."""

    def __init__(self, owner: AsyncOpenGlSceneViewer, gl: Any, glut: Any) -> None:
        self.owner = owner
        self.gl = gl
        self.glut = glut
        self.window_id: int | None = None
        self.window_destroyed = False
        self.shader_program: int | None = None
        self.mvp_location: int | None = None
        self.point_size_location: int | None = None
        self.vbo: int | None = None
        self.vao: int | None = None
        self.vbo_capacity_bytes = 0
        self.vertex_count = 0
        self.inset_texture: int | None = None
        self.inset_rgb_u8: np.ndarray | None = None
        self.inset_texture_dirty = False
        self.current_snapshot: SceneViewerSnapshot | None = None
        self.fov_y_deg = 70.0
        self.latest_scene_center = np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        self.latest_scene_radius = 0.5
        self.scene_center = self.latest_scene_center.copy()
        self.scene_radius = self.latest_scene_radius
        self.has_scene = False
        self.yaw_deg = -25.0
        self.pitch_deg = 12.0
        self.zoom = 1.0
        self.rotating = False
        self.panning = False
        self.last_mouse = (0, 0)
        self.show_depth_inset = False
        self.view_dirty = True
        self.last_title_update_s = 0.0
        self.last_control_state = ""

    def initialize(self) -> None:
        global _GLUT_INITIALIZED
        gl, glut = self.gl, self.glut
        with _GLUT_INIT_LOCK:
            if not _GLUT_INITIALIZED:
                glut.glutInit()
                _GLUT_INITIALIZED = True
        glut.glutInitDisplayMode(
            glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH
        )
        glut.glutInitWindowSize(self.owner.width_px, self.owner.height_px)
        glut.glutInitWindowPosition(30, 30)
        self.window_id = int(
            glut.glutCreateWindow(self.owner.window_name.encode("utf-8"))
        )
        if self.window_id <= 0:
            raise RuntimeError("FreeGLUT failed to create the viewer window")
        self._disable_vsync()
        glut.glutSetOption(
            glut.GLUT_ACTION_ON_WINDOW_CLOSE,
            glut.GLUT_ACTION_CONTINUE_EXECUTION,
        )

        gl.glClearColor(*_BACKGROUND_RGB, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        self.shader_program = self._create_point_shader()
        self.mvp_location = gl.glGetUniformLocation(self.shader_program, "u_mvp")
        self.point_size_location = gl.glGetUniformLocation(
            self.shader_program,
            "u_point_size",
        )

        self.vbo = int(gl.glGenBuffers(1))
        self.vao = int(gl.glGenVertexArrays(1))
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        self.inset_texture = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.inset_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        glut.glutDisplayFunc(self._display_callback)
        glut.glutReshapeFunc(self._reshape)
        glut.glutKeyboardFunc(self._keyboard)
        glut.glutSpecialFunc(self._special_key)
        glut.glutMouseFunc(self._mouse)
        glut.glutMotionFunc(self._motion)
        glut.glutCloseFunc(self._window_closed)
        self._update_window_title(force=True)

    @staticmethod
    def _disable_vsync() -> None:
        """Prevent buffer swaps in the viewer thread from stalling inference."""

        if sys.platform != "win32":
            return
        opengl32 = ctypes.WinDLL("opengl32.dll")
        get_proc_address = opengl32.wglGetProcAddress
        get_proc_address.argtypes = [ctypes.c_char_p]
        get_proc_address.restype = ctypes.c_void_p
        address = get_proc_address(b"wglSwapIntervalEXT")
        invalid_addresses = {
            None,
            0,
            1,
            2,
            3,
            ctypes.c_void_p(-1).value,
        }
        if address in invalid_addresses:
            return
        swap_interval = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)(address)
        swap_interval(0)

    def run(self) -> None:
        interval_s = 1.0 / self.owner.render_fps
        next_render = 0.0
        while (
            not self.owner._close_requested.is_set()
            and not self.owner._quit_requested.is_set()
            and not self.window_destroyed
        ):
            self.glut.glutSetWindow(self.window_id)
            self.glut.glutMainLoopEvent()
            if self.window_destroyed or self.owner._quit_requested.is_set():
                break
            now = time.perf_counter()
            needs_frame = self.view_dirty or self.owner._has_pending_snapshot()
            if needs_frame and now >= next_render:
                snapshot = self.owner._take_pending_snapshot()
                if snapshot is not None:
                    self._consume_snapshot(snapshot)
                self._draw()
                self.view_dirty = False
                next_render = now + interval_s
            wait_s = (
                min(0.010, max(0.0, next_render - time.perf_counter()))
                if needs_frame
                else 0.010
            )
            self.owner._wake.wait(wait_s)
            self.owner._wake.clear()

    def destroy(self) -> None:
        if self.window_id is None or self.window_destroyed:
            return
        gl, glut = self.gl, self.glut
        glut.glutSetWindow(self.window_id)
        if self.shader_program:
            gl.glDeleteProgram(self.shader_program)
        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            gl.glDeleteBuffers(1, [self.vbo])
        if self.inset_texture:
            gl.glDeleteTextures(1, [self.inset_texture])
        glut.glutDestroyWindow(self.window_id)
        self.window_destroyed = True
        self.window_id = None

    def _display_callback(self) -> None:
        # Drawing is invoked explicitly from run() so failures propagate back
        # to the application instead of being swallowed by a GLUT callback.
        self.view_dirty = True

    def _consume_snapshot(self, snapshot: SceneViewerSnapshot) -> None:
        self.current_snapshot = snapshot
        vertices = deproject_sampled_rgbd(
            snapshot,
            depth_min_m=self.owner.depth_min_m,
            depth_max_m=self.owner.depth_max_m,
        )
        self.vertex_count = int(vertices.shape[0])
        self._upload_cloud(vertices, snapshot.sampled_depth_m_f32.size)
        self.fov_y_deg = float(
            np.clip(
                math.degrees(
                    2.0
                    * math.atan(
                        snapshot.calibration.height_px
                        / (2.0 * snapshot.calibration.fy_px)
                    )
                ),
                30.0,
                100.0,
            )
        )
        self._update_scene_bounds(vertices[:, :3])
        self._rebuild_inset()
        self._update_window_title()

    def _upload_cloud(self, vertices: np.ndarray, maximum_points: int) -> None:
        gl = self.gl
        required_capacity = int(maximum_points * 6 * np.dtype(np.float32).itemsize)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        if required_capacity > self.vbo_capacity_bytes:
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                required_capacity,
                None,
                gl.GL_STREAM_DRAW,
            )
            self.vbo_capacity_bytes = required_capacity
        if vertices.nbytes:
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _update_scene_bounds(self, xyz: np.ndarray) -> None:
        if xyz.shape[0] == 0:
            return
        lower = np.min(xyz, axis=0)
        upper = np.max(xyz, axis=0)
        center = ((lower + upper) * 0.5).astype(np.float32)
        radius = float(np.linalg.norm(upper - lower) * 0.5)
        if not np.isfinite(center).all() or not math.isfinite(radius):
            return
        self.latest_scene_center = center
        self.latest_scene_radius = float(np.clip(radius, 0.15, 20.0))
        if not self.has_scene:
            self._refit_view()
            self.has_scene = True

    def _refit_view(self) -> None:
        self.scene_center = self.latest_scene_center.copy()
        self.scene_radius = self.latest_scene_radius
        self.yaw_deg = -25.0
        self.pitch_deg = 12.0
        self.zoom = 1.0
        self.view_dirty = True

    def _rebuild_inset(self) -> None:
        if self.current_snapshot is None:
            return
        if self.show_depth_inset:
            self.inset_rgb_u8 = _depth_inset(
                self.current_snapshot,
                self.owner.depth_min_m,
                self.owner.depth_max_m,
            )
        else:
            self.inset_rgb_u8 = _segmentation_inset(
                self.current_snapshot,
                self.owner.overlay_alpha,
            )
        self.inset_texture_dirty = True
        self.view_dirty = True

    def _draw(self) -> None:
        gl = self.gl
        gl.glViewport(0, 0, self.owner.width_px, self.owner.height_px)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        projection, modelview, mvp = self._camera_matrices()
        self._draw_cloud(mvp)
        self._draw_geometry(projection, modelview)
        self._draw_overlay()
        self.glut.glutSwapBuffers()

    def _camera_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        aspect = self.owner.width_px / max(1, self.owner.height_px)
        near = 0.01
        far = max(20.0, self.scene_radius * 20.0)
        projection = _perspective(self.fov_y_deg, aspect, near, far)
        distance = max(0.35, self.scene_radius * 2.35) * self.zoom
        modelview = (
            _translation(0.0, 0.0, -distance)
            @ _rotation_x(self.pitch_deg)
            @ _rotation_y(self.yaw_deg)
            @ _translation(*(-self.scene_center))
        )
        return projection, modelview, projection @ modelview

    def _draw_cloud(self, mvp: np.ndarray) -> None:
        if self.vertex_count == 0:
            return
        gl = self.gl
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(self.shader_program)
        gl.glUniformMatrix4fv(
            self.mvp_location,
            1,
            gl.GL_TRUE,
            np.ascontiguousarray(mvp, dtype=np.float32),
        )
        gl.glUniform1f(self.point_size_location, self.owner.point_size_px)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.vertex_count)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def _draw_geometry(self, projection: np.ndarray, modelview: np.ndarray) -> None:
        snapshot = self.current_snapshot
        if snapshot is None or (
            snapshot.observation is None and snapshot.geometry is None
        ):
            return
        gl = self.gl
        gl.glUseProgram(0)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadMatrixf(np.ascontiguousarray(projection.T))
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadMatrixf(np.ascontiguousarray(modelview.T))
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(-1.0, -1.0)
        if snapshot.observation is not None:
            self._draw_observation(snapshot.observation)
        if snapshot.geometry is not None:
            for cable in snapshot.geometry.cables:
                color = _CABLE_COLORS[cable.cable_id % len(_CABLE_COLORS)]
                self._draw_cable(cable, color)
            if snapshot.geometry.cube is not None:
                self._draw_cube(snapshot.geometry.cube)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glDisable(gl.GL_BLEND)

    def _draw_observation(self, observation: CableObservationFrame) -> None:
        """Draw measured evidence only; invalid or hidden depth remains absent."""

        for edge in observation.graph_edges:
            self._draw_valid_polyline(
                edge.xyz_camera_m_f32,
                edge.depth_valid_bool,
                (0.12, 0.95, 0.90, 0.48),
                1.5,
            )

        for cable_id, routes in enumerate(observation.routes_by_cable):
            base = _CABLE_COLORS[cable_id % len(_CABLE_COLORS)]
            for rank, route in enumerate(routes[:3]):
                alpha = (0.96, 0.34, 0.18)[rank]
                width = (4.5, 2.5, 1.5)[rank]
                self._draw_valid_polyline(
                    route.xyz_camera_m_f32,
                    route.depth_valid_bool,
                    (base[0], base[1], base[2], alpha),
                    width,
                )

        gl = self.gl
        gl.glPointSize(11.0)
        gl.glBegin(gl.GL_POINTS)
        for endpoint in observation.endpoints:
            if not endpoint.depth_valid:
                continue
            color = _CABLE_COLORS[endpoint.cable_id % len(_CABLE_COLORS)]
            gl.glColor4f(color[0], color[1], color[2], 1.0)
            gl.glVertex3fv(endpoint.xyz_camera_m_f32)
        gl.glEnd()
        gl.glPointSize(1.0)

        pairing_colors = (
            (1.00, 0.10, 0.78, 0.92),
            (1.00, 0.82, 0.16, 0.25),
            (0.82, 0.84, 0.88, 0.16),
        )
        for crossing in observation.crossings:
            edge_to_arm = {
                int(edge_id): index
                for index, edge_id in enumerate(crossing.incident_edge_ids_i32)
            }
            for pairing in crossing.pairings:
                color = pairing_colors[min(pairing.rank, 2)]
                gl.glColor4f(*color)
                gl.glLineWidth(3.5 if pairing.rank == 0 else 1.0)
                gl.glBegin(gl.GL_LINES)
                for first_edge, second_edge in pairing.edge_pairs_i32:
                    first = edge_to_arm[int(first_edge)]
                    second = edge_to_arm[int(second_edge)]
                    if not (
                        crossing.arm_depth_valid_bool[first]
                        and crossing.arm_depth_valid_bool[second]
                    ):
                        continue
                    gl.glVertex3fv(crossing.arm_xyz_camera_m_f32[first])
                    gl.glVertex3fv(crossing.arm_xyz_camera_m_f32[second])
                gl.glEnd()
        gl.glLineWidth(1.0)

    def _draw_valid_polyline(
        self,
        xyz: np.ndarray,
        valid: np.ndarray,
        color: tuple[float, float, float, float],
        width: float,
    ) -> None:
        gl = self.gl
        gl.glColor4f(*color)
        gl.glLineWidth(width)
        start = 0
        count = int(valid.shape[0])
        while start < count:
            while start < count and not bool(valid[start]):
                start += 1
            stop = start
            while stop < count and bool(valid[stop]):
                stop += 1
            if stop - start >= 2:
                gl.glBegin(gl.GL_LINE_STRIP)
                for point in xyz[start:stop]:
                    gl.glVertex3fv(point)
                gl.glEnd()
            start = stop + 1
        gl.glLineWidth(1.0)

    def _draw_cable(
        self,
        cable: CableGeometry,
        color: tuple[float, float, float, float],
    ) -> None:
        if cable.mesh_vertices_camera_m_f32.shape[0]:
            self._draw_indexed_mesh(
                cable.mesh_vertices_camera_m_f32,
                cable.mesh_triangles_i32,
                color,
            )
        gl = self.gl
        gl.glColor4f(color[0], color[1], color[2], 1.0)
        gl.glLineWidth(3.0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, cable.centerline_camera_m_f32)
        gl.glDrawArrays(
            gl.GL_LINE_STRIP,
            0,
            cable.centerline_camera_m_f32.shape[0],
        )
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def _draw_cube(self, cube: CubeGeometry) -> None:
        if cube.mesh_vertices_cube_m_f32.shape[0] == 0:
            return
        gl = self.gl
        gl.glPushMatrix()
        gl.glMultMatrixd(np.ascontiguousarray(cube.pose_camera_from_cube_f64.T))
        self._draw_indexed_mesh(
            cube.mesh_vertices_cube_m_f32,
            cube.mesh_triangles_i32,
            (1.0, 0.78, 0.05, 0.82),
        )
        gl.glPopMatrix()

    def _draw_indexed_mesh(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        color: tuple[float, float, float, float],
    ) -> None:
        gl = self.gl
        indices = np.ascontiguousarray(triangles, dtype=np.uint32)
        gl.glColor4f(*color)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, vertices)
        gl.glDrawElements(
            gl.GL_TRIANGLES,
            int(indices.size),
            gl.GL_UNSIGNED_INT,
            indices,
        )
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def _draw_overlay(self) -> None:
        gl = self.gl
        width, height = self.owner.width_px, self.owner.height_px
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        if self.inset_rgb_u8 is not None:
            self._draw_inset()
        gl.glDisable(gl.GL_BLEND)

    def _draw_inset(self) -> None:
        gl = self.gl
        image = self.inset_rgb_u8
        if image is None:
            return
        if self.inset_texture_dirty:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.inset_texture)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGB,
                image.shape[1],
                image.shape[0],
                0,
                gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE,
                image,
            )
            self.inset_texture_dirty = False

        margin = 18
        available_width = self.owner.width_px - 2 * margin
        available_height = self.owner.height_px - 2 * margin
        if available_width < 40 or available_height < 40:
            return
        inset_width = min(
            self.owner.inset_width_px,
            available_width,
            int(available_height * image.shape[1] / image.shape[0]),
        )
        inset_height = max(1, int(round(inset_width * image.shape[0] / image.shape[1])))
        x0 = self.owner.width_px - margin - inset_width
        y1 = self.owner.height_px - margin
        y0 = y1 - inset_height

        gl.glColor4f(0.02, 0.025, 0.03, 0.92)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x0 - 4, y0 - 4)
        gl.glVertex2f(x0 + inset_width + 4, y0 - 4)
        gl.glVertex2f(x0 + inset_width + 4, y1 + 4)
        gl.glVertex2f(x0 - 4, y1 + 4)
        gl.glEnd()

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.inset_texture)
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 1.0); gl.glVertex2f(x0, y0)
        gl.glTexCoord2f(1.0, 1.0); gl.glVertex2f(x0 + inset_width, y0)
        gl.glTexCoord2f(1.0, 0.0); gl.glVertex2f(x0 + inset_width, y1)
        gl.glTexCoord2f(0.0, 0.0); gl.glVertex2f(x0, y1)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)

    def _update_window_title(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        snapshot = self.current_snapshot
        if snapshot is None:
            suffix = "waiting for synchronized RGB-D"
        else:
            inset = "depth" if self.show_depth_inset else "PIDNet"
            control_text = (
                snapshot.status_lines[-1].strip()
                if snapshot.status_lines
                else ""
            )
            control_state = control_text.split("|", 1)[0].strip()
            if control_state != self.last_control_state:
                force = True
                self.last_control_state = control_state
            suffix = (
                f"frame {snapshot.key.sequence_index} | points {self.vertex_count:,} | "
                f"viewer dropped {self.owner.dropped_snapshots} | inset {inset}"
            )
            if control_text:
                suffix += f" | {control_text}"
        if not force and now - self.last_title_update_s < 0.5:
            return
        self.glut.glutSetWindowTitle(f"{self.owner.window_name} | {suffix}")
        self.last_title_update_s = now

    def _reshape(self, width: int, height: int) -> None:
        self.owner.width_px = max(1, int(width))
        self.owner.height_px = max(1, int(height))
        self.view_dirty = True
        self.owner._wake.set()

    def _keyboard(self, key: bytes, _x: int, _y: int) -> None:
        if key in (b"q", b"Q", b"\x1b"):
            self.owner._request_action(ViewerAction.QUIT)
        elif key in (b"r", b"R"):
            self.owner._request_action(ViewerAction.TOGGLE_RECORDING)
        elif key == b" ":
            self.owner._request_action(ViewerAction.TOGGLE_PAUSE)
        elif key in (b"n", b"N"):
            self.owner._request_action(ViewerAction.STEP)
        elif key in (b"d", b"D"):
            self.show_depth_inset = not self.show_depth_inset
            self._rebuild_inset()
            self._update_window_title(force=True)
        self.view_dirty = True

    def _special_key(self, key: int, _x: int, _y: int) -> None:
        home_key = getattr(self.glut, "GLUT_KEY_HOME", None)
        if home_key is not None and key == home_key:
            self._refit_view()

    def _mouse(self, button: int, state: int, x: int, y: int) -> None:
        glut = self.glut
        if button == glut.GLUT_LEFT_BUTTON:
            self.rotating = state == glut.GLUT_DOWN
            self.last_mouse = (x, y)
        elif button == glut.GLUT_RIGHT_BUTTON:
            self.panning = state == glut.GLUT_DOWN
            self.last_mouse = (x, y)
        elif state == glut.GLUT_DOWN and button == 3:
            self.zoom = max(0.20, self.zoom * 0.90)
        elif state == glut.GLUT_DOWN and button == 4:
            self.zoom = min(8.0, self.zoom * 1.10)
        self.view_dirty = True

    def _motion(self, x: int, y: int) -> None:
        previous_x, previous_y = self.last_mouse
        dx, dy = x - previous_x, y - previous_y
        if self.rotating:
            self.yaw_deg += dx * 0.45
            self.pitch_deg = float(np.clip(self.pitch_deg + dy * 0.35, -85.0, 85.0))
        elif self.panning:
            scale = max(0.00005, self.scene_radius * 0.0015 * self.zoom)
            yaw = math.radians(self.yaw_deg)
            pitch = math.radians(self.pitch_deg)
            right = np.asarray((math.cos(yaw), 0.0, math.sin(yaw)), dtype=np.float32)
            up = np.asarray(
                (
                    math.sin(yaw) * math.sin(pitch),
                    math.cos(pitch),
                    -math.cos(yaw) * math.sin(pitch),
                ),
                dtype=np.float32,
            )
            self.scene_center = self.scene_center - dx * scale * right + dy * scale * up
        self.last_mouse = (x, y)
        self.view_dirty = True

    def _window_closed(self) -> None:
        self.window_destroyed = True
        self.owner._request_action(ViewerAction.QUIT)

    def _create_point_shader(self) -> int:
        gl = self.gl
        vertex = self._compile_shader(gl.GL_VERTEX_SHADER, _POINT_VERTEX_SHADER)
        fragment = self._compile_shader(gl.GL_FRAGMENT_SHADER, _POINT_FRAGMENT_SHADER)
        program = int(gl.glCreateProgram())
        gl.glAttachShader(program, vertex)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)
        gl.glDeleteShader(vertex)
        gl.glDeleteShader(fragment)
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            message = gl.glGetProgramInfoLog(program)
            gl.glDeleteProgram(program)
            raise RuntimeError(
                "OpenGL point shader link failed: "
                + message.decode("utf-8", errors="replace")
            )
        return program

    def _compile_shader(self, shader_type: int, source: str) -> int:
        gl = self.gl
        shader = int(gl.glCreateShader(shader_type))
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            message = gl.glGetShaderInfoLog(shader)
            gl.glDeleteShader(shader)
            raise RuntimeError(
                "OpenGL point shader compilation failed: "
                + message.decode("utf-8", errors="replace")
            )
        return shader


__all__ = [
    "AsyncOpenGlSceneViewer",
    "ViewerAction",
    "deproject_sampled_rgbd",
]
