"""Asynchronous OpenGL viewport for ZED registered-depth point clouds."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np

from .contracts import StereoCalibration, StereoFrame


_GLUT_LOCK = threading.Lock()
_GLUT_INITIALIZED = False
_FREEGLUT: Any | None = None


@dataclass(frozen=True, slots=True)
class PointCloudSnapshot:
    frame: StereoFrame
    calibration: StereoCalibration
    observed_curve_camera_m: np.ndarray | tuple[np.ndarray, ...] | None = None
    estimated_centerline_camera_m: np.ndarray | None = None
    status: str = "ZED point cloud"
    segmentation_rgb: np.ndarray | None = None
    skeleton_graph_rgb: np.ndarray | None = None
    cable_mask: np.ndarray | None = None
    top_panel_title: str = "RGB + PIDNET SEGMENTATION"
    bottom_panel_title: str = "SKELETON + OBSERVATION SAMPLES"


def deproject_for_viewer(
    frame: StereoFrame,
    calibration: StereoCalibration,
    *,
    stride: int,
    depth_min_m: float,
    depth_max_m: float,
    cable_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return interleaved XYZRGB in OpenGL's right-handed camera frame."""

    if frame.depth_m is None:
        raise ValueError("The point-cloud viewer requires registered depth.")
    depth = frame.depth_m[::stride, ::stride]
    bgr = frame.left_bgr[::stride, ::stride]
    valid = np.isfinite(depth) & (depth >= depth_min_m) & (depth <= depth_max_m)
    rows, columns = np.nonzero(valid)
    vertices = np.empty((len(rows), 6), dtype=np.float32)
    if len(rows) == 0:
        return vertices
    z = depth[rows, columns]
    k = calibration.left_intrinsics
    u = columns.astype(np.float32) * stride
    v = rows.astype(np.float32) * stride
    vertices[:, 0] = (u - k[0, 2]) * z / k[0, 0]
    vertices[:, 1] = -(v - k[1, 2]) * z / k[1, 1]
    vertices[:, 2] = -z
    colors = bgr[rows, columns]
    vertices[:, 3] = colors[:, 2] / 255.0
    vertices[:, 4] = colors[:, 1] / 255.0
    vertices[:, 5] = colors[:, 0] / 255.0
    if cable_mask is not None:
        highlight = np.asarray(cable_mask, dtype=bool)
        if highlight.shape != frame.depth_m.shape:
            raise ValueError("Cable highlight mask must match registered depth.")
        selected = highlight[::stride, ::stride][rows, columns]
        vertices[selected, 3:6] = np.asarray((1.0, 0.55, 0.05), dtype=np.float32)
    return np.ascontiguousarray(vertices)


def _load_gl() -> tuple[Any, Any]:
    global _FREEGLUT
    if sys.platform == "win32" and _FREEGLUT is None:
        folder = Path(sys.prefix) / "Lib" / "site-packages" / "OpenGL" / "DLLS"
        bits = "64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "32"
        candidates = [folder / "freeglut.dll"] + [
            folder / f"freeglut{bits}.{version}.dll" for version in ("vc14", "vc10", "vc9")
        ]
        for candidate in candidates:
            if candidate.is_file():
                _FREEGLUT = ctypes.WinDLL(str(candidate))
                break
        if _FREEGLUT is None:
            raise RuntimeError(f"FreeGLUT was not found in {folder}.")
        import OpenGL.platform

        OpenGL.platform.PLATFORM.GLUT = _FREEGLUT
    from OpenGL import GL, GLUT

    return GL, GLUT


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    scale = 1.0 / math.tan(math.radians(fov_y) * 0.5)
    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = scale / max(aspect, 1.0e-6)
    result[1, 1] = scale
    result[2, 2] = (far + near) / (near - far)
    result[2, 3] = 2.0 * far * near / (near - far)
    result[3, 2] = -1.0
    return result


def _translation(x: float, y: float, z: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    result[:3, 3] = (x, y, z)
    return result


def _rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    result = np.eye(4, dtype=np.float32)
    result[1, 1], result[1, 2] = c, -s
    result[2, 1], result[2, 2] = s, c
    return result


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    result = np.eye(4, dtype=np.float32)
    result[0, 0], result[0, 2] = c, s
    result[2, 0], result[2, 2] = -s, c
    return result


_VERTEX_SHADER = """
#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 color;
uniform mat4 mvp;
uniform float point_size;
out vec3 vertex_color;
void main(){vertex_color=color; gl_Position=mvp*vec4(position,1.0); gl_PointSize=point_size;}
"""
_FRAGMENT_SHADER = """
#version 330 core
in vec3 vertex_color;
out vec4 output_color;
void main(){output_color=vec4(vertex_color,1.0);}
"""


class AsyncZedPointCloudViewer:
    """Capacity-one viewer; camera acquisition and fitting never wait for GL."""

    def __init__(
        self,
        *,
        width_px: int = 1500,
        height_px: int = 850,
        stride: int = 4,
        depth_min_m: float = 0.20,
        depth_max_m: float = 4.0,
    ) -> None:
        self.width_px, self.height_px = int(width_px), int(height_px)
        self.stride = int(stride)
        self.depth_min_m, self.depth_max_m = float(depth_min_m), float(depth_max_m)
        self._condition = threading.Condition()
        self._pending: PointCloudSnapshot | None = None
        self._close = threading.Event()
        self._quit = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="zed-pointcloud-viewer", daemon=True)
        self._thread.start()
        if not self._ready.wait(10.0):
            raise RuntimeError("Timed out creating the ZED point-cloud viewport.")
        self._raise_error()

    @property
    def quit_requested(self) -> bool:
        self._raise_error()
        return self._quit.is_set()

    def publish(self, snapshot: PointCloudSnapshot) -> None:
        self._raise_error()
        with self._condition:
            self._pending = snapshot
            self._condition.notify()

    def close(self) -> None:
        self._close.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("ZED point-cloud viewport did not close.")
        self._raise_error()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("ZED point-cloud viewport failed.") from self._error

    def _run(self) -> None:
        renderer = None
        try:
            gl, glut = _load_gl()
            renderer = _Renderer(self, gl, glut)
            renderer.initialize()
            self._ready.set()
            renderer.loop()
        except BaseException as error:
            self._error = error
            self._quit.set()
            self._ready.set()
        finally:
            if renderer is not None:
                renderer.destroy()


class _Renderer:
    def __init__(self, owner: AsyncZedPointCloudViewer, gl: Any, glut: Any) -> None:
        self.owner, self.gl, self.glut = owner, gl, glut
        self.window = 0
        self.program = self.vao = self.vbo = 0
        self.panel_textures: list[int] = []
        self.panel_images: tuple[np.ndarray | None, np.ndarray | None] = (None, None)
        self.panel_dirty = False
        self.capacity = self.vertex_count = 0
        self.mvp_location = self.size_location = -1
        self.snapshot: PointCloudSnapshot | None = None
        self.center = np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        self.radius = 0.5
        self.fitted = False
        self.fitted_to_cable = False
        self.yaw, self.pitch, self.zoom = -25.0, 12.0, 1.0
        self.rotating = self.panning = False
        self.last_mouse = (0, 0)
        self.dirty = True
        self.destroyed = False

    def initialize(self) -> None:
        global _GLUT_INITIALIZED
        gl, glut = self.gl, self.glut
        with _GLUT_LOCK:
            if not _GLUT_INITIALIZED:
                glut.glutInit()
                _GLUT_INITIALIZED = True
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
        glut.glutInitWindowSize(self.owner.width_px, self.owner.height_px)
        self.window = int(glut.glutCreateWindow(b"Cable Twin - 3D View"))
        glut.glutSetOption(glut.GLUT_ACTION_ON_WINDOW_CLOSE, glut.GLUT_ACTION_CONTINUE_EXECUTION)
        gl.glClearColor(0.012, 0.015, 0.019, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        self.program = self._program()
        self.mvp_location = gl.glGetUniformLocation(self.program, "mvp")
        self.size_location = gl.glGetUniformLocation(self.program, "point_size")
        self.vbo = int(gl.glGenBuffers(1)); self.vao = int(gl.glGenVertexArrays(1))
        gl.glBindVertexArray(self.vao); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0); gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        gl.glBindVertexArray(0)
        generated = gl.glGenTextures(2)
        self.panel_textures = [int(value) for value in generated]
        for texture in self.panel_textures:
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        glut.glutDisplayFunc(self._display); glut.glutReshapeFunc(self._reshape)
        glut.glutKeyboardFunc(self._keyboard); glut.glutSpecialFunc(self._special)
        glut.glutMouseFunc(self._mouse); glut.glutMotionFunc(self._motion)
        glut.glutCloseFunc(self._closed)

    def loop(self) -> None:
        while not self.owner._close.is_set() and not self.owner._quit.is_set() and not self.destroyed:
            self.glut.glutSetWindow(self.window); self.glut.glutMainLoopEvent()
            with self.owner._condition:
                snapshot = self.owner._pending
                self.owner._pending = None
            if snapshot is not None:
                self._consume(snapshot)
            if self.dirty:
                self._draw(); self.dirty = False
            with self.owner._condition:
                if self.owner._pending is None and not self.owner._close.is_set():
                    self.owner._condition.wait(timeout=0.01)

    def _consume(self, snapshot: PointCloudSnapshot) -> None:
        self.snapshot = snapshot
        vertices = deproject_for_viewer(
            snapshot.frame, snapshot.calibration, stride=self.owner.stride,
            depth_min_m=self.owner.depth_min_m, depth_max_m=self.owner.depth_max_m,
            cable_mask=snapshot.cable_mask,
        )
        self.vertex_count = len(vertices)
        required = max(vertices.nbytes, 1)
        gl = self.gl; gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        if required > self.capacity:
            gl.glBufferData(gl.GL_ARRAY_BUFFER, required, None, gl.GL_STREAM_DRAW); self.capacity = required
        if vertices.nbytes:
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        cable_points = self._cable_points(snapshot)
        if len(cable_points) and not self.fitted_to_cable:
            lower, upper = cable_points.min(0), cable_points.max(0)
            self.center = ((lower + upper) * 0.5).astype(np.float32)
            self.radius = max(0.25, float(np.linalg.norm(upper - lower) * 0.75))
            self.fitted = True
            self.fitted_to_cable = True
        elif len(vertices) and not self.fitted:
            lower, upper = vertices[:, :3].min(0), vertices[:, :3].max(0)
            self.center = ((lower + upper) * 0.5).astype(np.float32)
            self.radius = max(0.15, float(np.linalg.norm(upper - lower) * 0.5))
            self.fitted = True
        self.panel_images = (snapshot.segmentation_rgb, snapshot.skeleton_graph_rgb)
        self.panel_dirty = True
        title = f"Cable Twin - 3D View | {snapshot.status} | {self.vertex_count:,} points"
        self.glut.glutSetWindowTitle(title)
        self.dirty = True

    @staticmethod
    def _cable_points(snapshot: PointCloudSnapshot) -> np.ndarray:
        values: list[np.ndarray] = []
        if snapshot.estimated_centerline_camera_m is not None:
            values.append(np.asarray(snapshot.estimated_centerline_camera_m, dtype=np.float32))
        observed = snapshot.observed_curve_camera_m
        if observed is not None:
            segments = observed if isinstance(observed, tuple) else (observed,)
            values.extend(np.asarray(segment, dtype=np.float32) for segment in segments)
        valid = [
            value
            for value in values
            if value.ndim == 2
            and value.shape[1] == 3
            and len(value) >= 2
            and np.all(np.isfinite(value))
        ]
        if not valid:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(valid) * np.asarray((1.0, -1.0, -1.0), dtype=np.float32)

    def _matrices(self, render_width: int) -> tuple[np.ndarray, np.ndarray]:
        projection = _perspective(70.0, render_width / max(self.owner.height_px, 1), 0.01, 30.0)
        distance = max(0.4, 2.35 * self.radius) * self.zoom
        view = _translation(0.0, 0.0, -distance) @ _rotation_x(self.pitch) @ _rotation_y(self.yaw) @ _translation(*(-self.center))
        return projection, view

    def _draw(self) -> None:
        gl = self.gl
        left_width = min(520, max(360, int(round(0.32 * self.owner.width_px))))
        right_width = max(1, self.owner.width_px - left_width)
        gl.glViewport(left_width, 0, right_width, self.owner.height_px)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        projection, view = self._matrices(right_width); mvp = projection @ view
        if self.vertex_count:
            gl.glUseProgram(self.program)
            gl.glUniformMatrix4fv(self.mvp_location, 1, gl.GL_TRUE, np.ascontiguousarray(mvp))
            gl.glUniform1f(self.size_location, 2.0)
            gl.glBindVertexArray(self.vao); gl.glDrawArrays(gl.GL_POINTS, 0, self.vertex_count)
            gl.glBindVertexArray(0); gl.glUseProgram(0)
        if self.snapshot is not None:
            observed = self.snapshot.observed_curve_camera_m
            if observed is not None:
                segments = observed if isinstance(observed, tuple) else (observed,)
                self._draw_curves(
                    segments, projection, view,
                    color=(1.0, 0.72, 0.10), line_width=3.0, point_size=0.0,
                )
            estimate = self.snapshot.estimated_centerline_camera_m
            if estimate is not None:
                self._draw_curves(
                    (estimate,), projection, view,
                    color=(0.10, 1.0, 0.35), line_width=5.0, point_size=8.0,
                )
        self._draw_panels(left_width)
        self.glut.glutSwapBuffers()

    def _draw_curves(
        self,
        segments: tuple[np.ndarray, ...],
        projection: np.ndarray,
        view: np.ndarray,
        *,
        color: tuple[float, float, float],
        line_width: float,
        point_size: float,
    ) -> None:
        gl = self.gl
        gl.glMatrixMode(gl.GL_PROJECTION); gl.glLoadMatrixf(projection.T)
        gl.glMatrixMode(gl.GL_MODELVIEW); gl.glLoadMatrixf(view.T)
        gl.glColor3f(*color); gl.glLineWidth(line_width)
        for segment in segments:
            points = np.asarray(segment, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
                continue
            points = points * np.asarray((1.0, -1.0, -1.0), dtype=np.float32)
            gl.glBegin(gl.GL_LINE_STRIP)
            for point in points: gl.glVertex3fv(point)
            gl.glEnd()
            if point_size > 0.0:
                gl.glPointSize(point_size); gl.glBegin(gl.GL_POINTS)
                for point in points: gl.glVertex3fv(point)
                gl.glEnd()
        gl.glLineWidth(1.0)

    def _draw_panels(self, left_width: int) -> None:
        gl = self.gl
        gl.glViewport(0, 0, self.owner.width_px, self.owner.height_px)
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glMatrixMode(gl.GL_PROJECTION); gl.glLoadIdentity()
        gl.glOrtho(0, self.owner.width_px, 0, self.owner.height_px, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW); gl.glLoadIdentity()
        if self.panel_dirty:
            for texture, image in zip(self.panel_textures, self.panel_images, strict=True):
                if image is None:
                    continue
                rgb = np.ascontiguousarray(image, dtype=np.uint8)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D, 0, gl.GL_RGB, rgb.shape[1], rgb.shape[0],
                    0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb,
                )
            self.panel_dirty = False
        gl.glColor3f(0.025, 0.032, 0.042)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(0, 0); gl.glVertex2f(left_width, 0)
        gl.glVertex2f(left_width, self.owner.height_px); gl.glVertex2f(0, self.owner.height_px)
        gl.glEnd()
        margin = 10
        half = self.owner.height_px // 2
        single_panel = self.panel_images[0] is not None and self.panel_images[1] is None
        for index, (texture, image) in enumerate(zip(self.panel_textures, self.panel_images, strict=True)):
            if image is None:
                continue
            if single_panel:
                area_y0, area_y1 = 0, self.owner.height_px
            else:
                area_y0 = half if index == 0 else 0
                area_y1 = self.owner.height_px if index == 0 else half
            available_w = left_width - 2 * margin
            available_h = area_y1 - area_y0 - 2 * margin
            scale = min(available_w / image.shape[1], available_h / image.shape[0])
            width = max(1, int(round(image.shape[1] * scale)))
            height = max(1, int(round(image.shape[0] * scale)))
            x0 = (left_width - width) // 2
            y0 = area_y0 + (area_y1 - area_y0 - height) // 2
            x1, y1 = x0 + width, y0 + height
            gl.glEnable(gl.GL_TEXTURE_2D); gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glColor3f(1.0, 1.0, 1.0)
            gl.glBegin(gl.GL_QUADS)
            gl.glTexCoord2f(0.0, 1.0); gl.glVertex2f(x0, y0)
            gl.glTexCoord2f(1.0, 1.0); gl.glVertex2f(x1, y0)
            gl.glTexCoord2f(1.0, 0.0); gl.glVertex2f(x1, y1)
            gl.glTexCoord2f(0.0, 0.0); gl.glVertex2f(x0, y1)
            gl.glEnd(); gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glColor3f(0.22, 0.26, 0.30); gl.glLineWidth(1.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex2f(left_width, 0); gl.glVertex2f(left_width, self.owner.height_px)
        if not single_panel:
            gl.glVertex2f(0, half); gl.glVertex2f(left_width, half)
        gl.glEnd()
        if self.snapshot is not None:
            self._draw_text(
                14,
                self.owner.height_px - 26,
                self.snapshot.top_panel_title,
            )
            if self.panel_images[1] is not None:
                self._draw_text(14, half - 26, self.snapshot.bottom_panel_title)
        self._draw_text(left_width + 16, self.owner.height_px - 26, "ZED 3D POINT CLOUD + CABLE STATE")
        if self.snapshot is not None:
            self._draw_text(left_width + 16, self.owner.height_px - 50, self.snapshot.status)
            if self.snapshot.observed_curve_camera_m is not None:
                self._draw_text(left_width + 16, 44, "YELLOW: 3D OBSERVATION")
            if self.snapshot.estimated_centerline_camera_m is not None:
                self._draw_text(left_width + 250, 44, "GREEN: PF + DDER")
        self._draw_text(left_width + 16, 18, "LEFT DRAG rotate | RIGHT DRAG pan | WHEEL zoom | R reset")
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _draw_text(self, x: int, y: int, value: str) -> None:
        self.gl.glColor3f(0.92, 0.94, 0.96)
        self.gl.glRasterPos2f(float(x), float(y))
        font = self.glut.GLUT_BITMAP_HELVETICA_18
        for character in value:
            self.glut.glutBitmapCharacter(font, ord(character))

    def _refit(self) -> None:
        self.fitted = False; self.fitted_to_cable = False
        self.yaw, self.pitch, self.zoom = -25.0, 12.0, 1.0; self.dirty = True

    def _display(self) -> None: self.dirty = True
    def _reshape(self, width: int, height: int) -> None:
        self.owner.width_px, self.owner.height_px = max(1, width), max(1, height); self.dirty = True
    def _keyboard(self, key: bytes, _x: int, _y: int) -> None:
        if key in (b"q", b"Q", b"\x1b"): self.owner._quit.set()
        elif key in (b"r", b"R"): self._refit()
    def _special(self, key: int, _x: int, _y: int) -> None:
        if key == getattr(self.glut, "GLUT_KEY_HOME", -1): self._refit()
    def _mouse(self, button: int, state: int, x: int, y: int) -> None:
        if button == self.glut.GLUT_LEFT_BUTTON: self.rotating = state == self.glut.GLUT_DOWN
        elif button == self.glut.GLUT_RIGHT_BUTTON: self.panning = state == self.glut.GLUT_DOWN
        elif state == self.glut.GLUT_DOWN and button == 3: self.zoom = max(0.2, self.zoom * 0.9)
        elif state == self.glut.GLUT_DOWN and button == 4: self.zoom = min(8.0, self.zoom * 1.1)
        self.last_mouse = (x, y); self.dirty = True
    def _motion(self, x: int, y: int) -> None:
        dx, dy = x - self.last_mouse[0], y - self.last_mouse[1]
        if self.rotating:
            self.yaw += 0.45 * dx; self.pitch = float(np.clip(self.pitch + 0.35 * dy, -85.0, 85.0))
        elif self.panning:
            scale = self.radius * 0.0015 * self.zoom
            yaw = math.radians(self.yaw); pitch = math.radians(self.pitch)
            right = np.asarray((math.cos(yaw), 0.0, math.sin(yaw)), dtype=np.float32)
            up = np.asarray((math.sin(yaw)*math.sin(pitch), math.cos(pitch), -math.cos(yaw)*math.sin(pitch)), dtype=np.float32)
            self.center = self.center - dx * scale * right + dy * scale * up
        self.last_mouse = (x, y); self.dirty = True
    def _closed(self) -> None: self.destroyed = True; self.owner._quit.set()

    def _program(self) -> int:
        gl = self.gl
        shaders = []
        for kind, source in ((gl.GL_VERTEX_SHADER, _VERTEX_SHADER), (gl.GL_FRAGMENT_SHADER, _FRAGMENT_SHADER)):
            shader = int(gl.glCreateShader(kind)); gl.glShaderSource(shader, source); gl.glCompileShader(shader)
            if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
                raise RuntimeError(gl.glGetShaderInfoLog(shader).decode(errors="replace"))
            shaders.append(shader)
        program = int(gl.glCreateProgram())
        for shader in shaders: gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        for shader in shaders: gl.glDeleteShader(shader)
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            raise RuntimeError(gl.glGetProgramInfoLog(program).decode(errors="replace"))
        return program

    def destroy(self) -> None:
        if self.window and not self.destroyed:
            self.glut.glutSetWindow(self.window)
            if self.program: self.gl.glDeleteProgram(self.program)
            if self.vao: self.gl.glDeleteVertexArrays(1, [self.vao])
            if self.vbo: self.gl.glDeleteBuffers(1, [self.vbo])
            if self.panel_textures: self.gl.glDeleteTextures(len(self.panel_textures), self.panel_textures)
            self.glut.glutDestroyWindow(self.window); self.destroyed = True
