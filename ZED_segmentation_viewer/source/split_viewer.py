"""Minimal split RGB/OpenGL point-cloud viewer for the segmentation pipeline."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys
from threading import Lock

import numpy as np


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
YELLOW = (1.00, 1.00, 0.00)

FEATURE_CONTROLS = (
    ("1", "endpoint_direction_proposal", "PROPOSAL"),
    ("2", "connected_trace", "GRAPH TRACE"),
    ("3", "route_length_score", "ROUTE LENGTH"),
    ("4", "endpoint_tangent_score", "TANGENT SCORE"),
    ("5", "smoothness", "SMOOTH"),
    ("6", "fixed_length", "FIXED LENGTH"),
    ("7", "temporal_prediction", "TEMPORAL"),
    ("8", "motion_adaptive_noise", "ADAPTIVE"),
    ("9", "global_particles", "RETRACK 10%"),
    ("0", "kink_limit", "KINK"),
)

POINT_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
uniform mat4 u_mvp;
uniform float u_point_size;
out vec3 v_color;
void main() {
    v_color = in_color;
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
    ):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.left_panel_width = max(260, int(left_panel_width))
        self.left_panel_ratio = float(np.clip(self.left_panel_width / self.width, 0.25, 0.55))
        self.point_size = float(np.clip(point_size, 1.0, 8.0))
        self.title = str(title)
        self.window_id = None
        self.available = False

        self.lock = Lock()
        self.pending = None
        self.pending_status = "Waiting for camera"
        self.rgb_image = None
        self.rgb_texture_dirty = False
        self.vertices = np.empty((0, 6), dtype=np.float32)
        self.vertex_count = 0
        self.status = "Waiting for camera"
        self.stats = {
            "total_pixels": 0,
            "valid_points": 0,
            "invalid_points": 0,
            "cable_points": 0,
            "endpoint_1_points": 0,
            "endpoint_2_points": 0,
            "crossing_points": 0,
        }
        self.pipeline_stats = {}
        self.frame_index = -1
        self.inference_ms = 0.0
        self.processing_ms = 0.0
        self.latency_ms = 0.0
        self.observation = None
        self.particle_filter = None
        self.show_top_particles = True
        self.feature_states = {}
        self.feature_callback = None
        self.feature_hitboxes = []

        self.vbo = None
        self.vao = None
        self.rgb_texture = None
        self.shader_program = None
        self.mvp_location = None
        self.point_size_location = None
        self.fov_y_deg = 70.0
        self.scene_center = np.asarray((0.0, 0.0, -2.0), dtype=np.float32)
        self.scene_radius = 2.0
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
        glBindTexture(GL_TEXTURE_2D, self.rgb_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
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
            glutPostRedisplay()
            glutMainLoopEvent()
        except Exception:
            self.available = False
        return self.available

    def close(self):
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
            glutDestroyWindow(window_id)
        except Exception:
            pass

    def set_camera_fov(self, degrees):
        if degrees is not None and np.isfinite(degrees):
            self.fov_y_deg = float(np.clip(degrees, 35.0, 100.0))

    def update_status(self, text):
        with self.lock:
            self.pending_status = str(text)

    def set_feature_controls(self, features, callback):
        self.feature_states = {
            name: bool(getattr(features, name)) for _key, name, _label in FEATURE_CONTROLS
        }
        self.feature_callback = callback

    def update_frame(self, result, pipeline_stats):
        vertices = np.asarray(result.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 6:
            raise ValueError(f"Viewer vertices must have shape Nx6; got {vertices.shape}.")
        image = np.asarray(result.rgb_image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Viewer RGB image must have shape HxWx3; got {image.shape}.")
        payload = {
            "frame_index": int(result.frame_index),
            "image": np.ascontiguousarray(image),
            "vertices": np.ascontiguousarray(vertices),
            "stats": dict(result.stats),
            "pipeline": dict(pipeline_stats),
            "center": np.asarray(result.scene_center, dtype=np.float32).reshape(3),
            "radius": float(result.scene_radius),
            "inference_ms": float(result.inference_ms),
            "processing_ms": float(result.processing_ms),
            "latency_ms": float(result.latency_ms),
            "observation": result.observation,
            "particle_filter": result.particle_filter,
        }
        with self.lock:
            self.pending = payload
            self.pending_status = "Live cable tracking"

    def reset_view(self):
        self.yaw_deg = -35.0
        self.pitch_deg = 22.0
        self.zoom = 1.0

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
        self.vertices = payload["vertices"]
        self.vertex_count = len(self.vertices)
        self.stats = payload["stats"]
        self.pipeline_stats = payload["pipeline"]
        self.frame_index = payload["frame_index"]
        self.inference_ms = payload["inference_ms"]
        self.processing_ms = payload["processing_ms"]
        self.latency_ms = payload["latency_ms"]
        self.observation = payload["observation"]
        self.particle_filter = payload["particle_filter"]
        self._smooth_scene(payload["center"], payload["radius"])
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STREAM_DRAW)
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
        self._draw_particle_filter()
        self._draw_camera_origin()
        self._draw_cloud_overlay(cloud_width, self.height)
        self._draw_divider(left_width)
        glutSwapBuffers()

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
        self._legend(margin + 380, height - 55, YELLOW, "crossing")
        self._line(0, height - header, width, height - header, UI_STROKE)
        if self.rgb_image is None:
            self._text(margin, height - header - 30, self.status, UI_MUTED)
            return

        image_h, image_w = self.rgb_image.shape[:2]
        available_w = max(1, width - 2 * margin)
        available_h = max(1, height - header - 2 * margin)
        scale = min(available_w / image_w, available_h / image_h)
        draw_w = max(1, int(round(image_w * scale)))
        draw_h = max(1, int(round(image_h * scale)))
        x0 = (width - draw_w) // 2
        y0 = (height - header - draw_h) // 2
        x1, y1 = x0 + draw_w, y0 + draw_h
        glBindTexture(GL_TEXTURE_2D, self.rgb_texture)
        if self.rgb_texture_dirty:
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
                self.rgb_image,
            )
            self.rgb_texture_dirty = False
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
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
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

    def _draw_particle_filter(self):
        if self.particle_filter is None:
            return
        glUseProgram(0)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        colors = (BLUE, GREEN)
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
            supported = np.asarray(cable.dense_supported, dtype=bool)
            if len(dense) >= 2:
                glLineWidth(4.0)
                glBegin(GL_LINES)
                for index in range(len(dense) - 1):
                    if index < len(supported) and index + 1 < len(supported):
                        segment_supported = bool(supported[index] and supported[index + 1])
                    else:
                        segment_supported = False
                    glColor4f(*(color if segment_supported else (1.0, 0.18, 0.15)), 1.0)
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
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    def _draw_cloud_overlay(self, width, height):
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        header = 213
        self._rect(0, height - header, width, header, (0.018, 0.021, 0.025))
        self._rect(0, height - header, 5, header, UI_ACCENT)
        self._rect(0, height - header, width, 1, UI_STROKE)
        self._text(18, height - 29, "Full-resolution 3D point cloud", UI_TEXT)
        self._text(
            18,
            height - 53,
            f"Frame {self.frame_index} | NN {self.inference_ms:.1f} ms | "
            f"viewer prep {self.processing_ms:.1f} ms | display latency {self.latency_ms:.1f} ms | "
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
        x = self._metric(x, y, "ENDPOINT 2", f"{self.stats.get('endpoint_2_points', 0):,}", GREEN)
        self._metric(x, y, "CROSSING", f"{self.stats.get('crossing_points', 0):,}", YELLOW)

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

        if self.particle_filter is not None:
            for cable_index, cable in enumerate(self.particle_filter.cables):
                diagnostic = cable.diagnostics
                y = height - 151 - cable_index * 22
                color = (BLUE, GREEN)[cable_index]
                self._text(
                    18,
                    y,
                    f"PF{cable_index + 1} {diagnostic.status} | "
                    f"route={diagnostic.selected_route_index + 1}/"
                    f"{diagnostic.route_candidate_count} "
                    f"dL={diagnostic.route_length_error_mm:.1f} mm | "
                    f"trace={diagnostic.trace_mean_mm:.1f}/{diagnostic.trace_p95_mm:.1f} mm | "
                    f"support={100.0 * diagnostic.support_fraction:.0f}% | "
                    f"turn={diagnostic.turn_rms_degrees:.1f} deg | "
                    f"ESS={diagnostic.effective_sample_size:.0f}",
                    color,
                    GLUT_BITMAP_HELVETICA_12,
                )
            self._text(
                18,
                height - 195,
                f"Observation {self.pipeline_stats.get('observation_ms', 0.0):.1f} ms | "
                f"PF {self.pipeline_stats.get('particle_filter_ms', 0.0):.1f} ms "
                f"(CUDA {self.pipeline_stats.get('particle_filter_gpu_ms', 0.0):.1f} ms) | "
                f"total tracking {self.pipeline_stats.get('tracking_ms', 0.0):.1f} ms",
                UI_MUTED,
                GLUT_BITMAP_HELVETICA_12,
            )

        self._rect(0, 0, width, 82, (0.018, 0.021, 0.025))
        self.feature_hitboxes = []
        x = 18
        chip_y = 56
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
        phase_text = "CROSSING: PHASE 2"
        if x + self._text_width(phase_text, GLUT_BITMAP_HELVETICA_12) + 16 > width:
            x = 18
            chip_y -= 25
        self._text(x + 5, chip_y + 6, phase_text, YELLOW, GLUT_BITMAP_HELVETICA_12)
        self._text(
            18,
            10,
            f"Orbit: left drag    Pan: right drag    Zoom: wheel    Reset: R    "
            f"Top particles: P ({'on' if self.show_top_particles else 'off'})    "
            f"Point size: +/- ({self.point_size:.1f})    Click a feature or press 0-9    Quit: Q",
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

    def _smooth_scene(self, center, radius):
        radius = float(np.clip(radius, 0.25, 20.0))
        if not self.has_scene:
            self.scene_center = center.copy()
            self.scene_radius = radius
            self.has_scene = True
            return
        self.scene_center = (0.9 * self.scene_center + 0.1 * center).astype(np.float32)
        self.scene_radius = 0.9 * self.scene_radius + 0.1 * radius

    def _reshape(self, width, height):
        self.width = max(1, int(width))
        self.height = max(1, int(height))

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
        else:
            try:
                character = key.decode("ascii")
            except (AttributeError, UnicodeDecodeError):
                character = ""
            for control_key, name, _label in FEATURE_CONTROLS:
                if character == control_key:
                    self._toggle_feature(name)
                    break

    def _special_key(self, key, _x, _y):
        if key == GLUT_KEY_LEFT:
            self.yaw_deg -= 4.0
        elif key == GLUT_KEY_RIGHT:
            self.yaw_deg += 4.0
        elif key == GLUT_KEY_UP:
            self.pitch_deg = min(85.0, self.pitch_deg + 4.0)
        elif key == GLUT_KEY_DOWN:
            self.pitch_deg = max(-85.0, self.pitch_deg - 4.0)

    def _mouse_in_cloud(self, x):
        left, _right = self._panel_sizes()
        return x >= left

    def _mouse(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            bottom_y = self.height - y
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
