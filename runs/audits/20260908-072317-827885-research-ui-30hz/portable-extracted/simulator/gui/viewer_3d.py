"""Interactive 3D view for the point mass, DDER cable, target, and force."""

from __future__ import annotations

import importlib
import os
from typing import Sequence

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget


def _load_vtk_view_dependencies() -> None:
    # Load the native dependency explicitly before PyVista lazily loads the
    # context interactor. This avoids resolving vtkViewsCore from inside the
    # vtkViewsContext2D extension loader (the reported Windows import failure).
    importlib.import_module("vtkmodules.vtkViewsCore")
    importlib.import_module("vtkmodules.vtkViewsContext2D")


def _polyline(points: np.ndarray):
    import pyvista as pv

    values = np.asarray(points, dtype=np.float64)
    mesh = pv.PolyData(values)
    mesh.lines = np.concatenate(([len(values)], np.arange(len(values)))).astype(
        np.int64
    )
    return mesh


class PointCableViewer3D(QWidget):
    """Persistent scene; the drone glyph represents the force-controlled point."""

    backend_name = "PYVISTA / VTK"
    NORMAL_CONTROL_COLOR = '#2563eb'
    FORCE_CONTROL_COLOR = '#f97316'

    def set_controller_mode(self, force_control: bool) -> None:
        """Testing-only visual state; normal control includes hover and recovery."""
        color = self.FORCE_CONTROL_COLOR if force_control else self.NORMAL_CONTROL_COLOR
        for actor in (self._drone_actor, self._point_actor, self._force_actor):
            actor.prop.color = color

    def __init__(
        self,
        initial_cable_positions_m: np.ndarray,
        target_position_m: Sequence[float],
        desired_direction_world: Sequence[float],
        target_radius_m: float,
        parent: QWidget | None = None,
    ) -> None:
        _load_vtk_view_dependencies()
        import pyvista as pv
        from pyvistaqt import QtInteractor

        super().__init__(parent)

        self._pv = pv
        self._show_force = True
        self._show_trails = True
        self._last_force = np.zeros(3, dtype=np.float64)
        self._target = np.asarray(target_position_m, dtype=np.float64)
        self._desired_direction = np.asarray(
            desired_direction_world, dtype=np.float64
        )
        self._target_radius_m = float(target_radius_m)
        self._last_root = np.asarray(initial_cable_positions_m[0], dtype=np.float64)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self, auto_update=False)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#edf1f5")
        self._build_environment()

        cable = np.asarray(initial_cable_positions_m, dtype=np.float64)
        self._cable_mesh = _polyline(cable)
        self.plotter.add_mesh(
            self._cable_mesh,
            name="cable",
            color="#17a9c4",
            line_width=7,
            render_lines_as_tubes=True,
            smooth_shading=True,
            pickable=False,
        )
        self._nodes_mesh = pv.PolyData(cable[1:-1])
        self.plotter.add_mesh(
            self._nodes_mesh,
            name="cableNodes",
            color="#69d5e6",
            point_size=9,
            render_points_as_spheres=True,
            pickable=False,
        )
        self._point_actor = self.plotter.add_mesh(
            pv.Sphere(radius=0.037, theta_resolution=28, phi_resolution=28),
            name="controlledPoint",
            color="#f59e0b",
            metallic=0.12,
            smooth_shading=True,
            pickable=False,
        )
        drone = pv.Cube(x_length=.07, y_length=.05, z_length=.025)
        for x, y in ((-.1, -.1), (-.1, .1), (.1, -.1), (.1, .1)):
            drone = drone.merge(pv.Line((0., 0., 0.), (x, y, 0.)).tube(radius=.008))
            drone = drone.merge(pv.Cylinder(center=(x, y, 0.), direction=(0., 0., 1.), radius=.045, height=.008))
        self._drone_actor = self.plotter.add_mesh(drone, color="#f59e0b", name="drone", pickable=False)
        self._drone_actor.SetVisibility(False)
        self._tip_actor = self.plotter.add_mesh(
            pv.Sphere(radius=0.025, theta_resolution=24, phi_resolution=24),
            name="cableTip",
            color="#f97316",
            smooth_shading=True,
            pickable=False,
        )
        self._root_trail_mesh = _polyline(np.vstack((cable[0], cable[0])))
        self._root_trail_actor = self.plotter.add_mesh(
            self._root_trail_mesh,
            name="rootTrail",
            color="#f59e0b",
            line_width=2,
            opacity=0.75,
            pickable=False,
        )
        self._tip_trail_mesh = _polyline(np.vstack((cable[-1], cable[-1])))
        self._tip_trail_actor = self.plotter.add_mesh(
            self._tip_trail_mesh,
            name="tipTrail",
            color="#f97316",
            line_width=2,
            opacity=0.8,
            pickable=False,
        )
        self._force_mesh = pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            scale=1.0,
            tip_length=0.22,
            tip_radius=0.075,
            shaft_radius=0.023,
        )
        self._force_template_points = self._force_mesh.points.copy()
        self._force_actor = self.plotter.add_mesh(
            self._force_mesh,
            name="commandForce",
            color="#dc2626",
            smooth_shading=True,
            pickable=False,
        )
        self._force_actor.SetVisibility(False)
        self.set_target(
            self._target,
            self._desired_direction,
            self._target_radius_m,
            render=False,
        )
        self.update_state(cable, np.zeros(3), cable[[0]], cable[[-1]])
        self.set_camera_preset("Perspective")

    def _build_environment(self) -> None:
        pv = self._pv
        ground = pv.Plane(
            center=(0.5, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            i_size=3.4,
            j_size=3.4,
            i_resolution=24,
            j_resolution=24,
        )
        self.plotter.add_mesh(
            ground,
            color="#f7f8fa",
            opacity=0.7,
            show_edges=True,
            edge_color="#cbd2da",
            line_width=1,
            pickable=False,
        )
        self.plotter.add_axes(
            line_width=2,
            labels_off=False,
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )
        self.plotter.show_grid(
            color="#7c8796",
            location="outer",
            ticks="outside",
            font_size=9,
        )

    def set_target(
        self,
        target_position_m: Sequence[float],
        desired_direction_world: Sequence[float],
        target_radius_m: float,
        *,
        render: bool = True,
    ) -> None:
        pv = self._pv
        self._target = np.asarray(target_position_m, dtype=np.float64)
        direction = np.asarray(desired_direction_world, dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        self._desired_direction = (
            direction / direction_norm if direction_norm > 1.0e-12 else np.array((1.0, 0.0, 0.0))
        )
        self._target_radius_m = float(target_radius_m)
        sphere = pv.Sphere(
            radius=max(self._target_radius_m, 0.005),
            center=self._target,
            theta_resolution=30,
            phi_resolution=30,
        )
        self.plotter.add_mesh(
            sphere,
            name="targetTolerance",
            color="#22c55e",
            opacity=0.25,
            smooth_shading=True,
            pickable=False,
        )
        center = pv.Sphere(radius=0.018, center=self._target)
        self.plotter.add_mesh(
            center,
            name="targetCenter",
            color="#15803d",
            smooth_shading=True,
            pickable=False,
        )
        arrow_start = self._target - 0.22 * self._desired_direction
        arrow = pv.Arrow(
            start=arrow_start,
            direction=self._desired_direction,
            scale=0.22,
            tip_length=0.25,
            tip_radius=0.08,
            shaft_radius=0.025,
        )
        self.plotter.add_mesh(
            arrow,
            name="desiredStrikeDirection",
            color="#16a34a",
            smooth_shading=True,
            pickable=False,
        )
        if render:
            self.plotter.render()

    def update_state(
        self,
        cable_positions_m: np.ndarray,
        commanded_force_world_n: np.ndarray,
        root_trail_m: np.ndarray,
        tip_trail_m: np.ndarray,
        *, trail_phases=None,
    ) -> None:
        cable = np.asarray(cable_positions_m, dtype=np.float64)
        self._cable_mesh.points = cable
        self._cable_mesh.Modified()
        self._nodes_mesh.points = cable[1:-1]
        self._nodes_mesh.Modified()
        self._point_actor.SetPosition(*cable[0])
        self._drone_actor.SetPosition(*cable[0])
        self._tip_actor.SetPosition(*cable[-1])
        self._last_root = cable[0]
        for mesh, points, actor in (
            (self._root_trail_mesh, root_trail_m, self._root_trail_actor),
            (self._tip_trail_mesh, tip_trail_m, self._tip_trail_actor)):
            if trail_phases is None:
                mesh.cell_data.clear()
                actor.GetMapper().ScalarVisibilityOff()
                self._update_trail(mesh, points)
            else:
                self._update_phase_trail(mesh, points, trail_phases, actor)
        self._update_force_vector(np.asarray(commanded_force_world_n, dtype=np.float64))
        self.plotter.render()

    @staticmethod
    def _update_phase_trail(mesh, points, phases, actor):
        values = np.asarray(points, dtype=np.float64)
        phases = np.asarray(phases)
        if len(values) != len(phases) or len(values) == 0:
            raise ValueError('Each trail point requires its controller phase')
        if len(values) == 1:
            values = np.repeat(values, 2, axis=0)
            phases = np.repeat(phases, 2)
        mesh.cell_data.clear()
        mesh.verts = np.empty(0, dtype=np.int64)
        mesh.points = values
        indices = np.arange(len(values)-1)
        mesh.lines = np.column_stack((np.full(len(indices), 2), indices, indices+1)).ravel()
        # The destination sample records which controller produced that segment.
        colors = np.where((phases[1:] == 1)[:, None], [249, 115, 22],
            np.where((phases[1:]==3)[:,None],[13,148,136],[37, 99, 235])).astype(np.uint8)
        mesh.cell_data['controller_rgb'] = colors
        mesh.set_active_scalars('controller_rgb', preference='cell')
        mapper = actor.GetMapper()
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()
        mesh.Modified()

    @staticmethod
    def _update_trail(mesh, points: np.ndarray) -> None:
        values = np.asarray(points, dtype=np.float64)
        if len(values) < 2:
            values = np.vstack((values[0], values[0]))
        mesh.points = values
        mesh.lines = np.concatenate(([len(values)], np.arange(len(values)))).astype(
            np.int64
        )
        mesh.Modified()

    def _update_force_vector(self, force_world_n: np.ndarray) -> None:
        self._last_force = np.asarray(force_world_n, dtype=np.float64).copy()
        norm = float(np.linalg.norm(force_world_n))
        if not self._show_force or norm < 1.0e-10:
            self._force_actor.SetVisibility(False)
            return
        scale = min(0.46, 0.18 * norm)
        direction = force_world_n / norm
        source = np.array((1.0, 0.0, 0.0))
        cosine = float(np.clip(np.dot(source, direction), -1.0, 1.0))
        if cosine < -1.0 + 1.0e-10:
            rotation = np.diag((-1.0, 1.0, -1.0))
        else:
            cross = np.cross(source, direction)
            skew = np.array(
                (
                    (0.0, -cross[2], cross[1]),
                    (cross[2], 0.0, -cross[0]),
                    (-cross[1], cross[0], 0.0),
                )
            )
            denominator = max(1.0 + cosine, 1.0e-12)
            rotation = np.eye(3) + skew + (skew @ skew) / denominator
        self._force_mesh.points = (
            scale * (self._force_template_points @ rotation.T) + self._last_root
        )
        self._force_mesh.Modified()
        self._force_actor.SetVisibility(True)

    def set_show_force(self, enabled: bool) -> None:
        self._show_force = bool(enabled)
        if self._show_force:
            self._update_force_vector(self._last_force)
        else:
            self._force_actor.SetVisibility(False)
        self.plotter.render()

    def set_live_flight(self, enabled: bool) -> None:
        self._drone_actor.SetVisibility(enabled)
        self._point_actor.SetVisibility(not enabled)
        self.plotter.render()

    def set_show_trails(self, enabled: bool) -> None:
        self._show_trails = bool(enabled)
        for name in ("rootTrail", "tipTrail"):
            actor = self.plotter.renderer.actors.get(name)
            if actor is not None:
                actor.SetVisibility(self._show_trails)
        self.plotter.render()

    def set_camera_preset(self, name: str) -> None:
        focus = np.array((0.45, 0.0, 0.9))
        if name == "Side XZ":
            self.plotter.camera_position = [
                (focus[0], -3.1, focus[2]),
                tuple(focus),
                (0.0, 0.0, 1.0),
            ]
        elif name == "Front YZ":
            self.plotter.camera_position = [
                (3.4, focus[1], focus[2]),
                tuple(focus),
                (0.0, 0.0, 1.0),
            ]
        elif name == "Top XY":
            self.plotter.camera_position = [
                (focus[0], focus[1], 4.0),
                tuple(focus),
                (0.0, 1.0, 0.0),
            ]
        else:
            self.plotter.camera_position = [
                (2.8, -3.2, 2.45),
                tuple(focus),
                (0.0, 0.0, 1.0),
            ]
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def close(self) -> bool:
        self.plotter.close()
        return super().close()

    def set_tracked_pose(self,origin,rotation):
        """Display a tracking-frame glyph separately from the cable attachment."""
        matrix=np.eye(4);matrix[:3,:3]=np.asarray(rotation);matrix[:3,3]=np.asarray(origin)
        self._drone_actor.position=(0.,0.,0.);self._drone_actor.user_matrix=matrix
        self.plotter.render()


class MatplotlibPointCableViewer(QWidget):
    def set_controller_mode(self, force_control: bool) -> None:
        self._controller_color = '#f97316' if force_control else '#2563eb'

    """Offscreen-safe fallback used by tests and remote sessions."""

    backend_name = "MATPLOTLIB FALLBACK"

    def __init__(
        self,
        initial_cable_positions_m: np.ndarray,
        target_position_m: Sequence[float],
        desired_direction_world: Sequence[float],
        target_radius_m: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._show_force = True
        self._live_flight = False
        self._show_trails = True
        self._target = np.asarray(target_position_m, dtype=float)
        self._direction = np.asarray(desired_direction_world, dtype=float)
        self._radius = float(target_radius_m)
        self._cable = np.asarray(initial_cable_positions_m, dtype=float)
        self._force = np.zeros(3)
        self._root_trail = self._cable[[0]]
        self._tip_trail = self._cable[[-1]]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axis = self.figure.add_subplot(111, projection="3d")
        layout.addWidget(self.canvas)
        self._draw()

    def set_target(
        self,
        target_position_m: Sequence[float],
        desired_direction_world: Sequence[float],
        target_radius_m: float,
        *,
        render: bool = True,
    ) -> None:
        self._target = np.asarray(target_position_m, dtype=float)
        self._direction = np.asarray(desired_direction_world, dtype=float)
        self._radius = float(target_radius_m)
        if render:
            self._draw()

    def update_state(
        self,
        cable_positions_m: np.ndarray,
        commanded_force_world_n: np.ndarray,
        root_trail_m: np.ndarray,
        tip_trail_m: np.ndarray,
        *, trail_phases=None,
    ) -> None:
        self._cable = np.asarray(cable_positions_m, dtype=float)
        self._force = np.asarray(commanded_force_world_n, dtype=float)
        self._root_trail = np.asarray(root_trail_m, dtype=float)
        self._tip_trail = np.asarray(tip_trail_m, dtype=float)
        self._trail_phases = trail_phases
        self._draw()

    def _draw(self) -> None:
        axis = self.axis
        axis.clear()
        cable = self._cable
        axis.plot(*cable.T, color="#17a9c4", linewidth=3.2)
        axis.scatter(*cable[1:-1].T, color="#69d5e6", s=14)
        controller_color = getattr(self, '_controller_color', '#f59e0b')
        axis.scatter(*cable[0], color=controller_color, marker="D", s=85, label="drone" if self._live_flight else "point")
        if self._live_flight:
            for x, y in ((-.1, -.1), (-.1, .1), (.1, -.1), (.1, .1)):
                rotor = cable[0] + np.array([x, y, 0.])
                axis.plot(*np.vstack((cable[0], rotor)).T, color=controller_color, linewidth=3)
                axis.scatter(*rotor, color=controller_color, s=45)
        axis.scatter(*cable[-1], color="#f97316", s=70, label="tip")
        axis.scatter(*self._target, color="#22c55e", s=120, alpha=0.5, label="target")
        direction_norm = max(float(np.linalg.norm(self._direction)), 1.0e-12)
        direction = self._direction / direction_norm
        axis.quiver(*(self._target - 0.22 * direction), *(0.22 * direction), color="#16a34a")
        if self._show_force and np.linalg.norm(self._force) > 1.0e-10:
            force = self._force / np.linalg.norm(self._force) * min(
                0.46, 0.18 * np.linalg.norm(self._force)
            )
            axis.quiver(*cable[0], *force, color=getattr(self, '_controller_color', '#dc2626'), linewidth=2)
        if self._show_trails:
            phases = getattr(self, '_trail_phases', None)
            if phases is None:
                axis.plot(*self._root_trail.T, color="#f59e0b", alpha=0.7)
                axis.plot(*self._tip_trail.T, color="#f97316", alpha=0.75)
            else:
                from mpl_toolkits.mplot3d.art3d import Line3DCollection
                colors = ['#f97316' if p == 1 else '#2563eb' for p in phases[1:]]
                for trail in (self._root_trail, self._tip_trail):
                    if len(trail) > 1:
                        segments = np.stack((trail[:-1], trail[1:]), axis=1)
                        axis.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=2, alpha=.8))
        axis.set_xlim(-0.7, 1.7)
        axis.set_ylim(-1.2, 1.2)
        axis.set_zlim(0.0, 2.3)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=19, azim=-62)
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_zlabel("Z [m]")
        axis.legend(loc="lower left", fontsize=8)
        axis.grid(True, alpha=0.25)
        self.canvas.draw_idle()

    def set_show_force(self, enabled: bool) -> None:
        self._show_force = bool(enabled)
        self._draw()

    def set_live_flight(self, enabled: bool) -> None:
        self._live_flight = enabled
        self._draw()

    def set_show_trails(self, enabled: bool) -> None:
        self._show_trails = bool(enabled)
        self._draw()

    def set_camera_preset(self, name: str) -> None:
        if name == "Top XY":
            self.axis.view_init(elev=90, azim=-90)
        elif name == "Front YZ":
            self.axis.view_init(elev=0, azim=0)
        elif name == "Side XZ":
            self.axis.view_init(elev=0, azim=-90)
        else:
            self.axis.view_init(elev=19, azim=-62)
        self.canvas.draw_idle()


def create_viewer(
    initial_cable_positions_m: np.ndarray,
    target_position_m: Sequence[float],
    desired_direction_world: Sequence[float],
    target_radius_m: float,
    parent: QWidget | None = None,
) -> PointCableViewer3D | MatplotlibPointCableViewer:
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        return MatplotlibPointCableViewer(
            initial_cable_positions_m,
            target_position_m,
            desired_direction_world,
            target_radius_m,
            parent,
        )
    return PointCableViewer3D(
        initial_cable_positions_m,
        target_position_m,
        desired_direction_world,
        target_radius_m,
        parent,
    )
