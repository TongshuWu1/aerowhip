"""Retained PyVista/VTK viewer for the aerial-cable simulator."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QVBoxLayout, QWidget
import torch
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPolyLine
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkGlyph3DMapper, vtkPolyDataMapper
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ..uav.quaternion import quaternion_to_rotation_matrix_xyzw


def pose_transform_matrix_xyzw(
    position_m: np.ndarray | Sequence[float],
    orientation_xyzw: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return one body-to-world homogeneous transform.

    The rotation uses the same quaternion function as the physical rigid
    attachment. There is no graphics-only Euler-angle propagation path.
    """

    position = np.asarray(position_m, dtype=np.float64).reshape(3)
    quaternion = torch.as_tensor(
        np.array(orientation_xyzw, dtype=np.float64, copy=True).reshape(1, 4),
        dtype=torch.float64,
    )
    rotation = (
        quaternion_to_rotation_matrix_xyzw(quaternion)[0]
        .detach()
        .cpu()
        .numpy()
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def _vtk_matrix4(values: np.ndarray) -> vtkMatrix4x4:
    matrix = vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            matrix.SetElement(row, column, float(values[row, column]))
    return matrix


def _cylinder_between(
    start: Sequence[float],
    end: Sequence[float],
    radius: float,
) -> pv.PolyData:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    displacement = end_array - start_array
    length = float(np.linalg.norm(displacement))
    return pv.Cylinder(
        center=0.5 * (start_array + end_array),
        direction=displacement / length,
        radius=radius,
        height=length,
        resolution=18,
        capping=True,
    )


def _crazyflie_mesh() -> tuple[pv.PolyData, pv.PolyData]:
    """Build a clean, asymmetric Crazyflie-like body in body coordinates."""

    components: list[pv.PolyData] = [
        pv.Cube(
            center=(0.0, 0.0, 0.0),
            x_length=0.055,
            y_length=0.045,
            z_length=0.018,
        )
    ]
    motor_locations = (
        (0.066, 0.066, 0.0),
        (0.066, -0.066, 0.0),
        (-0.066, 0.066, 0.0),
        (-0.066, -0.066, 0.0),
    )
    for motor in motor_locations:
        components.append(_cylinder_between((0.0, 0.0, 0.0), motor, 0.0045))
        components.append(
            pv.Cylinder(
                center=motor,
                direction=(0.0, 0.0, 1.0),
                radius=0.026,
                height=0.0025,
                resolution=30,
                capping=True,
            )
        )
    body = components[0]
    for component in components[1:]:
        body = body.merge(component, merge_points=False)
    nose = pv.Cone(
        center=(0.041, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        height=0.032,
        radius=0.012,
        resolution=24,
        capping=True,
    )
    return body.clean(), nose


class CableViewer3D(QWidget):
    """Persistent scientific 3D scene driven only by immutable snapshots."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        marker_node_indices: Sequence[int],
        node_count: int,
        cable_length_m: float,
        cable_diameter_m: float,
        initial_uav_position_m: Sequence[float],
        visual_cable_radius_scale: float = 2.0,
    ) -> None:
        super().__init__(parent)
        self.marker_node_indices = tuple(int(value) for value in marker_node_indices)
        self.node_count = int(node_count)
        self.cable_length_m = float(cable_length_m)
        self.initial_uav = np.asarray(initial_uav_position_m, dtype=np.float64)
        self.visual_cable_radius_scale = float(visual_cable_radius_scale)
        self._show_commanded = False
        self._show_body_frame = False
        self._follow_uav = False
        self._last_uav_position: np.ndarray | None = None
        self.last_update_ms = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Rendering is driven only by the GUI's latest-snapshot timer.  The
        # PyVistaQt default adds an independent 5-Hz redraw timer, which causes
        # duplicate GPU work and avoidable contention with CUDA physics.
        self.plotter = QtInteractor(self, auto_update=False)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#edf1f5")
        self.plotter.disable_parallel_projection()

        self._build_environment()
        self._build_uav_actors()
        self._build_cable_pipeline(
            physical_radius_m=0.5 * float(cable_diameter_m)
        )
        self._build_marker_pipeline()
        self._build_attachment_actor()
        self._configure_lighting()
        self.set_camera_preset("Perspective")

    def _build_environment(self) -> None:
        ground = pv.Plane(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            i_size=3.2,
            j_size=3.2,
            i_resolution=24,
            j_resolution=24,
        )
        self.plotter.add_mesh(
            ground,
            color="#f7f8fa",
            opacity=0.72,
            show_edges=True,
            edge_color="#cbd2da",
            line_width=1.0,
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

    def _build_uav_actors(self) -> None:
        body, nose = _crazyflie_mesh()
        self._simulated_uav_actors = [
            self.plotter.add_mesh(
                body,
                color="#263648",
                metallic=0.18,
                roughness=0.55,
                smooth_shading=True,
                pickable=False,
            ),
            self.plotter.add_mesh(
                nose,
                color="#f08c32",
                metallic=0.05,
                roughness=0.55,
                smooth_shading=True,
                pickable=False,
            ),
        ]
        self._commanded_uav_actors = [
            self.plotter.add_mesh(
                body,
                color="#f1a33c",
                opacity=0.20,
                smooth_shading=True,
                pickable=False,
            ),
            self.plotter.add_mesh(
                nose,
                color="#ffd166",
                opacity=0.30,
                smooth_shading=True,
                pickable=False,
            ),
        ]
        for actor in self._commanded_uav_actors:
            actor.SetVisibility(False)

        colors = ("#d94841", "#2f9e44", "#2878c7")
        self._body_frame_actors = []
        for direction, color in zip(np.eye(3), colors, strict=True):
            arrow = pv.Arrow(
                start=(0.0, 0.0, 0.0),
                direction=direction,
                scale=0.105,
                tip_length=0.22,
                tip_radius=0.08,
                shaft_radius=0.025,
            )
            actor = self.plotter.add_mesh(
                arrow, color=color, smooth_shading=True, pickable=False
            )
            actor.SetVisibility(False)
            self._body_frame_actors.append(actor)

        # Actor slots intentionally separate commanded, simulated, and future
        # measured pose semantics. No real-data actor is populated yet.
        self.pose_actor_groups: dict[str, list[vtkActor] | None] = {
            "commanded": self._commanded_uav_actors,
            "simulated": self._simulated_uav_actors,
            "measured": None,
        }

    def _build_cable_pipeline(self, *, physical_radius_m: float) -> None:
        points = np.zeros((self.node_count, 3), dtype=np.float64)
        self._cable_points = vtkPoints()
        self._cable_points.SetData(numpy_to_vtk(points, deep=True))
        polyline = vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(self.node_count)
        for index in range(self.node_count):
            polyline.GetPointIds().SetId(index, index)
        cells = vtkCellArray()
        cells.InsertNextCell(polyline)
        self._cable_polydata = vtkPolyData()
        self._cable_polydata.SetPoints(self._cable_points)
        self._cable_polydata.SetLines(cells)
        self._cable_tube = vtkTubeFilter()
        self._cable_tube.SetInputData(self._cable_polydata)
        self._cable_tube.SetRadius(
            physical_radius_m * self.visual_cable_radius_scale
        )
        self._cable_tube.SetNumberOfSides(14)
        self._cable_tube.CappingOn()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(self._cable_tube.GetOutputPort())
        self._cable_actor = vtkActor()
        self._cable_actor.SetMapper(mapper)
        self._cable_actor.GetProperty().SetColor(0.10, 0.62, 0.76)
        self._cable_actor.GetProperty().SetInterpolationToPhong()
        self.plotter.renderer.AddActor(self._cable_actor)

    def _glyph_actor(
        self,
        count: int,
        *,
        radius_m: float,
        color: tuple[float, float, float],
    ) -> tuple[vtkPoints, vtkActor]:
        points = vtkPoints()
        points.SetData(numpy_to_vtk(np.zeros((count, 3)), deep=True))
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        sphere = vtkSphereSource()
        sphere.SetRadius(radius_m)
        sphere.SetThetaResolution(18)
        sphere.SetPhiResolution(14)
        mapper = vtkGlyph3DMapper()
        mapper.SetInputData(polydata)
        mapper.SetSourceConnection(sphere.GetOutputPort())
        mapper.ScalingOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetInterpolationToPhong()
        self.plotter.renderer.AddActor(actor)
        return points, actor

    def _build_marker_pipeline(self) -> None:
        self._ordinary_marker_indices = tuple(
            value
            for value in self.marker_node_indices
            if value != self.node_count - 1
        )
        self._marker_points, self._marker_actor = self._glyph_actor(
            len(self._ordinary_marker_indices),
            radius_m=0.010,
            color=(0.96, 0.54, 0.16),
        )
        self._tip_points, self._tip_actor = self._glyph_actor(
            1,
            radius_m=0.014,
            color=(0.84, 0.19, 0.18),
        )

    def _build_attachment_actor(self) -> None:
        self._attachment_actor = self.plotter.add_mesh(
            pv.Sphere(radius=0.012, theta_resolution=20, phi_resolution=16),
            color="#6f42c1",
            smooth_shading=True,
            pickable=False,
        )

    def _configure_lighting(self) -> None:
        self.plotter.remove_all_lights()
        key = pv.Light(
            position=(2.2, -2.4, 3.2),
            focal_point=(0.0, 0.0, 0.7),
            color="#ffffff",
            intensity=0.85,
        )
        fill = pv.Light(
            position=(-2.0, 1.6, 2.0),
            focal_point=(0.0, 0.0, 0.7),
            color="#dce8ff",
            intensity=0.42,
        )
        self.plotter.add_light(key)
        self.plotter.add_light(fill)

    @staticmethod
    def _apply_transform(
        actors: Sequence[vtkActor], transform: np.ndarray
    ) -> None:
        for actor in actors:
            actor.SetUserMatrix(_vtk_matrix4(transform))

    @staticmethod
    def _update_vtk_points(points: vtkPoints, values: np.ndarray) -> None:
        target = vtk_to_numpy(points.GetData())
        target[...] = np.asarray(values, dtype=target.dtype)
        points.GetData().Modified()
        points.Modified()

    @staticmethod
    def _replace_vtk_points(points: vtkPoints, values: np.ndarray) -> None:
        """Replace a dynamic measured-marker set without inventing hidden nodes."""

        points.SetData(numpy_to_vtk(np.asarray(values, dtype=np.float64), deep=True))
        points.GetData().Modified()
        points.Modified()

    def set_show_commanded_pose(self, enabled: bool) -> None:
        self._show_commanded = bool(enabled)
        for actor in self._commanded_uav_actors:
            actor.SetVisibility(self._show_commanded)
        self.plotter.render()

    def set_show_body_frame(self, enabled: bool) -> None:
        self._show_body_frame = bool(enabled)
        for actor in self._body_frame_actors:
            actor.SetVisibility(self._show_body_frame)
        self.plotter.render()

    def set_follow_uav(self, enabled: bool) -> None:
        self._follow_uav = bool(enabled)
        if enabled and self._last_uav_position is not None:
            self._follow_camera(self._last_uav_position, force=True)

    def set_camera_preset(self, name: str) -> None:
        if name == "Follow UAV":
            self._follow_uav = True
            if self._last_uav_position is not None:
                self._follow_camera(self._last_uav_position, force=True)
            self.plotter.render()
            return
        self._follow_uav = False
        center = self.initial_uav + np.array(
            (0.0, 0.0, -0.42 * self.cable_length_m)
        )
        distance = max(2.1 * self.cable_length_m, 1.6)
        if name == "Front":
            position = center + np.array((0.0, -distance, 0.16 * distance))
            view_up = (0.0, 0.0, 1.0)
        elif name == "Side":
            position = center + np.array((distance, 0.0, 0.16 * distance))
            view_up = (0.0, 0.0, 1.0)
        elif name == "Top":
            position = center + np.array((0.0, 0.0, distance))
            view_up = (0.0, 1.0, 0.0)
        else:
            position = center + distance * np.array((0.82, -1.05, 0.55))
            view_up = (0.0, 0.0, 1.0)
        self.plotter.camera_position = [position, center, view_up]
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def _follow_camera(self, uav_position: np.ndarray, *, force: bool = False) -> None:
        focal = np.asarray(self.plotter.camera.focal_point)
        position = np.asarray(self.plotter.camera.position)
        target = np.asarray(uav_position, dtype=np.float64)
        if force or np.linalg.norm(target - focal) > 0.0:
            offset = position - focal
            self.plotter.camera.focal_point = target
            self.plotter.camera.position = target + offset
            self.plotter.reset_camera_clipping_range()

    def update_state(
        self,
        cable_positions_m: np.ndarray,
        *,
        uav_position_m: np.ndarray,
        uav_orientation_xyzw: np.ndarray,
        attachment_position_m: np.ndarray,
        commanded_uav_position_m: np.ndarray | None,
        commanded_uav_orientation_xyzw: np.ndarray | None,
    ) -> float:
        """Update persistent actor transforms/coordinates and return milliseconds."""

        started = time.perf_counter()
        positions = np.asarray(cable_positions_m, dtype=np.float64)
        if positions.shape != (self.node_count, 3):
            raise ValueError(
                f"Cable display positions must have shape ({self.node_count}, 3)."
            )
        uav_position = np.asarray(uav_position_m, dtype=np.float64).reshape(3)
        uav_orientation = np.asarray(
            uav_orientation_xyzw, dtype=np.float64
        ).reshape(4)
        attachment_position = np.asarray(
            attachment_position_m, dtype=np.float64
        ).reshape(3)

        self._update_vtk_points(self._cable_points, positions)
        self._cable_polydata.Modified()
        self._cable_tube.Modified()
        ordinary_positions = positions[
            np.asarray(self._ordinary_marker_indices, dtype=np.int64)
        ]
        self._update_vtk_points(self._marker_points, ordinary_positions)
        self._update_vtk_points(self._tip_points, positions[-1:])
        self._attachment_actor.SetPosition(*attachment_position)

        simulated_transform = pose_transform_matrix_xyzw(
            uav_position, uav_orientation
        )
        self._apply_transform(self._simulated_uav_actors, simulated_transform)
        self._apply_transform(self._body_frame_actors, simulated_transform)

        has_commanded_pose = (
            commanded_uav_position_m is not None
            and commanded_uav_orientation_xyzw is not None
        )
        if has_commanded_pose:
            command_transform = pose_transform_matrix_xyzw(
                np.asarray(commanded_uav_position_m),
                np.asarray(commanded_uav_orientation_xyzw),
            )
            self._apply_transform(self._commanded_uav_actors, command_transform)
        for actor in self._commanded_uav_actors:
            actor.SetVisibility(self._show_commanded and has_commanded_pose)

        self._last_uav_position = uav_position.copy()
        if self._follow_uav:
            self._follow_camera(uav_position)
        self.plotter.render()
        self.last_update_ms = 1000.0 * (time.perf_counter() - started)
        return self.last_update_ms

    def update_measured_sites(
        self,
        measured_sites_m: np.ndarray,
        measured_site_valid: np.ndarray,
        *,
        uav_position_m: np.ndarray,
        uav_orientation_xyzw: np.ndarray,
        attachment_position_m: np.ndarray,
        commanded_uav_position_m: np.ndarray | None,
        commanded_uav_orientation_xyzw: np.ndarray | None,
    ) -> float:
        """Render only measured root/cable sites for processed-take playback.

        Lines join valid adjacent measured sites. No latent DDER node is
        reconstructed for this visualization path.
        """

        started = time.perf_counter()
        positions = np.asarray(measured_sites_m, dtype=np.float64)
        valid = np.asarray(measured_site_valid, dtype=bool)
        if positions.shape != (self.node_count, 3) or valid.shape != (self.node_count,):
            raise ValueError("Measured playback sites/validity have the wrong shape.")
        safe_positions = positions.copy()
        safe_positions[~valid] = 0.0
        self._update_vtk_points(self._cable_points, safe_positions)
        cells = vtkCellArray()
        start = 0
        while start < self.node_count:
            while start < self.node_count and not valid[start]:
                start += 1
            stop = start
            while stop < self.node_count and valid[stop]:
                stop += 1
            if stop - start >= 2:
                line = vtkPolyLine()
                line.GetPointIds().SetNumberOfIds(stop - start)
                for local, index in enumerate(range(start, stop)):
                    line.GetPointIds().SetId(local, index)
                cells.InsertNextCell(line)
            start = stop + 1
        self._cable_polydata.SetLines(cells)
        self._cable_polydata.Modified()
        self._cable_tube.Modified()

        ordinary = [index for index in self._ordinary_marker_indices if valid[index]]
        self._replace_vtk_points(
            self._marker_points,
            positions[np.asarray(ordinary, dtype=np.int64)] if ordinary else np.empty((0, 3)),
        )
        tip_visible = bool(valid[-1])
        if tip_visible:
            self._replace_vtk_points(self._tip_points, positions[-1:])
        self._tip_actor.SetVisibility(tip_visible)
        self._attachment_actor.SetPosition(*np.asarray(attachment_position_m).reshape(3))

        uav_position = np.asarray(uav_position_m, dtype=np.float64).reshape(3)
        uav_orientation = np.asarray(uav_orientation_xyzw, dtype=np.float64).reshape(4)
        simulated_transform = pose_transform_matrix_xyzw(uav_position, uav_orientation)
        self._apply_transform(self._simulated_uav_actors, simulated_transform)
        self._apply_transform(self._body_frame_actors, simulated_transform)
        has_command = commanded_uav_position_m is not None and commanded_uav_orientation_xyzw is not None
        if has_command:
            self._apply_transform(
                self._commanded_uav_actors,
                pose_transform_matrix_xyzw(commanded_uav_position_m, commanded_uav_orientation_xyzw),
            )
        for actor in self._commanded_uav_actors:
            actor.SetVisibility(self._show_commanded and has_command)
        self._last_uav_position = uav_position.copy()
        if self._follow_uav:
            self._follow_camera(uav_position)
        self.plotter.render()
        self.last_update_ms = 1000.0 * (time.perf_counter() - started)
        return self.last_update_ms

    def close(self) -> None:
        self.plotter.close()
