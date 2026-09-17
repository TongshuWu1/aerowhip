"""Measured solid geometry and a translucent saved-prediction ghost."""
import numpy as np
from .viewer_3d import PointCableViewer3D


def finite_segments(points, time=None):
    points = np.asarray(points, dtype=float)
    finite = np.isfinite(points).all(axis=1)
    connected = finite[:-1] & finite[1:]
    if time is not None and len(time) > 1:
        connected &= np.diff(time) <= 1.5*np.median(np.diff(time))
    ids = np.flatnonzero(connected)
    lines = np.column_stack([np.full(len(ids), 2), ids, ids+1]).ravel()
    return np.nan_to_num(points), lines


class AdaptationScene(PointCableViewer3D):
    def __init__(self, data, parent=None):
        points = np.concatenate([data['measured_cable'].reshape(-1,3), data['predicted_cable'].reshape(-1,3),
                                  data['measured_origin'], data['predicted_origin'], data['target'][None]])
        points = points[np.isfinite(points).all(axis=1)]
        self.comparison_bounds = np.column_stack([points.min(axis=0)-.2, points.max(axis=0)+.2]).ravel()
        self.comparison_bounds[4] = min(self.comparison_bounds[4], 0.)
        super().__init__(data['predicted_cable'][0], data['target'],
                         data['task']['desired_strike_direction_world'],
                         data['task'].get('success',{}).get('tip_target_distance_m',data['task'].get('target_marker_radius_m',.02)), parent)
        pv = self._pv
        # Keep the shared environment/camera, replace animated actors with paired geometry.
        for name in ('cable', 'cableNodes', 'controlledPoint', 'drone', 'cableTip',
                     'rootTrail', 'tipTrail', 'commandForce'):
            self.plotter.renderer.actors[name].SetVisibility(False)
        self.bodies = {}
        drone = self._drone_actor.mapper.dataset.copy()
        for key, color, opacity in (('measured', '#ea580c', 1.), ('predicted', '#2563eb', .28)):
            body = {}
            body['drone'] = self.plotter.add_mesh(drone.copy(), color=color, opacity=opacity, name=key+'Drone', render=False)
            for part, width in (('cable', 5), ('origin_trail', 2), ('tip_trail', 3)):
                mesh = pv.PolyData(np.zeros((2,3)), lines=np.array([2,0,1]))
                actor = self.plotter.add_mesh(mesh, color=color, opacity=opacity, line_width=width,
                                              name=key+part, render=False)
                body[part] = (mesh, actor)
            mesh = pv.PolyData(np.zeros((1,3)))
            body['markers'] = (mesh, self.plotter.add_mesh(mesh, color=color, opacity=opacity,
                point_size=8, render_points_as_spheres=True, name=key+'Markers', render=False))
            body['tip'] = self.plotter.add_mesh(pv.Sphere(radius=.023), color=color, opacity=opacity,
                                                name=key+'Tip', render=False)
            self.bodies[key] = body
        self.plotter.add_text('SOLID ORANGE  measured\nBLUE GHOST  predicted execution',
                              position='upper_left', font_size=10, color='#172033', name='comparisonLegend')
        self.set_scene_bounds(self.comparison_bounds)
        self.set_camera_preset('Perspective')

    def _build_environment(self):
        bounds = self.comparison_bounds
        center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, 0.]
        plane = self._pv.Plane(center=center, direction=(0,0,1), i_size=bounds[1]-bounds[0],
                               j_size=bounds[3]-bounds[2], i_resolution=18, j_resolution=18)
        self.plotter.add_mesh(plane, color='#f7f8fa', show_edges=True, edge_color='#d1dae3',
                              opacity=.65, name='comparisonFloor', pickable=False)
        self.plotter.add_axes(xlabel='X',ylabel='Y',zlabel='Z')
        self.plotter.show_grid(bounds=bounds, color='#7c8796', location='outer', font_size=9,
                               xtitle='X [m]',ytitle='Y [m]',ztitle='Z [m]')

    def set_camera_preset(self, name):
        b = self.comparison_bounds
        center = np.array([(b[0]+b[1])/2,(b[2]+b[3])/2,(b[4]+b[5])/2])
        direction = {'Side XZ':(0,-1,0),'Front YZ':(1,0,0),'Top XY':(0,0,1)}.get(name,(1,-2,1))
        up=(0,1,0) if name=='Top XY' else (0,0,1)
        self.plotter.camera_position=[center+np.array(direction),center,up]
        self.plotter.reset_camera(bounds=b,render=False)
        self.plotter.reset_camera_clipping_range(); self.render()

    def draw(self, data, index, *, measured=True, predicted=True, trails=True, opacity=.28):
        for key, enabled in (('measured', measured), ('predicted', predicted)):
            body = self.bodies[key]
            origin, rotation, cable = (data[key+'_'+part][index] for part in ('origin','rotation','cable'))
            alpha = 1. if key == 'measured' else opacity
            matrix = np.eye(4); matrix[:3,:3] = rotation; matrix[:3,3] = origin
            valid = np.isfinite(matrix).all()
            body['drone'].SetVisibility(bool(enabled and valid))
            if valid:
                body['drone'].position = (0.,0.,0.)
                body['drone'].user_matrix = matrix
            body['drone'].prop.opacity = alpha
            for part, points, times in (
                ('cable', cable, None),
                ('origin_trail', data[key+'_origin'][:index+1], data['time'][:index+1]),
                ('tip_trail', data[key+'_cable'][:index+1,-1], data['time'][:index+1])):
                mesh, actor = body[part]
                coords, lines = finite_segments(points, times)
                mesh.points = coords; mesh.lines = lines
                actor.SetVisibility(enabled and len(lines)>0 and (part == 'cable' or trails))
                actor.prop.opacity = alpha
            mesh, actor = body['markers']
            sites = cable[data['predicted_marker_indices']] if key == 'predicted' else cable[1:]
            pts = sites[np.isfinite(sites).all(axis=1)]
            # Replace topology too: marker visibility changes when observations are missing.
            mesh.copy_from(self._pv.PolyData(pts if len(pts) else np.zeros((1,3))))
            actor.SetVisibility(enabled and len(pts)>0); actor.prop.opacity = alpha
            body['tip'].SetVisibility(bool(enabled and np.isfinite(cable[-1]).all()))
            body['tip'].prop.opacity = alpha
            if np.isfinite(cable[-1]).all(): body['tip'].position = cable[-1]
        self.render()
