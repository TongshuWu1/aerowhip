"""Display a two-target whip with both fixed centers visible."""
import numpy as np


def add_targets(viewer,targets,radius):
    import pyvista as pv
    targets=np.asarray(targets)
    viewer.plotter.add_mesh(pv.Sphere(radius=radius,center=targets[1]),color='#a855f7',opacity=.45,
        name='SecondWhipTarget',render=False,reset_camera=False)
    viewer.plotter.add_point_labels(targets,['T1','T2'],font_size=13,point_size=0,always_visible=True,
        shape_opacity=.65,name='WhipTargetLabels',render=False,reset_camera=False)


def progress_text(time,hit_times):
    return ' → '.join(f'T{i+1} '+('hit' if np.isfinite(t) and time>=t else 'pending') for i,t in enumerate(hit_times))
