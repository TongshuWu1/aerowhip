"""Animate saved actual candidate states without running physics on the UI thread."""
from pathlib import Path
import time
import numpy as np
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QComboBox,QCheckBox,QPushButton,QSlider
from simulator.workflow import read_json
from .research_widgets import note


def load_snapshot(path):
    with np.load(path,allow_pickle=False) as source:data={key:source[key].copy() for key in source.files}
    if str(data['schema'])!='mppi_live_v1':raise ValueError('Unsupported live snapshot')
    q=data['cable_positions_m'];p=data['origin_positions_m'];r=data['origin_rotations'];t=data['time_s']
    if q.ndim!=4 or q.shape[-1]!=3 or p.shape!=q.shape[:2]+(3,) or r.shape!=q.shape[:2]+(3,3) or t.shape!=(q.shape[1],):raise ValueError('Invalid live geometry dimensions')
    if not all(np.isfinite(x).all() for x in (q,p,r,t,data['target_position_m'])) or np.any(np.diff(t)<=0):raise ValueError('Invalid live snapshot geometry/time')
    return data


class MPPILiveView(QWidget):
    def __init__(self):
        super().__init__();self.job=None;self.data=None;self.viewer=None;self.active=False;self.playing=True;self.fingerprint=None;self.clock=0.
        body=QVBoxLayout(self);self.status=note('Live 3D starts with the first completed candidate batch of a new run.');body.addWidget(self.status)
        row=QHBoxLayout();body.addLayout(row);self.candidate=QComboBox();row.addWidget(self.candidate,1);self.candidate.currentIndexChanged.connect(self.reset_play)
        self.follow=QCheckBox('Follow newest iteration');self.follow.setChecked(True);row.addWidget(self.follow)
        self.play=QPushButton('Pause');self.play.clicked.connect(self.toggle_play);row.addWidget(self.play)
        self.speed=QComboBox()
        for value in (.25,.5,1.,2.):self.speed.addItem(f'{value:g}×',value)
        self.speed.setCurrentIndex(2);row.addWidget(self.speed)
        self.camera=QComboBox();self.camera.addItems(['Perspective','Side XZ','Top XY','Front YZ']);self.camera.currentTextChanged.connect(self.set_camera);row.addWidget(self.camera)
        fit=QPushButton('Fit camera');fit.clicked.connect(self.fit_camera);row.addWidget(fit)
        self.paths=QCheckBox('Other sampled tip paths');self.paths.setChecked(True);self.paths.toggled.connect(self.draw_paths);row.addWidget(self.paths)
        self.host=QVBoxLayout();body.addLayout(self.host,1);self.empty=note('The current older run has no sampled-trajectory snapshots. Keep it running; new runs can enable Live 3D in setup.');self.host.addWidget(self.empty)
        self.slider=QSlider(Qt.Orientation.Horizontal);self.slider.setRange(0,0);self.slider.valueChanged.connect(self.seek);body.addWidget(self.slider)
        self.frame_note=note('');body.addWidget(self.frame_note)
        body.addWidget(note('Sampled simulation in the fitted model. Selected drone/cable animation; faint blue paths: other displayed samples; green path: committed motion. Proposals are not flight commands.'))
        self.timer=QTimer(self);self.timer.setInterval(33);self.timer.timeout.connect(self.tick)

    def set_job(self,job):
        job=Path(job) if job else None
        if self.job==job:return
        self.job=job;self.data=None;self.fingerprint=None;self.candidate.clear();self.slider.setRange(0,0);self.clear_viewer();self.empty.show();self.frame_note.setText('')
        self.status.setText('Waiting for a live candidate snapshot.');self.poll()

    def clear_viewer(self):
        if self.viewer is not None:self.host.removeWidget(self.viewer);self.viewer.close();self.viewer.deleteLater();self.viewer=None

    def poll(self):
        if self.job is None or not self.active:return
        path=self.job/'live.npz'
        if not path.exists():
            self.status.setText('No live snapshot yet. Older runs do not record candidate geometry; enable Live 3D for a new run.');return
        if not self.follow.isChecked() and self.data is not None:return
        key=(path.stat().st_mtime_ns,path.stat().st_size)
        if key==self.fingerprint:return
        try:data=load_snapshot(path)
        except (OSError,ValueError,KeyError) as exc:self.status.setText('Waiting for a readable snapshot: '+str(exc));return
        self.fingerprint=key;self.data=data;self.candidate.blockSignals(True);selected=max(0,self.candidate.currentIndex());self.candidate.clear()
        for i,label in enumerate(data['labels']):
            state='infeasible' if data['failed'][i] else 'modeled hit' if data['success'][i] else 'no modeled hit'
            self.candidate.addItem(f'{label} · {state} · score {data["scores"][i]:.2f}')
        self.candidate.setCurrentIndex(min(selected,self.candidate.count()-1));self.candidate.blockSignals(False)
        self.ensure_viewer();self.reset_play();self.draw_paths()

    def ensure_viewer(self):
        if self.viewer is not None or self.data is None:return
        from .viewer_3d import create_viewer
        cfg=read_json(self.job/'settings.json',{});task=cfg.get('task',{})
        self.viewer=create_viewer(self.data['cable_positions_m'][0,0],self.data['target_position_m'],task.get('strike_direction',[1,0,0]),task.get('target_radius_m',.05),self)
        if 'target_positions_m' in self.data:
            from .whip_targets import add_targets
            add_targets(self.viewer,self.data['target_positions_m'],task.get('target_radius_m',.05))
        self.viewer.set_live_flight(True);self.host.addWidget(self.viewer);self.empty.hide();self.fit_camera()

    def fit_camera(self):
        if self.viewer is None or self.data is None:return
        points=np.concatenate([self.data['cable_positions_m'].reshape(-1,3),self.data['target_position_m'][None]])
        if 'target_positions_m' in self.data:points=np.vstack((points,self.data['target_positions_m']))
        bounds=np.column_stack([points.min(0)-.3,points.max(0)+.3]).ravel();bounds[4]=min(bounds[4],0.)
        self.viewer.set_scene_bounds(bounds);self.viewer.set_camera_preset(self.camera.currentText())

    def set_camera(self,name):
        if self.viewer is not None:self.viewer.set_camera_preset(name)

    def draw_paths(self,*_):
        if self.viewer is None or self.data is None or not hasattr(self.viewer,'plotter'):return
        import pyvista as pv
        for i in range(getattr(self,'path_count',0)):self.viewer.plotter.remove_actor(f'liveSample{i}',render=False)
        self.path_count=len(self.data['labels'])
        for i,points in enumerate(self.data['cable_positions_m'][:,:,-1]):
            name=f'liveSample{i}';self.viewer.plotter.remove_actor(name,render=False)
            if self.paths.isChecked() and i!=self.candidate.currentIndex():self.viewer.plotter.add_mesh(pv.lines_from_points(points),color='#2563eb',line_width=1,opacity=.25,name=name,render=False,reset_camera=False)
        for key in ('committed_origin_m','committed_tip_m'):
            points=self.data[key]
            if len(points)>1:self.viewer.plotter.add_mesh(pv.lines_from_points(points),color='#16a34a',line_width=3,name=key,render=False,reset_camera=False)
        self.viewer.render()

    def reset_play(self,*_):
        if self.data is None:return
        i=max(0,self.candidate.currentIndex());self.clock=0.;self.last_tick=time.perf_counter()
        self.slider.setRange(0,int(self.data['frame_counts'][i])-1);self.slider.setValue(0);self.draw(0);self.draw_paths()

    def seek(self,index):
        if self.data is None:return
        self.clock=float(self.data['time_s'][index]-self.data['time_s'][0]);self.last_tick=time.perf_counter();self.draw(index)

    def draw(self,index):
        if self.data is None or self.viewer is None:return
        i=max(0,self.candidate.currentIndex());d=self.data;q=d['cable_positions_m'][i]
        self.viewer.update_state(q[index],np.zeros(3),q[:index+1,0],q[:index+1,-1],tracked_origin_m=d['origin_positions_m'][i,index],tracked_rotation=d['origin_rotations'][i,index],render=True)
        self.status.setText(f'Batch iteration {int(d["iteration"])} · {int(d["command_step"])} committed commands · selected proposal from iteration {int(d["series_iterations"][i])} · snapshot age {max(0,time.time()-float(d["created_unix_s"])):.0f} s')
        self.frame_note.setText(f'Simulated time {d["time_s"][index]:.3f} s · {self.clock:.2f} s into lookahead · candidate {int(d["candidate_ids"][i])} · trajectory end {d["termination_time_s"][i]:.3f} s')
        if 'target_hit_times_s' in d:
            from .whip_targets import progress_text
            self.frame_note.setText(self.frame_note.text()+' · '+progress_text(float(d['time_s'][index]),d['target_hit_times_s'][i]))

    def toggle_play(self):self.playing=not self.playing;self.play.setText('Pause' if self.playing else 'Play');self.last_tick=time.perf_counter()

    def tick(self):
        if not self.active:return
        now=time.perf_counter()
        if now-getattr(self,'last_poll',0)>.75:self.last_poll=now;self.poll()
        if self.data is None or not self.playing:return
        now=time.perf_counter();self.clock+=max(0,now-self.last_tick)*self.speed.currentData();self.last_tick=now
        span=self.data['time_s'][self.slider.maximum()]-self.data['time_s'][0]
        if span<=0:return
        self.clock%=span
        index=min(self.slider.maximum(),max(0,int(np.searchsorted(self.data['time_s'],self.data['time_s'][0]+self.clock,side='right')-1)))
        self.slider.blockSignals(True);self.slider.setValue(index);self.slider.blockSignals(False);self.draw(index)

    def set_active(self,active):
        self.active=active
        if active:self.last_tick=time.perf_counter();self.timer.start();self.poll()
        else:self.timer.stop()

    def shutdown(self):self.timer.stop();self.clear_viewer();return True
