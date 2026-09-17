"""Native measurement playback. No model, resampling, fitting or normalization."""
from pathlib import Path
import time
import numpy as np
from PySide6.QtCore import Qt,QTimer,QThread,Signal,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QComboBox,QPushButton,QSlider,QCheckBox,QFileDialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
from simulator.workflow import read_json
from simulator.geometry import normalized_rotations_xyzw
from experimental_data.adaptation_rounds import read_optitrack
from experimental_data.paths import flight_batch_roots
from .research_widgets import note


def discover_takes(root):
    root=Path(root);found={}
    for folder in sorted((root/'data/raw_takes').glob('*')):
        meta=read_json(folder/'experiment.json',{});name=meta.get('files',{}).get('optitrack')
        paths=[folder/name] if name else [p for p in folder.glob('*.csv') if not p.name.startswith('experiment_')]
        for path in paths:
            if path.is_file():found[str(path.resolve())]=dict(label=folder.name,role=meta.get('fit_role','unassigned'),drone=meta.get('drone_label'))
    # Include failed/excluded raw flights too; display never grants fit eligibility.
    for folder in flight_batch_roots(root):
        for path in sorted(folder.glob('**/flight_take/*.csv')):
            if not path.name.startswith('experiment_'):found[str(path.resolve())]=dict(label=path.parent.parent.name+' / '+path.stem,role='flight recording')
    return found


def load_take(path,drone=None):
    data=read_optitrack(path,drone_label=drone);rotation,valid=normalized_rotations_xyzw(data['quaternion'])
    data['rotation']=rotation;data['pose_valid']=valid&np.isfinite(data['drone']).all(1)
    data['elapsed']=data['time']-data['time'][0]
    data['marker_valid']=np.isfinite(data['cable']).all(-1)
    return data


class TakeLoader(QThread):
    loaded=Signal(object);failed=Signal(str)
    def __init__(self,path,drone):super().__init__();self.path=path;self.drone=drone
    def run(self):
        try:self.loaded.emit(load_take(self.path,self.drone))
        except Exception as exc:self.failed.emit(str(exc))


class RecordedTakesReplay(QWidget):
    def __init__(self,root):
        super().__init__();self.root=Path(root);self.data=None;self.worker=None;self.active=False;self.playing=False;self.opened={}
        body=QVBoxLayout(self);row=QHBoxLayout();body.addLayout(row)
        self.takes=QComboBox();row.addWidget(self.takes,1)
        self.refresh_button=QPushButton('Refresh takes');self.refresh_button.clicked.connect(self.refresh);row.addWidget(self.refresh_button)
        self.open_button=QPushButton('Open OptiTrack CSV…');self.open_button.clicked.connect(self.open_csv);row.addWidget(self.open_button)
        files=QPushButton('Open source folder');files.clicked.connect(self.open_folder);row.addWidget(files)
        self.description=note('Select a recorded take. Playback uses original global coordinates and native timestamps.');body.addWidget(self.description)
        self.figure=Figure(facecolor='white',layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure);body.addWidget(self.canvas,1)
        toolbar=NavigationToolbar2QT(self.canvas,self);body.addWidget(toolbar)
        row=QHBoxLayout();body.addLayout(row)
        self.previous=QPushButton('Previous');self.previous.clicked.connect(lambda:self.change_take(-1));row.addWidget(self.previous)
        self.play=QPushButton('Play');self.play.setEnabled(False);self.play.clicked.connect(self.toggle_play);row.addWidget(self.play)
        self.next=QPushButton('Next');self.next.clicked.connect(lambda:self.change_take(1));row.addWidget(self.next)
        self.speed=QComboBox()
        for value in (.25,.5,1.,2.):self.speed.addItem(f'{value:g}×',value)
        self.speed.setCurrentIndex(2);row.addWidget(self.speed)
        self.camera=QComboBox();self.camera.addItems(['Perspective','Side XZ','Top XY','Front YZ']);self.camera.currentTextChanged.connect(self.set_camera);row.addWidget(self.camera)
        self.auto_next=QCheckBox('Play all takes');row.addWidget(self.auto_next)
        self.timeline=QSlider(Qt.Orientation.Horizontal);self.timeline.setRange(0,0);self.timeline.valueChanged.connect(self.seek);body.addWidget(self.timeline)
        self.time_note=note('');body.addWidget(self.time_note)
        body.addWidget(note('Orange: measured tracked origin and orientation axes. Blue: measured cable markers. Gaps stay missing; lines between markers are a visual guide. No predicted motion or command alignment is added.'))
        self.timer=QTimer(self);self.timer.setInterval(33);self.timer.timeout.connect(self.tick)
        self.takes.currentIndexChanged.connect(self.select_take);self.refresh()

    def refresh(self):
        if self.worker is not None:return
        selected=self.takes.currentData();self.entries=discover_takes(self.root);self.entries.update(self.opened)
        self.takes.blockSignals(True);self.takes.clear()
        for path,row in self.entries.items():self.takes.addItem(row['label']+' · '+row.get('role','unassigned'),path)
        self.takes.setCurrentIndex(max(0,self.takes.findData(selected)));self.takes.blockSignals(False)
        if self.active:self.select_take()

    def select_take(self):
        self.pause();self.data=None;self.play.setEnabled(False);self.timeline.setRange(0,0)
        self.figure.clear();self.canvas.draw_idle();self.time_note.setText('')
        path=self.takes.currentData()
        if not path:self.description.setText('No recorded takes yet. Import in Recordings or open an OptiTrack CSV.');return
        if not self.active:return
        if self.worker is not None:return
        self.description.setText('Loading '+Path(path).name+'…');self.set_loading(True)
        self.loading_path=path
        self.worker=TakeLoader(path,self.entries[path].get('drone'));self.worker.loaded.connect(self.loaded)
        self.worker.failed.connect(lambda error:self.description.setText('Could not replay this file: '+error))
        self.worker.finished.connect(self.load_finished);self.worker.start()

    def set_loading(self,value):
        for w in (self.takes,self.refresh_button,self.open_button,self.previous,self.next):w.setEnabled(not value)

    def load_finished(self):
        self.worker.deleteLater();self.worker=None;self.set_loading(False)
        if self.loading_path!=self.takes.currentData():self.select_take();return
        if getattr(self,'continue_playback',False):
            self.continue_playback=False
            if self.data is not None and self.active:self.toggle_play()

    def loaded(self,data):
        if self.loading_path!=self.takes.currentData():return
        self.data=data;self.figure.clear();self.ax=self.figure.add_subplot(111,projection='3d')
        points=np.concatenate([data['drone'],data['cable'].reshape(-1,3)]);points=points[np.isfinite(points).all(1)]
        if not len(points):self.description.setText('No finite recorded positions.');self.data=None;return
        low=points.min(0)-.1;high=points.max(0)+.1
        self.ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]')
        self.ax.set_box_aspect(high-low,zoom=1.15)
        trail=data['drone'].copy();gaps=np.r_[False,np.diff(data['time'])>1.5*np.median(np.diff(data['time']))];trail[gaps]=np.nan
        self.ax.plot(*trail.T,color='#cbd5e1',linewidth=1)
        self.cable,=self.ax.plot([],[],[],color='#2563eb',marker='o',markersize=4,linewidth=2)
        self.drone,=self.ax.plot([],[],[],color='#ea580c',marker='o',markersize=8)
        self.axes=[self.ax.plot([],[],[],color=c,linewidth=2)[0] for c in ('#dc2626','#16a34a','#2563eb')]
        self.set_camera(self.camera.currentText());self.timeline.setRange(0,len(data['time'])-1);self.timeline.setValue(0);self.draw_frame(0);self.play.setEnabled(True)
        missing=int((~data['marker_valid'].all(1)).sum())
        self.description.setText(f'{self.takes.currentText()} · {data["drone_label"]} · {data["elapsed"][-1]:.2f} s · {len(data["time"]):,} frames · {missing:,} frames with missing cable markers · raw global metres')

    def set_camera(self,name):
        if self.data is None or not hasattr(self,'ax'):return
        from matplotlib.ticker import MaxNLocator,NullLocator
        for axis,label in zip((self.ax.xaxis,self.ax.yaxis,self.ax.zaxis),'XYZ'):
            axis.set_major_locator(MaxNLocator(5));axis.set_label_text(label+' [m]')
        hidden={'Side XZ':self.ax.yaxis,'Top XY':self.ax.zaxis,'Front YZ':self.ax.xaxis}.get(name)
        if hidden is not None:hidden.set_major_locator(NullLocator());hidden.set_label_text('')
        self.ax.set_proj_type('persp' if name=='Perspective' else 'ortho')
        elevation,azimuth={'Perspective':(22,-60),'Side XZ':(0,-90),'Top XY':(90,-90),'Front YZ':(0,0)}[name]
        self.ax.view_init(elev=elevation,azim=azimuth);self.canvas.draw_idle()

    def draw_frame(self,index):
        if self.data is None:return
        d=self.data;p=d['drone'][index];c=d['cable'][index]
        self.cable.set_data_3d(*c.T);self.drone.set_data_3d(*p[:,None])
        for j,line in enumerate(self.axes):
            points=np.stack([p,p+.08*d['rotation'][index,:,j]]) if d['pose_valid'][index] else np.full((2,3),np.nan)
            line.set_data_3d(*points.T)
        self.time_note.setText(f'{d["elapsed"][index]:.2f} / {d["elapsed"][-1]:.2f} s · native time {d["time"][index]:.3f} s · frame {int(d["frame"][index])} · visible cable markers {d["marker_valid"][index].sum()}/10 · XYZ [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] m')
        self.canvas.draw_idle()

    def seek(self,index):
        if self.data is None:return
        self.play_time=float(self.data['elapsed'][index]);self.last_tick=time.perf_counter();self.draw_frame(index)

    def toggle_play(self):
        if self.data is None:return
        if self.playing:self.pause();return
        if self.timeline.value()==self.timeline.maximum():self.timeline.setValue(0)
        self.play_time=float(self.data['elapsed'][self.timeline.value()]);self.last_tick=time.perf_counter()
        self.playing=True;self.play.setText('Pause');self.timer.start()

    def pause(self):self.playing=False;self.timer.stop();self.play.setText('Play')

    def tick(self):
        if not self.active or not self.playing or self.data is None:return
        now=time.perf_counter();self.play_time+=(now-self.last_tick)*self.speed.currentData();self.last_tick=now
        i=min(np.searchsorted(self.data['elapsed'],self.play_time,side='right')-1,self.timeline.maximum())
        self.timeline.blockSignals(True);self.timeline.setValue(int(i));self.timeline.blockSignals(False);self.draw_frame(int(i))
        if self.play_time>=self.data['elapsed'][-1]:
            self.pause()
            if self.auto_next.isChecked() and self.takes.currentIndex()+1<self.takes.count():self.continue_playback=True;self.change_take(1)

    def change_take(self,delta):
        if self.worker is None:self.takes.setCurrentIndex(max(0,min(self.takes.count()-1,self.takes.currentIndex()+delta)))

    def open_csv(self):
        paths,_=QFileDialog.getOpenFileNames(self,'Open native OptiTrack takes',str(self.root/'data/raw_takes'),'CSV (*.csv)')
        for path in paths:self.opened[path]=dict(label=Path(path).stem,role='opened raw file')
        if paths:self.refresh();self.takes.setCurrentIndex(self.takes.findData(paths[0]))

    def open_folder(self):
        if self.takes.currentData():QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.takes.currentData()).parent)))

    def open_take(self,name):
        self.refresh()
        for i in range(self.takes.count()):
            if Path(self.takes.itemData(i)).parent.name==name:self.takes.setCurrentIndex(i);break

    def set_page_active(self,active):
        self.active=active
        if not active:self.pause()
        elif self.data is None and self.worker is None:self.select_take()

    def shutdown(self):
        self.pause();return self.worker is None
