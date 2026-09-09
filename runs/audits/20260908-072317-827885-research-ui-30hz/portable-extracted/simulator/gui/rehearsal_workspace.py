"""Native 30 Hz offline rehearsal: configure, generate, inspect, export."""
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from PySide6.QtCore import Qt,QTimer,Signal,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QFrame,QPushButton,
    QComboBox,QDoubleSpinBox,QTabWidget,QSplitter,QSlider,QFileDialog,QTableWidget,QTableWidgetItem,QHeaderView,QProgressBar)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from simulator.workflow import read_json,stamp
from simulator.research_config import workspace_configs
from .research_widgets import note,BackgroundJob
from .model_workspace import style_axes


class RehearsalWorkspace(QWidget):
    flight_finished=Signal()
    def __init__(self,root):
        super().__init__();self.root=Path(root);self.thread=None;self.viewer=None;self.arrays=None;self.directory=None;self.active=False;self.playing=False;self.legacy=None
        outer=QVBoxLayout(self);outer.setContentsMargins(20,12,20,16);outer.setSpacing(10)
        self.tabs=QTabWidget();outer.addWidget(self.tabs,1)
        current=QWidget();layout=QVBoxLayout(current);self.tabs.addTab(current,'30 Hz workspace')
        row=QHBoxLayout();row.addWidget(QLabel('Policy'));self.checkpoints=QComboBox();row.addWidget(self.checkpoints,1)
        refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh_checkpoints);row.addWidget(refresh);layout.addLayout(row)
        self.policy_note=note('');layout.addWidget(self.policy_note)
        split=QSplitter(Qt.Orientation.Horizontal);layout.addWidget(split,1)
        controls=QFrame();controls.setObjectName('contentCard');cl=QVBoxLayout(controls);cl.setContentsMargins(16,14,16,14);controls.setMaximumWidth(300)
        title=QLabel('Launch setup');title.setStyleSheet('font-size: 14pt; font-weight: 650;');cl.addWidget(title)
        self.start_spins=[];self.target_spins=[]
        for label,fields in [('Tracked-origin start [m]',self.start_spins),('Target position [m]',self.target_spins)]:
            cl.addWidget(note(label));form=QFormLayout()
            for axis in 'XYZ':
                spin=QDoubleSpinBox();spin.setRange(-20,20);spin.setDecimals(4);spin.setSingleStep(.01);fields.append(spin);form.addRow(axis,spin)
            cl.addLayout(form)
        cl.addWidget(note('Settled level hover · zero initial velocity\nHanging cable · 10 s pre-hold\nTarget fixed when the plan is frozen'))
        self.start=QPushButton('Generate rehearsal');self.start.setObjectName('primaryButton');self.start.clicked.connect(self.generate);cl.addWidget(self.start)
        self.progress=QProgressBar();self.progress.setRange(0,1000);self.progress.setValue(0);cl.addWidget(self.progress)
        self.save=QPushButton('Save complete CSV…');self.save.clicked.connect(self.save_csv);self.save.setEnabled(False);cl.addWidget(self.save)
        self.package=QPushButton('Export policy + rehearsal ZIP…');self.package.clicked.connect(self.export_package);self.package.setEnabled(False);cl.addWidget(self.package)
        self.open=QPushButton('Open generated files');self.open.clicked.connect(self.open_files);self.open.setEnabled(False);cl.addWidget(self.open)
        cl.addStretch();split.addWidget(controls)
        self.views=QTabWidget();split.addWidget(self.views);split.setSizes([270,850])
        scene=QWidget();scene_layout=QVBoxLayout(scene);self.views.addTab(scene,'3D execution')
        self.scene_note=note('Generate a plan to compare the commanded tracked origin with the predicted drone and cable.');scene_layout.addWidget(self.scene_note)
        self.viewer_host=QVBoxLayout();scene_layout.addLayout(self.viewer_host,1)
        bar=QHBoxLayout();self.play=QPushButton('Play');self.play.clicked.connect(self.toggle_play);self.play.setEnabled(False);bar.addWidget(self.play)
        self.speed=QComboBox();self.speed.addItems(['0.25×','0.5×','1×','2×']);self.speed.setCurrentIndex(2);bar.addWidget(self.speed)
        self.camera=QComboBox();self.camera.addItems(['Perspective','Side XZ','Top XY','Front YZ']);self.camera.currentTextChanged.connect(lambda s:self.viewer.set_camera_preset(s) if self.viewer else None);bar.addWidget(self.camera)
        self.timeline=QSlider(Qt.Orientation.Horizontal);self.timeline.setRange(0,0);self.timeline.valueChanged.connect(self.draw_frame);bar.addWidget(self.timeline,1);scene_layout.addLayout(bar)
        self.time_note=note('Blue: commanded tracked origin · orange: predicted whip · green: predicted recovery');scene_layout.addWidget(self.time_note)
        plots=QWidget();pl=QVBoxLayout(plots);self.views.addTab(plots,'Force, PVA and tracking')
        self.figure=Figure(figsize=(9,6),layout='constrained',facecolor='white');self.canvas=FigureCanvasQTAgg(self.figure);pl.addWidget(self.canvas)
        pl.addWidget(note('Virtual force includes gravity and is only a simulator input. FullState CSV contains position, velocity and acceleration. Recovery is a separate analytic reference and does not affect the whip objective.'))
        data=QWidget();dl=QVBoxLayout(data);self.views.addTab(data,'Command table')
        self.table=QTableWidget();self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);self.table.verticalHeader().hide();dl.addWidget(self.table)
        self.status=note('Ready · GPU planning runs in an isolated process.');layout.addWidget(self.status)
        self.job=BackgroundJob(root);self.job.finished.connect(self.finished);self.job.progress.connect(self.job_progress);layout.addWidget(self.job)
        legacy=QWidget();ll=QVBoxLayout(legacy);self.tabs.addTab(legacy,'Legacy policies')
        ll.addWidget(note('Preserved 20 Hz checkpoints keep their original model and rehearsal semantics. They cannot be retimed into a new 30 Hz policy.'))
        load=QPushButton('Open legacy rehearsal');self.legacy_open=load;ll.addWidget(load)
        def open_legacy():
            if self.legacy is None:
                from .fullstate_page import FullStatePage
                self.legacy=FullStatePage(root);self.legacy.flight_finished.connect(self.flight_finished.emit);ll.addWidget(self.legacy,1);load.hide()
            self.legacy.set_page_active(self.active)
        load.clicked.connect(open_legacy)
        self.tabs.currentChanged.connect(lambda _:self.set_page_active(self.active))
        self.timer=QTimer(self);self.timer.setInterval(33);self.timer.timeout.connect(self.tick)
        self.checkpoints.currentIndexChanged.connect(self.selection_changed);self.refresh_checkpoints()

    def refresh_checkpoints(self):
        if self.job.running:return
        from simulator.policy_library import deleted_checkpoints,checkpoint_key
        deleted=deleted_checkpoints(self.root);previous=self.checkpoints.currentData();self.checkpoints.blockSignals(True);self.checkpoints.clear()
        for run in sorted((self.root/'runs/ppo').glob('*'),reverse=True):
            model=read_json(run/'model.json',{})
            if model.get('fullstate_execution',{}).get('schema')!='tracked_pose_execution_v1':continue
            for p in sorted((run/'checkpoints').glob('*.pt'),key=lambda p:(p.name!='best_validation.pt',p.name)):
                if checkpoint_key(self.root,p) not in deleted:self.checkpoints.addItem(read_json(run/'run.json',{}).get('display_name',run.name)+' / '+p.name,str(p.resolve()))
        index=self.checkpoints.findData(previous);self.checkpoints.setCurrentIndex(index if index>=0 else 0);self.checkpoints.blockSignals(False);self.selection_changed()

    def selection_changed(self):
        path=self.checkpoints.currentData()
        if self.arrays is not None and self.metadata.get('checkpoint')!=path:self.clear_result()
        model,task,config=([read_json(Path(path).parent.parent/(n+'.json')) for n in ('model','task','ppo')] if path else workspace_configs(self.root))
        origin=np.asarray(task['initial_root_position_m'])-np.asarray(model['recorded_data']['optitrack_to_attachment_offset_body_m'])
        for spins,values in [(self.start_spins,origin),(self.target_spins,task['target_position_m'])]:
            for spin,v in zip(spins,values):spin.setValue(float(v))
        self.policy_note.setText('Native 30 Hz PPO · both residuals · saved model travels with the policy' if path else 'No native 30 Hz checkpoint yet. Start a new PPO run; its saved checkpoints will appear here.')
        self.start.setEnabled(bool(path) and not self.job.running)

    def generate(self):
        if self.job.running:return
        self.clear_result()
        self.playing=False;self.timer.stop();self.directory=self.root/'runs/rehearsals'/stamp()
        command=[sys.executable,'-u','tools/rehearse_research.py','--checkpoint',self.checkpoints.currentData(),
            '--output',str(self.directory),'--origin',*[str(s.value()) for s in self.start_spins],
            '--target',*[str(s.value()) for s in self.target_spins],'--device','cuda']
        try:
            self.job.start(self.directory,command);self.start.setEnabled(False);self.checkpoints.setEnabled(False)
            for w in self.start_spins+self.target_spins+[self.save,self.package,self.open,self.play]:w.setEnabled(False)
            self.status.setText('Generating frozen force plan and complete FullState reference…');self.progress.setValue(0)
        except Exception as e:self.status.setText(str(e))

    def job_progress(self,value):
        self.progress.setValue(int(1000*value.get('step',0)/max(value.get('total',1),1)));self.status.setText(value['label'])

    def finished(self,code):
        self.start.setEnabled(bool(self.checkpoints.currentData()));self.checkpoints.setEnabled(True)
        for w in self.start_spins+self.target_spins:w.setEnabled(True)
        if code:
            self.status.setText('Generation failed. Open the job log; no new CSV is available.');self.flight_finished.emit();return
        self.load_result(self.directory);self.flight_finished.emit()

    def load_result(self,directory):
        self.directory=Path(directory);self.metadata=read_json(self.directory/'rehearsal.json')
        with np.load(self.directory/'rehearsal.npz',allow_pickle=False) as data:self.arrays={k:data[k].copy() for k in data.files}
        m=self.metadata;self.progress.setValue(1000)
        self.status.setText(f'Whip {m["whip_end_s"]:.2f} s · total CSV {m["total_duration_s"]:.2f} s · '
            f'predicted {"valid hit" if m["predicted_valid_hit"] else "miss"} · closest tip {m["minimum_tip_distance_m"]*100:.1f} cm. Recovery prediction is unvalidated.')
        if not m['recovery_prediction_complete']:self.status.setText(self.status.text()+f' Prediction stopped at {m["prediction_valid_through_s"]:.2f} s after a model-domain failure.')
        for w in [self.save,self.package,self.open,self.play]:w.setEnabled(True)
        self.timeline.setRange(0,len(self.arrays['prediction_time_s'])-1);self.timeline.setValue(0)
        self.draw_plots();self.fill_table()
        if self.active:self.ensure_viewer();self.draw_frame(0)

    def ensure_viewer(self):
        if self.viewer is not None or self.arrays is None:return
        from .viewer_3d import create_viewer
        task=read_json(self.directory/'task.json');self.viewer=create_viewer(self.arrays['cable_positions_m'][0],task['target_position_m'],task['desired_strike_direction_world'],task['success']['tip_target_distance_m'],self)
        self.viewer.set_live_flight(True);self.viewer_host.addWidget(self.viewer)

    def draw_frame(self,index):
        if self.arrays is None or not self.active:return
        self.ensure_viewer();a=self.arrays;t=a['prediction_time_s'][index];phase=1 if t<=self.metadata['whip_end_s'] else 2
        q=a['cable_positions_m'];phases=np.where(a['prediction_time_s'][:index+1]<=self.metadata['whip_end_s'],1,3)
        self.viewer.update_state(q[index],np.zeros(3),q[:index+1,0],q[:index+1,-1],trail_phases=phases)
        if hasattr(self.viewer,'set_tracked_pose'):self.viewer.set_tracked_pose(a['origin_positions_m'][index],a['origin_rotations'][index])
        if hasattr(self.viewer,'plotter'):
            import pyvista as pv
            if not getattr(self,'_reference_source',None)==str(self.directory):
                self.viewer.plotter.add_mesh(pv.lines_from_points(a['commands'][:,:3]),color='#2563eb',line_width=2,name='FullStateReference')
                self._reference_source=str(self.directory);self.viewer.set_target(a['target_position_m'],[1,0,0],read_json(self.directory/'task.json')['success']['tip_target_distance_m'])
            self.viewer.plotter.add_mesh(pv.PolyData(a['origin_positions_m'][index:index+1]),color='#0f172a',point_size=12,render_points_as_spheres=True,name='TrackedOrigin')
        self.scene_note.setText(f'{t:.2f} s · {"Frozen whip" if phase==1 else "Recovery / final hold — unvalidated"} · tracking-frame glyph and cable attachment · policy {self.metadata["checkpoint_sha256"][:8]}')

    def toggle_play(self):
        if self.arrays is None:return
        self.playing=not self.playing;self.play.setText('Pause' if self.playing else 'Play')
        if self.timeline.value()==self.timeline.maximum():self.timeline.setValue(0)
        self.last_tick=time.perf_counter();self.play_time=float(self.arrays['prediction_time_s'][self.timeline.value()]);self.timer.start()

    def tick(self):
        if not self.playing or not self.active:return
        now=time.perf_counter();self.play_time+=(now-self.last_tick)*[.25,.5,1,2][self.speed.currentIndex()];self.last_tick=now
        i=min(int(np.searchsorted(self.arrays['prediction_time_s'],self.play_time)),self.timeline.maximum());self.timeline.setValue(i)
        if i==self.timeline.maximum():self.playing=False;self.play.setText('Replay');self.timer.stop()

    def draw_plots(self):
        a=self.arrays;t=a['command_time_s'];p=a['commands'];self.figure.clear();axes=self.figure.subplots(2,2).ravel()
        colors=['#2563eb','#0d9488','#ea580c']
        for j,(axis,color) in enumerate(zip('XYZ',colors)):
            axes[0].plot(a['force_time_s'],a['virtual_force_n'][:,j],color=color,label=axis)
            axes[1].plot(t,p[:,j],color=color,label=axis+' commanded')
            axes[1].plot(a['prediction_time_s'],a['origin_positions_m'][:,j],color=color,ls='--',alpha=.65)
            axes[2].plot(t,p[:,6+j],color=color,label=axis)
        speed=np.linalg.norm(p[:,3:6],axis=-1);axes[3].plot(t,speed,color='#2563eb',label='Commanded speed [m/s]')
        prediction=a['origin_positions_m'];planned=np.column_stack([np.interp(a['prediction_time_s'],t,p[:,j]) for j in range(3)])
        error_axis=axes[3].twinx();error_axis.plot(a['prediction_time_s'],np.linalg.norm(prediction-planned,axis=-1)*100,color='#ea580c',label='Tracking error')
        error_axis.set_ylabel('Tracking error [cm]',color='#ea580c');error_axis.tick_params(axis='y',colors='#ea580c',labelsize=8);error_axis.spines['top'].set_visible(False)
        for ax,title,unit in zip(axes,['Virtual total force','Origin position (dashed = predicted)','Commanded acceleration','Speed and prediction error'],['N','m','m/s²','Commanded speed [m/s]']):
            ax.set(title=title,xlabel='CSV time [s]',ylabel=unit);ax.axvline(self.metadata['whip_end_s'],color='#94a3b8',ls=':',lw=1);ax.legend(fontsize=7,frameon=False)
        style_axes(axes);self.canvas.draw_idle()

    def fill_table(self):
        from deployment.research_rehearsal import FIELDS
        rows=np.c_[self.arrays['command_time_s'],self.arrays['commands']];self.table.setColumnCount(len(FIELDS));self.table.setRowCount(len(rows));self.table.setHorizontalHeaderLabels(FIELDS)
        for i,row in enumerate(rows):
            for j,v in enumerate(row):self.table.setItem(i,j,QTableWidgetItem(f'{v:.5f}'))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def save_csv(self):
        path,_=QFileDialog.getSaveFileName(self,'Save complete 30 Hz trajectory',str(self.root/'policies/fullstate_30hz.csv'),'CSV (*.csv)')
        if path:shutil.copy2(self.directory/'fullstate_30hz.csv',path);self.status.setText('Complete CSV saved: '+path)

    def export_package(self):
        path,_=QFileDialog.getSaveFileName(self,'Export frozen policy and rehearsal',str(self.root/'policies'/('PPO-30Hz-'+stamp()+'.zip')),'ZIP (*.zip)')
        if path:
            from deployment.research_rehearsal import export_package
            export_package(self.directory,Path(path));self.status.setText('Policy, both model residuals, CSV and provenance exported: '+path)

    def open_files(self):
        if self.directory:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.directory)))

    def set_page_active(self,active):
        self.active=active and self.tabs.currentIndex()==0
        if self.active:self.ensure_viewer();self.draw_frame(self.timeline.value())
        else:self.playing=False;self.timer.stop();self.play.setText('Play')
        if self.legacy:self.legacy.set_page_active(active and self.tabs.currentIndex()==1)

    def shutdown(self):
        self.playing=False;self.timer.stop()
        if self.viewer is not None:self.viewer.close();self.viewer=None
        # Generation is a detached offline job and can finish after closing the UI.
        return self.legacy.shutdown() if self.legacy else True

    def clear_result(self):
        self.playing=False;self.timer.stop();self.arrays=None
        if self.viewer is not None:self.viewer.close();self.viewer.deleteLater();self.viewer=None
        self._reference_source=None;self.figure.clear();self.canvas.draw_idle();self.table.setRowCount(0);self.timeline.setRange(0,0)
        for w in (self.save,self.package,self.open,self.play):w.setEnabled(False)
        self.scene_note.setText('Generate a rehearsal for the selected policy and launch setup.')
