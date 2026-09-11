"""Read-only flight comparison; never generates or fits a new prediction."""
from pathlib import Path
import time as clock
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QSlider, QCheckBox, QDoubleSpinBox, QTabWidget)
from .research_widgets import note
from experimental_data.adaptation_check import load_comparison
from .flight_generation_selector import FlightGenerationSelector


class ComparisonLoader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, root, batch, take, rehearsal, parent=None):
        super().__init__(parent)
        self.args = root, batch, take, rehearsal

    def run(self):
        try: self.loaded.emit(load_comparison(*self.args))
        except Exception as error: self.failed.emit(str(error))


class AdaptationCheckPage(QWidget):
    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = Path(root); self.data = None; self.viewer = None; self.worker = None
        self.active = False; self.playing = False; self.rehearsal = None
        layout = QVBoxLayout(self)
        self.library_selector=FlightGenerationSelector(self.root);layout.addWidget(self.library_selector)
        self.batches=self.library_selector.batches;self.refresh=self.library_selector.refresh_button
        row2 = QHBoxLayout(); layout.addLayout(row2)
        self.takes = QComboBox(); row2.addWidget(QLabel('Real-flight take')); row2.addWidget(self.takes,1)
        self.load = QPushButton('Load real flight + ghost'); row2.addWidget(self.load)
        self.details=QCheckBox('Details');row2.addWidget(self.details)
        self.advanced=QWidget();advanced_layout=QVBoxLayout(self.advanced);advanced_layout.setContentsMargins(0,0,0,0)
        row=QHBoxLayout();advanced_layout.addLayout(row)
        self.browse=QPushButton('Open batch…');row.addWidget(self.browse)
        self.reference=QPushButton('Choose another saved prediction…');row.addWidget(self.reference);row.addStretch()
        layout.addWidget(self.advanced);self.advanced.hide();self.details.toggled.connect(self.advanced.setVisible)
        self.status = note('Select a batch and load a flight. Its saved prediction is matched by the executed CSV.'); layout.addWidget(self.status)
        self.flight_legend=note('Solid orange: OptiTrack measurement · Blue ghost: saved predicted execution, not the commanded PVA. World XYZ in metres. Missing markers remain gaps.')
        layout.addWidget(self.flight_legend)
        self.normalize_height=QCheckBox('Show hover-normalized Z (diagnostic; not raw flight performance)')
        self.normalize_height.setEnabled(False);self.normalize_height.toggled.connect(self.apply_height_view);advanced_layout.addWidget(self.normalize_height)
        self.views = QTabWidget(); layout.addWidget(self.views,1)
        self.scene_page = QWidget(); self.scene_layout = QVBoxLayout(self.scene_page); self.scene_layout.setContentsMargins(0,0,0,0)
        self.empty_scene=QLabel('Choose a model generation, plan and real-flight take.');self.empty_scene.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_scene.setWordWrap(True);self.empty_scene.setStyleSheet('color:#64748b;font-size:18px;padding:32px;')
        self.scene_layout.addWidget(self.empty_scene,1)
        self.views.addTab(self.scene_page,'3D comparison')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self.figure = Figure(figsize=(9,5), tight_layout=True); self.canvas = FigureCanvasQTAgg(self.figure)
        self.views.addTab(self.canvas,'Errors over time')
        options = QHBoxLayout(); layout.addLayout(options)
        self.measured = QCheckBox('Measured'); self.measured.setChecked(True)
        self.ghost = QCheckBox('Plan ghost'); self.ghost.setChecked(True)
        self.trails = QCheckBox('Trails'); self.trails.setChecked(True)
        for widget in (self.measured,self.ghost,self.trails): options.addWidget(widget); widget.toggled.connect(self.redraw)
        options.addWidget(QLabel('Ghost opacity'))
        self.opacity = QDoubleSpinBox(); self.opacity.setRange(.05,1); self.opacity.setSingleStep(.05); self.opacity.setValue(.28)
        options.addWidget(self.opacity); self.opacity.valueChanged.connect(self.redraw)
        self.camera = QComboBox(); self.camera.addItems(['Perspective','Side XZ','Front YZ','Top XY']); options.addWidget(self.camera)
        self.camera.currentTextChanged.connect(lambda text: self.viewer.set_camera_preset(text) if self.viewer else None)
        options.addStretch()
        transport = QHBoxLayout(); layout.addLayout(transport)
        self.play = QPushButton('Play'); self.play.clicked.connect(self.toggle_play); transport.addWidget(self.play)
        self.strike = QPushButton('Predicted strike'); self.strike.clicked.connect(self.jump_strike); transport.addWidget(self.strike)
        self.whip = QCheckBox('Whip only'); transport.addWidget(self.whip); self.whip.toggled.connect(self.set_range)
        self.loop = QCheckBox('Loop'); transport.addWidget(self.loop)
        self.speed = QComboBox()
        for value in (.25,.5,1.,2.): self.speed.addItem(f'{value:g}×',value)
        self.speed.setCurrentIndex(2); transport.addWidget(self.speed)
        self.timeline = QSlider(Qt.Orientation.Horizontal); self.timeline.setRange(0,0); transport.addWidget(self.timeline,1)
        self.timeline.valueChanged.connect(self.redraw); self.timeline.sliderPressed.connect(self.pause)
        self.time_label = QLabel('0.000 s'); transport.addWidget(self.time_label)
        self.metrics = note(''); layout.addWidget(self.metrics)
        self.timer = QTimer(self); self.timer.setInterval(33); self.timer.timeout.connect(self.tick)
        self.browse.clicked.connect(self.open_batch)
        self.reference.clicked.connect(self.choose_reference); self.load.clicked.connect(self.load_flight)
        self.library_selector.session_changed.connect(self.batch_changed)
        self.takes.currentIndexChanged.connect(self.clear_result)
        self.views.currentChanged.connect(self.redraw)
        self.flight_controls=[self.status,self.flight_legend,self.metrics]+[
            group.itemAt(i).widget() for group in (row,row2,options,transport)
            for i in range(group.count()) if group.itemAt(i).widget() is not None]
        self.refresh_batches()
        self.whip.setChecked(True)

    def refresh_batches(self):
        self.library_selector.refresh()

    def batch_changed(self):
        self.rehearsal = None; self.reference.setText('Choose another saved prediction…')
        self.takes.clear()
        session=self.library_selector.selected_session()
        for take in session['takes'] if session else []:
            number=take['name'].rsplit('_',1)[-1]
            self.takes.addItem(f'Take {number} · {take["role"].capitalize()}',take['name'])
            self.takes.setItemData(self.takes.count()-1,take['name'],Qt.ItemDataRole.ToolTipRole)
        self.clear_result()
        self.status.setVisible(bool(session))
        if not session:
            self.status.setText(self.library_selector.summary.text());self.empty_scene.setText(self.library_selector.summary.text())
        elif not session['takes']:
            self.status.setText('No recorded takes for this plan yet.');self.empty_scene.setText('This model has a saved plan. Its measured flights will appear here after recording.')
        self.load.setEnabled(bool(session and session['takes']))

    def select_flight(self,batch,take):
        if not self.library_selector.select_batch(batch):return False
        index=self.takes.findData(take)
        if index<0:return False
        self.takes.setCurrentIndex(index);return True

    def open_batch(self):
        path = QFileDialog.getExistingDirectory(self,'Open batch containing simulation_csv and flight_take',str(self.root/'rehearsal_csv_and_result_in_real_flight'))
        if path:
            if not self.library_selector.select_batch(path):
                self.status.setText('This folder is not a flight batch in the project library. Import it through Recordings first.')

    def choose_reference(self):
        path = QFileDialog.getExistingDirectory(self,'Choose original saved rehearsal (must match CSV)',str(self.root/'runs/rehearsals'))
        if path:
            self.clear_result(); self.rehearsal = path; self.reference.setText(Path(path).name)

    def clear_result(self, *_):
        self.pause(); self.data = None
        if self.viewer is not None: self.viewer.close(); self.viewer.deleteLater(); self.viewer = None
        self.empty_scene.setText('Select a take, then load its real flight and original plan ghost.');self.empty_scene.show()
        self.flight_legend.hide()
        self.figure.clear(); self.canvas.draw_idle(); self.timeline.setRange(0,0)
        self.play.setEnabled(False); self.strike.setEnabled(False); self.metrics.setText(''); self.time_label.setText('—')
        self.status.setText('Ready to load selected flight; no comparison loaded.')

    def load_flight(self):
        if self.worker is not None or not self.takes.currentData(): return
        self.status.show()
        self.clear_result(); self.status.setText('Matching CSV and aligning measured drone logs…')
        self.set_loading(True)
        self.worker = ComparisonLoader(self.root, self.batches.currentData(), self.takes.currentData(), self.rehearsal, self)
        self.worker.loaded.connect(self.loaded); self.worker.failed.connect(self.failed); self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def set_loading(self, busy):
        self.library_selector.setEnabled(not busy)
        for widget in (self.takes,self.browse,self.reference): widget.setEnabled(not busy)
        self.load.setEnabled(not busy and self.takes.count()>0)

    def worker_finished(self):
        self.worker.deleteLater(); self.worker = None; self.set_loading(False)

    def failed(self, message): self.status.setText('Could not load: '+message)

    def loaded(self, data):
        self.empty_scene.hide()
        self.status.show();self.flight_legend.show()
        self.raw_data=data
        self.normalize_height.blockSignals(True);self.normalize_height.setChecked(False);self.normalize_height.blockSignals(False)
        self.normalize_height.setEnabled('hover_normalized' in data)
        self.data = dict(data,evaluation_frame='raw_global_xyz')
        self.flight_legend.setText('Raw global OptiTrack XYZ · orange measured motion · blue original saved prediction. No height correction applied.')
        a = data['alignment']; start,end = data['tracking_span']
        coverage = 'complete whip' if start<=0 and end>=data['metadata']['whip_end_s'] else 'WARNING: incomplete whip coverage'
        self.status.setToolTip(f"{data['take']} · {coverage} · tracking {start:.2f} to {end:.2f} s relative to CSV\n"
            f"Saved prediction: {data['rehearsal'].name} · exact CSV match · measured state: OptiTrack {data.get('drone_rigid_body') or ''}. "
            f"Clock alignment: {a['method']}; offset {a['offset_s']:.6f} s. Controller log: commands/timing only.")
        timing='synchronized clocks' if a.get('clock_verified') else 'estimated clock alignment'
        self.status.setText(f"{data['take']} · {coverage} · original MPPI/PVA forecast · exact CSV match · {timing}")
        self.play.setEnabled(True); self.strike.setEnabled(data['metadata'].get('predicted_hit_time_s') is not None)
        self.set_range(); self.draw_plots(); self.redraw()

    def apply_height_view(self,enabled):
        if not hasattr(self,'raw_data') or self.data is None:return
        from experimental_data.hover_calibration import normalized_evaluation
        if enabled:
            self.data=normalized_evaluation(self.raw_data)
            bias=self.raw_data['height_calibration']['bias_z_m']
            self.flight_legend.setText(f'Retrospective hover-normalized diagnostic: Z minus {bias*100:.2f} cm. Not raw flight performance; original forecast unchanged.')
        else:
            self.data=dict(self.raw_data,evaluation_frame='raw_global_xyz')
            self.flight_legend.setText('Raw global OptiTrack XYZ · orange measured motion · blue original saved prediction. No height correction applied.')
        self.draw_plots();self.redraw()

    def set_range(self, *_):
        if self.data is None: return
        self.pause(); t = self.data['time']
        last = np.searchsorted(t,self.data['metadata']['whip_end_s'],side='right')-1 if self.whip.isChecked() else len(t)-1
        self.timeline.setRange(0,max(0,int(last))); self.timeline.setValue(0)
        if self.figure.axes:
            self.figure.axes[-1].set_xlim(t[0], max(t[0]+.01,t[max(0,int(last))])); self.canvas.draw_idle()
        self.redraw()

    def draw_plots(self):
        d = self.data; self.figure.clear(); axes = self.figure.subplots(2,1,sharex=True)
        axes[0].plot(d['time'],d['drone_error']*100,label='Drone'); axes[0].plot(d['time'],d['tip_error']*100,label='Cable tip')
        axes[0].set_ylabel('Prediction error [cm]'); axes[0].legend()
        axes[1].plot(d['time'],d['target_error']*100,color='#ea580c',label='Measured tip → target')
        axes[1].plot(d['time'],np.linalg.norm(d['predicted_cable'][:,-1]-d['target'],axis=1)*100,color='#2563eb',label='Predicted tip → target')
        axes[1].set_ylabel('Target distance [cm]'); axes[1].set_xlabel('Time from CSV onset [s]'); axes[1].legend()
        self.plot_cursors=[]
        for axis in axes:
            axis.axvline(d['metadata']['whip_end_s'],color='gray',linestyle=':',label='Whip end')
            if d['metadata'].get('predicted_hit_time_s') is not None: axis.axvline(d['metadata']['predicted_hit_time_s'],color='purple',linestyle='--',label='Predicted strike')
            self.plot_cursors.append(axis.axvline(d['time'][self.timeline.value()],color='#172033',alpha=.6,linewidth=1))
            axis.grid(alpha=.2)
        axes[0].legend(ncol=2,fontsize=8)
        axes[-1].set_xlim(d['time'][0],max(d['time'][0]+.01,d['time'][self.timeline.maximum()]))
        self.canvas.draw_idle()

    def redraw(self, *_):
        if self.data is None: return
        d = self.data; i = self.timeline.value(); t = d['time'][i]
        self.time_label.setText(f'{t:.3f} s')
        fmt = lambda value: f'{100*value:.1f} cm' if np.isfinite(value) else 'missing'
        self.metrics.setText(f"{'WHIP' if t<=d['metadata']['whip_end_s'] else 'RECOVERY / HOLD'} · Drone prediction error: {fmt(d['drone_error'][i])} · Tip prediction error: {fmt(d['tip_error'][i])} · Tip → target: {fmt(d['target_error'][i])}")
        if self.views.currentIndex()==1:
            for cursor in getattr(self,'plot_cursors',[]): cursor.set_xdata([t,t])
            self.canvas.draw_idle()
        if not self.active or self.views.currentIndex()!=0: return
        try:
            if self.viewer is None:
                from .adaptation_scene import AdaptationScene
                self.viewer = AdaptationScene(d,self); self.scene_layout.addWidget(self.viewer)
                self.viewer.set_camera_preset(self.camera.currentText())
            self.viewer.draw(d,i,measured=self.measured.isChecked(),predicted=self.ghost.isChecked(),trails=self.trails.isChecked(),opacity=self.opacity.value())
        except Exception as error:
            self.pause(); self.status.setText('3D scene error: '+str(error))

    def jump_strike(self):
        if self.data is not None and self.data['metadata'].get('predicted_hit_time_s') is not None:
            self.pause(); self.timeline.setValue(int(np.argmin(abs(self.data['time']-self.data['metadata']['predicted_hit_time_s']))))

    def pause(self):
        self.playing = False; self.timer.stop(); self.play.setText('Play')

    def toggle_play(self):
        if self.playing: self.pause(); return
        if self.data is None: return
        if self.timeline.value() >= self.timeline.maximum(): self.timeline.setValue(0)
        self.playing=True; self.last_tick=clock.monotonic(); self.play_time=self.data['time'][self.timeline.value()]
        self.play.setText('Pause'); self.timer.start()

    def tick(self):
        if not self.playing or self.data is None: return
        now=clock.monotonic(); self.play_time+=(now-self.last_tick)*self.speed.currentData(); self.last_tick=now
        index=int(np.searchsorted(self.data['time'],self.play_time,side='right')-1)
        self.timeline.setValue(min(max(index,0),self.timeline.maximum()))
        if index>=self.timeline.maximum():
            if self.loop.isChecked():
                self.timeline.setValue(0);self.play_time=self.data['time'][0];self.last_tick=now
            else:self.pause()

    def set_page_active(self, active):
        self.active = active
        if not active: self.pause()
        else: self.redraw()

    def shutdown(self):
        self.pause()
        if self.worker is not None:
            if self.worker.isRunning(): return False
        if self.viewer is not None: self.viewer.close(); self.viewer = None
        return True
