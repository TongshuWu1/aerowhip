"""Read-only flight comparison; never generates or fits a new prediction."""
from pathlib import Path
import time as clock
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QSlider, QCheckBox, QDoubleSpinBox, QTabWidget)
from .research_widgets import note
from experimental_data.adaptation_check import discover_batches, flight_names, load_comparison


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
        row = QHBoxLayout(); layout.addLayout(row)
        row.addWidget(QLabel('Adaptation'))
        self.batches = QComboBox(); self.batches.setMinimumWidth(270); row.addWidget(self.batches,1)
        self.refresh = QPushButton('Refresh'); row.addWidget(self.refresh)
        self.browse = QPushButton('Open batch…'); row.addWidget(self.browse)
        row2 = QHBoxLayout(); layout.addLayout(row2)
        self.takes = QComboBox(); row2.addWidget(QLabel('Flight')); row2.addWidget(self.takes,1)
        self.reference = QPushButton('Choose saved prediction…'); row2.addWidget(self.reference)
        self.load = QPushButton('Load flight'); row2.addWidget(self.load)
        self.status = note('Select a batch and load a flight. Its saved prediction is matched by the executed CSV.'); layout.addWidget(self.status)
        self.flight_legend=note('Solid orange: OptiTrack measurement · Blue ghost: saved predicted execution, not the commanded PVA. World XYZ in metres. Missing markers remain gaps.')
        layout.addWidget(self.flight_legend)
        self.normalize_height=QCheckBox('Show hover-normalized Z (diagnostic; not raw flight performance)')
        self.normalize_height.setEnabled(False);self.normalize_height.toggled.connect(self.apply_height_view);layout.addWidget(self.normalize_height)
        self.normalize_height.hide()  # Normalized is now the sole evaluation view.
        self.views = QTabWidget(); layout.addWidget(self.views,1)
        self.scene_page = QWidget(); self.scene_layout = QVBoxLayout(self.scene_page); self.scene_layout.setContentsMargins(0,0,0,0)
        self.views.addTab(self.scene_page,'3D comparison')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self.figure = Figure(figsize=(9,5), tight_layout=True); self.canvas = FigureCanvasQTAgg(self.figure)
        self.views.addTab(self.canvas,'Errors over time')
        from .adaptation_progress_page import AdaptationProgressPage
        self.progress_page=AdaptationProgressPage(root)
        self.views.addTab(self.progress_page,'Adaptation progress')
        options = QHBoxLayout(); layout.addLayout(options)
        self.measured = QCheckBox('Measured'); self.measured.setChecked(True)
        self.ghost = QCheckBox('Prediction'); self.ghost.setChecked(True)
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
        self.speed = QComboBox()
        for value in (.25,.5,1.,2.): self.speed.addItem(f'{value:g}×',value)
        self.speed.setCurrentIndex(2); transport.addWidget(self.speed)
        self.timeline = QSlider(Qt.Orientation.Horizontal); self.timeline.setRange(0,0); transport.addWidget(self.timeline,1)
        self.timeline.valueChanged.connect(self.redraw); self.timeline.sliderPressed.connect(self.pause)
        self.time_label = QLabel('0.000 s'); transport.addWidget(self.time_label)
        self.metrics = note(''); layout.addWidget(self.metrics)
        self.timer = QTimer(self); self.timer.setInterval(33); self.timer.timeout.connect(self.tick)
        self.refresh.clicked.connect(self.refresh_batches); self.browse.clicked.connect(self.open_batch)
        self.reference.clicked.connect(self.choose_reference); self.load.clicked.connect(self.load_flight)
        self.batches.currentIndexChanged.connect(self.batch_changed)
        self.takes.currentIndexChanged.connect(self.clear_result)
        self.views.currentChanged.connect(self.redraw)
        self.flight_controls=[self.status,self.flight_legend,self.metrics]+[
            group.itemAt(i).widget() for group in (row,row2,options,transport)
            for i in range(group.count()) if group.itemAt(i).widget() is not None]
        self.views.currentChanged.connect(self.switch_view)
        self.refresh_batches(); self.clear_result()
        self.whip.setChecked(True)

    def switch_view(self,index):
        for widget in self.flight_controls:widget.setVisible(index!=2)
        if index==2:
            self.pause()
            if self.progress_page.real.data is None:self.progress_page.real.refresh()

    def refresh_batches(self):
        previous = self.batches.currentData(); self.batches.blockSignals(True); self.batches.clear()
        for folder in discover_batches(self.root): self.batches.addItem(f'{folder.parent.name} / {folder.name}',str(folder))
        index = self.batches.findData(previous)
        self.batches.setCurrentIndex(max(index,0)); self.batches.blockSignals(False); self.batch_changed()

    def batch_changed(self):
        self.rehearsal = None; self.reference.setText('Choose saved prediction…')
        self.takes.clear()
        if self.batches.currentData(): self.takes.addItems(flight_names(self.batches.currentData()))
        self.clear_result()

    def open_batch(self):
        path = QFileDialog.getExistingDirectory(self,'Open batch containing simulation_csv and flight_take',str(self.root/'rehearsal_csv_and_result_in_real_flight'))
        if path:
            index = self.batches.findData(path)
            if index < 0: self.batches.addItem(Path(path).parent.name+' / '+Path(path).name,path); index = self.batches.count()-1
            self.batches.setCurrentIndex(index)

    def choose_reference(self):
        path = QFileDialog.getExistingDirectory(self,'Choose original saved rehearsal (must match CSV)',str(self.root/'runs/rehearsals'))
        if path:
            self.clear_result(); self.rehearsal = path; self.reference.setText(Path(path).name)

    def clear_result(self, *_):
        self.pause(); self.data = None
        if self.viewer is not None: self.viewer.close(); self.viewer.deleteLater(); self.viewer = None
        self.figure.clear(); self.canvas.draw_idle(); self.timeline.setRange(0,0)
        self.play.setEnabled(False); self.strike.setEnabled(False); self.metrics.setText(''); self.time_label.setText('—')
        self.status.setText('Ready to load selected flight; no comparison loaded.')

    def load_flight(self):
        if self.worker is not None or not self.takes.currentText(): return
        self.clear_result(); self.status.setText('Matching CSV and aligning measured drone logs…')
        self.set_loading(True)
        self.worker = ComparisonLoader(self.root, self.batches.currentData(), self.takes.currentText(), self.rehearsal, self)
        self.worker.loaded.connect(self.loaded); self.worker.failed.connect(self.failed); self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def set_loading(self, busy):
        for widget in (self.batches,self.takes,self.refresh,self.browse,self.reference,self.load): widget.setEnabled(not busy)

    def worker_finished(self):
        self.worker.deleteLater(); self.worker = None; self.set_loading(False)

    def failed(self, message): self.status.setText('Could not load: '+message)

    def loaded(self, data):
        from experimental_data.hover_calibration import normalized_evaluation
        try:normalized_evaluation(data)
        except ValueError as error:
            self.failed(str(error));return
        self.raw_data=data
        self.normalize_height.blockSignals(True);self.normalize_height.setChecked(False);self.normalize_height.blockSignals(False)
        self.normalize_height.setEnabled('hover_normalized' in data)
        self.data = normalized_evaluation(data)
        self.flight_legend.setText('Hover-normalized OptiTrack motion · orange measured motion · blue original saved prediction. All displayed errors use normalized Z.')
        a = data['alignment']; start,end = data['tracking_span']
        coverage = 'complete whip' if start<=0 and end>=data['metadata']['whip_end_s'] else 'WARNING: incomplete whip coverage'
        self.status.setText(f"{data['take']} · {coverage} · tracking {start:.2f} to {end:.2f} s relative to CSV\n"
            f"Saved prediction: {data['rehearsal'].name} · exact CSV match · measured state: OptiTrack {data.get('drone_rigid_body') or ''}. "
            f"Clock alignment: {a['method']}; offset {a['offset_s']:.6f} s. Controller log: commands/timing only.")
        self.play.setEnabled(True); self.strike.setEnabled(data['metadata'].get('predicted_hit_time_s') is not None)
        self.set_range(); self.draw_plots(); self.redraw()
        # A valid per-batch calibration is the user's default working view.
        # Keep raw data untouched so toggling/reloading never applies it twice.
        self.normalize_height.setChecked('hover_normalized' in data)

    def apply_height_view(self,enabled):
        if not hasattr(self,'raw_data') or self.data is None:return
        from experimental_data.hover_calibration import normalized_evaluation
        self.data=normalized_evaluation(self.raw_data)
        bias=self.raw_data['height_calibration']['bias_z_m']
        self.flight_legend.setText(f'Hover-normalized evaluation: drone and all cable Z minus {bias*100:.2f} cm. Ghost/reference target unchanged. All displayed errors use this normalization.')
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
        if index>=self.timeline.maximum(): self.pause()

    def set_page_active(self, active):
        self.active = active
        if not active: self.pause()
        else: self.redraw()

    def shutdown(self):
        self.pause()
        if not self.progress_page.shutdown():return False
        if self.worker is not None:
            if self.worker.isRunning(): return False
        if self.viewer is not None: self.viewer.close(); self.viewer = None
        return True
