"""Offline CEM planning, iteration diagnostics and complete trajectory inspection."""
from pathlib import Path
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QComboBox,
    QLineEdit,QPushButton,QSpinBox,QDoubleSpinBox,QTabWidget,QFileDialog,QSplitter)
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from simulator.workflow import read_json,stamp
from planning.cem_run import DEFAULTS,prepare_job
from .research_widgets import note,BackgroundJob
from .rehearsal_workspace import RehearsalWorkspace


class CEMPage(QWidget):
    def __init__(self,root):
        super().__init__();self.root=Path(root);self.directory=None;self.active=False
        layout=QVBoxLayout(self);self.tabs=QTabWidget();layout.addWidget(self.tabs)
        setup=QWidget();body=QVBoxLayout(setup);self.tabs.addTab(setup,'Optimize spline')
        body.addWidget(note('Optimize a quintic position spline and duration. Candidates use the fitted drone, cable and both residuals. PPO supplies only the starting trajectory; no policy training is needed.'))
        row=QHBoxLayout();self.seed=QComboBox();row.addWidget(self.seed,1)
        refresh=QPushButton('Refresh seeds');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        browse=QPushButton('Browse seed…');browse.clicked.connect(self.browse_seed);row.addWidget(browse);body.addLayout(row)
        split=QSplitter(Qt.Orientation.Horizontal);body.addWidget(split,1)
        from PySide6.QtWidgets import QScrollArea
        controls=QWidget();form=QFormLayout(controls)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(controls);split.addWidget(scroll)
        self.name=QLineEdit('CEM spline');form.addRow('Run name',self.name)
        self.spins={}
        labels={'population':'Parallel candidates','iterations':'CEM iterations','control_points':'Spline control points',
                'position_std_m':'Position exploration [m]','duration_std_s':'Duration exploration [s]',
                'minimum_duration_s':'Minimum whip time [s]','maximum_duration_s':'Maximum whip time [s]',
                'maximum_height_m':'Maximum height [m]','minimum_height_m':'Minimum cable/drone height [m]',
                'random_seed':'Random seed'}
        for key,label in labels.items():
            integer=key in ('population','iterations','control_points','random_seed')
            spin=QSpinBox() if integer else QDoubleSpinBox()
            spin.setRange(0,1000000 if integer else 20)
            if not integer:spin.setDecimals(3);spin.setSingleStep(.01)
            spin.setValue(DEFAULTS[key]);self.spins[key]=spin;form.addRow(label,spin)
        self.start_spins=[];self.target_spins=[]
        for label,spins in [('Tracked start',self.start_spins),('Target',self.target_spins)]:
            for axis in 'XYZ':
                spin=QDoubleSpinBox();spin.setDecimals(6);spin.setRange(-20,20);spin.setSingleStep(.01);spins.append(spin)
                form.addRow(f'{label} {axis} [m]',spin)
        self.device=QComboBox();self.device.addItems(['cuda','cpu']);form.addRow('Device',self.device)
        self.start=QPushButton('Optimize with CEM');self.start.setObjectName('primaryButton');self.start.clicked.connect(self.start_run);form.addRow(self.start)
        self.stop=QPushButton('Stop optimization');self.stop.setEnabled(False);self.stop.clicked.connect(self.stop_run);form.addRow(self.stop)
        right=QWidget();rl=QVBoxLayout(right);split.addWidget(right);split.setSizes([330,700])
        rl.addWidget(note('Objective: valid directed first-tip hit; earlier hits; smaller miss distance; useful tip speed; smooth acceleration. Hit gate is inherited from the seed. Complete recovery must also fit the height and command limits before export.'))
        self.seed_note=note('');rl.addWidget(self.seed_note)
        self.figure=Figure(figsize=(7,4),layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure);rl.addWidget(self.canvas,1)
        resultrow=QHBoxLayout();self.results=QComboBox();resultrow.addWidget(self.results,1)
        inspect=QPushButton('Open rehearsal & export');inspect.clicked.connect(self.inspect);resultrow.addWidget(inspect);rl.addLayout(resultrow)
        self.status=note('Ready');rl.addWidget(self.status)
        self.job=BackgroundJob(root);body.addWidget(self.job);self.job.progress.connect(self.progress);self.job.finished.connect(self.finished)
        rehearsal=QWidget();rehearsal_layout=QVBoxLayout(rehearsal)
        rehearsal_layout.addWidget(note('CEM generates the complete rehearsal during optimization. Select a saved CEM run to replay its exact prediction and export its complete 30 Hz CSV. Saved results retain their original launch coordinates.'))
        selection=QHBoxLayout();self.rehearsal_results=QComboBox();selection.addWidget(self.rehearsal_results,1)
        self.rehearse=QPushButton('Start rehearsal');self.rehearse.clicked.connect(self.start_rehearsal);selection.addWidget(self.rehearse)
        refresh_results=QPushButton('Refresh');refresh_results.clicked.connect(self.refresh);selection.addWidget(refresh_results)
        open_result=QPushButton('Open saved CEM…');open_result.clicked.connect(self.open_rehearsal);selection.addWidget(open_result)
        rehearsal_layout.addLayout(selection)
        self.preview=RehearsalWorkspace(root,inspection_only=True);rehearsal_layout.addWidget(self.preview,1)
        self.tabs.addTab(rehearsal,'Rehearsal & Export')
        self.tabs.currentChanged.connect(self.tab_changed)
        self.seed.currentIndexChanged.connect(self.seed_changed)
        self.results.currentIndexChanged.connect(lambda _:self.draw_history(self.results.currentData()) if not self.job.running else None)
        self.results.currentIndexChanged.connect(self.rehearsal_results.setCurrentIndex)
        self.rehearsal_results.currentIndexChanged.connect(self.result_changed)
        self.refresh()

    def refresh(self):
        if self.job.running:return
        old=self.seed.currentData();selected=self.results.currentData();self.seed.blockSignals(True);self.seed.clear()
        self.results.blockSignals(True);self.rehearsal_results.blockSignals(True);self.results.clear();self.rehearsal_results.clear()
        paths=sorted((self.root/'runs/rehearsals').glob('*/rehearsal.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        paths+=sorted((self.root/'runs/cem').glob('*/rehearsal.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        for path in paths:
            meta=read_json(path,{})
            if meta.get('schema') not in ('research_fullstate_30hz_v1','cem_fullstate_30hz_v1'):continue
            self.seed.addItem(path.parent.name,str(path.parent.resolve()))
            if meta.get('planner'):
                label=meta.get('display_name',path.parent.name)+' / '+path.parent.name
                for combo in (self.results,self.rehearsal_results):combo.addItem(label,str(path.parent.resolve()))
        index=self.results.findData(selected)
        for combo in (self.results,self.rehearsal_results):combo.setCurrentIndex(max(index,0));combo.blockSignals(False)
        self.rehearse.setEnabled(self.results.count()>0);self.draw_history(self.results.currentData())
        index=self.seed.findData(old);self.seed.setCurrentIndex(index if index>=0 else 0);self.seed.blockSignals(False);self.seed_changed()

    def browse_seed(self):
        path=QFileDialog.getExistingDirectory(self,'Choose a completed native rehearsal',str(self.root/'runs/rehearsals'))
        if path:self.seed.addItem(Path(path).name,path);self.seed.setCurrentIndex(self.seed.count()-1)

    def seed_changed(self):
        path=self.seed.currentData()
        self.start.setEnabled(bool(path) and not self.job.running)
        if not path:return
        try:
            meta=read_json(Path(path)/'rehearsal.json');task=read_json(Path(path)/'task.json')
            from simulator.launch_setup import launch_positions
            origin,target=launch_positions(self.root,meta['initial_tracking_origin_m'],meta['target_position_m'])
            for spins,values in [(self.start_spins,origin),(self.target_spins,target)]:
                for spin,value in zip(spins,values):spin.setValue(value)
            gate=task['success'];self.seed_note.setText(f'Seed whip {meta["whip_end_s"]:.2f} s · hit radius {gate["tip_target_distance_m"]*100:g} cm · minimum directed speed {gate["minimum_directed_tip_speed_m_s"]:g} m/s · angle ≤ {gate["maximum_tip_velocity_to_desired_direction_error_deg"]:g}°. Initial level hover and hanging cable; seed is refitted and re-simulated.')
        except (OSError,ValueError,KeyError) as error:self.status.setText(str(error));self.start.setEnabled(False)

    def start_run(self):
        if self.job.running:return
        settings={k:s.value() for k,s in self.spins.items()};settings.update(display_name=self.name.text().strip() or 'CEM spline',device=self.device.currentText())
        directory=self.root/'runs/cem'/stamp()
        try:
            command=prepare_job(self.root,self.seed.currentData(),directory,settings,
                [s.value() for s in self.start_spins],[s.value() for s in self.target_spins])
            self.job.start(directory,command);self.directory=directory;self.set_running(True);self.status.setText('Evaluating the seed and first population…')
        except (OSError,ValueError,KeyError) as error:self.status.setText('Cannot start: '+str(error))

    def set_running(self,running):
        for widget in [self.seed,self.name,self.device,*self.spins.values(),*self.start_spins,*self.target_spins]:widget.setEnabled(not running)
        self.start.setEnabled(not running and bool(self.seed.currentData()));self.stop.setEnabled(running)

    def stop_run(self):
        if self.job.running:
            (self.directory/'STOP_REQUESTED').write_text('User requested stop',encoding='utf-8');self.status.setText('Stopping at the next simulation checkpoint…');self.stop.setEnabled(False)

    def progress(self,value):
        self.status.setText(value.get('label','Running'))
        self.draw_history(self.directory)

    def draw_history(self,directory):
        history=read_json(Path(directory)/'history.json',[]) if directory else []
        self.figure.clear();axes=self.figure.subplots(2,1)
        if not history:
            axes[0].text(.5,.5,'Start CEM to see optimization progress',ha='center',transform=axes[0].transAxes)
        x=[r['iteration'] for r in history]
        axes[0].plot(x,[r['best_score'] for r in history],color='#2563eb');axes[0].set(ylabel='Best objective')
        axes[1].plot(x,[r['success_fraction']*100 for r in history],label='Valid hits',color='#0d9488')
        axes[1].plot(x,[r['feasible_fraction']*100 for r in history],label='Feasible candidates',color='#ea580c')
        axes[1].set(xlabel='CEM iteration',ylabel='Population [%]',ylim=(0,100));axes[1].legend();self.canvas.draw_idle()

    def finished(self,code):
        self.set_running(False);self.refresh();run=read_json(self.directory/'run.json',{})
        self.progress(dict(label=run.get('status','FAILED')))
        if code:self.status.setText(run.get('message','Optimization failed; open the job log.'));return
        if run.get('status')=='STOPPED':self.status.setText('Stopped. Candidate history is saved; no new CSV exported.');return
        index=self.results.findData(str(self.directory.resolve()));self.results.setCurrentIndex(index)
        self.status.setText('Complete. Inspect the predicted hit, full recovery, PVA and CSV before export.')
        self.inspect()

    def inspect(self):
        path=self.results.currentData()
        if path:
            try:
                if self.preview.directory!=Path(path):self.preview.clear_result();self.preview.load_result(path)
                self.tabs.setCurrentIndex(1)
            except (OSError,ValueError,KeyError) as error:self.status.setText(str(error))

    def result_changed(self,index):
        self.results.setCurrentIndex(index)
        if self.tabs.currentIndex()==1:self.inspect()

    def tab_changed(self,index):
        if index==1:self.inspect()
        self.set_page_active(self.active)

    def start_rehearsal(self):
        self.inspect()
        if self.preview.arrays is not None:
            self.preview.playing=False;self.preview.timeline.setValue(0);self.preview.toggle_play()

    def open_rehearsal(self):
        path=QFileDialog.getExistingDirectory(self,'Open a completed CEM rehearsal',str(self.root/'runs/cem'))
        if not path:return
        if read_json(Path(path)/'rehearsal.json',{}).get('schema')!='cem_fullstate_30hz_v1':
            self.status.setText('Choose a completed CEM result folder.');return
        index=self.results.findData(path)
        if index<0:
            for combo in (self.results,self.rehearsal_results):
                combo.blockSignals(True);combo.addItem(Path(path).name,path);combo.blockSignals(False)
            index=self.results.count()-1
        self.results.setCurrentIndex(index);self.rehearsal_results.setCurrentIndex(index);self.rehearse.setEnabled(True);self.inspect()

    def set_page_active(self,active):
        self.active=active;self.preview.set_page_active(active and self.tabs.currentIndex()==1)

    def shutdown(self):
        return self.preview.shutdown()
