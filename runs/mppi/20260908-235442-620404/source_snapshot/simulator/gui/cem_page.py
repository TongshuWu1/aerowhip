"""Offline CEM planning, iteration diagnostics and complete trajectory inspection."""
from pathlib import Path
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QComboBox,
    QLineEdit,QPushButton,QSpinBox,QDoubleSpinBox,QTabWidget,QFileDialog,QSplitter,QGroupBox)
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from simulator.workflow import read_json,stamp
from planning.cem_run import DEFAULTS,MPPI_DEFAULTS,prepare_job,validate_settings
from planning.cem_objective import task_with_settings
from planning.cem_launch import resolve_launch_setup
from .research_widgets import note,BackgroundJob
from .rehearsal_workspace import RehearsalWorkspace


class CEMPage(QWidget):
    def __init__(self,root,optimizer='cem'):
        super().__init__();self.root=Path(root);self.directory=None;self.active=False
        self.optimizer=optimizer;self.planner=optimizer.upper()
        self.defaults=MPPI_DEFAULTS if optimizer=='mppi' else DEFAULTS
        self.config_path=self.root/f'config/{optimizer}.json'
        self.runs_path=self.root/f'runs/{optimizer}'
        layout=QVBoxLayout(self);self.tabs=QTabWidget();layout.addWidget(self.tabs)
        setup=QWidget();body=QVBoxLayout(setup);self.tabs.addTab(setup,'Optimize spline')
        body.addWidget(note('Optimize a quintic position spline and duration. Candidates use the fitted drone, cable and both residuals. PPO supplies only the starting trajectory; no policy training is needed.'))
        row=QHBoxLayout();self.seed=QComboBox();row.addWidget(self.seed,1)
        refresh=QPushButton('Refresh seeds');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        browse=QPushButton('Browse seed…');browse.clicked.connect(self.browse_seed);row.addWidget(browse);body.addLayout(row)
        split=QSplitter(Qt.Orientation.Horizontal);body.addWidget(split,1)
        from PySide6.QtWidgets import QScrollArea
        controls=QWidget();form=QFormLayout(controls)
        if optimizer=='mppi':
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(controls);split.addWidget(scroll)
        self.name=QLineEdit(self.defaults['display_name']);form.addRow('Run name',self.name)
        self.model_path=QLineEdit(self.defaults.get('model_path',''))
        if optimizer=='mppi':
            model_row=QHBoxLayout();model_row.addWidget(self.model_path)
            self.browse_model=QPushButton('Browse model…');self.browse_model.clicked.connect(self.choose_model);model_row.addWidget(self.browse_model)
            form.addRow('Model JSON (M1 default)',model_row)
            form.addRow(note('Offline MPPI · fixed covariance · importance-weighted updates. This model is frozen in each new run, independently of the seed and PPO.'))
        launch_group=QGroupBox(f'{self.planner} launch setup');launch_form=QFormLayout(launch_group)
        self.start_spins=[];self.target_spins=[]
        for label,spins in [('Tracked-origin start [m]',self.start_spins),('Target position [m]',self.target_spins)]:
            launch_form.addRow(note(label))
            for axis in 'XYZ':
                spin=QDoubleSpinBox();spin.setDecimals(6);spin.setRange(-20,20);spin.setSingleStep(.01);spins.append(spin)
                launch_form.addRow(axis,spin)
        launch_form.addRow(note('CEM only · world XYZ in metres. Settled hover and hanging cable. Changing seeds keeps this setup.'))
        self.save_launch=QPushButton(f'Save {self.planner} launch setup');self.save_launch.clicked.connect(self.save_launch_setup)
        launch_form.addRow(self.save_launch);form.addRow(launch_group)
        self.spins={}
        labels={'population':'Random candidates (+2 for MPPI)','iterations':'Optimizer iterations','elite_fraction':'Elite fraction','control_points':'Spline control points',
                'position_std_m':'Position exploration [m]','duration_std_s':'Duration exploration [s]',
                'minimum_duration_s':'Minimum whip time [s]','maximum_duration_s':'Maximum whip time [s]',
                'maximum_height_m':'Maximum height [m]','minimum_height_m':'Minimum cable/drone height [m]',
                'random_seed':'Random seed'}
        if optimizer=='mppi':
            labels.pop('elite_fraction');labels['temperature']='MPPI temperature [score units]'
        for key,label in labels.items():
            integer=key in ('population','iterations','control_points','random_seed')
            spin=QSpinBox() if integer else QDoubleSpinBox()
            spin.setRange(0,1000000 if integer else 20)
            if not integer:spin.setDecimals(3);spin.setSingleStep(.01)
            if key=='elite_fraction':spin.setRange(.001,.5)
            if key=='temperature':spin.setRange(.001,100000);spin.setSingleStep(1)
            spin.setValue(self.defaults[key]);self.spins[key]=spin;form.addRow(label,spin)
        self.device=QComboBox();self.device.addItems(['cuda','cpu']);form.addRow('Device',self.device)
        self.start=QPushButton(f'Optimize with {self.planner}');self.start.setObjectName('primaryButton');self.start.clicked.connect(self.start_run)
        self.stop=QPushButton('Stop optimization');self.stop.setEnabled(False);self.stop.clicked.connect(self.stop_run)
        right=QWidget();rl=QVBoxLayout(right);split.addWidget(right);split.setSizes([330,700])
        rl.addWidget(note('Tune the objective and hit criteria in Task & rewards. Complete recovery must also fit the height and command limits before export.'))
        self.seed_note=note('');rl.addWidget(self.seed_note)
        self.figure=Figure(figsize=(7,4),layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure);rl.addWidget(self.canvas,1)
        resultrow=QHBoxLayout();self.results=QComboBox();resultrow.addWidget(self.results,1)
        inspect=QPushButton('Open rehearsal & export');inspect.clicked.connect(self.inspect);resultrow.addWidget(inspect);rl.addLayout(resultrow)
        self.status=note('Ready');rl.addWidget(self.status)
        actionbar=QHBoxLayout();actionbar.addWidget(self.start);actionbar.addWidget(self.stop);body.addLayout(actionbar)
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
        from .cem_settings_page import CEMSettingsPage
        self.settings_page=CEMSettingsPage(self.planner);self.tabs.addTab(self.settings_page,'Task & rewards')
        self.settings_page.save_requested.connect(self.save_settings)
        self.settings_page.load_requested.connect(self.load_run_settings)
        saved=read_json(self.config_path,{})
        saved.setdefault('launch_setup',resolve_launch_setup())
        self._criteria_initialized=bool(saved.get('success'))
        self.apply_settings(saved)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.seed.currentIndexChanged.connect(self.seed_changed)
        self.results.currentIndexChanged.connect(lambda _:self.draw_history(self.results.currentData()) if not self.job.running else None)
        self.results.currentIndexChanged.connect(self.rehearsal_results.setCurrentIndex)
        self.rehearsal_results.currentIndexChanged.connect(self.result_changed)
        self.refresh()
        if self.optimizer=='mppi':
            from PySide6.QtWidgets import QLabel
            for widget in self.findChildren(QLabel)+self.findChildren(QPushButton):
                widget.setText(widget.text().replace('CEM','MPPI'))

    def choose_model(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose fitted model JSON',str(self.root/'data/model_candidates'),'JSON (*.json)')
        if path:self.model_path.setText(path)

    def refresh(self):
        if self.job.running:return
        old=self.seed.currentData();selected=self.results.currentData();self.seed.blockSignals(True);self.seed.clear()
        self.results.blockSignals(True);self.rehearsal_results.blockSignals(True);self.results.clear();self.rehearsal_results.clear()
        paths=sorted((self.root/'runs/rehearsals').glob('*/rehearsal.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        paths+=sorted((self.root/'runs/cem').glob('*/rehearsal.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        paths+=sorted((self.root/'runs/mppi').glob('*/rehearsal.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        for path in paths:
            meta=read_json(path,{})
            if meta.get('schema') not in ('research_fullstate_30hz_v1','cem_fullstate_30hz_v1','mppi_fullstate_30hz_v1'):continue
            self.seed.addItem(path.parent.name,str(path.parent.resolve()))
            if meta.get('schema')==f'{self.optimizer}_fullstate_30hz_v1':
                label=meta.get('display_name',path.parent.name)+' / '+path.parent.name
                for combo in (self.results,self.rehearsal_results):combo.addItem(label,str(path.parent.resolve()))
        index=self.results.findData(selected)
        for combo in (self.results,self.rehearsal_results):combo.setCurrentIndex(max(index,0));combo.blockSignals(False)
        self.rehearse.setEnabled(self.results.count()>0);self.draw_history(self.results.currentData())
        if old is None and self.optimizer=='mppi':old=str((self.root/'runs/rehearsals/20260908-203914-039721').resolve())
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
            if not self._criteria_initialized:
                self.settings_page.set_values({k:task[k] for k in ('success','desired_strike_direction_world') if k in task});self._criteria_initialized=True
            self.seed_note.setText(f'Seed whip {meta["whip_end_s"]:.2f} s · initial level hover and hanging cable; seed is refitted and re-simulated. Next-run hit criteria are shown in Task & rewards.')
        except (OSError,ValueError,KeyError) as error:self.status.setText(str(error));self.start.setEnabled(False)

    def start_run(self):
        if self.job.running:return
        directory=self.runs_path/stamp()
        try:
            settings=self.settings_values()
            command=prepare_job(self.root,self.seed.currentData(),directory,settings,
                [s.value() for s in self.start_spins],[s.value() for s in self.target_spins])
            self.job.start(directory,command);self.directory=directory;self.set_running(True);self.status.setText('Evaluating the seed and first population…')
        except (OSError,ValueError,KeyError) as error:self.status.setText('Cannot start: '+str(error))

    def set_running(self,running):
        self.settings_page.setEnabled(not running)
        for widget in [self.seed,self.name,self.device,self.save_launch,self.model_path,*self.spins.values(),*self.start_spins,*self.target_spins]:widget.setEnabled(not running)
        if hasattr(self,'browse_model'):self.browse_model.setEnabled(not running)
        self.start.setEnabled(not running and bool(self.seed.currentData()));self.stop.setEnabled(running)

    def settings_values(self):
        settings={k:s.value() for k,s in self.spins.items()}
        settings.update(display_name=self.name.text().strip() or self.defaults['display_name'],device=self.device.currentText(),**self.settings_page.values())
        settings['optimizer']=self.optimizer
        if self.optimizer=='mppi':
            if not self.model_path.text().strip():raise ValueError('Select a fitted model JSON for MPPI.')
            settings['model_path']=self.model_path.text().strip()
        settings['launch_setup']=self.launch_values()
        validate_settings(dict(self.defaults,**settings))
        task_with_settings({'success':{},'desired_strike_direction_world':[1,0,0]},settings)
        return settings

    def apply_settings(self,settings):
        if settings.get('model_path'):self.model_path.setText(settings['model_path'])
        if 'launch_setup' in settings:
            launch=resolve_launch_setup(settings['launch_setup'])
            for spins,key in [(self.start_spins,'initial_tracking_origin_m'),(self.target_spins,'target_position_m')]:
                for spin,value in zip(spins,launch[key]):spin.setValue(value)
        for key,spin in self.spins.items():
            if key in settings:spin.setValue(settings[key])
        if settings.get('device') in ('cuda','cpu'):self.device.setCurrentText(settings['device'])
        self.settings_page.set_values(settings)

    def launch_values(self):
        return resolve_launch_setup(dict(initial_tracking_origin_m=[s.value() for s in self.start_spins],
                                         target_position_m=[s.value() for s in self.target_spins]))

    def save_launch_setup(self):
        if self.job.running:return
        from experimental_data.io import atomic_json
        try:
            settings=read_json(self.config_path,{})
            settings['launch_setup']=self.launch_values()
            atomic_json(self.config_path,settings)
            self.status.setText(f'Saved {self.planner} launch setup for new plans.')
        except (ValueError,OSError) as error:self.status.setText('Cannot save launch setup: '+str(error))

    def save_settings(self):
        from experimental_data.io import atomic_json
        try:
            settings=self.settings_values();atomic_json(self.config_path,settings)
            self.settings_page.status.setText(f'Saved {self.planner} launch setup, reward, hit criteria and search settings for new runs.')
        except (ValueError,OSError) as error:self.settings_page.status.setText('Cannot save: '+str(error))

    def load_run_settings(self):
        path=self.results.currentData()
        if not path:self.settings_page.status.setText('Select a saved planner run on Optimize spline or Rehearsal & Export first.');return
        try:
            settings=read_json(Path(path)/f'{self.optimizer}.json');task=read_json(Path(path)/'task.json')
            if self.optimizer=='mppi':settings['model_path']=str(Path(path)/'model.json')
            if 'launch_setup' not in settings:
                meta=read_json(Path(path)/'rehearsal.json')
                settings['launch_setup']={key:meta[key] for key in ('initial_tracking_origin_m','target_position_m')}
            settings.setdefault('reward',{})
            for key in ('success','desired_strike_direction_world'):settings.setdefault(key,task[key])
            self.apply_settings(dict(self.defaults,**settings));self._criteria_initialized=True
            self.settings_page.status.setText('Loaded settings from '+Path(path).name+'. Save to keep them after restarting.')
        except (ValueError,OSError,KeyError) as error:self.settings_page.status.setText('Cannot load: '+str(error))

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
            axes[0].text(.5,.5,f'Start {self.planner} to see optimization progress',ha='center',transform=axes[0].transAxes)
        x=[r['iteration'] for r in history]
        axes[0].plot(x,[r['best_score'] for r in history],color='#2563eb');axes[0].set(ylabel='Best objective')
        axes[1].plot(x,[r['success_fraction']*100 for r in history],label='Valid hits',color='#0d9488')
        axes[1].plot(x,[r['feasible_fraction']*100 for r in history],label='Feasible candidates',color='#ea580c')
        if self.optimizer=='mppi' and history:
            axes[1].plot(x,[r.get('effective_sample_size',0)/max(1,r.get('proposal_samples',1))*100 for r in history],label='Effective samples',linestyle=':')
        axes[1].set(xlabel=f'{self.planner} iteration',ylabel='Population [%]',ylim=(0,100));axes[1].legend();self.canvas.draw_idle()

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
        path=QFileDialog.getExistingDirectory(self,f'Open a completed {self.planner} rehearsal',str(self.runs_path))
        if not path:return
        if read_json(Path(path)/'rehearsal.json',{}).get('schema')!=f'{self.optimizer}_fullstate_30hz_v1':
            self.status.setText(f'Choose a completed {self.planner} result folder.');return
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
