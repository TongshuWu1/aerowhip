"""The application's sole offline planner; shared saved-result infrastructure."""
from .cem_page import CEMPage
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QComboBox,QLabel,QDoubleSpinBox,QPushButton
from .research_widgets import note
from simulator.workflow import read_json


class MPPIPage(CEMPage):
    def __init__(self, root):
        super().__init__(root, optimizer='mppi')
        from planning.mppi_force_run import DEFAULTS
        self.defaults=DEFAULTS
        self.name.setText(read_json(self.config_path,{}).get('display_name',DEFAULTS['display_name']))
        for key in ('control_points','position_std_m','duration_std_s','minimum_duration_s','maximum_duration_s'):
            self.search_form.removeRow(self.spins.pop(key))
        for key,label in [('horizon_s','Force sequence horizon [s]'),('action_std','Exploration [normalized force]')]:
            spin=QDoubleSpinBox();spin.setDecimals(6);spin.setRange(.001,5);spin.setValue(DEFAULTS[key])
            spin.setSingleStep(1/30 if key=='horizon_s' else .01);self.spins[key]=spin;self.search_form.addRow(label,spin)
        from .mppi_force_settings import ForceRewardSettings
        from planning.mppi_contract import REWARD_DEFAULTS
        reward=REWARD_DEFAULTS
        old_settings=self.settings_page;criteria=old_settings.values();self.tabs.removeTab(2)
        self.settings_page=ForceRewardSettings(reward);self.tabs.addTab(self.settings_page,'Task & rewards')
        self.settings_page.set_values({k:v for k,v in criteria.items() if k!='reward'})
        old_settings.deleteLater()
        self.settings_page.save_requested.connect(self.save_settings);self.settings_page.load_requested.connect(self.load_run_settings)
        from planning.mppi_contract import ACTION_DEFAULTS,LAUNCH_DEFAULTS
        self.action_spins={}
        for key,label,value in [('x','X force scale [N]',2.),('y','Y force scale [N]',2.),('z','Z force scale [N]',1.6),
                                ('maximum_force_norm_n','Force norm limit [N]',3.2),('minimum_vertical_force_n','Minimum vertical force [N]',0.)]:
            spin=QDoubleSpinBox();spin.setDecimals(5);spin.setRange(0,100);spin.setValue(value)
            self.action_spins[key]=spin;self.search_form.addRow(label,spin)
        self._force_ready=True
        self.apply_settings({'launch_setup':LAUNCH_DEFAULTS})
        self.apply_settings(read_json(self.config_path,{}))
        for i in range(self.seed_controls.count()):
            widget=self.seed_controls.itemAt(i).widget()
            if widget:widget.hide()
        self.tabs.setTabText(0,'1 · Plan setup')
        self.tabs.setTabText(1,'2 · Rehearsal & export')
        self.tabs.setTabText(2,'Task & rewards')
        self.start.setText('Generate new plan with MPPI')
        self.results.hide()  # One saved-result selector, on the rehearsal page.
        self.inspect_button.setText('Go to rehearsal & export')
        self.result_scope=QComboBox()
        self.result_scope.addItems(['Current launch and model','Saved runs — original coordinates'])
        self.tabs.widget(1).layout().insertWidget(0,self.result_scope)
        self.result_banner=note('');self.layout().insertWidget(0,self.result_banner)
        self.rehearse.setText('Play saved prediction')
        self.preview.tabs.tabBar().hide()
        # No duplicate editable-looking launch form in the result viewer.
        for spin in self.preview.start_spins+self.preview.target_spins:
            spin.setReadOnly(True)
        for label in self.preview.findChildren(QLabel):
            if label.text()=='Launch setup':label.setText('Saved trajectory setup')
        for label in self.findChildren(QLabel):
            text=label.text()
            if 'quintic position spline' in text:
                label.setText('MPPI independently optimizes 30 Hz XYZ forces from hover using the fitted drone and cable simulator, then generates a complete FullState trajectory.')
            elif 'fixed covariance' in text:
                label.setText('MPPI samples independent XYZ force actions at every 30 Hz tick. The selected model and both residuals are frozen in each run.')
            elif 'elite fraction' in text or 'control points' in text:
                label.setText('Horizon, exploration, temperature, batch size and iterations are on Plan setup.')
        for label in self.findChildren(QLabel):
            label.setText(label.text().replace('Optimize spline','Plan setup').replace('Changing seeds keeps this setup.','Independent of PPO settings.'))
        self.result_scope.currentIndexChanged.connect(self.setup_changed)
        for spin in self.start_spins+self.target_spins:spin.valueChanged.connect(self.setup_changed)
        self.model_path.textChanged.connect(self.setup_changed)
        self.refresh()

    def result_allowed(self,path,meta):
        if hasattr(self,'result_scope') and self.result_scope.currentIndex()==1:return True
        if meta.get('representation')!='independent_force_30hz_v2':return False
        launch=self.launch_values()
        try:
            if any(not np.allclose(meta[key],value,rtol=0,atol=1e-8) for key,value in launch.items()):return False
            model=Path(self.model_path.text())
            if not model.is_absolute():model=self.root/model
            saved=read_json(path/'model.json')
            current=read_json(model)
            # Ignore only asset file locations; compare physical model contents,
            # including both residual hashes. A portable job can be the source.
            for config in (saved,current):
                for field in ('motion_residual','fullstate_execution'):config[field].pop('checkpoint',None)
            return saved==current
        except (OSError,ValueError,KeyError,TypeError):return False

    def setup_changed(self,*_):
        if self.job.running:return
        self.preview.clear_result()
        self.refresh()

    def refresh(self):
        # MPPI owns its run library. Do not discover PPO rehearsals or seeds.
        if self.job.running:return
        selected=self.results.currentData()
        for combo in (self.results,self.rehearsal_results):combo.blockSignals(True);combo.clear()
        for path in sorted(self.runs_path.glob('*/rehearsal.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            meta=read_json(path,{})
            if meta.get('schema') not in ('mppi_fullstate_30hz_v1','mppi_force_fullstate_30hz_v1') or not self.result_allowed(path.parent,meta):continue
            for combo in (self.results,self.rehearsal_results):
                combo.addItem(meta.get('display_name',path.parent.name)+' / '+path.parent.name,str(path.parent.resolve()))
        index=self.results.findData(selected)
        for combo in (self.results,self.rehearsal_results):combo.setCurrentIndex(max(index,0));combo.blockSignals(False)
        self.rehearse.setEnabled(self.results.count()>0);self.draw_history(self.results.currentData())
        self.seed_changed()
        if hasattr(self,'result_banner'):
            if not self.results.currentData():self.preview.clear_result()
            self.update_result_banner()

    def seed_changed(self):
        self.start.setEnabled(not self.job.running)
        self.seed_note.setText('Independent initialization: settled hanging hover. No policy, checkpoint or prior rehearsal is required. Every candidate is a new 30 Hz XYZ force sequence.')

    def settings_values(self):
        if not getattr(self,'_force_ready',False):return super().settings_values()
        from planning.mppi_force_run import DEFAULTS,validate_settings
        settings=dict(DEFAULTS,**{k:s.value() for k,s in self.spins.items()})
        settings.update(display_name=self.name.text().strip() or DEFAULTS['display_name'],device=self.device.currentText(),
            model_path=self.model_path.text().strip(),launch_setup=self.launch_values(),**self.settings_page.values())
        # UI rounds 1/30 increments to six decimals; freeze an exact packet grid.
        settings['horizon_s']=round(settings['horizon_s']*30)/30
        settings['action']={k:self.action_spins[k].value() for k in ('maximum_force_norm_n','minimum_vertical_force_n')}
        settings['action']['delta_force_scale_n']=[self.action_spins[k].value() for k in 'xyz']
        validate_settings(settings)
        return settings

    def apply_settings(self,settings):
        # The shared shell builds its legacy reward widget before replacement.
        # Never interpret MPPI's force objective as the historical spline score.
        values=settings if getattr(self,'_force_ready',False) else {k:v for k,v in settings.items() if k!='reward'}
        super().apply_settings(values)
        if hasattr(self,'action_spins'):
            action=settings.get('action',{})
            for key in ('maximum_force_norm_n','minimum_vertical_force_n'):
                if key in action:self.action_spins[key].setValue(action[key])
            for key,value in zip('xyz',action.get('delta_force_scale_n',[])):self.action_spins[key].setValue(value)

    def load_run_settings(self):
        path=self.results.currentData()
        if path and read_json(Path(path)/'mppi.json',{}).get('representation')!='independent_force_30hz_v2':
            self.settings_page.status.setText('Historical planner settings cannot replace the independent MPPI contract. Enter the desired settings on Plan setup.');return
        super().load_run_settings()

    def prepare_planner_job(self,*args):
        from planning.mppi_force_run import prepare_job
        return prepare_job(*args)

    @staticmethod
    def coordinates(values):
        return '['+', '.join(f'{v:g}' for v in values)+']'

    def update_result_banner(self):
        if not hasattr(self,'result_banner'):return
        launch=self.launch_values();path=self.results.currentData()
        if hasattr(self,'result_scope') and self.result_scope.currentIndex()==1:
            meta=read_json(Path(path)/'rehearsal.json',{}) if path else {}
            text='SAVED RUN — original coordinates. '
            if meta:text+=f'{Path(path).name}: start {self.coordinates(meta["initial_tracking_origin_m"])} → target {self.coordinates(meta["target_position_m"])} m. '
            text+='Changing Plan setup does not move this saved CSV.'
        else:
            text=f'CURRENT SETUP: start {self.coordinates(launch["initial_tracking_origin_m"])} → target {self.coordinates(launch["target_position_m"])} m. '
            text+=f'Saved matching result: {Path(path).name}.' if path else 'No completed trajectory for this launch and model. Generate a new plan; no CSV is available.'
            if not path:
                for status in sorted(self.runs_path.glob('*/run.json'),key=lambda p:p.stat().st_mtime,reverse=True):
                    settings=read_json(status.parent/'mppi.json',{})
                    if settings.get('launch_setup')==launch and settings.get('representation')=='independent_force_30hz_v2':
                        run=read_json(status,{})
                        text+=f' Latest attempt: {run.get("status","unknown")}. '+run.get('message','')
                        break
        self.result_banner.setText(text)
        self.rehearse.setEnabled(bool(path))
        self.preview.setVisible(bool(path))

    def inspect(self):
        if self.job.running:
            self.preview.clear_result();self.preview.hide()
            if self.tabs.currentIndex()!=1:self.tabs.setCurrentIndex(1)
            return
        if not self.results.currentData():
            self.preview.clear_result()
            if self.tabs.currentIndex()!=1:self.tabs.setCurrentIndex(1)
        else:super().inspect()
        self.update_result_banner()

    def result_changed(self,index):
        super().result_changed(index)
        if not self.results.currentData():self.preview.clear_result()
        self.update_result_banner()

    def start_run(self):
        if not self.job.running:
            self.preview.clear_result()
            self.result_scope.setCurrentIndex(0)
        if self.job.running:return
        from simulator.workflow import stamp
        directory=self.runs_path/stamp()
        try:
            command=self.prepare_planner_job(self.root,directory,self.settings_values())
            self.job.start(directory,command);self.directory=directory;self.set_running(True)
            self.status.setText('Evaluating hover initialization and first MPPI population…')
        except (OSError,ValueError,KeyError) as error:self.status.setText('Cannot start: '+str(error))
        if self.job.running:
            self.preview.hide()
            self.result_banner.setText('Generating a NEW trajectory for the frozen launch and model. Rehearsal/CSV will be available when this run completes.')

    def set_running(self,running):
        super().set_running(running)
        self.start.setEnabled(not running)
        for spin in getattr(self,'action_spins',{}).values():spin.setEnabled(not running)
        if hasattr(self,'result_scope'):self.result_scope.setEnabled(not running)
        self.rehearsal_results.setEnabled(not running)

    def open_rehearsal(self):
        self.result_scope.setCurrentIndex(1)
        super().open_rehearsal()
        self.update_result_banner()
