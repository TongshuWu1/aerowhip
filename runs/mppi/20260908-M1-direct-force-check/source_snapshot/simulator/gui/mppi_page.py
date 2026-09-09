"""The application's sole offline planner; shared saved-result infrastructure."""
from .cem_page import CEMPage
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QComboBox,QLabel,QDoubleSpinBox
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
        for key,label in [('horizon_s','Force sequence horizon [s]'),('action_std','Exploration [normalized PPO action]')]:
            spin=QDoubleSpinBox();spin.setDecimals(6);spin.setRange(.001,5);spin.setValue(DEFAULTS[key])
            spin.setSingleStep(1/30 if key=='horizon_s' else .01);self.spins[key]=spin;self.search_form.addRow(label,spin)
        from .mppi_force_settings import ForceRewardSettings
        source=self.seed.currentData()
        reward=read_json(Path(source)/'ppo.json')['reward'] if source and (Path(source)/'ppo.json').exists() else read_json(self.root/'config/research_30hz/ppo.json',{}).get('reward',{})
        old_settings=self.settings_page;self.tabs.removeTab(2)
        self.settings_page=ForceRewardSettings(reward);self.tabs.addTab(self.settings_page,'PPO reward settings')
        old_settings.deleteLater()
        self.settings_page.save_requested.connect(self.save_settings);self.settings_page.load_requested.connect(self.load_run_settings)
        self._force_ready=True
        self.apply_settings(read_json(self.config_path,{}))
        self.tabs.setTabText(0,'1 · Plan setup')
        self.tabs.setTabText(1,'2 · Rehearsal & export')
        self.tabs.setTabText(2,'PPO reward settings')
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
                label.setText('Optimize 30 Hz XYZ force actions through the same virtual planning, FullState conversion and fitted execution scoring as PPO.')
            elif 'fixed covariance' in text:
                label.setText('MPPI samples independent XYZ force actions at every 30 Hz tick. The selected model and both residuals are frozen in each run.')
            elif 'elite fraction' in text or 'control points' in text:
                label.setText('Horizon, exploration, temperature, batch size and iterations are on Plan setup.')
        self.result_scope.currentIndexChanged.connect(self.setup_changed)
        for spin in self.start_spins+self.target_spins:spin.valueChanged.connect(self.setup_changed)
        self.model_path.textChanged.connect(self.setup_changed)
        self.refresh()

    def result_allowed(self,path,meta):
        if hasattr(self,'result_scope') and self.result_scope.currentIndex()==1:return True
        if meta.get('schema')!='mppi_force_fullstate_30hz_v1':return False
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
        super().refresh()
        if hasattr(self,'seed'):
            self.seed.blockSignals(True)
            for i in reversed(range(self.seed.count())):
                try:
                    with np.load(Path(self.seed.itemData(i))/'rehearsal.npz') as data:valid='virtual_force_n' in data.files
                except (OSError,ValueError):valid=False
                if not valid:self.seed.removeItem(i)
            self.seed.blockSignals(False);self.seed_changed()
        if hasattr(self,'result_banner'):
            if not self.results.currentData():self.preview.clear_result()
            self.update_result_banner()

    def seed_changed(self):
        super().seed_changed()
        if hasattr(self,'seed_note'):
            self.seed_note.setText('Seed supplies 30 Hz forces and its PPO action/reward/termination contract. Forces are evaluated afresh at the new launch and target; no position spline or translated trajectory is used.')

    def settings_values(self):
        if not getattr(self,'_force_ready',False):return super().settings_values()
        from planning.mppi_force_run import DEFAULTS,validate_settings
        settings=dict(DEFAULTS,**{k:s.value() for k,s in self.spins.items()})
        settings.update(display_name=self.name.text().strip() or DEFAULTS['display_name'],device=self.device.currentText(),
            model_path=self.model_path.text().strip(),launch_setup=self.launch_values(),**self.settings_page.values())
        # UI rounds 1/30 increments to six decimals; freeze an exact packet grid.
        settings['horizon_s']=round(settings['horizon_s']*30)/30
        validate_settings(settings)
        return settings

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
                    if settings.get('launch_setup')==launch:
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
        super().start_run()
        if self.job.running:
            self.preview.hide()
            self.result_banner.setText('Generating a NEW trajectory for the frozen launch and model. Rehearsal/CSV will be available when this run completes.')

    def set_running(self,running):
        super().set_running(running)
        if hasattr(self,'result_scope'):self.result_scope.setEnabled(not running)
        self.rehearsal_results.setEnabled(not running)

    def open_rehearsal(self):
        self.result_scope.setCurrentIndex(1)
        super().open_rehearsal()
        self.update_result_banner()
