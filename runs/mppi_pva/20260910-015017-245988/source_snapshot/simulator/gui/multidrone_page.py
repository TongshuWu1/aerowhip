"""Prepare calibrated batch replays and launch the separate Isaac renderer."""
import os
from pathlib import Path
import subprocess
import sys
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QSpinBox, QLineEdit, QLabel, QFileDialog)
from .research_widgets import BackgroundJob, note
from simulator.workflow import read_json, stamp


class MultiDronePage(QWidget):
    def __init__(self, root):
        super().__init__()
        self.root=Path(root);self.directory=None;self.viewer_process=None;self.record_folder=None;self.encoding_folder=None
        self.viewer_close_flag=None;self.pending_live_run=None
        layout=QVBoxLayout(self)
        title=QLabel('Parallel drone + cable replay');title.setStyleSheet('font-size:20px;font-weight:600;');layout.addWidget(title)
        layout.addWidget(note('Generate independent open-loop executions from a saved native 30 Hz PPO. Isaac Sim displays our calibrated drone and cable models with both residuals. This is checkpoint replay, not live training footage.'))
        row=QHBoxLayout();self.checkpoints=QComboBox();row.addWidget(self.checkpoints,1)
        refresh=QPushButton('Refresh policies');refresh.clicked.connect(self.refresh);row.addWidget(refresh);layout.addLayout(row)
        row=QHBoxLayout();row.addWidget(QLabel('Drones'))
        self.count=QComboBox();self.count.addItems(['64','256','1024']);self.count.setCurrentText('256');row.addWidget(self.count)
        row.addWidget(QLabel('Scenario seed'));self.seed=QSpinBox();self.seed.setRange(0,2147483647);self.seed.setValue(20260908);row.addWidget(self.seed);row.addStretch();layout.addLayout(row)
        row=QHBoxLayout();self.generate_button=QPushButton('Generate batch replay');self.generate_button.setObjectName('primaryButton');self.generate_button.clicked.connect(self.generate);row.addWidget(self.generate_button)
        self.open_saved=QPushButton('Open saved batch…');self.open_saved.clicked.connect(self.choose);row.addWidget(self.open_saved);row.addStretch();layout.addLayout(row)
        layout.addWidget(note('The saved policy and its frozen source/configuration determine the motion. Successful and unsuccessful attempts are retained. Grid translations affect only display.'))
        self.result=note('No batch loaded.');layout.addWidget(self.result)
        row=QHBoxLayout();row.addWidget(QLabel('Isaac Python'))
        prefs=read_json(self.root/'config/multidrone_viewer.json',{})
        self.python=QLineEdit(prefs.get('isaac_python',str(Path.home()/'env_isaaclab/Scripts/python.exe')));row.addWidget(self.python,1)
        browse=QPushButton('Browse…');browse.clicked.connect(self.browse);row.addWidget(browse);layout.addLayout(row)
        row=QHBoxLayout();self.launch=QPushButton('Open Isaac scene');self.launch.clicked.connect(self.launch_viewer);row.addWidget(self.launch)
        self.files=QPushButton('Open replay files');self.files.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.directory))));row.addWidget(self.files)
        self.video=QPushButton('Record 30 fps MP4');self.video.clicked.connect(lambda:self.launch_viewer(record=True));row.addWidget(self.video);row.addStretch();layout.addLayout(row)
        layout.addWidget(note('Isaac controls: play/pause, scrub, playback speed, grid overview and per-drone close-up. Recording saves one half-speed loop as an MP4 and retains its PNG frames. Native camera controls remain available.'))
        self.status=note('Ready');layout.addWidget(self.status)
        live_row=QHBoxLayout();self.live_runs=QComboBox();live_row.addWidget(self.live_runs,1)
        follow=QPushButton('Open recorded batches (older runs)');follow.clicked.connect(lambda:self.follow_training(self.live_runs.currentData()));live_row.addWidget(follow);layout.addLayout(live_row)
        self.job=BackgroundJob(root);self.job.finished.connect(self.finished);layout.addWidget(self.job)
        layout.addStretch()
        self.timer=QTimer(self);self.timer.setInterval(1500);self.timer.timeout.connect(self.poll_viewer)
        self.refresh();self.controls()

    def controls(self):
        running=self.job.running
        viewing=self.viewer_process is not None and self.viewer_process.poll() is None
        self.generate_button.setEnabled(bool(self.checkpoints.currentData()) and not running and not viewing)
        self.checkpoints.setEnabled(not running);self.open_saved.setEnabled(not running and not viewing)
        for w in (self.launch,self.video):w.setEnabled(self.directory is not None and (self.directory/'replay.json').exists() and not running and not viewing)
        self.files.setEnabled(self.directory is not None)

    def refresh(self):
        if self.job.running:return
        previous=self.checkpoints.currentData();self.checkpoints.clear()
        from simulator.policy_library import deleted_checkpoints,checkpoint_key
        deleted=deleted_checkpoints(self.root)
        for run in sorted((self.root/'runs/ppo').glob('*'),reverse=True):
            if read_json(run/'model.json',{}).get('fullstate_execution',{}).get('schema')!='tracked_pose_execution_v1':continue
            for p in sorted((run/'checkpoints').glob('*.pt'),key=lambda p:(p.name!='best_validation.pt',p.name)):
                if checkpoint_key(self.root,p) not in deleted:
                    self.checkpoints.addItem(read_json(run/'run.json',{}).get('display_name',run.name)+' / '+p.name,str(p.resolve()))
        index=self.checkpoints.findData(previous)
        if index>=0:self.checkpoints.setCurrentIndex(index)
        if hasattr(self,'live_runs'):
            chosen=self.live_runs.currentData();self.live_runs.clear()
            for run in sorted((self.root/'runs/ppo').glob('*'),reverse=True):
                cfg=read_json(run/'ppo.json',read_json(run/'launch_config/ppo.json',{}))
                if cfg.get('live_scene',{}).get('enabled'):
                    self.live_runs.addItem(read_json(run/'run.json',{}).get('display_name',run.name),str(run.resolve()))
            i=self.live_runs.findData(chosen)
            if i>=0:self.live_runs.setCurrentIndex(i)
        if hasattr(self,'job'):self.controls()

    def generate(self):
        if self.job.running:return
        self.directory=self.root/'runs/presentations'/stamp()
        try:
            self.job.start(self.directory,[sys.executable,'-u','tools/export_multidrone.py','--checkpoint',self.checkpoints.currentData(),
                '--output',str(self.directory),'--num-envs',self.count.currentText(),'--batch-size','64','--seed',str(self.seed.value())])
            self.result.setText('Generating independent scenarios…');self.status.setText('GPU generation runs separately; no policy optimization.');self.controls()
        except (OSError,ValueError) as e:self.status.setText(str(e))

    def finished(self,code):
        if code==0:self.load(self.directory)
        else:self.status.setText('Generation failed; open job log for details.')
        self.controls()

    def load(self,path):
        meta=read_json(Path(path)/'replay.json',{})
        if meta.get('schema')!='multidrone_replay_v1':raise ValueError('Select a completed multi-drone replay folder.')
        self.directory=Path(path)
        if self.count.findText(str(meta['num_envs']))>=0:self.count.setCurrentText(str(meta['num_envs']))
        self.result.setText(f'{meta["num_envs"]} independent scenarios · {meta["success_count"]} valid hits · {meta["failure_count"]} invalid attempts\nCheckpoint at {meta["training_attempts"]:,} attempts · seed {meta["seed"]}\n{path}')
        self.status.setText('Ready to open in Isaac Sim.');self.controls()

    def choose(self):
        path=QFileDialog.getExistingDirectory(self,'Open batch replay',str(self.root/'runs/presentations'))
        if path:
            try:self.load(path)
            except (ValueError,OSError,KeyError) as e:self.status.setText(str(e))

    def browse(self):
        path,_=QFileDialog.getOpenFileName(self,'Select Isaac environment Python',str(Path.home()),'Python (python.exe)')
        if path:self.python.setText(path)

    def launch_viewer(self,checked=False,record=False):
        if self.viewer_process is not None and self.viewer_process.poll() is None:return
        python=Path(self.python.text())
        if not python.is_file():self.status.setText('Select the Python executable from your Isaac Sim environment.');return
        from experimental_data.io import atomic_json
        atomic_json(self.root/'config/multidrone_viewer.json',dict(isaac_python=str(python)))
        command=[str(python),'-u',str(self.root/'tools/view_multidrone_isaac.py'),'--replay',str(self.directory),'--num-envs',self.count.currentText()]
        self.viewer_close_flag=self.directory/('close-viewer-'+stamp()+'.flag')
        command+=['--close-flag',str(self.viewer_close_flag)]
        self.record_folder=self.directory/('frames-'+stamp()) if record else None
        self.encoding_folder=None
        if record:command+=['--record-dir',str(self.record_folder),'--headless']
        try:
            environment=os.environ.copy()
            for key in ('PYTHONPATH','QT_QPA_PLATFORM_PLUGIN_PATH','QT_PLUGIN_PATH'):environment.pop(key,None)
            with (self.directory/'isaac.log').open('ab') as stream:
                self.viewer_process=subprocess.Popen(command,cwd=self.root,env=environment,stdout=stream,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.timer.start();self.status.setText('Isaac is starting. First launch may take a few minutes; details are in isaac.log.');self.controls()
        except OSError as e:self.status.setText(str(e))

    def poll_viewer(self):
        if self.viewer_process is not None and self.viewer_process.poll() is not None:
            code=self.viewer_process.returncode;self.timer.stop()
            if code==0 and self.record_folder is not None:
                folder=self.record_folder;self.record_folder=None
                try:
                    with (self.directory/'isaac.log').open('ab') as stream:
                        self.viewer_process=subprocess.Popen([sys.executable,str(self.root/'tools/encode_multidrone_video.py'),str(folder)],
                            cwd=self.root,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    self.encoding_folder=folder
                    self.timer.start();self.status.setText('Encoding MP4 in '+str(folder));return
                except OSError as e:self.status.setText('Video encoding failed: '+str(e));self.controls();return
            if code==0 and self.encoding_folder is not None:
                self.status.setText('MP4 saved: '+str(self.encoding_folder/'replay.mp4'))
            else:self.status.setText('Isaac closed.' if code==0 else 'Viewer / recording failed; see isaac.log in the replay folder.')
            self.controls()
            if self.pending_live_run:
                run=self.pending_live_run;self.pending_live_run=None;self.follow_training(run)

    def follow_training(self,run):
        if not run:self.status.setText('No older batch recordings. New Isaac training opens its scene directly from the PPO launcher.');return
        if self.viewer_process is not None and self.viewer_process.poll() is None:
            self.pending_live_run=run
            if self.record_folder is None and self.encoding_folder is None and self.viewer_close_flag:
                self.viewer_close_flag.touch()
            self.status.setText('Switching Isaac to training run '+str(run));self.refresh();return
        run=Path(run)
        config=read_json(run/'ppo.json',read_json(run/'launch_config/ppo.json',{}))
        if not config.get('live_scene',{}).get('enabled'):
            self.status.setText('This run predates live training recording; use checkpoint replay.');return
        python=Path(self.python.text())
        if not python.is_file():self.status.setText('Select your Isaac Python executable.');return
        script=run/'source_snapshot/tools/view_multidrone_isaac.py'
        if not script.is_file():script=self.root/'tools/view_multidrone_isaac.py'
        env=os.environ.copy()
        for key in ('PYTHONPATH','QT_QPA_PLATFORM_PLUGIN_PATH','QT_PLUGIN_PATH'):env.pop(key,None)
        try:
            self.viewer_close_flag=run/('close-viewer-'+stamp()+'.flag')
            with (run/'isaac-live.log').open('ab') as stream:
                self.viewer_process=subprocess.Popen([str(python),'-u',str(script),'--run',str(run),
                    '--num-envs',str(config['training']['collection_batch']),'--close-flag',str(self.viewer_close_flag)],cwd=self.root,env=env,stdout=stream,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.record_folder=self.encoding_folder=None
            self.status.setText('Isaac is following training run '+run.name+'; it waits for the first completed batch.')
            self.timer.start();self.refresh();self.controls()
        except OSError as e:self.status.setText('Training is running, but Isaac could not launch: '+str(e))
