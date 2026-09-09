"""Research overview and honest, directly sourced bootstrap diagnostics."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QFrame,QPushButton,
    QTabWidget,QTableWidget,QTableWidgetItem,QHeaderView,QComboBox,QSplitter,QScrollArea)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
from simulator.workflow import read_json
from simulator.research_config import workspace_configs,BOOTSTRAP_JOB
from .research_widgets import note

BLUE='#2563eb';TEAL='#0d9488';ORANGE='#ea580c';INK='#172033'


def card(title,value,caption):
    frame=QFrame();frame.setObjectName('contentCard');layout=QVBoxLayout(frame);layout.setContentsMargins(18,12,18,12)
    layout.addWidget(note(title));big=QLabel(value);big.setStyleSheet('font-size: 22pt; font-weight: 650; color: #172033;')
    layout.addWidget(big);layout.addWidget(note(caption));return frame


def table(headers,rows):
    widget=QTableWidget(len(rows),len(headers));widget.setHorizontalHeaderLabels(headers)
    widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);widget.verticalHeader().hide()
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    for i,row in enumerate(rows):
        for j,value in enumerate(row):widget.setItem(i,j,QTableWidgetItem(str(value)))
    return widget


def style_axes(axes):
    for ax in np.asarray(axes,dtype=object).flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.tick_params(labelsize=8)


def metric_3d(ax,*clouds):
    points=np.concatenate([np.asarray(p).reshape(-1,3) for p in clouds]);points=points[np.isfinite(points).all(-1)]
    if not len(points):return
    low=points.min(0)-.05;high=points.max(0)+.05
    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]));ax.set_box_aspect(high-low)


class ModelWorkspace(QWidget):
    def __init__(self,root):
        super().__init__();self.root=Path(root);self.job=self.root/'data/bootstrap_model_runs'/BOOTSTRAP_JOB
        model,task,ppo=workspace_configs(root);outer=QVBoxLayout(self);outer.setContentsMargins(22,16,22,18);outer.setSpacing(14)
        row=QHBoxLayout()
        for args in [('MODEL','Bootstrap M0','Nominal drone + drone NN + cable physics + cable NN'),
                     ('CONTROL CLOCK','30 / 30 Hz','Virtual force policy / FullState command'),
                     ('MEASURED MASS','175 g','157 g drone + 18 g cable assembly')]:row.addWidget(card(*args),1)
        outer.addLayout(row)
        pipeline=QLabel('Initial drone state + target  →  virtual force plan  →  FullState reference  →  predicted drone pose  →  cable motion')
        pipeline.setWordWrap(True);pipeline.setObjectName('pipelineBanner');outer.addWidget(pipeline)
        tabs=QTabWidget();outer.addWidget(tabs,1)
        overview=QWidget();layout=QVBoxLayout(overview);tabs.addTab(overview,'Fit evidence')
        layout.addWidget(note('Historical development data · 8 preliminary takes for cable; 3 whip takes for drone and cable. '
            'Bars use the observed CSV maneuver and measured past cable initialization. These are trajectory errors, not hitting errors.'))
        self.figure=Figure(figsize=(9,3.8),layout='constrained',facecolor='white');canvas=FigureCanvasQTAgg(self.figure);layout.addWidget(canvas,1)
        results=read_json(self.job/'combined/results.json',{})
        if results:
            axes=self.figure.subplots(1,2);names=list(results['final_all_three']);x=np.arange(len(names))
            for mode,color,label in [('nominal_drone_physics_cable',BLUE,'Nominal physics'),('both_residuals',TEAL,'Both NNs')]:
                i=0 if mode.startswith('nominal') else 1
                values=[100*results['final_all_three'][n]['cases'][mode]['tip_rmse_m'] for n in names]
                axes[0].bar(x+(i-.5)*.32,values,.30,color=color,label=label)
            leftout=[100*results['leave_out_'+n][n]['cases']['both_residuals']['tip_rmse_m'] for n in names]
            axes[1].bar(x,leftout,.52,color=ORANGE,label='Leave-one-take-out')
            for ax,title in zip(axes,['All-data fit: cable tip error','Development check: excluded take']):
                ax.set_title(title,fontsize=11,loc='left');ax.set_xticks(x,[n.replace('whip1_','Take ') for n in names]);ax.set_ylabel('Tip RMS [cm]');ax.legend(frameon=False,fontsize=8)
            style_axes(axes)
        layout.addWidget(note('Both NNs are retained by your choice. They do not improve every cable-tip comparison. '
            'Recovery lies outside the fitted maneuver scope; collect corrected-route recordings for the next adaptation.'))
        parameters=QWidget();pl=QVBoxLayout(parameters);tabs.addTab(parameters,'Parameters and geometry')
        execution=model['fullstate_execution'];drone=read_json(execution['checkpoint']);params=drone['nominal']['parameters'];cable=model['cable']
        rows=[('Cable EI',f'{cable["EI_n_m2"]:g}','N·m²'),('Cable internal damping',f'{cable["Cb_n_m2_s"]:g}','N·m²·s'),
              ('Fixed external drag',f'{cable["external_drag_s_inv"]:g}','s⁻¹'),('Drone NN','15 → 16 → 16 → 3','bounded ±0.5 m/s²'),
              ('Cable NN','node / axis damping','bounded 0–2 s⁻¹'),('Physics / cable integration','150 / 1200','Hz'),
              ('Tracking origin → attachment',str(model['recorded_data']['optitrack_to_attachment_offset_body_m']),'m, tracking frame; rotates with R'),
              ('Attachment → first cable marker',str(cable['marker_interval_lengths_m'][0]),'m arc length'),
              ('Start / target variation','5 / 5','cm radius; target known before planning')]
        rows += [(k,f'{v:g}' if isinstance(v,(float,int)) else str(v),'fitted effective response') for k,v in params.items()]
        pl.addWidget(table(['Component / parameter','Value','Meaning / units'],rows))
        pl.addWidget(note('FullState positions refer to the OptiTrack tracked origin. The cable root is the rotated attachment point. '
            'The fitted drone is already loaded: cable reaction is not added a second time.'))
        footer=QHBoxLayout()
        for title,path in [('Open model bundle',self.job/'bundle'),('Read fit report',self.root/'docs/BOOTSTRAP_COMPLETE_MODEL_20260908.md')]:
            b=QPushButton(title);b.clicked.connect(lambda _,p=path:QDesktopServices.openUrl(QUrl.fromLocalFile(str(p))));footer.addWidget(b)
        footer.addStretch();outer.addLayout(footer)


class DiagnosticsWorkspace(QWidget):
    def __init__(self,root):
        super().__init__();self.root=Path(root);self.base=self.root/'data/bootstrap_model_runs'/BOOTSTRAP_JOB/'combined'
        outer=QVBoxLayout(self);outer.setContentsMargins(22,16,22,18);outer.setSpacing(12)
        row=QHBoxLayout();self.split=QComboBox();self.split.addItem('All-data fit — development','final_all_three');self.split.addItem('Leave-one-take-out — development','leave_out')
        self.take=QComboBox();self.take.addItems(['whip1_001','whip1_002','whip1_003'])
        self.case=QComboBox()
        for label,key in [('Both residuals','both_residuals'),('Nominal physics','nominal_drone_physics_cable'),
            ('Drone NN only','residual_drone_physics_cable'),('Cable NN only','nominal_drone_residual_cable'),
            ('Both NNs, hanging initialization','both_residuals_hanging_initialization'),('Cable NN, measured attachment','measured_attachment_residual_cable')]:self.case.addItem(label,key)
        for title,w in [('Assessment',self.split),('Take',self.take),('Model',self.case)]:row.addWidget(QLabel(title));row.addWidget(w,1)
        outer.addLayout(row);self.metrics=note('');outer.addWidget(self.metrics)
        self.figure=Figure(figsize=(10,6),layout='constrained',facecolor='white');self.canvas=FigureCanvasQTAgg(self.figure)
        outer.addWidget(NavigationToolbar2QT(self.canvas,self));outer.addWidget(self.canvas,1)
        outer.addWidget(note('Blue: measured OptiTrack. Orange: uninterrupted model prediction. Only the original maneuver score mask contributes to RMS. '
            'No future measured drone/cable resets. The first marker is C1; the tip is C10. Rotate the 3D axes with the mouse.'))
        for w in (self.split,self.take,self.case):w.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self):
        name=self.take.currentText();split=self.split.currentData();folder=self.base/(split if split!='leave_out' else 'leave_out_'+name)
        path=folder/(name+'.npz');self.figure.clear()
        if not path.exists():self.metrics.setText('No saved prediction for this selection.');self.canvas.draw_idle();return
        with np.load(path,allow_pickle=False) as data:a={k:data[k] for k in data.files}
        case=self.case.currentData();measured=a['measured_markers_m'];predicted=a[case+'_markers_m'];mask=a['marker_score_mask'].astype(bool)
        valid=mask&np.isfinite(measured).all(-1)&np.isfinite(predicted).all(-1);error=np.linalg.norm(predicted-measured,axis=-1)*100
        metric=read_json(folder/'results.json')[name]['cases'][case];times=a['time_s'];t=times-times[0]
        self.metrics.setText(f'Tip RMS {metric["tip_rmse_m"]*100:.2f} cm   ·   Tip maximum {metric["tip_max_m"]*100:.2f} cm   ·   '
            f'All-marker RMS {metric["marker_rmse_m"]*100:.2f} cm   ·   {metric["tip_samples"]} scored tip samples')
        grid=self.figure.add_gridspec(2,2);scene=self.figure.add_subplot(grid[:,0],projection='3d');scene.set_title('Measured and predicted trajectories',loc='left',fontsize=11)
        for values,color,label in [(measured,BLUE,'Measured tip'),(predicted,ORANGE,'Predicted tip')]:
            scene.plot(*values[:,-1].T,color=color,label=label,lw=2)
            for i in np.linspace(0,len(values)-1,5,dtype=int):scene.plot(*values[i].T,color=color,alpha=.22,lw=1)
        scene.plot(*a['measured_attachment_m'].T,color=BLUE,ls='--',alpha=.65,label='Measured attachment')
        scene.plot(*a[case+'_attachment_m'].T,color=ORANGE,ls='--',alpha=.65,label='Predicted attachment')
        scene.set(xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]');scene.legend(fontsize=8,loc='upper left');scene.view_init(20,-70)
        metric_3d(scene,measured,predicted,a['measured_attachment_m'])
        error_ax=self.figure.add_subplot(grid[0,1]);marker_ax=self.figure.add_subplot(grid[1,1])
        error_ax.plot(t,np.where(valid[:,-1],error[:,-1],np.nan),color=ORANGE,label='Cable tip')
        root_error=np.linalg.norm(a[case+'_attachment_m']-a['measured_attachment_m'],axis=-1)*100
        error_ax.plot(t,np.where(valid.any(-1),root_error,np.nan),color=TEAL,label='Attachment')
        error_ax.set(title='Where the error grows',xlabel='Time since prediction start [s]',ylabel='Position error [cm]');error_ax.legend(frameon=False,fontsize=8)
        marker_ax.bar(np.arange(1,11),100*np.asarray(metric['per_marker_rmse_m']),color=TEAL,width=.65)
        marker_ax.set(title='Error along the cable',xlabel='Marker (C10 = tip)',ylabel='RMS [cm]',xticks=np.arange(1,11))
        style_axes([error_ax,marker_ax]);self.canvas.draw_idle()


class RecordedFlightDiagnostics(QWidget):
    """Compare actual logged references and OptiTrack, with preserved phase masks."""
    def __init__(self,root):
        super().__init__();self.root=Path(root);outer=QVBoxLayout(self);outer.setContentsMargins(20,14,20,16)
        row=QHBoxLayout();self.take=QComboBox();self.scope=QComboBox();self.scope.addItems(['CSV maneuver','Whole recorded interval'])
        row.addWidget(QLabel('Recording'));row.addWidget(self.take,1);row.addWidget(self.scope);refresh=QPushButton('Refresh');row.addWidget(refresh);outer.addLayout(row)
        self.metrics=note('');outer.addWidget(self.metrics)
        self.figure=Figure(figsize=(10,6),layout='constrained',facecolor='white');self.canvas=FigureCanvasQTAgg(self.figure)
        outer.addWidget(NavigationToolbar2QT(self.canvas,self));outer.addWidget(self.canvas,1)
        outer.addWidget(note('Reference = actual logged FullState command, aligned by the saved processing version. '
            'Cable trace = measured C10, with missing samples left as gaps. No planned cable path is assumed from the drone reference. '
            'Velocity comes from processed mocap. Legacy holds remain separate from the CSV maneuver.'))
        refresh.clicked.connect(self.refresh);self.take.currentIndexChanged.connect(self.draw);self.scope.currentIndexChanged.connect(self.draw);self.refresh()

    def refresh(self):
        previous=self.take.currentData();self.take.blockSignals(True);self.take.clear()
        from experimental_data.adaptation_rounds import preparation_enabled
        for round_dir in sorted((self.root/'data/adaptation_rounds').glob('adaptation*'),reverse=True):
            if not preparation_enabled(round_dir):continue
            for version in sorted((round_dir/'processed').glob('*/processing.json'),reverse=True):
                for report in read_json(version).get('reports',[]):
                    path=version.parent/report['trial_id']/'dataset.npz'
                    if path.exists():self.take.addItem(f'{round_dir.name} / {report["trial_id"]} / {version.parent.name}',str(path))
        i=self.take.findData(previous);self.take.setCurrentIndex(i if i>=0 else 0);self.take.blockSignals(False);self.draw()

    def draw(self):
        self.figure.clear();path=self.take.currentData()
        if not path:self.metrics.setText('Process a recording round to inspect planned versus measured motion.');self.canvas.draw_idle();return
        with np.load(path,allow_pickle=False) as data:a={k:data[k].copy() for k in data.files}
        p=a['drone_position_m'];reference=a['reference_fullstate'];cable=a['cable_position_m'];t=a['controller_time_s']
        selected=np.ones(len(t),dtype=bool);scope='whole recording'
        if self.scope.currentIndex()==0 and 'csv_maneuver_mask' in a:selected=a['csv_maneuver_mask'].astype(bool);scope='CSV maneuver (saved phase mask)'
        elif self.scope.currentIndex()==0:scope='whole recording — no verified CSV phase mask'
        valid=selected&a['drone_position_valid'].astype(bool)&a['reference_valid'].astype(bool)&np.isfinite(p).all(-1)&np.isfinite(reference[:,:3]).all(-1)
        if not valid.any():self.metrics.setText('No valid aligned position/reference pairs in this interval.');self.canvas.draw_idle();return
        origin=t[np.flatnonzero(selected)[0]];time=t-origin;delta=(p-reference[:,:3])*100;norm=np.linalg.norm(delta,axis=-1)
        self.metrics.setText(f'{scope} · tracked-origin RMS {np.sqrt(np.mean(norm[valid]**2)):.2f} cm · '
            f'maximum {norm[valid].max():.2f} cm · {valid.sum()} valid aligned samples')
        grid=self.figure.add_gridspec(2,2);scene=self.figure.add_subplot(grid[:,0],projection='3d')
        actual=np.where((selected&a['drone_position_valid'].astype(bool))[:,None],p,np.nan)
        desired=np.where((selected&a['reference_valid'].astype(bool))[:,None],reference[:,:3],np.nan)
        tip=np.where((selected&a['cable_valid'][:,-1].astype(bool))[:,None],cable[:,-1],np.nan)
        for values,color,label in [(desired,BLUE,'Commanded drone'),(actual,ORANGE,'Measured drone'),(tip,TEAL,'Measured cable tip')]:scene.plot(*values.T,color=color,label=label,lw=1.8)
        scene.set(title='Commanded and measured trajectories',xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]');scene.legend(fontsize=8);metric_3d(scene,desired,actual,tip);scene.view_init(20,-65)
        error_ax=self.figure.add_subplot(grid[0,1]);velocity_ax=self.figure.add_subplot(grid[1,1])
        for i,(axis,color) in enumerate(zip('XYZ',[BLUE,TEAL,ORANGE])):error_ax.plot(time,np.where(valid,delta[:,i],np.nan),color=color,label=axis)
        error_ax.set(title='Tracked-origin position error',xlabel='Time since selected interval start [s]',ylabel='Measured − commanded [cm]');error_ax.legend(frameon=False,fontsize=8)
        velocity_ax.plot(time,np.where(valid,np.linalg.norm(reference[:,3:6],axis=-1),np.nan),color=BLUE,label='Commanded')
        velocity_ax.plot(time,np.where(valid,np.linalg.norm(a['drone_velocity_m_s'],axis=-1),np.nan),color=ORANGE,label='Mocap-derived')
        velocity_ax.set(title='Drone speed',xlabel='Time since selected interval start [s]',ylabel='Speed [m/s]');velocity_ax.legend(frameon=False,fontsize=8)
        for ax in (error_ax,velocity_ax):ax.set_xlim(time[selected].min(),time[selected].max())
        style_axes([error_ax,velocity_ax]);self.canvas.draw_idle()


class DiagnosticsPage(QWidget):
    def __init__(self,root):
        super().__init__();layout=QVBoxLayout(self);layout.setContentsMargins(18,12,18,14);self.tabs=QTabWidget();layout.addWidget(self.tabs)
        self.model=DiagnosticsWorkspace(root);self.recordings=RecordedFlightDiagnostics(root)
        self.tabs.addTab(self.model,'Historical model fit');self.tabs.addTab(self.recordings,'Historical recorded commands')


class RecordingsWorkspace(QWidget):
    def __init__(self,root):
        super().__init__();outer=QVBoxLayout(self);outer.setContentsMargins(18,12,18,14)
        tabs=QTabWidget();outer.addWidget(tabs)
        from .flight_batch_page import FlightBatchPage
        self.current=FlightBatchPage(root);tabs.addTab(self.current,'Current flight batches')
        from .recording_rounds_page import RecordingRoundsPage
        self.rounds=RecordingRoundsPage(root);tabs.addTab(self.rounds,'Historical processing')
        page=QWidget();layout=QVBoxLayout(page);tabs.addTab(page,'Collection checklist')
        layout.addWidget(QLabel('Record the complete execution'))
        layout.addWidget(note('1. Start the controller logger and OptiTrack recording.\n\n'
            '2. Take off and hold for 10 seconds at the exported tracked-origin start.\n\n'
            '3. Execute the exact 30 Hz FullState CSV through whip, braking, return and final hold.\n\n'
            '4. Land, then stop OptiTrack and the controller logger.\n\n'
            '5. Preserve the native take, 100 Hz position/quaternion of the single cf drone rigid body, cable1 C1–C10, controller commands, exact CSV, target and contact/intervention notes.\n\n'
            'Keep successful and failed flights. Record exclusions as reviewed intervals; preserve original measurements. '
            'A new date alone does not identify a corrected-route experiment.'))
        layout.addStretch()
        legacy=QWidget();ll=QVBoxLayout(legacy);tabs.addTab(legacy,'Legacy tools')
        ll.addWidget(note('Historical preliminary-data tools. They operate on the preserved legacy configuration and do not change the selected M0 / 30 Hz workspace.'))
        button=QPushButton('Open historical data and fitting workspace');ll.addWidget(button);ll.addStretch()
        self.legacy=None
        def load():
            if self.legacy is None:
                from .calibration_page import BaselinePage
                self.legacy=BaselinePage(root);ll.addWidget(self.legacy,1);button.hide()
        button.clicked.connect(load)
