"""Three-step calibration workspace using the constrained geometry/drag workflow."""
from pathlib import Path
import shutil
import numpy as np
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QGroupBox,
    QTabWidget,QScrollArea,QPushButton,QComboBox,QFileDialog,QLabel)
from experimental_data.io import canonical_json_hash
from experimental_data.io import atomic_json
from simulator.workflow import read_json,apply_recommended_baseline,prepare_data_job
from .baseline_page import BaselinePage as RecordingControls
from .research_widgets import note


class BaselinePage(RecordingControls):
    def __init__(self,root,parent=None):
        self.modern_ready=False
        super().__init__(root,parent)
        outer=self.layout();old=[]
        while outer.count():
            item=outer.takeAt(0)
            if item.widget():old.append(item.widget());item.widget().hide()
        outer.addWidget(self.active);self.active.show()
        self.steps=QTabWidget();outer.addWidget(self.steps,1)
        self.build_recordings();self.build_setup();self.build_results()
        from .cable_residual_page import CableResidualPage
        self.residual_page=CableResidualPage(root)
        self.steps.addTab(self.residual_page,'4  Cable residual')
        self.residual_page.baseline_applied.connect(self.residual_applied)
        outer.addWidget(self.job);self.job.show()
        self.steps.currentChanged.connect(lambda index:self.job.setVisible(index!=3))
        self.modern_ready=True
        self.refresh_fits();self.refresh();self.inspect_take()
        for widget in old:
            if widget not in (self.active,self.job):widget.deleteLater()

    def build_recordings(self):
        page=QWidget();layout=QVBoxLayout(page)
        layout.addWidget(note('Preliminary data only. Fit takes tune the simulator; validation takes check prediction.'))
        actions=QHBoxLayout();actions.addWidget(self.import_button);actions.addWidget(self.process_button);actions.addStretch();layout.addLayout(actions)
        self.table.setMaximumHeight(16777215);layout.addWidget(self.table,1)
        layout.addWidget(self.take_note)
        # Recorded motion has its own canvas; candidate plots remain on Results.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self.result_figure,self.result_canvas=self.figure,self.canvas
        self.figure.set_layout_engine('constrained')
        self.record_figure=Figure(figsize=(7,2),facecolor='white');self.record_canvas=FigureCanvasQTAgg(self.record_figure)
        self.record_canvas.setMinimumHeight(150);layout.addWidget(self.record_canvas,1)
        layout.addWidget(note('Raw recordings are preserved. Recording roles control fitting access; historical all-data fits keep their masks and development splits in a separate audit.'))
        self.steps.addTab(page,'1  Recordings')

    def residual_applied(self):
        self.reset_inputs()
        self.refresh()
        self.baseline_applied.emit()

    def inspect_take(self):
        if not hasattr(self,'record_figure'):return super().inspect_take()
        original=(self.figure,self.canvas)
        self.figure,self.canvas=self.record_figure,self.record_canvas
        try:super().inspect_take()
        finally:self.figure,self.canvas=original

    def build_setup(self):
        scroll=QScrollArea();scroll.setWidgetResizable(True);page=QWidget();layout=QVBoxLayout(page)
        self.inputs_note=note('Fit inputs: active baseline. Fitting creates a candidate; it does not apply it.');layout.addWidget(self.inputs_note)
        method=QGroupBox('Standard fit · geometry and cable drag');ml=QVBoxLayout(method)
        ml.addWidget(note('1. Fit lateral attachment offset; keep measured height and cable lengths.\n'
            '2. Fit cable drag on fit takes; hold EI and internal damping fixed.\n'
            '3. Check continuous 2-second and 5-second predictions on validation takes.'))
        ml.addWidget(note('Past-only initialization · pivot attachment · neural residual off'))
        layout.addWidget(method)
        measured=QGroupBox('Measured inputs');mf=QFormLayout(measured)
        basic={('point_mass','mass_kg'):'Drone mass',('cable','bare_cable_mass_kg'):'Bare cable mass',
            ('recorded_data','optitrack_to_attachment_offset_body_m',2):'Top origin → attachment Z in tracking frame [m]',
            ('cable','marker_interval_lengths_m',0):'Attachment → c1 cable arc length [m]'}
        for key,label in basic.items():mf.addRow(label,self.fields[key])
        layout.addWidget(measured)
        layout.addWidget(note('The top-to-attachment offset rotates with the OptiTrack rigid body. '
            'The first cable span can bend; its straight-line marker distance can be shorter than its arc length. '
            'The tracked origin and cable attachment are separate from the unknown center of mass.'))
        self.material_note=note('');layout.addWidget(self.material_note)
        toggle=QPushButton('Edit fixed material values and remaining geometry');toggle.setCheckable(True);layout.addWidget(toggle)
        advanced=QGroupBox('Advanced fit inputs');af=QFormLayout(advanced)
        for key,spin in self.fields.items():
            if key in basic:continue
            if key[1]=='EI_n_m2':label='Fixed EI [N·m²]'
            elif key[1]=='Cb_n_m2_s':label='Fixed Cb [N·m²·s]'
            elif key[1]=='marker_interval_lengths_m':label=f'c{key[2]} → c{key[2]+1} [m]'
            elif key[1]=='moving_marker_masses_kg':label=f'c{key[2]+1} marker mass'
            else:label=f'Starting attachment {"XYZ"[key[2]]} [m]'
            af.addRow(label,spin)
        advanced.hide();toggle.toggled.connect(advanced.setVisible);layout.addWidget(advanced)
        buttons=QHBoxLayout();self.reset_button=QPushButton('Use active model inputs');self.reset_button.clicked.connect(self.reset_inputs)
        buttons.addWidget(self.reset_button);buttons.addStretch();self.fit_button.setText('Fit geometry and cable drag')
        self.fit_button.setObjectName('primaryButton');buttons.addWidget(self.fit_button)
        layout.addStretch();scroll.setWidget(page)
        wrapper=QWidget();wrapper_layout=QVBoxLayout(wrapper)
        wrapper_layout.addWidget(scroll,1);wrapper_layout.addLayout(buttons)
        self.steps.addTab(wrapper,'2  Fit setup');self.update_material()

    def build_results(self):
        page=QWidget();layout=QVBoxLayout(page)
        line=QHBoxLayout();line.addWidget(QLabel('Candidate'));line.addWidget(self.fit_runs,1)
        self.horizon=QComboBox();self.horizon.currentIndexChanged.connect(self.show_results);line.addWidget(self.horizon);layout.addLayout(line)
        layout.addWidget(self.fit_result)
        self.comparison_note=note('');layout.addWidget(self.comparison_note)
        self.view=QComboBox();self.view.addItems(['Validation errors','Measured vs predicted tip']);self.view.currentIndexChanged.connect(self.show_results)
        layout.addWidget(self.view);self.trial=QComboBox();self.trial.currentIndexChanged.connect(self.show_results);layout.addWidget(self.trial)
        self.canvas.setMinimumHeight(150);layout.addWidget(self.canvas,1);layout.addWidget(self.baseline_status)
        buttons=QHBoxLayout();buttons.addWidget(self.export_fit_button)
        self.refit_button=QPushButton('Use candidate as fit inputs');self.refit_button.clicked.connect(self.use_candidate_inputs)
        buttons.addWidget(self.refit_button);buttons.addStretch();self.apply_button.setText('Apply selected candidate');buttons.addWidget(self.apply_button)
        layout.addLayout(buttons);self.steps.addTab(page,'3  Results')

    def inputs_changed(self):
        if self.modern_ready:
            self.inputs_note.setText('Fit inputs edited. Saved candidates and the active baseline are unchanged.');self.update_material()

    def change_role(self,name,**changes):
        manifest=read_json(self.root/'data/dataset_manifest.json')
        if manifest['takes'][name]['role']=='untouched_test':return
        manifest['takes'][name].update(changes)
        atomic_json(self.root/'data/dataset_manifest.json',manifest)
        self.manifest=manifest
        if self.modern_ready:
            self.inputs_note.setText('Recording selection changed for the next fit. Saved results retain their original data selection.')

    def update_material(self):
        c=self.model_values()['cable'];self.material_note.setText(f'Fixed during fitting: EI {c["EI_n_m2"]:.4g} N·m² · Cb {c["Cb_n_m2_s"]:.4g} N·m²·s')

    def load_inputs(self,model,label):
        self.model=model
        for key,spin in self.fields.items():
            value=model[key[0]][key[1]];spin.blockSignals(True);spin.setValue(value if len(key)==2 else value[key[2]])
            self.loaded_field_values[key]=spin.value();spin.blockSignals(False)
        self.inputs_note.setText(label);self.update_material()

    def reset_inputs(self):self.load_inputs(read_json(self.root/'config/model.json'),'Fit inputs: active baseline.')

    def use_candidate_inputs(self):
        if self.candidate:
            self.load_inputs(read_json(self.candidate/'recommended_model.json'),'Fit inputs: selected candidate. Active baseline unchanged.')
            self.steps.setCurrentIndex(1)

    def start_job(self,kind):
        try:
            self.job_kind=kind;directory,command=prepare_data_job(self.root,kind,self.model_values())
            self.job.start(directory,command);self.steps.setEnabled(False)
        except (OSError,ValueError) as error:self.job.status.setText(str(error))

    def fit_progress(self,progress):pass
    def show_fit_view(self):pass

    def job_finished(self,code):
        self.steps.setEnabled(True);self.table.setEnabled(True);self.fit_runs.setEnabled(True)
        for widget in [*self.fields.values(),self.fit_button,self.process_button,self.import_button]:widget.setEnabled(True)
        self.refresh();self.refresh_fits()
        if code==0 and self.job_kind=='fit':
            self.fit_runs.setCurrentIndex(self.fit_runs.findData(str(self.job.directory)));self.steps.setCurrentIndex(2)

    def refresh_fits(self):
        if not self.modern_ready:return
        selected=self.fit_runs.currentData();self.fit_runs.blockSignals(True);self.fit_runs.clear()
        self.fit_runs.addItem('Select a completed candidate',None)
        paths=[]
        for parent in ('data/workflow_jobs','data/calibration_audits'):
            paths.extend(p.parent for p in (self.root/parent).glob('*/recommended_model.json')
                if (p.parent/'candidate_review.json').exists() and (p.parent/'evaluation.json').exists())
        for path in sorted(paths,reverse=True):self.fit_runs.addItem(path.name,str(path))
        if selected:self.fit_runs.setCurrentIndex(max(0,self.fit_runs.findData(selected)))
        elif paths:self.fit_runs.setCurrentIndex(1)
        self.fit_runs.blockSignals(False);self.select_fit()

    def select_fit(self):
        if not self.modern_ready:return
        value=self.fit_runs.currentData();self.candidate=Path(value) if value else None
        for button in (self.apply_button,self.refit_button,self.export_fit_button):button.setEnabled(bool(value))
        self.horizon.blockSignals(True);self.horizon.clear();self.trial.blockSignals(True);self.trial.clear()
        if self.candidate:
            model=read_json(self.candidate/'recommended_model.json');self.candidate_hash=canonical_json_hash(model)
            self.evaluation=read_json(self.candidate/'evaluation.json');self.recommended=read_json(self.candidate/'candidate_review.json')['recommended']
            self.reference='starting_inputs' if 'starting_inputs' in self.evaluation['2.0'] else 'original_fitted'
            for h in sorted(self.evaluation,key=float):self.horizon.addItem(f'{float(h):g}-second predictions',h)
            c=model['cable'];r=model['recorded_data']['optitrack_to_attachment_offset_body_m']
            self.fit_result.setText(f'SELECTED CANDIDATE · review before applying\nAttachment XYZ [{r[0]*1000:.2f}, {r[1]*1000:.2f}, {r[2]*1000:.2f}] mm · Drag {c.get("external_drag_s_inv",0):g}/s\nEI {c["EI_n_m2"]:.4g} N·m² · Cb {c["Cb_n_m2_s"]:.4g} N·m²·s · Neural residual off')
            self.prediction_path=self.candidate/f'{self.recommended}_2s_predictions.npz'
            if self.prediction_path.exists():
                with np.load(self.prediction_path) as data:
                    valid_names=self.evaluation['2.0'][self.recommended]['validation']['per_take']
                    for i,(take,start) in enumerate(zip(data['take'],data['start_frame'])):
                        if str(take) in valid_names:self.trial.addItem(f'{take} · start {start/100:.2f} s',i)
            self.comparison_note.setText('Comparison uses the saved starting model, not necessarily today’s active baseline.\nDevelopment validation · measured attachment input; this does not validate open-loop flight.')
            self.baseline_status.setText('Apply copies the complete model: geometry, material values, drag and solver settings.')
        else:
            self.fit_result.setText('Select a completed candidate. Legacy EI/Cb-only fits are separate from this workflow.')
            self.comparison_note.clear();self.baseline_status.clear()
        self.horizon.blockSignals(False);self.trial.blockSignals(False);self.show_results()

    def show_results(self):
        if not self.modern_ready:return
        self.figure.clear();motion=self.view.currentIndex()==1;self.trial.setVisible(motion)
        if not self.candidate:
            ax=self.figure.subplots();ax.text(.5,.5,'Fit a model or select a saved candidate',ha='center',transform=ax.transAxes);ax.axis('off')
        elif motion and self.trial.currentData() is not None:
            axes=self.figure.subplots(1,3);index=self.trial.currentData()
            with np.load(self.prediction_path) as data:
                for axis,ax in enumerate(axes):
                    time=np.arange(data['prediction'].shape[1])*.01
                    ax.plot(time,data['measured'][index,:,-1,axis],color='#172033',label='Measured')
                    ax.plot(time,data['prediction'][index,:,-1,axis],color='#268271',ls='--',label='Selected candidate')
                    ax.set_title(f'Tip {"XYZ"[axis]} [m]');ax.set_xlabel('Time [s]')
                axes[0].legend(frameon=False,fontsize=8)
        else:
            ax=self.figure.subplots();h=self.horizon.currentData() or '2.0'
            for dx,key,label,color in [(-.18,self.reference,'Starting model','#94a3b8'),(.18,self.recommended,'Selected candidate','#268271')]:
                values=[self.evaluation[h][key][role]['equal_take_marker_rmse_m']*1000 for role in ('training','validation')]
                bars=ax.bar(np.arange(2)+dx,values,.34,label=label,color=color);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
            ax.set_xticks([0,1],['Fit takes','Validation takes']);ax.set_ylabel('Mean take\nmarker RMSE [mm]');ax.set_ylim(top=ax.get_ylim()[1]*1.2)
            ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
        self.horizon.setEnabled(not motion);self.canvas.draw_idle()

    def apply_model(self):
        if not self.candidate:return
        try:
            version=apply_recommended_baseline(self.root,self.candidate,self.candidate_hash)
            self.reset_inputs();self.refresh();self.baseline_status.setText(f'Applied {version}. New runs use this model.');self.baseline_applied.emit()
        except (OSError,ValueError) as error:self.baseline_status.setText(str(error))

    def export_fit(self):
        if not self.candidate:return
        folder=QFileDialog.getExistingDirectory(self,'Export calibration figures and source data')
        if not folder:return
        try:
            destination=Path(folder)
            for ext in ('pdf','svg','png'):self.figure.savefig(destination/f'calibration_{self.view.currentIndex()}.{ext}',dpi=600,bbox_inches='tight')
            for name in ('evaluation.json','candidate_review.json','recommended_model.json','geometry_fit.json','drag_search.json'):
                if (self.candidate/name).exists():shutil.copy2(self.candidate/name,destination/name)
            for source in self.candidate.glob('*_2s_predictions.npz'):
                shutil.copy2(source,destination/source.name)
            if (self.candidate/'figures').exists():shutil.copytree(self.candidate/'figures',destination/'all_figures',dirs_exist_ok=True)
            self.baseline_status.setText('Exported PDF, SVG and PNG figures with source metrics and model.')
        except (OSError,ValueError) as error:self.baseline_status.setText(str(error))
