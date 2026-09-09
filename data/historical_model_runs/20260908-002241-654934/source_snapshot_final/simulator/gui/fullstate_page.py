"""Export the existing rehearsal's whip and original PID recovery as full state."""
import csv
from pathlib import Path
from PySide6.QtWidgets import QTableWidgetItem, QFileDialog, QHeaderView, QPushButton
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from simulator.workflow import read_json
from .testing_page import TestingPage


class FullStatePage(TestingPage):
    fullstate_mode = True

    def __init__(self, root):
        super().__init__(root)
        self.intro.setText('Unchanged whip → smooth braking → gentle return to hover at 30 Hz')
        self.assumption.setText('3D shows the source simulation. The exported CSV replaces its PID recovery with smooth braking and a slow return. Open the CSV trajectory plot to inspect that reference.')
        self.execute.setText('Execute rehearsal & create CSV')
        self.controller_legend.setText('Trails: orange = PPO whip; blue = original hover / PID recovery')
        self.open_csv.setText('Open full-state CSV')
        self.export_reference = self.open_controller_csv
        self.export_reference.clicked.disconnect()
        self.export_reference.setText('Save full-state bundle…')
        self.export_reference.clicked.connect(self.save_reference)
        self.csv_convention.setText('Kinematic acceleration: no mass division or gravity subtraction. Vehicle position/velocity feedback remains enabled.')
        self.plan_note.setText('Start rehearsal, wait for the force plan, then execute. The complete CSV is saved when the original recovery finishes.')
        self.open_reference_plot = QPushButton('Open CSV trajectory plot')
        self.open_reference_plot.setEnabled(False)
        self.open_reference_plot.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(
            str((self.csv_path.parent/'recovery_reference.png').resolve()))) if self.csv_path else None)
        self.layout().addWidget(self.open_reference_plot)

    def experiment_setup(self):
        return dict(super().experiment_setup(), reference_mode='fullstate', reference_rate_hz=30.,
                    reference_source='recorded_whip_gentle_recovery')

    def set_controller_visual(self, force_control):
        super().set_controller_visual(force_control)
        if force_control:
            self.controller_mode.setText('Source simulation: PPO whip')

    def show_plan(self, path, metadata):
        directory = Path(path).parent
        self.csv_kind.blockSignals(True)
        self.csv_kind.clear()
        if not (directory/'fullstate.json').exists():
            self.open_reference_plot.setEnabled(False)
            self.plan_metadata = metadata
            self.csv_path = None
            self.csv_kind.blockSignals(False)
            self.csv_kind.setEnabled(False)
            self.table.setRowCount(0)
            self.open_csv.setEnabled(False)
            self.export_reference.setEnabled(False)
            self.execute.setEnabled(True)
            self.plan_note.setText(f'Force plan ready · whip {metadata["cutoff_s"]:.3f} s. '
                'Execute rehearsal to record the whip and original PID recovery. Complete CSV follows recovery.')
            return
        self.plan_metadata = read_json(directory/'fullstate.json')
        self.csv_kind.addItem('Full-state trajectory · whip + recovery + hover',str(directory/'fullstate_30hz.csv'))
        self.csv_kind.blockSignals(False)
        self.csv_kind.setEnabled(True)
        self.show_csv_selection()
        self.export_reference.setEnabled(True)
        self.execute.setEnabled(False)

    def show_csv_selection(self):
        path = self.csv_kind.currentData()
        if not path:
            return
        self.csv_path = Path(path)
        with self.csv_path.open(newline='',encoding='utf-8') as stream:
            header,*rows = list(csv.reader(stream))
        self.table.setColumnCount(len(header))
        self.table.setHorizontalHeaderLabels(header)
        self.table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            for j,value in enumerate(row):
                try:
                    text = f'{float(value):.6f}'
                except ValueError:
                    text = value
                self.table.setItem(i,j,QTableWidgetItem(text))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        meta = self.plan_metadata
        self.open_reference_plot.setEnabled((self.csv_path.parent/'recovery_reference.png').is_file())
        if meta.get('schema') == 'gentle_recovery_fullstate_v4':
            recovery = meta['recovery']
            self.plan_note.setText(f'Whip unchanged: {meta["whip_end_s"]:.3f} s. '
                f'Smooth braking {recovery["brake_end_s"]:.2f} s · return {recovery["return_s"]:.1f} s '
                f'· hold {recovery["hold_s"]:.1f} s. Return feedforward tilt ≤{recovery["return_peak_tilt_surrogate_deg"]:.2f}°. '
                'Recovery is an analytic reference; vehicle tracking and cable motion are not validated.')
        elif meta.get('schema') == 'recorded_rehearsal_fullstate_v3':
            self.plan_note.setText(f'Whip {meta["whip_end_s"]:.3f} s · complete recorded trajectory {meta["total_duration_s"]:.3f} s · {len(rows)} rows. '
                f'Peak speed {meta["peak_speed_m_s"]:.2f} m/s · acceleration {meta["peak_acceleration_m_s2"]:.2f} m/s². '
                'Includes the original PID recovery and settled hover. Takeoff and landing remain external.')
        else:
            self.plan_note.setText('Historical export. Run a new rehearsal to export its unchanged whip and PID recovery.')
        if meta.get('reference_point')=='OptiTrack_cf7_origin':
            self.plan_note.setText(self.plan_note.text()+
                f'\nCSV commands the cf_7 origin; starting vehicle position {meta["initial_vehicle_position_m"]} m.'
                ' The 3D source shows the virtual attachment/cable trajectory.')
        self.open_csv.setEnabled(True)

    def save_reference(self):
        if self.csv_path is None:
            return
        destination,_ = QFileDialog.getSaveFileName(self,'Save full-state reference bundle',
            str(self.root/'policies'/f'{self.directory.name}-fullstate.zip'),'ZIP (*.zip)')
        if not destination:
            return
        import zipfile
        try:
            with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
                for path in self.csv_path.parent.iterdir():
                    if path.is_file():
                        archive.write(path,path.name)
            self.status.setText(f'Saved full-state reference bundle: {destination}')
        except Exception as error:
            self.status.setText(f'Export failed: {error}')
