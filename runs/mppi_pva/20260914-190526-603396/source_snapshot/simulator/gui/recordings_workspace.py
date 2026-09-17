"""Current raw preliminary and flight recording intake."""
from PySide6.QtWidgets import QWidget,QVBoxLayout,QTabWidget,QLabel
from .research_widgets import note
from .preliminary_recordings_page import PreliminaryRecordingsPage
from .flight_batch_page import FlightBatchPage


class RecordingsWorkspace(QWidget):
    def __init__(self,root):
        super().__init__();outer=QVBoxLayout(self);outer.setContentsMargins(18,12,18,14)
        self.tabs=QTabWidget();outer.addWidget(self.tabs)
        self.preliminary=PreliminaryRecordingsPage(root);self.tabs.addTab(self.preliminary,'Preliminary takes')
        self.current=FlightBatchPage(root);self.tabs.addTab(self.current,'Flights by model')
        page=QWidget();layout=QVBoxLayout(page);self.tabs.addTab(page,'Recording checklist')
        layout.addWidget(QLabel('Record the complete execution'))
        layout.addWidget(note('1. Start the controller logger and OptiTrack.\n\n'
            '2. Take off and hold at the exported tracked-origin start.\n\n'
            '3. Execute the exact frozen 30 Hz PVA CSV using the verified lab sender.\n\n'
            '4. Record recovery and landing, then stop both logs.\n\n'
            '5. Preserve native position/quaternion and C1–C10 markers, command receipts, '
            'the exact CSV, target location and contact/intervention notes.\n\n'
            'Keep hits and misses. Review clocks, masks and free-motion intervals before fitting.'))
        layout.addStretch()
