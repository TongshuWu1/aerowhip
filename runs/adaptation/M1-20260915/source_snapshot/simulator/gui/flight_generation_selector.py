"""Shared generation -> recorded plan/session navigation."""
from pathlib import Path
from PySide6.QtCore import Signal,Qt
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QPushButton
from experimental_data.flight_library import flight_library
from .research_widgets import note


class FlightGenerationSelector(QWidget):
    session_changed=Signal()

    def __init__(self,root,parent=None):
        super().__init__(parent);self.root=Path(root);self.library=dict(models={},sessions=[])
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        row=QHBoxLayout();layout.addLayout(row)
        row.addWidget(QLabel('Model generation'));self.generations=QComboBox();self.generations.setMinimumWidth(210);row.addWidget(self.generations)
        self.summary=note('');row.addWidget(self.summary,1)
        self.refresh_button=QPushButton('Refresh');row.addWidget(self.refresh_button)
        row=QHBoxLayout();layout.addLayout(row);row.addWidget(QLabel('Plan / session'))
        self.batches=QComboBox();row.addWidget(self.batches,1)
        self.generations.currentIndexChanged.connect(self.populate_sessions)
        self.batches.currentIndexChanged.connect(self.session_changed)
        self.refresh_button.clicked.connect(self.refresh)

    def refresh(self):
        previous=self.generations.currentData();batch=self.batches.currentData()
        try:self.library=flight_library(self.root)
        except (OSError,ValueError,KeyError) as exc:
            self.library=dict(models={},sessions=[]);self.summary.setToolTip(str(exc))
        self.generations.blockSignals(True);self.generations.clear()
        models=self.library['models']
        for key,model in models.items():
            self.generations.addItem(key,key)
            self.generations.setItemData(self.generations.count()-1,model.get('status',''),Qt.ItemDataRole.ToolTipRole)
        numbers=[int(k[1:]) for k in models if k.startswith('M') and k[1:].isdigit()]
        if numbers:
            future=f'M{max(numbers)+1}'
            self.generations.addItem(f'{future} · not created yet',future)
        if any(s['generation']=='unassigned' for s in self.library['sessions']):
            self.generations.addItem('Unassigned recordings','unassigned')
        self.generations.setCurrentIndex(max(0,self.generations.findData(previous)))
        self.generations.blockSignals(False);self.populate_sessions(preferred=batch)

    def populate_sessions(self,*_,preferred=None):
        generation=self.generations.currentData()
        sessions=[s for s in self.library['sessions'] if s['generation']==generation]
        self.batches.blockSignals(True);self.batches.clear()
        for i,s in enumerate(sessions,1):
            label=f'{s["planner"]} · session {i} · {len(s["takes"])} takes'
            if generation=='unassigned':label=Path(s['batch']).name+f' · {len(s["takes"])} takes'
            self.batches.addItem(label,s['batch']);self.batches.setItemData(i-1,s['batch'],Qt.ItemDataRole.ToolTipRole)
        self.batches.setCurrentIndex(max(0,self.batches.findData(preferred)))
        self.batches.blockSignals(False)
        model=self.library['models'].get(generation)
        if model:
            count=sum(len(s['takes']) for s in sessions)
            parent=f' · adapted from {model["parent"]}' if model.get('parent') else ' · initial model'
            session_label='flight session' if len(sessions)==1 else 'flight sessions'
            self.summary.setText(f'{generation}{parent} · {len(sessions)} {session_label} · {count} takes')
        elif generation=='unassigned':self.summary.setText('Recordings without a registered forecast model')
        elif generation:self.summary.setText(f'{generation} has not been created.')
        else:self.summary.setText('No registered model generations yet.')
        self.session_changed.emit()

    def selected_session(self):
        return next((s for s in self.library['sessions'] if s['batch']==self.batches.currentData()),None)

    def select_batch(self,batch):
        batch=str(Path(batch).resolve());self.refresh()
        session=next((s for s in self.library['sessions'] if s['batch']==batch),None)
        if session is None:return False
        self.generations.setCurrentIndex(self.generations.findData(session['generation']))
        self.batches.setCurrentIndex(self.batches.findData(batch));return True
