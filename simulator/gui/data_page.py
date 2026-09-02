"""Integrated processed-data inventory, trim editor, and PyVista replay."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fitting.production_status import get_dataset_role_summary
from simulator.parameters import SimulatorSettings
from simulator.production import PROJECT_ROOT
from .dataset_widget import TakesDatasetWidget
from .theme import MetricCard


class DataPage(QWidget):
    """One dataset page with a compact inventory and an editable workspace."""

    def __init__(
        self,
        settings: SimulatorSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings or SimulatorSettings.load(
            Path(PROJECT_ROOT) / "config" / "default.json"
        )
        self.workspace: TakesDatasetWidget | None = None
        self.setObjectName("dataPage")
        self.setStyleSheet("#dataPage { background: #f8fafc; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 18)
        layout.setSpacing(10)

        summary = QHBoxLayout()
        summary.setSpacing(9)
        self.takes_card = MetricCard("Accepted takes")
        self.episodes_card = MetricCard("Physical episodes")
        self.duration_card = MetricCard("Recorded duration")
        self.protected_card = MetricCard("Protected data")
        for card in (
            self.takes_card,
            self.episodes_card,
            self.duration_card,
            self.protected_card,
        ):
            summary.addWidget(card, 1)
        layout.addLayout(summary)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("datasetTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._tab_changed)
        layout.addWidget(self.tabs, 1)

        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_layout.setContentsMargins(0, 10, 0, 0)
        notice = QLabel(
            "Processed scientific takes only. Trimming is saved as reversible manifest "
            "metadata; source recordings and the protected test are never overwritten."
        )
        notice.setWordWrap(True)
        notice.setStyleSheet(
            "background: #eff6ff; color: #1e3a8a; border: 1px solid #bfdbfe; "
            "border-radius: 7px; padding: 8px 11px;"
        )
        overview_layout.addWidget(notice)
        self.table = QTableWidget(0, 4, self)
        self.table.setObjectName("take_role_table")
        self.table.setHorizontalHeaderLabels(("Take", "Role", "Duration", "Status"))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        overview_layout.addWidget(self.table, 1)
        self.tabs.addTab(overview, "Dataset inventory")

        self.workspace_host = QWidget()
        self.workspace_layout = QVBoxLayout(self.workspace_host)
        self.workspace_layout.setContentsMargins(0, 10, 0, 0)
        placeholder = QFrame()
        placeholder.setObjectName("toolbarCard")
        placeholder_layout = QVBoxLayout(placeholder)
        title = QLabel("Interactive trim and 3D take replay")
        title.setStyleSheet("font-size: 14pt; font-weight: 800; color: #0f172a;")
        explanation = QLabel(
            "Open the workspace to inspect measured UAV/cable motion and commanded pose in "
            "PyVista, edit use/exclude intervals, mark physical episode breaks, assign roles, "
            "and reprocess changed takes."
        )
        explanation.setWordWrap(True)
        open_button = QPushButton("OPEN DATA WORKSPACE")
        open_button.setObjectName("primaryButton")
        open_button.clicked.connect(self._ensure_workspace)
        placeholder_layout.addStretch(1)
        placeholder_layout.addWidget(title, 0, Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addWidget(explanation, 0, Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addWidget(open_button, 0, Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addStretch(1)
        self.workspace_placeholder = placeholder
        self.workspace_layout.addWidget(placeholder, 1)
        self.tabs.addTab(self.workspace_host, "Trim & PyVista replay")
        self.refresh()

    def _tab_changed(self, index: int) -> None:
        if index == 1:
            self._ensure_workspace()

    def _ensure_workspace(self) -> None:
        if self.workspace is not None:
            return
        self.workspace_placeholder.hide()
        self.workspace_layout.removeWidget(self.workspace_placeholder)
        self.workspace_placeholder.deleteLater()
        self.workspace = TakesDatasetWidget(self.settings, self.workspace_host)
        self.workspace_layout.addWidget(self.workspace, 1)

    def refresh(self) -> None:
        rows = get_dataset_role_summary()
        episodes = sum(int(row["physical_episode_count"]) for row in rows)
        duration = sum(float(row["physical_duration_s"]) for row in rows)
        protected = sum(str(row["role"]) == "Protected Test" for row in rows)
        self.takes_card.set_metric(str(len(rows)), "immutable take IDs")
        self.episodes_card.set_metric(str(episodes), "causally segmented")
        self.duration_card.set_metric(f"{duration:.1f} s", "physical observation")
        self.protected_card.set_metric(str(protected), "never evaluated or edited")
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            role = str(row["role"])
            status = str(row["cable_status"])
            values = (
                str(row["take_id"]),
                "Protected" if role == "Protected Test" else role.replace("Provisional ", ""),
                f"{float(row['physical_duration_s']):.2f} s",
                status,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if "PROTECTED" in value:
                    item.setForeground(Qt.GlobalColor.darkRed)
                self.table.setItem(row_index, column, item)
        if self.workspace is not None:
            self.workspace.refresh()

    def close(self) -> None:
        if self.workspace is not None:
            self.workspace.close()
        super().close()
