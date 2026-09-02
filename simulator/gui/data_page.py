"""Compact read-only dataset page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fitting.production_status import get_dataset_role_summary
from .theme import MetricCard


class DataPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
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
        self.table = QTableWidget(0, 4, self)
        self.table.setObjectName("take_role_table")
        self.table.setHorizontalHeaderLabels(("Take", "Role", "Duration", "Status"))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        advanced = QGroupBox("Advanced details")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_layout = QVBoxLayout(advanced)
        self.advanced_label = QLabel(
            "Episode counts, residual eligibility, masks, timestamps, and data contracts remain "
            "available in the saved dataset artifacts. They are intentionally omitted here."
        )
        self.advanced_label.setWordWrap(True)
        advanced_layout.addWidget(self.advanced_label)
        advanced.toggled.connect(self.advanced_label.setVisible)
        self.advanced_label.setVisible(False)
        layout.addWidget(advanced)
        self.refresh()

    def refresh(self) -> None:
        rows = get_dataset_role_summary()
        episodes = sum(int(row["physical_episode_count"]) for row in rows)
        duration = sum(float(row["physical_duration_s"]) for row in rows)
        protected = sum(str(row["role"]) == "Protected Test" for row in rows)
        self.takes_card.set_metric(str(len(rows)), "immutable take IDs")
        self.episodes_card.set_metric(str(episodes), "causally segmented")
        self.duration_card.set_metric(f"{duration:.1f} s", "physical observation")
        self.protected_card.set_metric(str(protected), "never evaluated")
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
