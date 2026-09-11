"""Construct and launch the direct-PVA research desktop application."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtWidgets import QApplication

from .main_window import SimulatorMainWindow


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication([sys.argv[0]])
    application.setApplicationName("Aerial Cable Research Simulator")
    application.setStyle('Fusion')
    application.setOrganizationName("Aerial Cable Research")
    window = SimulatorMainWindow(project_root)
    window.show()
    return application.exec() if owns_application else 0
