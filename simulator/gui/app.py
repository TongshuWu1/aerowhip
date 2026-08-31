"""PySide6/PyVista application construction kept separate from physics."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtWidgets import QApplication

from ..parameters import SimulatorSettings
from .main_window import SimulatorMainWindow


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    settings = SimulatorSettings.load(project_root / "config" / "default.json")
    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(sys.argv)
    application.setApplicationName("Aerial Cable Research Simulator")
    application.setOrganizationName("Aerial Cable Research")
    window = SimulatorMainWindow(settings)
    window.show()
    return application.exec() if owns_application else 0
