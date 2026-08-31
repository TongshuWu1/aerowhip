"""Capture the four normal production GUI pages for the Milestone-4B report."""

from __future__ import annotations

from pathlib import Path
import sys

repository_root = Path(__file__).resolve().parents[1]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from PySide6.QtWidgets import QApplication

from simulator.gui.main_window import SimulatorMainWindow
from simulator.parameters import SimulatorSettings
from simulator.production import PROJECT_ROOT


def main() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    settings = SimulatorSettings.load(PROJECT_ROOT / "config" / "default.json")
    window = SimulatorMainWindow(settings)
    window.resize(1320, 860)
    window.show()
    application.processEvents()
    window.simulator_page.load_latest_plan()
    output = PROJECT_ROOT / "reports"
    output.mkdir(parents=True, exist_ok=True)
    names = ("simulator", "data", "model", "planning")
    for index, name in enumerate(names):
        window.main_tabs.setCurrentIndex(index)
        application.processEvents()
        path = output / f"milestone4b_ui_{name}.png"
        if not window.grab().save(str(path), "PNG"):
            raise RuntimeError(f"Could not save {path}")
        print(path)
    window.close()
    application.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
