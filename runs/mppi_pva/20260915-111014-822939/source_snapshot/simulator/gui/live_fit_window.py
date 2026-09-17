"""Open the live fitting page without restarting the research app or fit worker."""
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication
from .pva_model_page import PVAModelPage
from .theme import APP_STYLE,load_application_font


def main():
    app=QApplication(sys.argv);load_application_font()
    page=PVAModelPage(Path(__file__).resolve().parents[2])
    page.setWindowTitle('AeroWhip · Models & fitting')
    page.setStyleSheet(APP_STYLE);page.resize(1240,820)
    page.tabs.setCurrentIndex(2 if '--correction' in sys.argv else 1)
    page.fit_start.hide()
    app.aboutToQuit.connect(page.shutdown)
    page.show()
    return app.exec()


if __name__=='__main__':
    sys.exit(main())
