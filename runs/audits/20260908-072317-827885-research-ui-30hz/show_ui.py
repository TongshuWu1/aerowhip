from pathlib import Path
import sys,json
root=Path.cwd();sys.path.insert(0,str(root))
from PySide6.QtCore import QTimer
from simulator.gui import app
Base=app.SimulatorMainWindow
class OpenedWindow(Base):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        QTimer.singleShot(300,self.show_training)
    def show_training(self):
        self.main_tabs.setCurrentIndex(2)
        (root/'runs/audits/20260908-072317-827885-research-ui-30hz/ui-ready.json').write_text(json.dumps(dict(title=self.windowTitle(),page=self.shell_page_title.text(),renderer=self.training_page.viewport.viewer.backend_name)))
app.SimulatorMainWindow=OpenedWindow
raise SystemExit(app.main())
