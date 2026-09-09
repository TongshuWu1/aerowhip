"""Render the actual Qt/VTK workflow without starting any fit or planner job."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from simulator.gui.pva_main_window import PVAResearchWindow
from experimental_data.io import atomic_json

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    out=ROOT/'runs/audits/pva-ui';out.mkdir(parents=True,exist_ok=True)
    app=QApplication([]);window=PVAResearchWindow(ROOT);window.show();errors=[]
    def capture(index):
        try:
            window.main_tabs.setCurrentIndex(index);app.processEvents()
            if index==4 and window.rehearsals.count():window.open_rehearsal();app.processEvents()
            if not window.grab().save(str(out/f'page-{index}.png')):raise ValueError('Screenshot save failed')
            if index==2:
                p=window.ppo_page;p.tabs.setCurrentIndex(2);app.processEvents();window.grab().save(str(out/'ppo-library.png'));p.tabs.setCurrentIndex(0)
        except Exception as e:errors.append(dict(page=index,error=str(e)))
        if index<5:QTimer.singleShot(600,lambda:capture(index+1))
        else:
            atomic_json(out/'result.json',dict(errors=errors,platform=app.platformName(),pages=6,started_jobs=False))
            window.close();QTimer.singleShot(300,app.quit)
    QTimer.singleShot(600,lambda:capture(0));app.exec()
    if errors:raise RuntimeError(str(errors))
