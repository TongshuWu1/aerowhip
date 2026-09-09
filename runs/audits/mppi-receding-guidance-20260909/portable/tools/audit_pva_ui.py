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
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'runs/audits/pva-ui')
    parser.add_argument('--rehearsal',type=Path);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    app=QApplication([]);app.setStyle('Fusion');window=PVAResearchWindow(ROOT);window.show();errors=[]
    def capture(index):
        try:
            window.main_tabs.setCurrentIndex(index);app.processEvents()
            if index==4 and window.rehearsals.count():
                if args.rehearsal:
                    selected=window.rehearsals.findData(str(args.rehearsal.resolve()))
                    if selected<0:raise ValueError('Requested rehearsal missing from library')
                    window.rehearsals.setCurrentIndex(selected)
                window.open_rehearsal();app.processEvents()
                import numpy as np
                time=window.inspector.metadata.get('predicted_hit_time_s') or 0.
                frame=int(np.argmin(abs(window.inspector.arrays['prediction_time_s']-time)))
                window.inspector.timeline.setValue(frame);window.inspector.draw_frame(frame);app.processEvents()
                viewer=window.inspector.viewer
                viewer.plotter.render();viewer.plotter.screenshot(str(out/'rehearsal-render.png'))
                pixels=viewer.plotter.screenshot(return_img=True)
                if np.ptp(pixels)<30:raise ValueError('Native VTK render is empty')
            if not window.grab().save(str(out/f'page-{index}.png')):raise ValueError('Screenshot save failed')
            if index==2:
                p=window.ppo_page;p.tabs.setCurrentIndex(2);app.processEvents();window.grab().save(str(out/'ppo-library.png'));p.tabs.setCurrentIndex(0)
            if index==3:
                p=window.mppi_page
                for tab,label in ((1,'progress'),(2,'library')):
                    p.tabs.setCurrentIndex(tab);p.poll();app.processEvents();window.grab().save(str(out/f'mppi-{label}.png'))
                p.tabs.setCurrentIndex(0)
        except Exception as e:errors.append(dict(page=index,error=str(e)))
        if index<5:QTimer.singleShot(600,lambda:capture(index+1))
        else:
            atomic_json(out/'result.json',dict(errors=errors,platform=app.platformName(),pages=6,started_jobs=False,
                rendered_rehearsal=str(window.inspector.directory) if window.inspector.arrays is not None else None))
            window.close();QTimer.singleShot(300,app.quit)
    QTimer.singleShot(600,lambda:capture(0));app.exec()
    if errors:raise RuntimeError(str(errors))
