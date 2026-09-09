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
    app=QApplication([]);app.setStyle('Fusion');window=PVAResearchWindow(ROOT);window.show();errors=[];startup={}
    def capture(index):
        try:
            if index==0:
                startup.update(page=window.main_tabs.currentIndex(),rehearsal=str(window.inspector.directory),
                    camera=window.inspector.camera.currentText(),speed=window.inspector.speed.currentText(),
                    frame=window.inspector.timeline.value())
                if window.inspector.viewer is not None:
                    window.inspector.viewer.plotter.render()
                    window.inspector.viewer.plotter.screenshot(str(out/'startup-render.png'))
                    startup['viewer_ready']=True
                    startup['viewport_size']=[window.inspector.viewer.width(),window.inspector.viewer.height()]
                    if min(startup['viewport_size'])<100:raise ValueError('Startup replay viewport did not lay out')
                window.grab().save(str(out/'startup-replay.png'))
            window.main_tabs.setCurrentIndex(index);app.processEvents()
            if index==4 and window.rehearsals.count():
                if args.rehearsal:
                    selected=window.rehearsals.findData(str(args.rehearsal.resolve()))
                    if selected<0:raise ValueError('Requested rehearsal missing from library')
                    window.rehearsals.setCurrentIndex(selected)
                window.open_rehearsal();app.processEvents()
                import numpy as np
                time=window.inspector.metadata.get('predicted_hit_time_s')
                if time is None:
                    arrays=window.inspector.arrays
                    closest=int(np.argmin(np.linalg.norm(arrays['cable_positions_m'][:,-1]-arrays['target_position_m'],axis=-1)))
                    time=arrays['prediction_time_s'][closest]
                frame=int(np.argmin(abs(window.inspector.arrays['prediction_time_s']-time)))
                window.inspector.timeline.setValue(frame);window.inspector.draw_frame(frame);app.processEvents()
                viewer=window.inspector.viewer
                viewer.plotter.render();viewer.plotter.screenshot(str(out/'rehearsal-render.png'))
                pixels=viewer.plotter.screenshot(return_img=True)
                if np.ptp(pixels)<30:raise ValueError('Native VTK render is empty')
                inspector=window.inspector
                if inspector.views.count()>3 and inspector.views.isTabVisible(3):
                    inspector.views.setCurrentIndex(3);app.processEvents();window.grab().save(str(out/'pull-and-release.png'))
                    inspector.views.setCurrentIndex(0);app.processEvents()
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
            atomic_json(out/'result.json',dict(errors=errors,platform=app.platformName(),pages=6,started_jobs=False,startup=startup,
                rendered_rehearsal=str(window.inspector.directory) if window.inspector.arrays is not None else None))
            window.close();QTimer.singleShot(300,app.quit)
    QTimer.singleShot(600,lambda:capture(0));app.exec()
    if errors:raise RuntimeError(str(errors))
