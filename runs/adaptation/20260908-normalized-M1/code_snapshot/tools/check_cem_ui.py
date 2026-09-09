"""Manual native Windows Qt/VTK smoke check; writes review images, never flies."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    from PySide6.QtWidgets import QApplication
    from simulator.gui.cem_page import CEMPage
    app=QApplication([]);page=CEMPage(root);page.resize(1300,900);page.show();app.processEvents()
    page.grab().save(str(output/'cem-setup.png'))
    index=page.results.findData(str(Path(args.result).resolve()))
    if index<0:raise ValueError('Result is not in the saved CEM library.')
    page.results.setCurrentIndex(index);page.inspect();page.set_page_active(True);app.processEvents()
    for index in [0,30,90,145,page.preview.timeline.maximum(),0,90,0]:
        page.preview.timeline.setValue(index);app.processEvents()
    page.preview.timeline.setValue(140);app.processEvents();page.grab().save(str(output/'cem-replay.png'))
    page.preview.viewer.plotter.screenshot(str(output/'cem-scene.png'),return_img=False)
    page.preview.views.setCurrentIndex(1);app.processEvents();page.grab().save(str(output/'cem-pva.png'))
    page.shutdown();page.close();app.processEvents()
    print('CEM setup, native 3D replay/rewind and PVA inspected successfully.')
