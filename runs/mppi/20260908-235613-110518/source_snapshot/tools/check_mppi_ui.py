"""Native Windows seven-page UI and MPPI saved-result replay smoke test."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    from PySide6.QtWidgets import QApplication
    from simulator.gui.main_window import SimulatorMainWindow
    from experimental_data.io import atomic_json
    import numpy as np
    app=QApplication([]);window=SimulatorMainWindow(root,{}, {}, {})
    window.resize(1500,980);window.show();app.processEvents()
    assert window.main_tabs.count()==7
    assert not hasattr(window,'cem_page')
    assert window.main_tabs.tabText(5)=='MPPI Planner'
    assert window.main_tabs.tabText(6)=='Adaptation Check'
    page=window.mppi_page;window.main_tabs.setCurrentIndex(5);app.processEvents()
    assert page.settings_values()['optimizer']=='mppi'
    window.grab().save(str(output/'mppi-setup.png'))
    index=page.results.findData(str(Path(args.result).resolve()))
    assert index>=0
    page.results.setCurrentIndex(index);page.inspect();app.processEvents()
    assert 'MPPI' in page.preview.result_label
    assert page.preview.save.isEnabled() and page.preview.package.isEnabled()
    assert 'virtual_force_n' not in page.preview.arrays
    for index in [0,30,90,145,page.preview.timeline.maximum(),0,90,0]:
        page.preview.timeline.setValue(index);app.processEvents()
    page.preview.timeline.setValue(140);app.processEvents()
    window.grab().save(str(output/'mppi-replay.png'))
    page.preview.viewer.plotter.screenshot(str(output/'mppi-scene.png'),return_img=False)
    page.preview.views.setCurrentIndex(1);app.processEvents()
    window.grab().save(str(output/'mppi-pva.png'))
    for i in [0,6,5]:window.main_tabs.setCurrentIndex(i);app.processEvents()
    atomic_json(output/'verification.json',dict(pages=7,adaptation_check_index=6,cem_page_removed=True,
        mppi_result=str(Path(args.result).resolve()),replay_and_rewind=True,
        native_viewer=True,csv_rows=page.preview.table.rowCount(),no_force_output=True))
    window.close();app.processEvents()
    print('Native seven-page UI, MPPI-only planner, VTK replay, rewind and PVA passed.')
