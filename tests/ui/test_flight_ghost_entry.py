from types import SimpleNamespace
from PySide6.QtWidgets import QApplication,QComboBox,QTabWidget,QWidget
from simulator.gui.pva_main_window import PVAResearchWindow


def test_recordings_shortcut_selects_and_loads_flight():
    app=QApplication.instance() or QApplication([])
    page=QWidget();page.worker=None;page.batches=QComboBox();page.takes=QComboBox();page.views=QTabWidget()
    page.batches.addItem('M0','batch');page.takes.addItems(['whip_m0_001','whip_m0_002'])
    def select(batch,take):
        page.batches.setCurrentIndex(page.batches.findData(batch));page.takes.setCurrentText(take);return True
    page.select_flight=select
    page.views.addTab(QWidget(),'3D')
    calls=[];page.load_flight=lambda:calls.append((page.batches.currentData(),page.takes.currentText()))
    workspace=QTabWidget();workspace.addTab(QWidget(),'Models');workspace.addTab(page,'Ghost')
    tabs=QTabWidget()
    for i in range(6):tabs.addTab(QWidget(),str(i))
    owner=SimpleNamespace(adaptation_check_page=page,flight_workspace=workspace,main_tabs=tabs)
    PVAResearchWindow.open_flight_comparison(owner,'batch','whip_m0_002')
    assert calls==[('batch','whip_m0_002')]
    assert workspace.currentWidget() is page and tabs.currentIndex()==5
    workspace.close();tabs.close()


def test_loop_keeps_both_sources_on_same_timeline(tmp_path,monkeypatch):
    import numpy as np
    from simulator.gui.adaptation_check_page import AdaptationCheckPage
    app=QApplication.instance() or QApplication([])
    page=AdaptationCheckPage(tmp_path)
    monkeypatch.setattr(page,'redraw',lambda *args:None)
    page.data={'time':np.array([0.,.01,.02])}
    page.timeline.setRange(0,2);page.loop.setChecked(True);page.playing=True
    page.last_tick=0.;page.play_time=0.
    monkeypatch.setattr('simulator.gui.adaptation_check_page.clock.monotonic',lambda:1.)
    page.tick()
    assert page.timeline.value()==0 and page.playing and page.play_time==0
    page.loop.setChecked(False);page.last_tick=0.;page.tick()
    assert page.timeline.value()==2 and not page.playing
    page.shutdown();page.close()
