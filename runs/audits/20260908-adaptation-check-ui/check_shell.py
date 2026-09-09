import os
os.environ['QT_API']='pyside6'; os.environ['QT_QPA_PLATFORM']='windows'
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from simulator.gui.main_window import SimulatorMainWindow
app=QApplication([]); app.setStyle('Fusion')
window=SimulatorMainWindow(ROOT,*[json.loads((ROOT/'config'/f'{name}.json').read_text()) for name in ('model','task','ppo')])
window.resize(1440,950); window.show()
page=window.adaptation_check_page
window.navigation_buttons[6].click(); page.load_flight()
end=time.monotonic()+30
while page.worker is not None and time.monotonic()<end: app.processEvents(); time.sleep(.01)
assert page.viewer is not None and page.data is not None,page.status.text()
assert '3D scene error' not in page.status.text(),page.status.text()
page.jump_strike(); app.processEvents()
app.primaryScreen().grabWindow(window.winId()).save(str(Path(__file__).parent/'full-window.png'))
page.views.setCurrentIndex(1); app.processEvents()
app.primaryScreen().grabWindow(window.winId()).save(str(Path(__file__).parent/'error-plots.png'))
assert window.main_tabs.count()==7
window.close(); app.processEvents()
print('Full application: seventh navigation page, asynchronous loading, plots and shutdown passed.')
