from pathlib import Path
import os,sys
os.environ.pop('QT_QPA_PLATFORM',None)
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from simulator.gui.fullstate_page import FullStatePage
app=QApplication([])
app.setFont(QFont('Segoe UI',10))
page=FullStatePage(ROOT)
page.directory=Path(__file__).resolve().parent
page.resize(1650,1000)
page.show()
page.set_page_active(True)
page.show_plan(str(page.directory/'plan_001/commands.csv'),{})
page.execute_sequence()
page.preview_timer.stop()
page.preview_started-=1.
page.preview_tick()
app.processEvents()
page.viewer.plotter.screenshot(str(page.directory/'native_recovery.png'))
page.grab().save(str(page.directory/'ui_recovery.png'))
page.preview_started-=7.
page.preview_tick()
app.processEvents()
page.table.scrollToBottom()
app.processEvents()
page.viewer.plotter.screenshot(str(page.directory/'native_complete.png'))
page.grab().save(str(page.directory/'ui_complete.png'))
assert 'finished' in page.status.text()
assert page.table.item(page.table.rowCount()-1,13).text()=='hover_hold'
assert page.shutdown()
page.close()
app.processEvents()
print('Full native VTK 3D preview and final CSV phase verified on Windows')
