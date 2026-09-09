"""Exercise the real Full-state UI and worker on Windows/CUDA; no vehicle."""
import json
from pathlib import Path
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root))
from simulator.gui.fullstate_page import FullStatePage
from deployment.fullstate_playback import load_reference

app=QApplication([])
page=FullStatePage(root)
page.resize(1550,1000)
checkpoint=root/'runs/ppo/20260906-201957-294110-measured-mass-seed653/checkpoints/best_validation.pt'
index=page.checkpoints.findData(str(checkpoint))
assert index>=0
page.checkpoints.setCurrentIndex(index)
page.show()
page.set_page_active(True)
state={'executed':False}
errors=[]

def poll():
    if page.execute.isEnabled() and not state['executed'] and page.worker is not None:
        state['executed']=True
        page.execute_sequence()

def done():
    try:
        rows,meta=load_reference(page.csv_path.parent)
        assert meta['schema']=='gentle_recovery_fullstate_v4'
        assert page.open_reference_plot.isEnabled()
        assert 'Whip unchanged' in page.plan_note.text()
        result=dict(native_windows_ui=True,directory=str(page.directory),rows=len(rows),
                    schema=meta['schema'],note=page.plan_note.text())
        (Path(__file__).parent/'native_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        page.grab().save(str(Path(__file__).parent/'native_ui.png'))
        print(json.dumps(result),flush=True)
    except Exception as error:
        errors.append(str(error));print(repr(error),flush=True)
    page.close();app.exit(1 if errors else 0)

page.flight_finished.connect(done)
timer=QTimer();timer.timeout.connect(poll);timer.start(100)
QTimer.singleShot(100,page.start_rehearsal)
QTimer.singleShot(90000,lambda: (errors.append('timeout'),page.stop_rehearsal()))
raise SystemExit(app.exec())
