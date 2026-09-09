"""Read-only native UI render audit. Never start training or fitting."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from simulator.gui.main_window import SimulatorMainWindow
from simulator.workflow import read_json

if __name__=='__main__':
    output=ROOT/'runs/audits/model-workspace-ui';output.mkdir(parents=True,exist_ok=True)
    files=[*list((ROOT/'config').rglob('*.json')),
        ROOT/'runs/ppo/20260908-195207-486249-seed655/checkpoints/best_validation.pt',
        ROOT/'rehearsal_csv_and_result_in_real_flight/20260908-195207-486249-seed655_best_validation/adp0/simulation_csv/fullstate_30hz.csv']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    app=QApplication.instance() or QApplication([])
    w=SimulatorMainWindow(ROOT,*[read_json(ROOT/'config'/n) for n in ['model.json','task.json','ppo.json']]);w.show()
    for index,label in [(0,'models'),(1,'recordings'),(2,'ppo'),(3,'historical'),(4,'rehearsal'),(5,'mppi'),(6,'flight-check')]:
        w.main_tabs.setCurrentIndex(index);app.processEvents();app.processEvents()
        w.grab().save(str(output/(label+'.png')))
    w.main_tabs.setCurrentIndex(0);w.model_page.tabs.setCurrentIndex(1);app.processEvents()
    w.grab().save(str(output/'adaptation-jobs.png'))
    assert not w.model_page.job.running and not w.training_page.job.running and not w.fullstate_page.job.running
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (output/'verification.json').write_text(json.dumps(dict(pages=w.main_tabs.count(),protected_files=len(hashes),unchanged=True,training_started=False,fitting_started=False),indent=2))
    w.close();app.processEvents();print('Verified seven native pages and unchanged protected configuration/policy/CSV.',flush=True)
