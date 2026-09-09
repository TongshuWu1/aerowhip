"""Native Windows review; does not apply a model or launch training."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication
from experimental_data.io import canonical_json_hash, sha256_file
from simulator.workflow import read_json
from simulator.gui.cable_residual_page import CableResidualPage
from simulator.cable.residual import cable_drag_description

job = ROOT/'data/cable_residual_runs/20260907-232213-673553'
app = QApplication.instance() or QApplication([])
page = CableResidualPage(ROOT)
page.resize(1180, 720)
page.runs.setCurrentIndex(page.runs.findData(str(job)))
page.show()
app.processEvents()
page.grab().save(str(job/'ui.png'))
review = read_json(job/'review.json')
assert page.table.rowCount() == 4
assert not page.job.running
assert page.apply.isEnabled() == review['accepted']
assert 'learned drag' in page.summary.text()
model = read_json(ROOT/'config/model.json')
assert canonical_json_hash(model) == '51a6349546d255cd9ebdf8913f5ff10fc70e58dc48a70f725518771a97fb3d11'
checkpoint = ROOT/'runs/ppo/20260906-201957-294110-measured-mass-seed653/checkpoints/best_validation.pt'
assert sha256_file(checkpoint) == '18f33e89fea72e683c40359853e40710f52ecb45623d53669ed6e240334078d7'
result = dict(native_windows_ui=True, rows=page.table.rowCount(), training_running=False,
              active_calibration_unchanged=True, selected_ppo_unchanged=True,
              candidate_applied=False, accepted=review['accepted'],
              summary=page.summary.text(), active_model_label=cable_drag_description(model))
(job/'ui_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
page.close()
app.processEvents()
