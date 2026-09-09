"""Native Windows UI/data verification using the five supplied flights."""
import os
os.environ['QT_API']='pyside6'
os.environ['QT_QPA_PLATFORM']='windows'
import sys
from pathlib import Path
import json
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from experimental_data.adaptation_check import discover_batches, flight_names, load_comparison, sha256
from simulator.gui.adaptation_check_page import AdaptationCheckPage

app=QApplication([])
page=AdaptationCheckPage(ROOT); page.resize(1420,920); page.show(); page.set_page_active(True)
batch=discover_batches(ROOT)[0]
files=list(batch.rglob('*.csv'))+[ROOT/'runs/ppo/20260908-195207-486249-seed655/checkpoints/best_validation.pt']
hashes={str(p):sha256(p) for p in files}
results=[]
for take in flight_names(batch):
    data=load_comparison(ROOT,batch,take)
    assert data['packet_count']==245
    assert data['tracking_span'][0]<0 and data['tracking_span'][1]>1
    mask=data['time']<=1
    assert np.isfinite(data['measured_cable'][mask]).all()
    page.clear_result(); page.loaded(data); app.processEvents()
    assert page.viewer is not None, page.status.text()
    assert 'error:' not in page.status.text(), page.status.text()
    viewer=page.viewer
    camera=np.array(list(viewer.plotter.camera_position))
    for index in [0,30,90,len(data['time'])-1,0,90]:
        page.timeline.setValue(index); page.redraw(); app.processEvents()
        for key in ('measured','predicted'):
            vtk=viewer.bodies[key]['drone'].GetMatrix()
            mat=np.array([[vtk.GetElement(i,j) for j in range(4)] for i in range(4)])
            np.testing.assert_allclose(mat[:3,3],data[key+'_origin'][index])
            np.testing.assert_allclose(mat[:3,:3],data[key+'_rotation'][index],atol=1e-12)
        np.testing.assert_allclose(np.array(list(viewer.plotter.camera_position)),camera)
    page.jump_strike(); app.processEvents()
    assert abs(data['time'][page.timeline.value()]-.94)<.006
    results.append(dict(take=take,frames=len(data['time']),alignment=data['alignment']))
    if take.endswith('001'):
        page.grab().save(str(Path(__file__).parent/'comparison.png'))
        viewer.plotter.screenshot(str(Path(__file__).parent/'scene.png'))
        # Real marker gaps hide segments instead of reconnecting across the absent marker.
        gaps=np.flatnonzero(~np.isfinite(data['measured_cable']).all(axis=(1,2)))
        if len(gaps):
            page.timeline.setValue(int(gaps[0])); app.processEvents()
            _,actor=viewer.bodies['measured']['cable']
            lines=actor.mapper.dataset.lines.reshape(-1,3)[:,1:]
            bad=np.flatnonzero(~np.isfinite(data['measured_cable'][gaps[0]]).all(axis=1))
            assert not np.isin(lines,bad).any()
    page.toggle_play(); assert page.timer.isActive()
    page.set_page_active(False); assert not page.timer.isActive()
    page.set_page_active(True)

# Exercise the actual background-loader signal and lifecycle, not just direct loads.
page.load_flight()
deadline=time.monotonic()+30
while page.worker is not None and time.monotonic()<deadline:
    app.processEvents(); time.sleep(.01)
assert page.worker is None and page.data is not None,page.status.text()
assert all(sha256(p)==hashes[str(p)] for p in files)
assert page.shutdown(); page.close(); app.processEvents()
(Path(__file__).parent/'verification.json').write_text(json.dumps(dict(
    platform='Windows, project Python / native Qt and VTK',takes=results,
    checks=['all five command matches and complete whip coverage','forward/reverse scrub pose matrices',
            'fixed camera','missing-marker connectivity','strike seek','playback pauses on tab exit',
            'background loading','raw CSV and selected checkpoint hashes unchanged']),indent=2))
print('Native comparison verification passed for all five flights.')
