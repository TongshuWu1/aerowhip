"""Independently check serialized M1 commands, model identity and UI loading."""
from pathlib import Path
import os
import sys
import json
import numpy as np
from scipy.interpolate import BSpline
from numpy.polynomial import polynomial as poly
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import model_identity
from experimental_data.whip_adaptation import verify_hashes
from deployment.research_rehearsal import FIELDS
from planning.reference_correction import command_valid
import torch


def main():
    export=ROOT/'exports/M2_local_fixed_tip_reference'
    summary=json.loads((export/'export.json').read_text())
    rehearsal=ROOT/summary['rehearsal'];job=ROOT/summary['correction_job']
    cfg=json.loads((job/'settings.json').read_text())
    source=ROOT/'runs/adaptation/M2-paper-20260913'
    verify_hashes(json.loads((source/'fit/result.json').read_text())['candidate_hashes'])
    verify_hashes(json.loads((job/'source_hashes.json').read_text()))
    assert model_identity(source/'candidate/model.json')[0]==model_identity(rehearsal/'model.json')[0]
    data=np.genfromtxt(export/'fullstate_30hz.csv',delimiter=',',names=True)
    assert list(data.dtype.names)==list(FIELDS)
    rows=np.column_stack([data[k] for k in data.dtype.names]);t=rows[:,0];commands=rows[:,1:]
    assert np.isfinite(rows).all() and np.allclose(np.diff(t),1/30,atol=1e-12,rtol=0)
    assert sha256_file(export/'fullstate_30hz.csv')==summary['csv_sha256']==sha256_file(rehearsal/'fullstate_30hz.csv')
    with np.load(rehearsal/'rehearsal.npz') as f:arrays={k:f[k] for k in f.files}
    np.testing.assert_allclose(commands,arrays['commands'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(t,arrays['command_time_s'],atol=1e-12,rtol=0)
    with np.load(job/'plan.npz') as f:plan={k:f[k] for k in f.files}
    origin=np.asarray(cfg['launch']['origin_m'])
    spline=BSpline(plan['spline_knots_s'],np.r_[np.tile(origin,(3,1)),plan['position_control_points_m']],int(plan['spline_degree']))
    handover=float(plan['fixed_handover_s']);prefix=round(handover*30)+1
    regenerated=np.c_[*[spline(t[:prefix],nu=i) for i in range(3)],np.zeros((prefix,2))]
    np.testing.assert_allclose(commands[:prefix],regenerated,atol=1e-10,rtol=0)
    np.testing.assert_allclose(commands[0,:3],origin,atol=1e-12)
    np.testing.assert_allclose(commands[-1,:3],origin,atol=1e-12)
    np.testing.assert_allclose(commands[[0,-1],3:],0,atol=1e-10)
    assert np.all(commands[:,9:]==0)
    assert bool(command_valid(torch.tensor(commands)[None],cfg['limits']).all())
    recovery=summary['recovery'];brake=np.asarray(recovery['brake_coefficients_normalized'])
    ret=np.asarray(recovery['return_coefficients_normalized']);bt=recovery['brake_end_s'];rt=recovery['return_s']
    for derivative in range(3):
        bd=poly.polyder(brake,m=derivative,axis=0)/bt**derivative
        rd=poly.polyder(ret,m=derivative,axis=0)/rt**derivative
        np.testing.assert_allclose(poly.polyval(0.,bd),commands[prefix-1,3*derivative:3*derivative+3],atol=1e-9)
        np.testing.assert_allclose(poly.polyval(1.,bd),poly.polyval(0.,rd),atol=1e-9)
        np.testing.assert_allclose(poly.polyval(1.,rd),origin if derivative==0 else np.zeros(3),atol=1e-9)
    assert recovery['settings']==json.loads((job/'reference.json').read_text())['recovery_settings']
    assert bt>=1. and summary['recovery_prediction_complete']
    assert summary['standard_fast_cable_max_difference_m']<=1e-5
    # Load the actual completed artifact using the normal Rehearsal inspector.
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication
    from simulator.gui.rehearsal_workspace import RehearsalWorkspace
    app=QApplication.instance() or QApplication([])
    page=RehearsalWorkspace(ROOT,inspection_only=True)
    try:
        page.load_result(rehearsal);page.canvas.draw();page.whip_canvas.draw()
        assert page.save.isEnabled() and page.play.isEnabled()
        assert 'fixed-reference correction' in page.status.text()
        message=page.status.text()
    finally:page.close();app.processEvents()
    receipt=dict(csv_sha256=summary['csv_sha256'],rows=len(rows),rate_hz=30,duration_s=float(t[-1]),
        finalized_M2_identity_unchanged=True,original_reference_hash_verified=True,
        exact_saved_commands=True,independent_bspline_derivatives=True,
        recovery_continuous_PVA=True,original_slower_brake_settings_unchanged=True,
        start_end_settled=True,command_envelope_passed=True,offscreen_UI_loaded=True,
        ui_status=message,hardware='Windows; optimizer ran on RTX 4080',physical_M2_flight_performed=False)
    atomic_json(export/'verification.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
