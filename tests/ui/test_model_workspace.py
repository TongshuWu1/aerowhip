import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import json
import shutil
import numpy as np
from PySide6.QtWidgets import QApplication
from simulator.gui.rehearsal_workspace import RehearsalWorkspace

ROOT=Path(__file__).resolve().parents[2]


def test_editing_launch_setup_invalidates_old_rehearsal_and_exports(tmp_path):
    app=QApplication.instance() or QApplication([]);shutil.copytree(ROOT/'config',tmp_path/'config')
    folder=tmp_path/'saved';folder.mkdir();n=31;t=np.arange(n)/30;q=np.zeros((n,12,3));q[:,:,2]=1.5-np.arange(12)/12
    commands=np.zeros((n,11));commands[:,2]=1.555
    np.savez(folder/'rehearsal.npz',command_time_s=t,commands=commands,command_phase=np.ones(n),prediction_time_s=t,
        cable_positions_m=q,origin_positions_m=commands[:,:3],origin_rotations=np.tile(np.eye(3),(n,1,1)),
        force_time_s=t,virtual_force_n=np.zeros((n,3)),target_position_m=[1,0,1.4])
    (folder/'rehearsal.json').write_text(json.dumps(dict(schema='research_fullstate_30hz_v1',checkpoint='saved.pt',checkpoint_sha256='a'*64,
        initial_tracking_origin_m=[0,0,1.555],target_position_m=[1,0,1.4],whip_end_s=1,total_duration_s=1,
        predicted_valid_hit=False,minimum_tip_distance_m=1,recovery_prediction_complete=True,prediction_valid_through_s=1)))
    page=RehearsalWorkspace(tmp_path);page.load_result(folder)
    assert page.save.isEnabled() and page.table.rowCount()==31 and page.viewer is None
    page.start_spins[0].setValue(.02)
    assert page.arrays is None and page.table.rowCount()==0 and not page.save.isEnabled() and not page.package.isEnabled()
    assert not hasattr(page,'job') and (folder/'rehearsal.npz').exists()
    page.shutdown();page.close();app.processEvents()
