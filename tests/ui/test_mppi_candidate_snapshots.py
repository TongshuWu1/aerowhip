from types import SimpleNamespace
import os
import numpy as np
import torch
from planning.mppi_live import RolloutCapture,series,publish
from simulator.gui.mppi_live_view import load_snapshot


def test_capture_clones_states_and_publishes_prior_best_without_relabeling(tmp_path):
    p=torch.zeros(4,3,dtype=torch.float64);q=torch.zeros(4,5,3,dtype=torch.float64)
    env=SimpleNamespace(pose=SimpleNamespace(position=p,rotation=torch.eye(3).repeat(4,1,1)),
        state=SimpleNamespace(positions_m=q),command=torch.zeros(4,9))
    capture=RolloutCapture();capture(env);p.add_(1);q.add_(2);capture(env)
    result=dict(failed=torch.tensor([False,False,True,False]),success=torch.zeros(4,dtype=torch.bool),duration_s=torch.ones(4))
    score=torch.tensor([3.,2.,-torch.inf,1.]);best=series(capture,0,7,score,result,'Best so far')
    actual=SimpleNamespace(index=14,target=torch.ones(1,3),initial_pose=env.pose,initial_state=env.state,frames=[])
    publish(tmp_path,actual,capture,score,result,best,8)
    data=load_snapshot(tmp_path/'live.npz')
    assert data['series_iterations'][0]==7 and data['iteration']==8 and data['command_step']==14
    np.testing.assert_array_equal(data['origin_positions_m'][0],[[0,0,0],[1,1,1]])
    np.testing.assert_array_equal(data['cable_positions_m'][0,0],np.zeros((5,3)))
    np.testing.assert_allclose(data['time_s'],[14/30,15/30])
    assert np.isneginf(data['scores']).any() and not (tmp_path/'live.tmp.npz').exists()


def test_quick_setup_preserves_horizon_reward_and_does_not_start(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.pva_workspace import PVAPlannerPage
    app=QApplication.instance() or QApplication([])
    page=PVAPlannerPage(tmp_path,'mppi');before=page.collect();page.quick_setup();after=page.collect()
    assert after['mppi']['samples']==256 and after['mppi']['minimum_iterations']==3 and after['mppi']['patience']==2
    assert after['mppi']['horizon_s']==before['mppi']['horizon_s'] and after['reward']==before['reward']
    assert after['visualization']['live_mppi'] and not list((tmp_path/'runs/mppi_pva').glob('*/status.json'))
    page.shutdown();page.close();app.processEvents()
