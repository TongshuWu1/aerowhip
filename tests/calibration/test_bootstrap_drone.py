from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
import torch
from simulator.drone_pose_residual import DronePoseResidual,save_residual,load_residual
from simulator.drone_pose_response import PoseResponseParameters,PoseResponseState,CommandSchedule,predict_pose,derivatives
from experimental_data.bootstrap_drone import TranslationBatch,trial_args
from experimental_data.nominal_pose_fit import PreparedTrial
from experimental_data.bootstrap_bundle import read,VERSION
from experimental_data.io import sha256_file


def test_residual_bound_translation_invariance_and_checkpoint(tmp_path):
    torch.manual_seed(4);net=DronePoseResidual().double()
    p=torch.randn(7,3,dtype=torch.float64);v=torch.randn_like(p);b=torch.randn_like(p);cmd=torch.randn(7,11,dtype=torch.float64)
    assert torch.count_nonzero(net(p,v,b,cmd))==0
    with torch.no_grad():net.net[-1].weight.normal_();net.net[-1].bias.fill_(3)
    actual=net(p,v,b,cmd);assert actual.abs().max()<=.5
    shifted=cmd.clone();shifted[:,:3]+=cmd.new_tensor([2.,3.,4.])
    torch.testing.assert_close(actual,net(p+p.new_tensor([2.,3.,4.]),v,b,shifted))
    path=tmp_path/'model.pt';save_residual(path,net);loaded=load_residual(path,sha256_file(path),'cpu')
    torch.testing.assert_close(actual,loaded(p,v,b,cmd))
    with pytest.raises(ValueError,match='hash'):load_residual(path,'bad','cpu')


def test_residual_is_realized_acceleration_not_direct_attitude_drive():
    z=torch.zeros(1,3,dtype=torch.float64);eye=torch.eye(3,dtype=z.dtype)[None]
    state=PoseResponseState(z,z,eye,z,z,eye,'explicit_calibration',0.)
    cmd=torch.zeros(1,11,dtype=z.dtype);params=PoseResponseParameters(4,4,3,3,1,1,.08,0,attitude_drive_model='independent_scale_v3')
    a,alpha=derivatives(state,cmd,params)
    def residual(*args):return z+z.new_tensor([.2,0,.1])
    corrected,corrected_alpha=derivatives(state,cmd,params,residual)
    torch.testing.assert_close(corrected-a,z+z.new_tensor([.2,0,.1]))
    torch.testing.assert_close(alpha,corrected_alpha,atol=0,rtol=0)


def test_training_recurrence_matches_full_pose_and_network_gradient():
    root=Path(__file__).resolve().parents[2];old=root/'data/nominal_drone_runs/20260908-062555-496491-legacy-whip-nominal-pose'
    if not old.exists():pytest.skip('Legacy data fixture unavailable')
    names=['whip1_001','whip1_003'];trials=[PreparedTrial(old/'inputs'/VERSION/n) for n in names]
    params=PoseResponseParameters(**read(old/'final_all_three/nominal_model.json')['parameters'])
    batch=TranslationBatch(trials,params,device='cpu');net=DronePoseResidual().double()
    with torch.no_grad():net.net[-1].bias[:]=torch.tensor([.05,-.02,.03])
    predictions,_=batch.predict(net)
    for t,p in zip(trials,predictions):
        out=predict_pose(**trial_args(t,params,device='cpu',maneuver_only=True),residual=net)
        torch.testing.assert_close(p,out['position_origin_m'][0],atol=1e-10,rtol=0)
    loss=batch.loss(net);g=torch.autograd.grad(loss,net.net[-1].bias)[0][0]
    initial=net.net[-1].bias[0].detach().clone();eps=1e-5
    with torch.no_grad():
        net.net[-1].bias[0]=initial+eps;up=batch.loss(net)
        net.net[-1].bias[0]=initial-eps;down=batch.loss(net);net.net[-1].bias[0]=initial
    torch.testing.assert_close(g,(up-down)/(2*eps),atol=1e-7,rtol=1e-4)
