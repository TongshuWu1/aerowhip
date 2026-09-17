"""Measured-data regression for the previously ill-conditioned fitting window."""
from pathlib import Path
import pytest
import torch
from experimental_data.current_adaptation import read
from experimental_data.preliminary_prepare import PreliminaryTrial
from experimental_data.current_adaptation_fit import cable_windows,join_windows,cable_objectives
from experimental_data.cuda_cable_fit import CudaCableFit
from simulator.research_execution import ResearchExecutionModel


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_smoothed_frame_one_second_residual_gradient_matches_numerical_difference(tmp_path):
    source=Path(__file__).resolve().parents[2]/'runs/adaptation/20260909-preliminary1-M0-v2'
    if not source.exists():pytest.skip('Local preliminary data not installed')
    model=read(source/'candidate/model.json');model['cable']['curvature_frame_regularization']=2e-5
    base=read(source/'source_candidate/model.json');windows=read(source/'windows.json')
    e=ResearchExecutionModel.from_mapping(model,root=source/'candidate',device='cuda',trainable_residuals=True)
    trials=[]
    for name in ('figure8_001-00641','figure8_001-02441'):
        t=PreliminaryTrial(source,next(w for w in windows if w['name']==name),base)
        t.cable_history_s=1.;t.cable_velocity_weight_tau_s=.02
        assert t.role=='training';trials.append(t)
    records,rejected=cable_windows(trials,e.physics,[0.],1.)
    assert len(records)==2 and not rejected
    data=join_windows(records)
    params=data['q'].new_tensor([model['cable']['EI_n_m2'],model['cable']['Cb_n_m2_s']])
    forward=CudaCableFit(e,data,params,block_steps=3)
    assert forward.verify_eager(data,tmp_path/'capture_eager.json')['passed']
    ids=list(e.cable.marker_node_indices[1:]);bias=e.physics.motion_residual.correction_head.bias
    q,_=forward(data,gradients=True);loss=cable_objectives(q,data['truth'],ids).mean()
    grad=torch.autograd.grad(loss,bias)[0];index=int(grad.abs().argmax())
    assert torch.isfinite(grad).all() and abs(float(grad[index]))>1e-4
    original=bias.detach().clone()
    for eps in (1e-4,1e-6):
        scores=[]
        for sign in (-1,1):
            with torch.no_grad():
                bias.copy_(original);bias[index]+=sign*eps
                q,_=forward(data,gradients=False)
                scores.append(cable_objectives(q,data['truth'],ids).mean())
        fd=(scores[1]-scores[0])/(2*eps)
        torch.testing.assert_close(grad[index],fd,rtol=2e-3,atol=1e-6)
    with torch.no_grad():bias.copy_(original)
    from experimental_data.whip_full_cable import gradient_check
    checked=gradient_check(e.physics.motion_residual,
        lambda:cable_objectives(forward(data,gradients=torch.is_grad_enabled())[0],data['truth'],ids).mean(),
        tmp_path/'all_tensor_gradients.json')
    assert checked['passed'] and len(checked['directions'])>3
