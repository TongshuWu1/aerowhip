from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from simulator.cable import CableConfiguration, DderModel, DderState, START_PINNED_FREE_END
from simulator.cable.residual import MotionResidual, with_acceleration_correction
from simulator.drone_pose_response import PoseResponseParameters, PoseResponseState, rotation_exp
from simulator.drone_pose_residual import DronePoseResidual
from simulator.research_pose import ResearchPoseModel
from simulator.research_execution import ResearchExecutionModel

ROOT=Path(__file__).resolve().parents[2]


def fixture(device='cpu',steps=12,batch=1):
    torch.set_num_threads(1);torch.manual_seed(71)
    payload=json.loads((ROOT/'config/research_30hz/model.json').read_text())
    cable=CableConfiguration.from_mapping(payload['cable'])
    physics=DderModel(cable.dder_parameters(EI=2e-6,Cb=1e-4))
    nn=MotionResidual(cable.node_count,hidden=16,mode='dissipative').to(device=device,dtype=torch.float64)
    with torch.no_grad():nn.net[-1].bias.fill_(.2)
    physics.motion_residual=with_acceleration_correction(nn,acceleration_limit=.5)
    drone=object.__new__(ResearchPoseModel)
    drone.parameters=PoseResponseParameters(6.,21.,6.,2.,.3,1.9,.022,.003,
        attitude_drive_model='independent_scale_v3',attitude_acceleration_scale_z=.46)
    drone.residual=DronePoseResidual().to(device=device,dtype=torch.float64)
    with torch.no_grad():drone.residual.net[-1].weight.normal_(std=.03)
    origin=torch.tensor([[-2.,0.,1.255]],device=device,dtype=torch.float64).expand(batch,-1).clone()
    zero=torch.zeros_like(origin);eye=torch.eye(3,device=device,dtype=origin.dtype)[None].expand(batch,-1,-1)
    rotation=rotation_exp(origin.new_tensor([[.02,-.03,.04]]).expand(batch,-1))
    initial=PoseResponseState(origin,zero,rotation,zero,zero,eye,'explicit_calibration',0.)
    offset=payload['recorded_data']['optitrack_to_attachment_offset_body_m']
    root=origin+torch.einsum('bij,j->bi',rotation,origin.new_tensor(offset))
    arc=origin.new_tensor([0.,*np.cumsum(cable.rest_lengths_m)])
    q=root[:,None].expand(-1,cable.node_count,-1).clone();q[:,:,2]-=arc
    q[:,:,0]+=.015*torch.sin(arc*4);q[:,:,1]+=.01*torch.sin(arc*3)
    q=physics.project_lengths(q,q[:,:1],pinned_endpoints=START_PINNED_FREE_END)
    state=DderState(q,torch.zeros_like(q))
    dt=1/150;times=np.arange(steps+1)*dt;pt=np.arange(0,times[-1]+1/30,1/30)
    packets=origin.new_zeros(batch,len(pt),11);packets[:,:,:3]=origin[:,None]
    packets[:,:,0]+=.03*torch.arange(len(pt),device=device)
    packets[:,:,3]=.2;packets[:,:,6]=.7;packets[:,:,7]=-.2;packets[:,:,8]=.3
    packets[:,:,9]=.05
    hover=packets[:,0].clone();hover[:,:3]=origin;hover[:,3:9]=0.
    model=ResearchExecutionModel(drone,cable,physics,offset,dt)
    return model,initial,state,packets,pt,times,hover


def run(f,*,packets=None,parameters=None,gradients=True,checkpoint_steps=0,graph=False):
    model,initial,state,values,pt,times,hover=f
    return model.predict(initial,state,values if packets is None else packets,pt,times,
        gradients=gradients,graph=graph,checkpoint_steps=checkpoint_steps,
        hover_command=hover,cable_parameters=parameters)


def test_recursive_forward_matches_inference_and_checkpointed_backward():
    f=fixture(batch=2)
    with torch.no_grad():f[0].physics.motion_residual.correction_head.bias.fill_(.07)
    p=f[3].clone().requires_grad_()
    ordinary=run(f,packets=p,gradients=False)
    direct=run(f,packets=p)
    checkpointed=run(f,packets=p,checkpoint_steps=4)
    for key in ordinary:
        torch.testing.assert_close(direct[key],ordinary[key],atol=2e-10,rtol=1e-10)
        torch.testing.assert_close(checkpointed[key],direct[key],atol=0,rtol=0)
    assert not ordinary['cable_positions_m'].requires_grad
    a=torch.autograd.grad(direct['cable_positions_m'][:,-1].square().sum(),p)[0]
    b=torch.autograd.grad(checkpointed['cable_positions_m'][:,-1].square().sum(),p)[0]
    assert a.norm()>0 and torch.isfinite(a).all()
    torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-9)


def score(output):
    # A tip-only derivative can be almost zero before the wave reaches the tip.
    q=output['cable_positions_m'][:,-1,1:]
    return (q*q.new_tensor([.3,-.7,.2])).sum()


@pytest.mark.parametrize('regularization',[0.,2e-7])
def test_full_chain_packet_and_material_gradients_match_finite_difference(regularization):
    f=fixture(steps=18)
    f[0].physics.parameters=replace(f[0].physics.parameters,curvature_frame_regularization=regularization)
    packets=f[3].clone().requires_grad_()
    material=packets.new_tensor([2e-6,1e-4],requires_grad=True)
    value=score(run(f,packets=packets,parameters=material,checkpoint_steps=6))
    gp,gm=torch.autograd.grad(value,(packets,material))
    assert torch.isfinite(gp).all() and torch.isfinite(gm).all()
    for row,col in [(0,0),(1,3),(0,6),(0,9)]:
        step=1e-5;plus=packets.detach().clone();minus=plus.clone()
        plus[0,row,col]+=step;minus[0,row,col]-=step
        fd=(score(run(f,packets=plus,parameters=material.detach(),gradients=False))-
            score(run(f,packets=minus,parameters=material.detach(),gradients=False)))/(2*step)
        assert gp[0,row,col].abs()>1e-8
        torch.testing.assert_close(gp[0,row,col],fd,rtol=3e-3,atol=2e-7)
    for index in range(2):
        step=float(material[index].detach())*1e-4
        plus=material.detach().clone();minus=plus.clone();plus[index]+=step;minus[index]-=step
        fd=(score(run(f,parameters=plus,gradients=False))-score(run(f,parameters=minus,gradients=False)))/(2*step)
        assert gm[index].abs()>1e-6
        torch.testing.assert_close(gm[index],fd,rtol=3e-3,atol=2e-4)


def test_drone_parameters_both_networks_and_attachment_rotation_receive_gradients():
    f=fixture(steps=18);model=f[0]
    gain=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    model.drone.parameters=replace(model.drone.parameters,feedforward_xy=gain)
    tilt=torch.tensor([[.02,-.03,.04]],dtype=torch.float64,requires_grad=True)
    initial=replace(f[1],rotation=rotation_exp(tilt))
    offset=initial.position.new_tensor(model.offset)
    root=initial.position+torch.einsum('bij,j->bi',initial.rotation,offset)
    # Translate the full initial cable with its own attached root.
    q=f[2].positions_m+(root-f[2].positions_m[:,0])[:,None]
    f=(model,initial,DderState(q,f[2].velocities_m_s),*f[3:])
    output=run(f,checkpoint_steps=6)
    head=model.physics.motion_residual.correction_head.bias
    drone_head=model.drone.residual.net[-1].bias
    grads=torch.autograd.grad(score(output),(gain,tilt,head,drone_head))
    assert all(torch.isfinite(g).all() and g.norm()>1e-8 for g in grads)
    epsilon=1e-5
    def gain_value(x):
        model.drone.parameters=replace(model.drone.parameters,feedforward_xy=x)
        return score(run(f,gradients=False))
    fd=(gain_value(.3+epsilon)-gain_value(.3-epsilon))/(2*epsilon)
    torch.testing.assert_close(grads[0],fd,rtol=3e-3,atol=2e-7)
    model.drone.parameters=replace(model.drone.parameters,feedforward_xy=.3)
    for bias,gradient in [(head,grads[2]),(drone_head,grads[3])]:
        index=int(gradient.abs().argmax())
        with torch.no_grad():
            saved=bias[index].clone();bias[index]=saved+epsilon
        plus=score(run(f,gradients=False))
        with torch.no_grad():bias[index]=saved-epsilon
        minus=score(run(f,gradients=False))
        with torch.no_grad():bias[index]=saved
        torch.testing.assert_close(gradient[index],(plus-minus)/(2*epsilon),rtol=3e-3,atol=2e-7)
    def rotated_value(angle):
        pose=replace(initial,rotation=rotation_exp(angle))
        root=pose.position+torch.einsum('bij,j->bi',pose.rotation,offset)
        q=f[2].positions_m.detach()+(root-f[2].positions_m[:,0].detach())[:,None]
        changed=(model,pose,DderState(q,f[2].velocities_m_s),*f[3:])
        return score(run(changed,gradients=False))
    plus=tilt.detach().clone();minus=plus.clone();plus[0,1]+=epsilon;minus[0,1]-=epsilon
    fd=(rotated_value(plus)-rotated_value(minus))/(2*epsilon)
    torch.testing.assert_close(grads[1][0,1],fd,rtol=3e-3,atol=2e-7)


def test_zero_delay_level_hover_has_finite_gradients():
    f=fixture(steps=4);model,initial,state,packets,pt,times,hover=f
    model.drone.parameters=replace(model.drone.parameters,delay_s=0.)
    initial=replace(initial,rotation=initial.rotation_command_from_tracking)
    packets=packets.detach().clone();packets[:,:,3:]=0.;packets[:,:,:3]=initial.position[:,None]
    packets.requires_grad_()
    out=model.drone.predict_differentiable(initial,packets,pt,times,model.offset,hover_command=hover)
    g=torch.autograd.grad(out['position_attachment_m'][:,-1].sum(),packets)[0]
    assert torch.isfinite(g).all() and g.norm()>0


def test_exact_delayed_commands_and_future_packets_do_not_leak():
    f=fixture(steps=12);model,initial,_,packets,pt,times,hover=f
    model.drone.parameters=replace(model.drone.parameters,delay_s=.06)
    packets=packets.detach().clone().requires_grad_()
    out=model.drone.predict_differentiable(initial,packets,pt,times,model.offset,hover_command=hover)
    g=torch.autograd.grad(out['position_origin_m'][:,5].sum(),packets,retain_graph=True)[0]
    assert torch.count_nonzero(g)==0
    g=torch.autograd.grad(out['position_origin_m'][:,-1].sum(),packets)[0]
    assert g[:,0].norm()>0 and torch.count_nonzero(g[:,1:])==0


def test_invalid_inputs_raise_instead_of_training_on_a_fallback():
    f=fixture();model,initial,state,packets,pt,times,hover=f
    with pytest.raises(ValueError,match='graph replay'):
        run(f,graph=True)
    model.drone.parameters=replace(model.drone.parameters,delay_s=torch.tensor(.06,requires_grad=True))
    with pytest.raises(ValueError,match='outer search'):
        run(f)
    model.drone.parameters=replace(model.drone.parameters,delay_s=0.)
    bad=packets.clone();bad[0,0,6:8]=0.
    bad[0,0,8]=-model.drone.parameters.gravity_m_s2/(1.9*.46)
    bad[0,0,3:6]=0.;bad[0,0,:3]=initial.position[0]
    with pytest.raises(ValueError,match='Invalid attitude'):
        run(f,packets=bad)
    with pytest.raises(ValueError,match='uniform'):
        model.predict(initial,state,packets,pt,times+np.linspace(0,.01,len(times)))
    wrong=DderState(state.positions_m+.1,state.velocities_m_s)
    with pytest.raises(ValueError,match='root does not match'):
        model.predict(initial,wrong,packets,pt,times)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_captured_inference_matches_differentiable_prediction():
    f=fixture('cuda',steps=12,batch=2)
    p=f[3].clone().requires_grad_()
    fast=run(f,packets=p,gradients=False,graph=True)
    learned=run(f,packets=p,checkpoint_steps=4)
    for key in fast:
        torch.testing.assert_close(learned[key],fast[key],atol=1e-9,rtol=1e-9)
    g=torch.autograd.grad(score(learned),p)[0]
    assert torch.isfinite(g).all() and g.norm()>0


def test_candidate_preparation_is_unfitted_and_preserves_source(tmp_path):
    from tools.prepare_residual_candidate import prepare
    from experimental_data.io import sha256_file
    from dataclasses import asdict
    from simulator.drone_pose_residual import DronePoseResidual,save_residual
    from simulator.cable.residual import MotionResidual
    from experimental_data.differentiable_fit import save_weights
    from experimental_data.io import atomic_json
    # Synthetic saved components; the active structural template is intentionally unfitted.
    old=json.loads((ROOT/'config/research_30hz/model.json').read_text(encoding='utf-8'))
    cable_weight=tmp_path/'cable.pt';save_weights(cable_weight,MotionResidual(12,mode='dissipative').double())
    drone_weight=tmp_path/'drone.pt';save_residual(drone_weight,DronePoseResidual().double())
    drone=tmp_path/'drone.json'
    atomic_json(drone,dict(nominal=dict(parameters=asdict(PoseResponseParameters(4.,4.,3.,3.,1.,1.,.1,.02))),
        residual=dict(checkpoint=drone_weight.name,sha256=sha256_file(drone_weight))))
    old['motion_residual']=dict(enabled=True,checkpoint=str(cable_weight),sha256=sha256_file(cable_weight))
    old['fullstate_execution'].update(enabled=True,checkpoint=str(drone),sha256=sha256_file(drone),source_job='synthetic-preservation-test')
    old['cable']['external_drag_s_inv']=0.
    source=tmp_path/'source.json';atomic_json(source,old)
    paths=[source,Path(old['motion_residual']['checkpoint']),Path(old['fullstate_execution']['checkpoint'])]
    before={p:sha256_file(p) for p in paths}
    folder=tmp_path/'candidate'
    manifest=prepare(source,folder,acceleration_limit=.5)
    assert manifest['status']=='UNFITTED' and not manifest['trained']
    assert not manifest['selected_for_deployment']
    assert before=={p:sha256_file(p) for p in paths}
    payload=json.loads((folder/'model.json').read_text(encoding='utf-8'))
    loaded=ResearchExecutionModel.from_mapping(payload,trainable_residuals=True)
    assert loaded.physics.motion_residual.mode=='dissipative_plus_acceleration'
    assert torch.count_nonzero(loaded.physics.motion_residual.correction_head.weight)==0
    for name,digest in manifest['files'].items():assert sha256_file(folder/name)==digest
    with pytest.raises(ValueError,match='never overwritten'):
        prepare(source,folder,acceleration_limit=.5)
