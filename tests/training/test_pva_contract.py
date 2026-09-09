from pathlib import Path
import numpy as np
import torch
import pytest
from simulator.pva_commands import integrate_jerk,jerk_packets,sphere_entry
from learning.pva_env import PVAEnvironment,defaults
from experimental_data.current_adaptation import read

ROOT=Path(__file__).resolve().parents[2]


def test_exact_jerk_integration_and_temporal_gradient():
    packet=torch.zeros(1,11,dtype=torch.float64);packet[:,:3]=torch.tensor([1.,2.,3.])
    packet[:,3:6]=1;packet[:,6:9]=2
    jerk=torch.tensor([[3.,-2.,1.]],dtype=torch.float64,requires_grad=True)
    full=integrate_jerk(packet,jerk,.1)
    half=integrate_jerk(integrate_jerk(packet,jerk,.05),jerk,.05)
    torch.testing.assert_close(full,half,atol=1e-14,rtol=0)
    full[:,:3].sum().backward();torch.testing.assert_close(jerk.grad,torch.full_like(jerk,.1**3/6))
    with pytest.raises(ValueError):integrate_jerk(packet,jerk,-1)


def test_sphere_entry_detects_between_sample_crossing_in_order():
    previous=torch.tensor([[[-2.,0.,0.],[-1.,0.,0.],[0.,2.,0.]]])
    current=previous+torch.tensor([[[4.,0.,0.],[4.,0.,0.],[4.,0.,0.]]])
    fractions=sphere_entry(previous,current,torch.zeros(1,3),.1)
    torch.testing.assert_close(fractions[0,:2],torch.tensor([.475,.225]))
    assert torch.isinf(fractions[0,2])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_streaming_model_matches_exact_packet_replay_and_preserves_batch_independence():
    source=ROOT/'runs/adaptation/20260909-pva-M0-bootstrap/source_candidate/model.json'
    if not source.exists():pytest.skip('Prepared cold seed required')
    settings=defaults();settings['task']['duration_s']=.2
    env=PVAEnvironment(read(source),settings,root=source.parent,batch_size=2,graph=True)
    actions=torch.zeros(2,6,3,device='cuda');actions[0,:,0]=.12;actions[1,:,1]=-.08
    initial_pose,initial_state=env.initial_pose,env.initial_state
    result=env.rollout(actions=actions,trace=True)
    assert not result['failed'].any()
    expected=env.engine.predict(initial_pose,initial_state,result['packets'],np.arange(7)/30,np.arange(31)/150,
        graph=True,hover_command=env.hover)
    q=torch.stack([f['cable'] for f in env.frames],1)
    p=torch.stack([f['origin'] for f in env.frames],1)
    torch.testing.assert_close(q,expected['cable_positions_m'][:,1:],atol=1e-9,rtol=0)
    torch.testing.assert_close(p,expected['position_origin_m'][:,1:],atol=1e-10,rtol=0)
    solo=PVAEnvironment(read(source),settings,root=source.parent,batch_size=1,graph=True)
    single=solo.rollout(actions=actions[:1],trace=True)
    torch.testing.assert_close(single['reward'],result['reward'][:1],atol=1e-8,rtol=0)
    torch.testing.assert_close(solo.state.positions_m,env.state.positions_m[:1],atol=1e-9,rtol=0)
