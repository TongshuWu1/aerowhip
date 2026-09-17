from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from tests.force_config import load_configs
from learning.point_force_env import PointForceWhipEnvironment
from learning.deployment_rollout import sample_batch,execute_batch
from simulator.cuda_graph_physics import CudaGraphPhysics

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph requires a GPU')


def test_graph_matches_eager_bent_states_and_preserves_history():
    torch.set_num_threads(1)
    m,t,c=load_configs();c=deepcopy(c);c['cuda_graph_physics']=False
    env=PointForceWhipEnvironment(m,t,c,batch_size=4,device=torch.device('cuda'))
    batch=sample_batch(env,{**c['deployment'],'nominal_fraction':0.},torch.Generator(device='cuda').manual_seed(21))
    s=batch.truth;g=CudaGraphPhysics(env.model,s,env.physics_dt_s)
    original=s.positions_m.clone();eager=s;actual=s
    generator=torch.Generator(device='cuda').manual_seed(44)
    for i in range(8):
        force=env.hover_force_world_n+torch.randn(4,3,device='cuda',dtype=s.positions_m.dtype,generator=generator)*.2
        eager=env.model.step_runtime(eager,force,env.physics_dt_s).state
        actual=g(actual,force)
        torch.testing.assert_close(actual.positions_m,eager.positions_m,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(actual.velocities_m_s,eager.velocities_m_s,atol=1e-10,rtol=1e-10)
        if i==0:saved=actual.positions_m;copy=saved.clone()
    assert torch.equal(saved,copy) and torch.equal(s.positions_m,original)


@pytest.mark.parametrize('cutoff_values',[[1,2,3,4],[0,2,0,4],[0,0,0,0]])
def test_graph_execution_preserves_randomized_plant_and_recovery(cutoff_values):
    torch.set_num_threads(1)
    m,t,c=load_configs();c=deepcopy(c);c['cuda_graph_physics']=False
    env=PointForceWhipEnvironment(m,t,c,batch_size=4,device=torch.device('cuda'))
    batch=sample_batch(env,{**c['deployment'],'nominal_fraction':0.},torch.Generator(device='cuda').manual_seed(21))
    forces=env.hover_force_world_n.expand(4,4,3).clone();forces[:,:,0]=.3
    cutoffs=torch.tensor(cutoff_values,device='cuda')
    settings={**c['deployment'],'recovery_duration_s':.02}
    reference=execute_batch(env,batch,forces,cutoffs,settings)
    env.ppo_config={**c,'cuda_graph_physics':True}
    updates=[]
    actual=execute_batch(env,batch,forces,cutoffs,settings,
                         progress=lambda *values: updates.append(values))
    total=max(cutoff_values)+round(settings['recovery_duration_s']/env.physics_dt_s) if any(cutoff_values) else 0
    assert [row[1] for row in updates]==list(range(1,total+1))
    assert all(row[2]==total for row in updates)
    torch.testing.assert_close(actual.execution_state.positions_m,reference.execution_state.positions_m,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(actual.episode_reward,reference.episode_reward,atol=1e-10,rtol=1e-10)
    assert torch.equal(actual.failed,reference.failed)
    torch.testing.assert_close(sum(actual.episode_component_sums.values()),actual.episode_reward)
    for key in actual.deployment:
        torch.testing.assert_close(actual.deployment[key],reference.deployment[key],equal_nan=True)
    for key in ('episode_nonfinite','episode_position_limit','episode_speed_limit'):
        assert torch.equal(getattr(actual,key),getattr(reference,key))
