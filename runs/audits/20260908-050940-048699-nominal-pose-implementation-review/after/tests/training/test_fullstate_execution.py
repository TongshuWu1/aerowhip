from copy import deepcopy
import pytest
import torch
from run_ppo import load_configs
from simulator.drone_tracking import DroneTrackingResidual
from simulator.cable import DderState
from simulator.cable.residual import MotionResidual
from simulator.cable import CableConfiguration
from experimental_data.differentiable_fit import save_weights
from experimental_data.io import sha256_file
from learning.point_force_env import PointForceWhipEnvironment
from learning.deployment_rollout import sample_batch,execute_batch
from learning.fullstate_rollout import frozen_reference


def make_environment(tmp_path,device='cpu',dissipative=False):
    torch.set_num_threads(1)
    model,task,config=load_configs();model=deepcopy(model)
    model.pop('motion_residual',None)
    if dissipative:
        model['cable']['external_drag_s_inv']=0.
        residual=MotionResidual(CableConfiguration.from_mapping(model['cable']).node_count,hidden=32,mode='dissipative').double()
        with torch.no_grad():residual.net[-1].bias.fill_(.1)
        residual_path=tmp_path/'cable.pt';save_weights(residual_path,residual)
        model['motion_residual']=dict(enabled=True,checkpoint=str(residual_path),sha256=sha256_file(residual_path),
            specification=residual.specification(),drag_mode='nn_only')
    network=DroneTrackingResidual().double()
    path=tmp_path/'tracking.pt'
    torch.save(dict(schema='effective_fullstate_drone_residual_v1',specification=network.specification(),
        state_dict=network.state_dict(),nominal=dict(gains=[4.]*3+[3.]*3+[1.]*3,delay_s=.06),history_s=.05,
        reference_point='cable_attachment',command_position_transform='logged_cf7_command_plus_initial_world_attachment_offset',
        attachment_offset_body_m=model['recorded_data']['optitrack_to_attachment_offset_body_m']),path)
    model['fullstate_execution']=dict(enabled=True,schema='effective_attachment_execution_v1',checkpoint=str(path),sha256=sha256_file(path),recovery_objective='excluded')
    task['episode_duration_s']=.2
    env=PointForceWhipEnvironment(model,task,config,batch_size=2,device=torch.device(device))
    batch=sample_batch(env,{**config['deployment'],'nominal_fraction':1.},torch.Generator(device=device).manual_seed(12))
    env.reset(batch.estimate)
    forces=[env.hover_force_world_n.expand(2,-1).clone() for _ in range(12)]
    for force in forces:force[:,0]=.2
    cutoffs=torch.tensor([8,12],device=device)
    return env,batch,forces,cutoffs


def test_virtual_reference_cannot_observe_execution_truth(tmp_path):
    env,batch,forces,cutoffs=make_environment(tmp_path)
    before=frozen_reference(env,batch,forces,cutoffs)
    batch.truth=DderState(batch.truth.positions_m+1,batch.truth.velocities_m_s+2)
    after=frozen_reference(env,batch,forces,cutoffs)
    for a,b in zip(before,after):torch.testing.assert_close(a,b)


def test_fullstate_mode_keeps_cutoffs_and_does_not_claim_recovery(tmp_path):
    env,batch,forces,cutoffs=make_environment(tmp_path)
    trace=[]
    score=execute_batch(env,batch,forces,cutoffs,env.ppo_config['deployment'],trace=lambda i,f,s,striking,hit:trace.append(striking.clone()))
    assert len(trace)==12 and torch.stack(trace).sum(0).tolist()==[8,12]
    assert not score.failed.any()
    assert not score.deployment['recovery_evaluated'].any()
    assert not score.deployment['joint_success'].any()
    assert (score.episode_component_sums['recovery']==0).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA comparison requires GPU')
@pytest.mark.parametrize('dissipative',[False,True])
def test_boundary_gpu_matches_cpu_for_dynamic_reference(tmp_path,dissipative):
    outputs=[]
    for device in ('cpu','cuda'):
        env,batch,forces,cutoffs=make_environment(tmp_path,device,dissipative)
        score=execute_batch(env,batch,forces,cutoffs,env.ppo_config['deployment'])
        assert not score.failed.any()
        outputs.append(score.execution_state.positions_m.cpu())
    torch.testing.assert_close(*outputs,atol=1e-5,rtol=1e-5)
