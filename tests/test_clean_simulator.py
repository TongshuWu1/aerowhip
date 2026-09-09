from pathlib import Path
import torch
from simulator.rollout import load_json
from simulator.research_config import snapshot_assets,validate_research_contract
from simulator.research_pose import ResearchPoseModel
from simulator.research_execution import ResearchExecutionModel
from experimental_data.io import sha256_file

ROOT=Path(__file__).resolve().parents[1]

def configs():
    return tuple(load_json(ROOT/'config'/f'{name}.json') for name in ('model','task','ppo'))

def test_unfitted_models_load_without_any_nn_checkpoint(tmp_path):
    model,task,ppo=configs();validate_research_contract(model,task,ppo)
    assert model['motion_residual']=={'enabled':False}
    assert ppo['bootstrap']=={'enabled':False}
    tracker=ResearchPoseModel(ROOT/model['fullstate_execution']['checkpoint'],model['fullstate_execution']['sha256'],'cpu')
    assert tracker.residual is None
    ResearchExecutionModel.from_mapping(model,root=ROOT,device='cpu',trainable_residuals=True)
    saved=snapshot_assets(model,tmp_path)
    assert not list(tmp_path.rglob('*.pt'))
    assert sha256_file(saved['fullstate_execution']['checkpoint'])==saved['fullstate_execution']['sha256']
    ResearchPoseModel(saved['fullstate_execution']['checkpoint'],saved['fullstate_execution']['sha256'],'cpu')

def test_force_to_fullstate_execution_without_fitted_weights():
    from learning.point_force_env import PointForceWhipEnvironment
    from learning.deployment_rollout import sample_batch,plan_batch
    from learning.research_rollout import execute_research_batch
    model,task,ppo=configs();task['episode_duration_s']=.1
    ppo['deployment']['nominal_fraction']=1.
    class HoverAgent:
        def deterministic_action(self,observation):return observation.new_zeros(len(observation),3)
    env=PointForceWhipEnvironment(model,task,ppo,batch_size=1,device=torch.device('cpu'))
    batch=sample_batch(env,ppo['deployment'],torch.Generator().manual_seed(1))
    with torch.no_grad():
        forces,cutoffs=plan_batch(env,HoverAgent(),batch)
        score=execute_research_batch(env,batch,forces,cutoffs,ppo['deployment'])
    assert not bool(score.failed.any())
    assert score.reference_packets.shape[1]==4
    assert bool(score.predicted_pose['valid'].all())
    expected=score.predicted_pose['position_origin_m'][:,0:1].expand_as(score.predicted_pose['position_origin_m'])
    torch.testing.assert_close(score.predicted_pose['position_origin_m'],expected,atol=1e-9,rtol=0)
