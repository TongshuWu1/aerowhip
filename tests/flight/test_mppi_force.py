from pathlib import Path
import numpy as np
import pytest
import torch
from simulator.workflow import read_json
from planning.mppi_force import ForceEvaluator
from planning.mppi_force_run import DEFAULTS,validate_settings

ROOT=Path(__file__).resolve().parents[2]
SEED=ROOT/'runs/rehearsals/20260908-203914-039721'


def test_direct_force_reproduces_saved_ppo_whip():
    if not torch.cuda.is_available() or not (SEED/'rehearsal.npz').exists():pytest.skip('Local CUDA and flown PPO rehearsal required')
    model,task,config=[read_json(SEED/f'{n}.json') for n in ('model','task','ppo')]
    with np.load(SEED/'rehearsal.npz') as saved:
        expected={k:saved[k].copy() for k in saved.files}
    evaluator=ForceEvaluator(model,task,config,DEFAULTS,'cuda')
    actions=evaluator.seed_actions(expected['virtual_force_n'],expected['force_time_s'])
    values,diagnostics,result=evaluator.evaluate([actions],record=True)
    score=result['score'];count=int(result['cutoffs'][0]);packets=score.reference_packets[0].cpu().numpy()
    np.testing.assert_allclose(result['forces'][:,0].cpu().numpy(),expected['virtual_force_n'][:count],atol=2e-12,rtol=0)
    np.testing.assert_allclose(packets,expected['commands'][:len(packets)],atol=1e-8,rtol=0)
    np.testing.assert_allclose(np.asarray(result['frames']),expected['cable_positions_m'][:count+1],atol=1e-8,rtol=0)
    np.testing.assert_allclose(score.predicted_pose['position_origin_m'][0].cpu().numpy(),expected['origin_positions_m'][:count+1],atol=1e-8,rtol=0)
    assert values[0]==float(score.episode_reward[0])
    assert bool(score.episode_success[0])==read_json(SEED/'rehearsal.json')['predicted_valid_hit']
    assert not bool(score.failed[0])


@pytest.mark.parametrize('change',[dict(horizon_s=1.01),dict(action_std=0),dict(temperature=float('nan'))])
def test_invalid_force_settings(change):
    with pytest.raises(ValueError):validate_settings(dict(DEFAULTS,**change))


def test_force_job_needs_no_ppo_or_rehearsal(tmp_path,monkeypatch):
    from planning.mppi_force_run import prepare_job
    from planning.mppi_contract import REWARD_DEFAULTS
    model=ROOT/'data/model_candidates/20260908-adp0-M1/model.json'
    if not model.exists():pytest.skip('M1 required')
    # Fail on any attempt to read a policy, rehearsal, or PPO config.
    original=Path.open
    def guarded(path,*args,**kwargs):
        if path.name=='ppo.json' or ('runs' in path.parts and any(n in path.parts for n in ('ppo','rehearsals'))):
            raise AssertionError(f'MPPI tried to read PPO: {path}')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    output=tmp_path/'job'
    settings=dict(DEFAULTS,reward={'time_to_success_weight_per_s':17.})
    command=prepare_job(ROOT,output,settings)
    actual=read_json(output/'rollout.json')
    assert actual['reward']['time_to_success_weight_per_s']==17.
    assert REWARD_DEFAULTS['time_to_success_weight_per_s']==10.
    assert actual['deployment']['termination']=='execution_success_or_timeout'
    assert not any((output/n).exists() for n in ('ppo.json','cem.json','seed.npz','seed_force.npz','seed_provenance.json'))
    assert 'ppo' not in actual and 'training' not in actual
    assert read_json(output/'mppi.json')['representation']=='independent_force_30hz_v2'
    assert read_json(output/'initialization.json')['method']=='hanging_hover'
    assert command[2].endswith('plan_mppi_force.py')


def test_independent_hover_and_time_objective():
    from planning.mppi_contract import rollout_contract,LAUNCH_DEFAULTS
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    model=read_json(ROOT/'data/model_candidates/20260908-adp0-M1/model.json')
    settings=dict(DEFAULTS,horizon_s=.2)
    task,config=rollout_contract(settings)
    task.update(initial_root_position_m=(np.array(LAUNCH_DEFAULTS['initial_tracking_origin_m'])+model['recorded_data']['optitrack_to_attachment_offset_body_m']).tolist(),
                initial_root_velocity_m_s=[0.,0.,0.],target_position_m=LAUNCH_DEFAULTS['target_position_m'])
    evaluator=ForceEvaluator(model,task,config,settings)
    actions=evaluator.hover_actions();assert np.all(actions==0) and len(actions)==18
    _,_,result=evaluator.evaluate([actions],record=True)
    hover=result['env'].hover_force_world_n[0].cpu().numpy()
    np.testing.assert_allclose(result['forces'][:,0].cpu(),np.broadcast_to(hover,(30,3)),atol=1e-12)
    assert not bool(result['score'].failed[0])
    assert not bool(result['score'].episode_success[0])
    assert float(result['score'].episode_component_sums['time'][0])==pytest.approx(-2.)
    # Changing only MPPI's time cost changes the same no-hit rollout by -2.
    config['reward']['time_to_success_weight_per_s']=20.
    other=ForceEvaluator(model,task,config,settings)
    values,_,_=other.evaluate([other.hover_actions()])
    assert values[0]-float(result['score'].episode_reward[0])==pytest.approx(-2.,abs=1e-6)


def test_force_batch_matches_independent_execution():
    if not torch.cuda.is_available() or not SEED.exists():pytest.skip('Local CUDA and PPO required')
    model,task,config=[read_json(SEED/f'{n}.json') for n in ('model','task','ppo')]
    evaluator=ForceEvaluator(model,task,config,DEFAULTS,'cuda')
    with np.load(SEED/'rehearsal.npz') as data:base=evaluator.seed_actions(data['virtual_force_n'],data['force_time_s'])
    perturbed=base.copy();perturbed[::3]-=.01
    batched,_,batch_result=evaluator.evaluate([base,perturbed])
    singles=[]
    for i,vector in enumerate((base,perturbed)):
        values,_,result=evaluator.evaluate([vector]);singles.append(values[0])
        assert bool(batch_result['score'].episode_success[i])==bool(result['score'].episode_success[0])
        np.testing.assert_allclose(batch_result['score'].reference_packets[i].cpu(),result['score'].reference_packets[0].cpu(),rtol=0,atol=1e-6)
        np.testing.assert_allclose(batch_result['score'].execution_state.positions_m[i].cpu(),result['score'].execution_state.positions_m[0].cpu(),rtol=0,atol=1e-5)
    # Different CUDA batch kernels (including float32 residual networks) are
    # not bitwise identical: compare return to 1e-5 and final cable geometry to
    # 10 micrometres (observed M0 difference ~1.6 micrometres).
    np.testing.assert_allclose(batched,singles,rtol=0,atol=1e-5)
