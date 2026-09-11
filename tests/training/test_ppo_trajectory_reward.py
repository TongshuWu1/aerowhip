"""Return equivalence, quarantine and immutable reward-reference checks."""
from copy import deepcopy
import pytest
import torch
from experimental_data.io import sha256_file
from learning.ppo_trajectory_reward import redistribute,freeze_reference,validate
from learning.pva_env import defaults


def test_shaping_telescopes_for_success_failure_and_timeout():
    # Four steps: row 0 exits immediately, row 1 after step 2, row 2 times out.
    masks=torch.tensor([[1,1,1],[0,1,1],[0,0,1],[0,0,1]],dtype=torch.float64)[...,None]
    dones=torch.tensor([[1,0,0],[1,1,0],[1,1,0],[1,1,1]],dtype=torch.float64)[...,None]
    phi=torch.tensor([[-4,-5,-6],[-3,-4,-5],[-2,-2,-4],[0,-1,-3],[0,0,-2]],dtype=torch.float64)[...,None]
    score=torch.tensor([50,-10000,75],dtype=torch.float64)
    reward=redistribute(score,phi,masks,dones)
    torch.testing.assert_close(reward.sum(0)[:,0],score-phi[0,:,0])
    assert not bool(reward[masks==0].any())
    # Different intermediate potentials cannot alter trajectory ranking.
    phi[1:]+=123
    torch.testing.assert_close(redistribute(score,phi,masks,dones).sum(0)[:,0],score-phi[0,:,0])


def test_partial_episode_is_rejected():
    with pytest.raises(ValueError,match='terminal'):
        redistribute(torch.zeros(2),torch.zeros(3,2,1),torch.ones(2,2,1),torch.zeros(2,2,1))


def test_reference_freezes_and_tamper_is_rejected(tmp_path):
    source=tmp_path/'source.npz';source.write_bytes(b'reference shape')
    target=tmp_path/'job';target.mkdir()
    cfg=dict(ppo_objective=dict(schema='mppi_preferred_fold_v1',reference_source=str(source),reference_sha256=sha256_file(source)),
             trajectory_objective=dict(reference_file='wave_reference.npz'))
    freeze_reference(cfg,target)
    assert (target/'wave_reference.npz').read_bytes()==source.read_bytes()
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):freeze_reference(cfg,target)


def test_reward_mode_validation():
    from planning.whip_objective import WAVE_OBJECTIVE
    cfg=defaults('ppo');cfg['trajectory_objective']=deepcopy(WAVE_OBJECTIVE)
    cfg['training']['success_priority']=False
    cfg['ppo_objective']=dict(schema='mppi_preferred_fold_v1',reference_sha256='a'*64,failure_penalty=10000,potential_scale=300)
    validate(cfg)
    cfg['training']['success_priority']=True
    with pytest.raises(ValueError,match='ranks'):validate(cfg)


def test_contact_first_checkpoint_selection_explicitly_matches_fixed_mppi_task():
    from planning.whip_objective import WAVE_OBJECTIVE
    from planning.ppo_progress import evaluation_better
    cfg=defaults('ppo');cfg['trajectory_objective']=deepcopy(WAVE_OBJECTIVE)
    cfg['task']['success_criterion']='tip_contact_v1'
    cfg['launch'].update(start_radius_m=0.,target_radius_m=0.)
    cfg['training']['success_priority']=True
    cfg['ppo_objective']=dict(schema='mppi_preferred_fold_v1',reference_sha256='a'*64,
        failure_penalty=10000,potential_scale=300,selection_criterion='tip_contact_then_score_v1')
    validate(cfg)
    assert evaluation_better(100.,1.,1000.,0.,success_priority=True)
    assert not evaluation_better(1000.,0.,100.,1.,success_priority=True)
    assert evaluation_better(101.,1.,100.,1.,success_priority=True)
    cfg['training']['success_priority']=False
    with pytest.raises(ValueError,match='success priority'):validate(cfg)
    cfg['training']['success_priority']=True;cfg['launch']['target_radius_m']=.05
    with pytest.raises(ValueError,match='fixed MPPI'):validate(cfg)
