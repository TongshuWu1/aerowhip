from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from learning.whip_wave import DEFAULTS,advance,bend_features,FIELDS
from learning.pva_env import PVAEnvironment
from simulator.workflow import read_json

ROOT=Path(__file__).resolve().parents[2]


def test_rigid_motion_has_no_bend_and_local_bend_uses_material_coordinate():
    q=torch.zeros(2,12,3,dtype=torch.float64);q[:,:,2]=-torch.arange(12,dtype=torch.float64)*.1
    q[1,:,0]=torch.arange(12,dtype=torch.float64)*.1+5
    angles,peak,where=bend_features(q,torch.arange(1,11)/11)
    assert angles.abs().max()<1e-10
    q[0,6:,0]=torch.arange(1,7)*.1
    _,peak,where=bend_features(q,torch.arange(1,11)/11)
    assert peak[0]>.7 and float(where[0])==pytest.approx(5/11)


def simulate(locations,amplitudes=None):
    stage=torch.zeros(1,dtype=torch.long);dwell=torch.zeros(1);credit=dwell.clone()
    bonuses=[]
    for i,loc in enumerate(locations):
        stage,dwell,credit,bonus=advance(stage,dwell,credit,torch.tensor([amplitudes[i] if amplitudes else .8]),
            torch.tensor([loc]),torch.ones(1,dtype=torch.bool),.01,DEFAULTS)
        bonuses.append(float(bonus))
    return int(stage),float(credit),sum(bonuses)


def test_order_persistence_amplitude_and_bounded_nonrepeat_reward():
    sequence=[.3]*5+[.6]*5+[.9]*5
    stage,credit,total=simulate(sequence*3)
    assert stage==3 and credit==pytest.approx(1) and total==pytest.approx(1)
    assert simulate([.9]*5+[.6]*5+[.3]*5)[0]==1
    assert simulate([.3,.6,.9]*10)[0]==0
    assert simulate([.3]*50)[0]==1
    assert simulate(sequence,[.01]*len(sequence))[0]==0


def test_wave_seed_and_sampling_expand_search_reproducibly():
    from planning.mppi_receding import wave_seed_bank,sample_noise
    env=SimpleNamespace(direction=torch.tensor([1.,0.,0.]),limit=torch.tensor([60.,60.,60.]),device=torch.device('cpu'),
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float64))
    bank=wave_seed_bank(env,60,656)
    assert bank.shape==(1025,60,3) and bank.abs().max()<1
    torch.testing.assert_close(bank,wave_seed_bank(env,60,656))
    s=dict(noise_scales=[.05,.15,.35],noise_std=.05,noise_correlation=.9)
    noise=sample_noise(env,s,3000,60,torch.Generator().manual_seed(2))
    std=[float(noise[k::3,0].std()) for k in range(3)]
    assert std[1]>2*std[0] and std[2]>5*std[0]


def test_wave_requires_pullback_and_mixture_requires_zero_prior():
    from learning.pva_env import defaults
    from planning.pva_job import validate_settings
    cfg=defaults('mppi');cfg['task'].update(DEFAULTS,require_wave=True,require_pullback=False)
    with pytest.raises(ValueError,match='requires pullback'):validate_settings(cfg)
    cfg=defaults('mppi');cfg['mppi'].update(noise_scales=[.05,.15,.35],control_prior=.5)
    with pytest.raises(ValueError,match='zero control prior'):validate_settings(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_wave_state_nonzero_branch_graph_eager_parity_and_contact_causality():
    job=ROOT/'runs/mppi_pva/20260909-135429-797997'
    if not (job/'plan.npz').exists():pytest.skip('Historical fixture unavailable')
    cfg=read_json(job/'settings.json');cfg['task'].update(DEFAULTS,require_wave=True)
    cfg['reward']['wave_progress']=120.
    model=read_json(job/'model.json')
    fast=PVAEnvironment(model,cfg,root=job,batch_size=2)
    ref=PVAEnvironment(model,cfg,root=job)
    with np.load(job/'plan.npz') as z:actions=ref.tensor(z['normalized_jerk'])[None]
    for k in range(25):ref.step(actions[:,k],trace=True)
    fast.branch_from(ref)
    a=fast.rollout(actions=actions[:,25:].expand(2,-1,-1),max_steps=actions.shape[1]-25)
    b=ref.rollout(actions=actions[:,25:],max_steps=actions.shape[1]-25,trace=True)
    for name in FIELDS:
        torch.testing.assert_close(getattr(fast,name)[:1],getattr(ref,name),atol=1e-9,rtol=0)
    torch.testing.assert_close(a['reward'][:1],b['reward'],atol=1e-8,rtol=0)
    if ref.success[0]:assert ref.wave_completion_time[0]<ref.termination_time[0]-ref.dt

    # Deliberately impossible pre-contact persistence must reject the old hit.
    cfg['task']['wave_dwell_s']=2.
    rejected=PVAEnvironment(model,cfg,root=job)
    result=rejected.rollout(actions=actions,max_steps=actions.shape[1])
    assert rejected.contact[0] and not result['success'][0]
