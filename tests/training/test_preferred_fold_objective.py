from types import SimpleNamespace
import numpy as np
import torch
from planning.whip_objective import record_encounter,sample_history,encounter_quality


def test_encounter_latches_first_any_contact_and_interpolates_all_fields():
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    previous=SimpleNamespace(positions_m=t([[[0,0,0],[1,0,0]]]),velocities_m_s=t([[[0,0,0],[2,0,0]]]))
    env=SimpleNamespace(target=t([[1.1,0,0]]),contact=torch.tensor([False]),dt=.02,
        pose=SimpleNamespace(position=t([[.2,0,0]]),velocity=t([[-2,0,0]])),
        origin0=t([[0,0,0]]),direction=t([[1,0,0]]),pull_peak=t([.3]),
        encounter_time=t([0]),encounter_distance=t([1]),encounter_q=previous.positions_m.clone(),
        encounter_tip_velocity=t([[0,0,0]]),encounter_drone_velocity=t([[0,0,0]]),
        encounter_origin=t([[0,0,0]]),encounter_backward=t([0]),encounter_tip_first=torch.tensor([False]))
    kwargs=dict(env=env,previous=previous,previous_origin=t([[0,0,0]]),previous_velocity=t([[0,0,0]]),
        q=previous.positions_m+t([[[.2,0,0],[.2,0,0]]]),v=previous.velocities_m_s*2,
        entry=t([[float('inf'),.25]]),nearest_fraction=t([.5]),running=torch.tensor([True]),clock_left=t([1.]),previous_pull_peak=t([.3]))
    record_encounter(**kwargs)
    torch.testing.assert_close(env.encounter_time,t([1.005]))
    torch.testing.assert_close(env.encounter_origin,t([[.05,0,0]]))
    torch.testing.assert_close(env.encounter_tip_velocity,t([[2.5,0,0]]))
    torch.testing.assert_close(env.encounter_drone_velocity,t([[-.5,0,0]]))
    torch.testing.assert_close(env.encounter_backward,t([.25]))
    assert env.encounter_tip_first.item()
    saved=env.encounter_q.clone();env.contact.fill_(True)
    kwargs['q']+=100;kwargs['entry']=t([[.1,.2]])
    record_encounter(**kwargs)
    torch.testing.assert_close(env.encounter_q,saved)


def test_forward_contact_cannot_collect_future_interval_peak_as_backward_travel():
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    previous=SimpleNamespace(positions_m=t([[[0,0,0],[1,0,0]]]),velocities_m_s=t([[[0,0,0],[2,0,0]]]))
    env=SimpleNamespace(target=t([[1.1,0,0]]),contact=torch.tensor([False]),dt=.02,
        pose=SimpleNamespace(position=t([[.2,0,0]]),velocity=t([[1,0,0]])),origin0=t([[0,0,0]]),
        direction=t([[1,0,0]]),pull_peak=t([.2]),encounter_time=t([0]),encounter_distance=t([1]),
        encounter_q=previous.positions_m.clone(),encounter_tip_velocity=t([[0,0,0]]),
        encounter_drone_velocity=t([[0,0,0]]),encounter_origin=t([[0,0,0]]),
        encounter_backward=t([0]),encounter_tip_first=torch.tensor([False]))
    record_encounter(env,previous,t([[0,0,0]]),t([[1,0,0]]),previous.positions_m+.2,
        previous.velocities_m_s,t([[float('inf'),.25]]),t([.5]),torch.tensor([True]),t([1.]),t([0.]))
    torch.testing.assert_close(env.encounter_backward,t([0.]))


def test_shape_sampling_cannot_collect_good_frames_after_encounter():
    q=torch.arange(5,dtype=torch.float64)[None,:,None,None].expand(2,5,3,3).clone()
    times=torch.arange(5,dtype=torch.float64);end=torch.tensor([1.5,2.3],dtype=torch.float64)
    end_q=end[:,None,None].expand(2,3,3).clone();phase=torch.linspace(0,1,21,dtype=torch.float64)
    expected=sample_history(q,times,end,phase,end_q)
    for i in range(2):q[i,times>end[i]]=999
    torch.testing.assert_close(sample_history(q,times,end,phase,end_q),expected)


def test_non_tip_first_contact_has_no_contact_or_cast_credit():
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    task=dict(minimum_directed_speed_m_s=4.,minimum_backward_speed_m_s=.5,minimum_backward_distance_m=.1)
    args=(t([.05]),t([[5,0,0]]),t([[-1,0,0]]),t([.2]),t([.8]),t([[1,0,0]]))
    bad=encounter_quality(*args,torch.tensor([False]),torch.tensor([True]),task,.35)
    good=encounter_quality(*args,torch.tensor([True]),torch.tensor([True]),task,.35)
    assert bad[0]==0 and bad[1]==0 and good[0]>0 and good[1]>0


def test_horizontal_outward_velocity_beats_equal_forward_speed_with_vertical_sweep():
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    task=dict(minimum_directed_speed_m_s=4.,minimum_backward_speed_m_s=.5,minimum_backward_distance_m=.1)
    def cast(v):return encounter_quality(t([.05]),t([v]),t([[-1,0,0]]),t([.2]),t([.8]),t([[1,0,0]]),
        torch.tensor([True]),torch.tensor([True]),task,.35)[1]
    assert cast([4,0,0])>cast([4,0,3])>cast([4,0,6])
