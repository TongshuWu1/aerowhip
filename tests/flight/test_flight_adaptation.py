from copy import deepcopy
from pathlib import Path
import csv
import json
import numpy as np
import pytest
import torch
from simulator.workflow import read_json,atomic_json
from simulator.point_mass import ForceControlledPointCable
from simulator.cable import DderState
from experimental_data.flight_trials import template,TRACKING_FIELDS,COMMAND_FIELDS,validate_and_prepare,import_trial
from experimental_data.flight_adaptation import coupled_rollout,load_trial,replay,fit_candidate
from learning.strike_adaptation import corrected_forces,sequence_loss

ROOT=Path(__file__).resolve().parents[2]


def make_flight(path,*,trial_id='synthetic_001',role='adaptation',steps=10,truth_drag=None,force_x=.5):
    torch.set_num_threads(1)
    template(path)
    payload=read_json(ROOT/'config/model.json');task=read_json(ROOT/'config/task.json')
    truth=deepcopy(payload)
    if truth_drag is not None:truth['cable']['external_drag_s_inv']=truth_drag
    model=ForceControlledPointCable.from_mapping(truth)
    initial=model.hanging_state(torch.tensor([0.,0.,1.5],dtype=torch.float64))
    hover=model.hover_force_world_n(dtype=torch.float64,device='cpu')
    force=hover[None,None].expand(steps,1,3).clone();force[:,:,0]=force_x
    with torch.no_grad():q,_=coupled_rollout(truth,initial,force)
    dt=payload['simulation']['dt_s'];start=.1;cutoff=start+steps*dt
    meta=read_json(path/'trial.json');meta.update(trial_id=trial_id,role=role,tracking_time_offset_s=0.,command_time_offset_s=0.,
        clock_alignment_verified=True,strike_start_s=start,planned_cutoff_s=cutoff,precontact_end_s=cutoff,
        contact_evidence='Synthetic no-contact trajectory',controller_version='synthetic_instantaneous_force')
    atomic_json(path/'trial.json',meta);atomic_json(path/'model.json',payload);atomic_json(path/'task.json',task)
    all_q=torch.cat((initial.positions_m[:,None].expand(-1,10,-1,-1),q),dim=1)[0].numpy()
    offset=np.array(payload['recorded_data']['optitrack_to_attachment_offset_body_m'])
    with (path/'tracking.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(TRACKING_FIELDS)
        for i,nodes in enumerate(all_q):
            row=[i*dt,*(nodes[0]-offset),0,0,0,1,1]
            for marker in model.cable_configuration.marker_node_indices[1:]:row.extend([*nodes[marker],1])
            writer.writerow(row)
    with (path/'commands.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(COMMAND_FIELDS)
        writer.writerow([0,*hover.tolist()])
        for i in range(0,steps,5):writer.writerow([start+i*dt,*force[i,0].tolist()])
    return payload,task


def test_causal_import_contact_exclusion_and_roundtrip(tmp_path):
    source=tmp_path/'source';payload,_=make_flight(source)
    meta,_,_,data,diagnostics=validate_and_prepare(source)
    assert data['time_s'][-1] < meta['precontact_end_s']-meta['strike_start_s']
    assert np.max(abs(data['initial_velocities_m_s'])) < 1e-10
    assert diagnostics['initialization_marker_rmse_m'] < 1e-7
    imported=import_trial(tmp_path,source)
    metrics=replay(imported,payload,tmp_path/'replay')
    assert metrics['coupled']['marker_rmse_m'] < 1e-7
    assert (tmp_path/'replay/tip_replay.pdf').exists()
    with pytest.raises(ValueError,match='already exists'):import_trial(tmp_path,source)
    # Future flight values cannot change causal initialization.
    data_csv=np.loadtxt(source/'tracking.csv',delimiter=',',skiprows=1)
    data_csv[12:,9]+=1
    np.savetxt(source/'tracking.csv',data_csv,delimiter=',',header=','.join(TRACKING_FIELDS),comments='')
    _,_,_,modified,_=validate_and_prepare(source)
    np.testing.assert_array_equal(modified['initial_velocities_m_s'],data['initial_velocities_m_s'])


def test_import_refuses_unknown_clock_and_invalid_markers(tmp_path):
    source=tmp_path/'source';make_flight(source)
    meta=read_json(source/'trial.json');meta['clock_alignment_verified']=False;atomic_json(source/'trial.json',meta)
    with pytest.raises(ValueError,match='clock alignment'):validate_and_prepare(source)
    meta.update(clock_alignment_verified=True,role='protected_test');atomic_json(source/'trial.json',meta)
    with pytest.raises(ValueError,match='Protected'):validate_and_prepare(source)
    meta['role']='adaptation';atomic_json(source/'trial.json',meta)
    rows=np.loadtxt(source/'tracking.csv',delimiter=',',skiprows=1);rows[11,12]=0
    np.savetxt(source/'tracking.csv',rows,delimiter=',',header=','.join(TRACKING_FIELDS),comments='')
    with pytest.raises(ValueError,match='Invalid drone/cable'):validate_and_prepare(source)


def test_force_derivative_and_runtime_parity():
    torch.set_num_threads(1)
    payload=read_json(ROOT/'config/model.json');model=ForceControlledPointCable.from_mapping(payload)
    initial=model.hanging_state(torch.tensor([0.,0.,1.5],dtype=torch.float64))
    force=model.hover_force_world_n(dtype=torch.float64,device='cpu')[None,None].expand(3,1,3).clone()
    force[:,:,0]=.4;force.requires_grad_()
    q,_=coupled_rollout(payload,initial,force,gradients=True)
    objective=q[:,-1,0,0].sum();gradient=torch.autograd.grad(objective,force)[0]
    direction=torch.zeros_like(force);direction[:,:,0]=1
    with torch.no_grad():
        plus=coupled_rollout(payload,initial,force+direction*1e-5)[0][:,-1,0,0].sum()
        minus=coupled_rollout(payload,initial,force-direction*1e-5)[0][:,-1,0,0].sum()
        state=initial
        for f in force:state=model.step_runtime(state,f,.01).state
    torch.testing.assert_close(q[:,-1],state.positions_m,atol=1e-8,rtol=1e-7)
    torch.testing.assert_close((gradient*direction).sum(),(plus-minus)/2e-5,atol=1e-7,rtol=.003)


def test_drag_fit_creates_candidate_without_validation_claim(tmp_path):
    source=tmp_path/'source';payload,_=make_flight(source,steps=4)
    imported=import_trial(tmp_path,source)
    result=fit_candidate([imported],payload,tmp_path/'fit',max_evaluations=2)
    assert not result['flight_ready'] and not result['validation_improved']
    assert not result['active_baseline_changed']
    assert result['selected_parameters']==['external_drag_s_inv']
    assert (tmp_path/'fit/model.json').exists()


def test_refinement_exports_once_and_never_claims_flight_release(tmp_path):
    from learning.strike_adaptation import refine
    source=tmp_path/'source';payload,_=make_flight(source,steps=4)
    imported=import_trial(tmp_path,source)
    config=read_json(ROOT/'config/ppo.json');config['deployment']['recovery_duration_s']=.5
    result=refine(imported,payload,config,tmp_path/'refinement',updates=1)
    assert result['command_rate_hz']==20
    assert result['cutoff_s']==pytest.approx(.04)
    assert not result['flight_ready'] and not result['policy_weights_changed']
    assert (tmp_path/'refinement/candidate_forces.csv').exists()


def test_transport_neutral_recorder_roundtrip(tmp_path):
    from experimental_data.flight_recorder import FlightRecorder
    source=tmp_path/'source';model,task=make_flight(source)
    meta=read_json(source/'trial.json');meta['trial_id']='ros_callback_fixture'
    with FlightRecorder(tmp_path/'recorded',model,task,metadata=meta) as recorder:
        for row in np.loadtxt(source/'tracking.csv',delimiter=',',skiprows=1):
            markers=row[9:].reshape(10,4)
            recorder.record_tracking(float(row[0]),row[1:4],row[4:8],markers[:,:3],marker_valid=markers[:,3])
        for row in np.loadtxt(source/'commands.csv',delimiter=',',skiprows=1):
            recorder.record_sent_force(float(row[0]),row[1:])
        recorder.record_controller(.1,{'gyro_rad_s':[0.,0.,0.]},clock='controller_boot',frame='body')
    imported=import_trial(tmp_path,tmp_path/'recorded')
    assert (imported/'raw/controller_imu.jsonl').exists()
    with pytest.raises(ValueError,match='closed'):recorder.record_sent_force(.3,[0,0,1.7])
