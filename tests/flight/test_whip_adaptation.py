from pathlib import Path
import numpy as np
import pytest
from experimental_data.io import atomic_json,sha256_file
from experimental_data.whip_adaptation import SCHEMA,protocol,prepare,verify_hashes


def minimal_review(tmp_path):
    batch=tmp_path/'batch';batch.mkdir();(batch/'simulation_csv').mkdir();(batch/'flight_take').mkdir()
    csv=batch/'simulation_csv/fullstate_30hz.csv';csv.write_text('frozen command')
    for n in ('take.csv','experiment_take.csv'):(batch/'flight_take'/n).write_text('raw')
    atomic_json(batch/'protocol.json',dict(schema=SCHEMA,frame='raw_global_xyz',normalization=False,
        frozen_hashes={},command_sha256=sha256_file(csv)))
    comp=tmp_path/'comparison';comp.mkdir();atomic_json(comp/'report.json',{})
    atomic_json(comp/'source_hashes.json',{str(csv):sha256_file(csv)})
    row=dict(role='adaptation',accepted=True,reviewed_by='test',clock_reviewed=True,
        same_controller_and_hardware=True,no_intervention=True,physical_contact='none',free_motion_end_s=1.)
    review=dict(schema=SCHEMA,comparison=str(comp),report_sha256=sha256_file(comp/'report.json'),takes={'take':row})
    return batch,comp,review


@pytest.mark.parametrize('key,value,message',[
    ('physical_contact','unknown','free-motion'),('free_motion_end_s',None,'free-motion'),
    ('free_motion_end_s',-1.,'free-motion'),('accepted',False,'explicit take review'),
    ('clock_reviewed',False,'clock'),('same_controller_and_hardware',False,'hardware'),
    ('no_intervention',False,'intervention'),('role','unassigned','whole adaptation'),
])
def test_unreviewed_or_unsupported_take_cannot_enter_fitting(tmp_path,key,value,message):
    batch,comp,review=minimal_review(tmp_path);review['takes']['take'][key]=value
    path=tmp_path/'review.json';atomic_json(path,review)
    with pytest.raises(ValueError,match=message):prepare(tmp_path,batch,comp,path,tmp_path/'job')
    assert not (tmp_path/'job').exists()


def test_forecast_command_and_review_identity_cannot_change(tmp_path):
    batch,comp,review=minimal_review(tmp_path)
    command=batch/'simulation_csv/fullstate_30hz.csv';command.write_text('changed')
    with pytest.raises(ValueError,match='Flown command'):protocol(batch)
    with pytest.raises(ValueError,match='Source changed'):verify_hashes({str(command):'not its hash'})


def test_protocol_rejects_retrospective_height_normalization(tmp_path):
    batch,_,_=minimal_review(tmp_path)
    from simulator.workflow import read_json
    p=read_json(batch/'protocol.json');p['normalization']=True;atomic_json(batch/'protocol.json',p)
    with pytest.raises(ValueError,match='raw-coordinate'):protocol(batch)


def test_predeclared_validation_take_cannot_be_reassigned_to_training(tmp_path):
    batch,comp,review=minimal_review(tmp_path)
    from simulator.workflow import read_json
    p=read_json(batch/'protocol.json');p['planned_roles']={'take':'validation'};atomic_json(batch/'protocol.json',p)
    path=tmp_path/'review.json';atomic_json(path,review)
    with pytest.raises(ValueError,match='predeclared whole-take split'):
        prepare(tmp_path,batch,comp,path,tmp_path/'job')


def test_invalid_controller_rows_break_new_protocol_command_coverage():
    from experimental_data.preliminary_prepare import recorded_packets
    from experimental_data.adaptation_rounds import COMMAND_COLUMNS
    from simulator.drone_pose_response import CommandSchedule
    import torch
    c={'time_s':np.arange(10)*.01,'cmd_age':np.tile([0.,.01,.02],4)[:10],
       'cmd_valid':np.ones(10)}
    # Three complete packets plus a final cached sample. Invalid at 0.01 s.
    c['cmd_age'][9]=.03;c['cmd_valid'][1]=0
    for n in COMMAND_COLUMNS:c[n]=np.zeros(10)
    times,values,until,end=recorded_packets(c,invalid_rows_break_coverage=True)
    schedule=CommandSchedule(times,torch.tensor(values[None],dtype=torch.float64),coverage_end_s=end,valid_until_s=until)
    with pytest.raises(ValueError,match='No fresh observed command'):schedule.sample(.015)
    assert np.isfinite(schedule.sample(.035).numpy()).all()


def test_masks_and_equal_take_weight_are_independent_of_candidate(tmp_path):
    import torch
    from experimental_data.whip_adaptation_fit import losses
    truth=np.zeros((3,3,3));valid=np.ones((3,2),bool);valid[1,0]=False
    q=torch.zeros((2,2,3,3,3),dtype=torch.float64)
    q[0,0,:,1:]=.02;q[1,0,:,1:]=.04
    q[0,1,1,1]=1e9 # masked observation cannot influence this candidate's score
    rows=[dict(truth=truth,valid=valid,grid=np.arange(3)),dict(truth=truth,valid=valid,grid=np.arange(3))]
    score=losses(q,rows,[1,2])
    expected=.5*((np.sqrt(1+3)-1)+(np.sqrt(1+12)-1))
    assert float(score[0])==pytest.approx(expected)
    assert float(score[1])==0


def test_prospective_batch_cannot_launch_legacy_five_take_neural_fit(tmp_path,monkeypatch):
    import tools.model_job as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    batch,_,_=minimal_review(tmp_path)
    with pytest.raises(ValueError,match='raw-coordinate whip workflow'):
        runner.run('prepare',tmp_path/'runs/adaptation/test',batch)


def test_pre_fit_diagnosis_never_reads_validation_records(tmp_path,monkeypatch):
    from experimental_data import whip_adaptation_fit as fitting
    from simulator.workflow import read_json
    model={'cable':{'external_drag_s_inv':.4}}
    p={'takes':{'whip_001':{'role':'adaptation'},'whip_002':{'role':'adaptation'},'whip_003':{'role':'validation'}}}
    monkeypatch.setattr(fitting,'load',lambda *args:(model,p,object()))
    seen=[]
    def records(job,names,*args):
        assert 'whip_003' not in names
        seen.extend(names);return names
    def evaluate(rows,engine,rate,folder):
        result={name:{'role':'adaptation'} for name in rows}
        atomic_json(folder/'metrics.json',result);return result
    monkeypatch.setattr(fitting,'records',records);monkeypatch.setattr(fitting,'evaluate',evaluate)
    fitting.diagnose(tmp_path,'cpu')
    assert seen==['whip_001','whip_002']
    review=read_json(tmp_path/'fit_review.template.json')
    assert review['diagnostic_role']=='adaptation'
    assert review['diagnostic_takes']==seen
