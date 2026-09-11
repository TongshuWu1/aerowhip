from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from experimental_data import model_evaluation as ev
from experimental_data.io import atomic_json,sha256_file


def test_model_identity_tracks_component_bytes_but_ignores_packaging(tmp_path):
    for name in ('a','b'):
        p=tmp_path/name;p.mkdir();(p/'weights.pt').write_bytes(b'weights')
        atomic_json(p/'drone.json',{'residual':{'checkpoint':'weights.pt'}})
        atomic_json(p/'model.json',{'cable':{'external_drag_s_inv':.4},'fullstate_execution':{'checkpoint':'drone.json'},'provenance':{'label':name}})
    assert ev.model_identity(tmp_path/'a/model.json')[0]==ev.model_identity(tmp_path/'b/model.json')[0]
    (tmp_path/'b/weights.pt').write_bytes(b'changed')
    assert ev.model_identity(tmp_path/'a/model.json')[0]!=ev.model_identity(tmp_path/'b/model.json')[0]


def test_coverage_excludes_initialization_and_preserves_missing_samples(tmp_path):
    t=np.array([-.1,0,.1,.2,.3]);truth=np.zeros((5,3,3));pred=truth.copy();pred[:,:,0]=.01
    mask=np.ones((5,2),bool);mask[2,-1]=False
    np.savez(tmp_path/'take.npz',time_s=t,measured_sites=truth,conditional_cable=pred,coupled_cable=pred,
        measured_origin=truth[:,0],predicted_origin=pred[:,0],mask=mask)
    p={'takes':{'take':{'end_s':.3,'role':'validation'}}}
    report=ev.summarize_diagnostics(tmp_path,p,[1,2])['take']['metrics']
    assert report['command_driven_tip']['coverage']==pytest.approx(2/3)
    assert report['command_driven_tip']['total_count']==3
    assert report['command_driven_tip']['rmse_m']==pytest.approx(.01)
    assert report['drone']['coverage']==1


def test_paired_summary_rejects_different_coverage_and_uses_whole_takes():
    row={'role':'validation','end_s':1.,'metrics':{'tip':{'valid_count':10,'total_count':10,'rmse_m':.1}}}
    report={'models':{m:{'takes':{'a':deepcopy(row),'b':deepcopy(row)}} for m in ['M0','M1','M2']}}
    report['models']['M1']['takes']['a']['metrics']['tip']['rmse_m']=.05
    names,values=ev.paired_summary(report,'tip')
    assert names==['a','b'] and np.mean(values[1])==pytest.approx(.075)
    report['models']['M2']['takes']['a']['metrics']['tip']['valid_count']=9
    with pytest.raises(ValueError,match='coverage'):ev.paired_summary(report,'tip')


def test_missing_and_failed_outcomes_are_not_successes():
    counts=ev.success_counts([{'outcome':'hit','execution':'unknown'}, {'outcome':'hit','execution':'feasible'},
        {'outcome':'unknown','execution':'unknown'}, {'outcome':'hit','execution':'infeasible'}])
    assert counts==dict(attempts=4,hits=1,failures=1,unknown=2,assessed_rate=.5)
    assert ev.success_counts([])['assessed_rate'] is None


def test_catalog_keeps_revisions_and_rejects_synthetic_registration(tmp_path):
    first=dict(schema=ev.CATALOG,models=[],flights=[])
    ev.save_catalog(tmp_path,first);old=ev.catalog_path(tmp_path).read_bytes()
    ev.save_catalog(tmp_path,{**first,'note':'new'})
    assert next((ev.catalog_path(tmp_path).parent/'history').glob('*.json')).read_bytes()==old
    with pytest.raises(ValueError,match='audit fixtures'):
        ev.add_candidate(tmp_path,tmp_path/'runs/audits/synthetic/job')


def test_flight_registration_binds_full_candidate_not_same_generation_sibling(tmp_path):
    batch=tmp_path/'rehearsal_csv_and_result_in_real_flight/new_flight'
    rehearsal=tmp_path/'runs/rehearsals_pva/full';rehearsal.mkdir(parents=True)
    atomic_json(rehearsal/'model.json',{'cable':{'external_drag_s_inv':.37}})
    signature,hashes=ev.model_identity(rehearsal/'model.json')
    command=batch/'simulation_csv/fullstate_30hz.csv';command.parent.mkdir(parents=True);command.write_text('actual command')
    atomic_json(batch/'protocol.json',dict(schema=ev.SCHEMA,frame='raw_global_xyz',normalization=False,
        parent_generation=1,model_id='M1-full',rehearsal=str(rehearsal),frozen_hashes=hashes,
        command_sha256=sha256_file(command),forecast_sha256='frozen forecast'))
    ev.save_catalog(tmp_path,dict(schema=ev.CATALOG,models=[dict(id='M1',signature='other model'),
        dict(id='M1-full',signature=signature)],flights=[]))
    comparison=tmp_path/'runs/data_review/flight';atomic_json(comparison/'report.json',dict(schema=ev.SCHEMA,batch=str(batch),takes={}))
    atomic_json(comparison/'source_hashes.json',hashes)
    ev.add_flight_report(tmp_path,comparison)
    assert ev.load_catalog(tmp_path)['flights'][0]['model']=='M1-full'


def test_incomplete_or_tampered_evaluation_cannot_display(tmp_path):
    atomic_json(tmp_path/'status.json',{'status':'failed'})
    with pytest.raises(ValueError,match='incomplete'):ev.load_evaluation(tmp_path)
    atomic_json(tmp_path/'status.json',{'status':'completed'})
    atomic_json(tmp_path/'report.json',{'schema':ev.REPORT})
    atomic_json(tmp_path/'evidence_hashes.json',{str(tmp_path/'report.json'):sha256_file(tmp_path/'report.json')})
    assert ev.load_evaluation(tmp_path)['schema']==ev.REPORT
    atomic_json(tmp_path/'report.json',{'schema':ev.REPORT,'changed':True})
    with pytest.raises(ValueError,match='Source changed'):ev.load_evaluation(tmp_path)


def test_candidate_generations_require_actual_parent_and_never_replace(tmp_path):
    from experimental_data.whip_adaptation import SCHEMA
    baseline=tmp_path/'m0.json';atomic_json(baseline,dict(cable={'external_drag_s_inv':.4}))
    signature,hashes=ev.model_identity(baseline)
    ev.save_catalog(tmp_path,dict(schema=ev.CATALOG,models=[dict(id='M0',signature=signature,hashes=hashes,training_sources=[])],flights=[]))
    for generation in (1,2):
        job=tmp_path/f'runs/adaptation/fit{generation}';job.mkdir(parents=True)
        parent=job/'source_candidate/model.json';atomic_json(parent,{'cable':{'external_drag_s_inv':.4+(generation-1)*.1}})
        atomic_json(job/'protocol.json',{'schema':SCHEMA,'parent_generation':generation-1})
        atomic_json(job/'fit/result.json',{'status':'completed','training_takes':['take']})
        atomic_json(job/'prepared_hashes.json',{str(parent):sha256_file(parent)})
        raw=job/'raw';raw.mkdir();(raw/'take.csv').write_text('observed take');(raw/'experiment_take.csv').write_text('commands')
        atomic_json(job/'source_hashes.json',{str(path):sha256_file(path) for path in raw.iterdir()})
        atomic_json(job/'candidate/model.json',{'cable':{'external_drag_s_inv':.4+generation*.1},'provenance':{'generation_index':generation,'parent_model_sha256':sha256_file(parent)}})
        atomic_json(job/'fit/result.json',{'status':'completed','training_takes':['take'],
            'candidate_hashes':{str(job/'candidate/model.json'):sha256_file(job/'candidate/model.json')}})
        assert ev.add_candidate(tmp_path,job)==f'M{generation}'
        with pytest.raises(ValueError,match='already exists'):ev.add_candidate(tmp_path,job)
    catalog=ev.load_catalog(tmp_path)
    assert [m['id'] for m in catalog['models']]==['M0','M1','M2']
    assert catalog['models'][-1]['parent']=='M1'


def test_outcome_reviews_are_bound_to_report_and_prior_review_is_preserved(tmp_path):
    report=tmp_path/'comparison/report.json';atomic_json(report,dict(takes={'take':{}}))
    flight=dict(comparison=str(report.parent),hashes={str(report):sha256_file(report)})
    ev.save_catalog(tmp_path,dict(schema=ev.CATALOG,models=[],flights=[flight]))
    ev.review_outcome(tmp_path,report.parent,'take','unknown','unknown','observer','Tip occluded near target')
    first=ev.load_catalog(tmp_path)['flights'][0]['outcomes']['take']
    ev.review_outcome(tmp_path,report.parent,'take','hit','feasible','observer','Separate recording reviewed with target and execution visible')
    current=ev.load_catalog(tmp_path)['flights'][0]
    assert Path(first['path']).is_file()
    assert ev.flight_outcomes(current)['take']['outcome']=='hit'
    atomic_json(report,dict(takes={'take':{}},changed=True))
    with pytest.raises(ValueError,match='another comparison'):ev.flight_outcomes(current)
