"""Operator-contract tests. No optimizer, simulation rollout, training or hardware."""
from copy import deepcopy
from pathlib import Path
import csv
import json
import shutil

import pytest

from deployment.lab_workflow import LabWorkspace,digest,planned_slots,portable_outputs,read,safe_name,write


@pytest.fixture
def lab(tmp_path,monkeypatch):
    from deployment import lab_seed
    root=tmp_path/'lab root';root.mkdir()
    model=root/'workspace/baseline/model/model.json'
    write(model,{'provenance':{'generation_index':0},'physical':1})
    rehearsal=root/'workspace/baseline/rehearsal';rehearsal.mkdir()
    shutil.copy2(model,rehearsal/'model.json')
    (rehearsal/'fullstate_30hz.csv').write_text('time_s,x\n0,0\n',encoding='utf-8')
    (rehearsal/'rehearsal.npz').write_bytes(b'synthetic forecast, never loaded')
    write(rehearsal/'rehearsal.json',dict(csv_sha256=digest(rehearsal/'fullstate_30hz.csv')))
    write(rehearsal/'task.json',{})
    manifest=dict(m0_model='workspace/baseline/model/model.json',m0_rehearsal='workspace/baseline/rehearsal',
                  preliminary='workspace/baseline/preliminary',preliminary_training_takes=[],mppi_settings='workspace/baseline/planner/settings.json')
    write(root/'workspace/baseline/manifest.json',manifest)
    monkeypatch.setattr(lab_seed,'load_baseline',lambda _:deepcopy(manifest))
    workspace=LabWorkspace(root,'day1');workspace.create()
    return workspace


def fake_comparison(root,batch,output):
    from experimental_data.whip_adaptation import SCHEMA
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    names=read(Path(batch)/'protocol.json')['planned_roles']
    present={n:r for n,r in names.items() if (Path(batch)/'flight_take'/(n+'.csv')).is_file()}
    write(output/'report.json',{'takes':{n:{} for n in present}})
    write(output/'source_hashes.json',{})
    write(output/'review.template.json',dict(schema=SCHEMA,comparison=str(output),report_sha256=digest(output/'report.json'),takes={n:{'role':r} for n,r in present.items()}))


def raw_pair(lab,index):
    folder=lab.root/'incoming';folder.mkdir(exist_ok=True)
    tracking=folder/f'tracking{index}.csv';controller=folder/f'controller{index}.csv'
    tracking.write_text(f'raw tracking {index}',encoding='utf-8');controller.write_text(f'raw controller {index}',encoding='utf-8')
    return tracking,controller


def import_mocked(lab,monkeypatch,stage='M0',take='whip_001',index=1):
    from experimental_data import whip_adaptation
    monkeypatch.setattr(whip_adaptation,'compare',fake_comparison)
    lab.export('M0')
    a,b=raw_pair(lab,index)
    return lab.import_take(stage,take,a,b,offset_s=.1,clock_source='Shared timestamp event')


def test_roles_and_order_are_frozen():
    slots=planned_slots()
    assert len(slots)==20
    assert [s['take'] for s in slots if s['stage']=='M0' and s['role']=='adaptation']==['whip_001','whip_002','whip_004']
    assert [(s['pair'],s['generation']) for s in slots if s['stage']=='final']==[(1,'M2'),(1,'M0'),(2,'M0'),(2,'M2'),(3,'M0'),(3,'M2'),(4,'M2'),(4,'M0'),(5,'M0'),(5,'M2')]
    assert all(s['role']=='final' for s in slots if s['stage']=='final')


@pytest.mark.parametrize('name',['../bad','CON','aux','a/b','bad:name','fig8vertical_002'])
def test_cross_platform_names(name):
    with pytest.raises(ValueError):safe_name(name)


def test_create_and_export_preserve_exact_m0_and_no_planning(lab,monkeypatch):
    import planning.pva_job
    monkeypatch.setattr(planning.pva_job,'run',lambda *_:pytest.fail('Unexpected planner'))
    before=digest(lab.root/'workspace/baseline/model/model.json')
    exported=lab.export('M0')
    assert digest(lab.path(exported['export']))==digest(lab.root/'workspace/baseline/rehearsal/fullstate_30hz.csv')
    assert digest(lab.root/'workspace/baseline/model/model.json')==before
    assert not Path(exported['export']).is_absolute()
    assert lab.export('M0')==exported
    with pytest.raises(ValueError,match='cannot be replanned'):lab.plan('M0')


def test_import_keeps_bytes_and_rejects_reuse(lab,monkeypatch):
    slot=import_mocked(lab,monkeypatch)
    assert read(lab.path(slot['batch'])/'protocol.json')['planned_roles']['whip_004']=='adaptation'
    assert read(lab.path(slot['comparison'])/'review.template.json')['report_sha256']==digest(lab.path(slot['comparison'])/'report.json')
    for path,sha in slot['raw_hashes'].items():assert digest(lab.path(path))==sha
    a,b=raw_pair(lab,1)
    with pytest.raises(ValueError,match='already belong'):lab.import_take('M0','whip_002',a,b,offset_s=.1,clock_source='Shared event')
    with pytest.raises(ValueError,match='immutable'):lab.import_take('M0','whip_001',a,b,offset_s=.1,clock_source='Shared event')


def test_invalid_import_is_retained_and_timing_can_be_repaired(lab,monkeypatch):
    from experimental_data import whip_adaptation
    lab.export('M0');a,b=raw_pair(lab,1)
    def fail(*_):raise ValueError('Clock mismatch')
    monkeypatch.setattr(whip_adaptation,'compare',fail)
    with pytest.raises(ValueError,match='Raw files were preserved'):
        lab.import_take('M0','whip_001',a,b,offset_s=0,clock_source='Initial timestamp')
    slot=lab.overview()['slots'][0];assert slot['status']=='imported'
    assert lab.path(slot['tracking']).read_bytes()==a.read_bytes()
    monkeypatch.setattr(whip_adaptation,'compare',fake_comparison)
    repaired=lab.align_take('M0','whip_001',offset_s=.2,clock_source='Corrected timestamp')
    assert repaired['offset_s']==.2 and repaired['review']=={}
    assert list((lab.path(slot['batch'])/'alignment_history').glob('*/time_alignment.json'))


def test_review_requires_evidence_and_freezes_after_prepare(lab,monkeypatch):
    import_mocked(lab,monkeypatch)
    with pytest.raises(ValueError,match='timing'):lab.review_take('M0','whip_001',reviewer='operator',free_motion_end_s=1.5)
    with pytest.raises(ValueError,match='reason'):lab.review_take('M0','whip_001',reviewer='operator',exclude=True)
    reviewed=lab.review_take('M0','whip_001',reviewer='operator',free_motion_end_s=1.5,clock_reviewed=True,same_hardware=True,no_intervention=True)
    assert reviewed['review']['role']=='adaptation'
    state=lab._load();state['updates']['M1']={'source_stage':'M0','job':'runs/adaptation/mock'};lab._save(state)
    with pytest.raises(ValueError,match='frozen'):lab.review_take('M0','whip_001',reviewer='another',exclude=True,notes='later')
    with pytest.raises(ValueError,match='frozen'):lab.align_take('M0','whip_001',offset_s=0,clock_source='later')


def test_final_data_cannot_enter_full_preparer_or_fitter(tmp_path):
    from experimental_data.whip_full_data import prepare
    from experimental_data.whip_full_fit import fit
    job=tmp_path/'diagnostics';write(job/'protocol.json',{'diagnostics_only':True})
    with pytest.raises(ValueError,match='cannot enter a fit'):prepare(tmp_path/'fit',job,tmp_path/'preliminary')
    with pytest.raises(ValueError,match='cannot enter a fit'):fit(job,'cpu')
    assert not (job/'fit').exists()


def test_public_prepare_cannot_refit_m0_or_use_final(lab):
    with pytest.raises(ValueError,match='retained M0'):lab.prepare_update('M0')
    state=lab._load();state['slots'][-1]['status']='imported';lab._save(state)
    with pytest.raises(ValueError,match='Final collection'):lab.prepare_update('M1')


def test_preparation_retry_preserves_failed_attempt_and_uses_new_paths(lab,monkeypatch):
    from experimental_data import whip_full_data
    observed=[]
    def records(state,stage,generation,output,**kwargs):
        output.mkdir(parents=True,exist_ok=False)
        (output/'source_evidence.txt').write_text('immutable preparation',encoding='utf-8')
        return output
    monkeypatch.setattr(lab,'_prepare_records',records)
    def prepare(job,source,preliminary,contract):
        observed.append((job,source,deepcopy(contract)))
        job.mkdir(parents=True,exist_ok=False)
        if len(observed)==1:raise ValueError('Synthetic external preflight error')
        write(job/'status.json',{'status':'prepared'})
    monkeypatch.setattr(whip_full_data,'prepare',prepare)
    with pytest.raises(ValueError,match='Synthetic'):lab.prepare_update('M1')
    assert lab._load()['preparation_attempts'][0]['status']=='failed'
    assert lab._load()['updates']=={}
    update=lab.prepare_update('M1')
    assert update['status']=='prepared'
    assert observed[0][0]!=observed[1][0] and observed[0][1]!=observed[1][1]
    assert observed[0][1].is_dir() and observed[0][0].is_dir()
    assert observed[1][2]['parent_id']=='day1-M0' and observed[1][2]['candidate_id']=='day1-M1'
    assert observed[1][2]['prior_whip_sources']==[]


def test_second_update_replays_only_first_prepared_whip_source(lab,monkeypatch):
    from experimental_data import whip_full_data
    state=lab._load();state['models']['M1']=deepcopy(state['models']['M0'])
    state['updates']['M1']={'job':'runs/adaptation/first','source':'experiments/day1/prepared/M0-original','source_stage':'M0'};lab._save(state)
    def records(state,stage,generation,output,**kwargs):
        assert stage==generation=='M1';output.mkdir(parents=True);return output
    monkeypatch.setattr(lab,'_prepare_records',records)
    def prepare(job,source,preliminary,contract):
        assert contract['prior_whip_sources']==['experiments/day1/prepared/M0-original']
        assert contract['parent_id']=='day1-M1' and contract['candidate_id']=='day1-M2'
        job.mkdir(parents=True);write(job/'status.json',{'status':'prepared'})
    monkeypatch.setattr(whip_full_data,'prepare',prepare)
    assert lab.prepare_update('M2')['status']=='prepared'


def test_explicit_failed_fit_retry_reuses_exact_frozen_inputs_without_fitting(lab,monkeypatch):
    from experimental_data import whip_full_data,whip_full_fit
    old=lab.root/'runs/adaptation/old';source=lab.directory/'prepared/frozen'
    source.mkdir(parents=True);(source/'bytes.npz').write_bytes(b'frozen raw input')
    contract={'schema':'whip_full_model_v1','candidate_id':'day1-M1','parent_id':'day1-M0','seed':1234,'prior_whip_sources':[]}
    write(old/'protocol.json',dict(full_update=contract,preliminary_source='workspace/baseline/preliminary'))
    write(old/'prepared_hashes.json',{lab.rel(source/'bytes.npz'):digest(source/'bytes.npz')});write(old/'source_hashes.json',{})
    write(old/'status.json',{'status':'failed'});(old/'failure.log').write_text('Original failure',encoding='utf-8')
    previous=dict(job=lab.rel(old),source=lab.rel(source),source_stage='M0',comparison='experiments/day1/comparison',status='failed')
    state=lab._load();state['updates']['M1']=previous;lab._save(state)
    monkeypatch.setattr(whip_full_fit,'fit',lambda *_:pytest.fail('Retry preparation must not start fitting'))
    monkeypatch.setattr(lab,'_prepare_records',lambda *_:pytest.fail('Frozen raw preparation must not be replaced'))
    def prepare(job,actual_source,preliminary,actual_contract):
        assert actual_source==source and actual_contract==contract
        assert preliminary==lab.root/'workspace/baseline/preliminary'
        job.mkdir(parents=True,exist_ok=False);write(job/'status.json',{'status':'prepared'})
    monkeypatch.setattr(whip_full_data,'prepare',prepare)
    result=lab.prepare_update('M1',retry=True)
    assert result['job']!=lab.rel(old) and result['source']==lab.rel(source)
    assert (old/'failure.log').read_text()=='Original failure'
    assert lab._load()['update_history']['M1']==[previous]
    with pytest.raises(ValueError,match='Only a failed or stopped'):lab.prepare_update('M1',retry=True)


def test_tampered_roles_and_input_bytes_fail(lab,monkeypatch):
    slot=import_mocked(lab,monkeypatch)
    lab.path(slot['tracking']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen input changed'):
        lab.review_take('M0','whip_001',reviewer='operator',exclude=True,notes='invalid')
    state=read(lab.directory/'study.json');state['slots'][0]['role']='validation';write(lab.directory/'study.json',state)
    with pytest.raises(ValueError,match='allocation changed'):lab.overview()


def test_timing_estimation_is_explicit_read_only_and_never_verified(lab,monkeypatch):
    import numpy as np
    from experimental_data import adaptation_rounds,preliminary_prepare
    a,b=raw_pair(lab,77);before=[a.read_bytes(),b.read_bytes(),(lab.directory/'study.json').read_bytes()]
    positions=np.zeros((40,3));positions[5]=np.nan;original=positions.copy()
    measured={'time':np.arange(40)*.01,'drone':positions}
    sentinel=object();calls=[]
    monkeypatch.setattr(adaptation_rounds,'read_optitrack',lambda path,drone_label=None,allow_external_filename=False:measured if allow_external_filename else pytest.fail('External filenames must be supported'))
    monkeypatch.setattr(adaptation_rounds,'read_controller',lambda path,allow_external_filename=False:sentinel if allow_external_filename else pytest.fail('External filenames must be supported'))
    def alignment(m,c):
        assert c is sentinel and np.isfinite(m['drone']).all() and len(m['time'])==39
        calls.append(True)
        return dict(offset_s=.23,rmse_m=.001,chunks=[{'offset_s':.22},{'offset_s':.24}],method='Existing measured-stream estimator',limitation='Unknown cached tracking latency.')
    monkeypatch.setattr(preliminary_prepare,'measured_clock_alignment',alignment)
    assert calls==[]
    result=lab.estimate_timing(a,b)
    assert calls==[True] and result['clock_verified'] is False and result['requires_operator_review']
    assert result['discarded_tracking_samples']==1 and result['chunk_spread_s']==pytest.approx(.02)
    assert result['optitrack_sha256']==digest(a)
    assert before==[a.read_bytes(),b.read_bytes(),(lab.directory/'study.json').read_bytes()]
    assert np.array_equal(positions,original,equal_nan=True)
    with pytest.raises(ValueError,match='Protected'):lab.estimate_timing(a.parent/'fig8vertical_002.csv',b)


def test_real_timing_parser_accepts_external_filenames_with_spaces(lab):
    import numpy as np
    from experimental_data.adaptation_rounds import COMMAND_COLUMNS
    tracking=lab.root/'Flight 1 tracking.csv';controller=lab.root/'Flight 1 controller.csv'
    names=['cf_7']*7+[f'cable1:c{k}' for k in range(1,11) for _ in range(3)]
    with tracking.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['Length Units','Meters','Coordinate Space','Global','Rotation Type','Quaternion']);writer.writerow([])
        writer.writerow(['','']+['Marker']*len(names));writer.writerow(['','']+names)
        writer.writerow(['','']+['id']*len(names));writer.writerow(['','']+['Rotation']*4+['Position']*(len(names)-4))
        writer.writerow(['Frame','Time (Seconds)']+list('XYZW')+list('XYZ')*11)
        for i,t in enumerate(np.arange(101)*.01):
            writer.writerow([i,t,0,0,0,1,t,t*t,1.5,*[v for k in range(1,11) for v in (t,0,1.5-.08*k)]])
    with controller.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['time_s','x','y','z','cmd_age','cmd_valid',*COMMAND_COLUMNS])
        for t in np.arange(-.2,1.21,.01):writer.writerow([t+.4,t,t*t,1.5,0,1,*([0]*11)])
    hashes=[digest(tracking),digest(controller)]
    result=lab.estimate_timing(tracking,controller)
    assert result['offset_s']==pytest.approx(.4,abs=1e-5)
    assert result['clock_verified'] is False and result['matched_updates']>=30
    assert [digest(tracking),digest(controller)]==hashes


def test_portability_rewrites_only_metadata_and_rebinds_hashes(tmp_path):
    root=tmp_path/'old root';root.mkdir();job=root/'runs/job';job.mkdir(parents=True)
    asset=job/'assets/net.pt';asset.parent.mkdir();asset.write_bytes(b'exact opaque weights')
    model=job/'model.json';write(model,dict(network={'checkpoint':str(asset),'sha256':digest(asset)},provenance={'source_job':str(job)}))
    protocol=job/'protocol.json';write(protocol,dict(model=str(model)))
    hashes=job/'prepared_hashes.json';write(hashes,{str(model):digest(model),str(protocol):digest(protocol),str(asset):digest(asset)})
    portable_outputs(root,[job])
    assert read(model)['network']['checkpoint']=='assets/net.pt'
    assert read(protocol)['model']=='runs/job/model.json'
    for path,sha in read(hashes).items():assert digest(root/path)==sha
    moved=tmp_path/'new root';shutil.copytree(root,moved)
    for path,sha in read(moved/'runs/job/prepared_hashes.json').items():assert digest(moved/path)==sha
    assert (moved/'runs/job/assets/net.pt').read_bytes()==b'exact opaque weights'


def test_partial_final_window_remains_flagged_in_observed_summary(lab,monkeypatch):
    import numpy as np
    from experimental_data import adaptation_check
    state=lab._load();state['models']['M2']=deepcopy(state['models']['M0']);lab._save(state)
    lab.export('M0');lab.export('M2');state=lab._load()
    for i,slot in enumerate(s for s in state['slots'] if s['stage']=='final' and s['pair']==1):
        a,b=raw_pair(lab,100+i)
        slot.update(status='reviewed',raw_hashes={lab.rel(a):digest(a),lab.rel(b):digest(b)},batch='unused',review={})
    lab._save(state)
    def comparison(*args):
        time=np.arange(0.,1.01,.01);cable=np.zeros((len(time),11,3));cable[:,-1,0]=.2
        return dict(time=time,measured_cable=cable,target=np.zeros(3))
    monkeypatch.setattr(adaptation_check,'load_comparison',comparison)
    result=lab.evaluate('cpu',predictions=False)
    assert len(result['physical'])==2 and all(r['minimum_tip_target_m']==.2 for r in result['physical'])
    assert all(not r['fully_observed'] for r in result['physical'])
    assert result['summary']['paired_count']==1
    assert result['summary']['fully_observed_pair_count']==0
    assert not result['paired'][0]['coverage']['M0']['window_complete']
    assert not lab._load()['results']['predictions_complete']
