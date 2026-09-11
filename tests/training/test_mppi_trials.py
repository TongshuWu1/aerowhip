from tools.run_mppi_trials import next_action
import json,sys
import pytest
from experimental_data.io import atomic_json


def test_only_planning_outcomes_allow_authorized_fallback():
    assert next_action({'status':'completed'},{'best_success':True,'best_failed':False})=='success'
    assert next_action({'status':'completed'},{'best_success':False,'best_failed':False})=='next'
    assert next_action({'status':'completed'},{'best_success':True,'best_failed':True})=='next'
    assert next_action({'status':'failed','error':'No feasible MPPI lookahead candidate; partial committed plan retained, no export'},{})=='next'
    for state in ('stopped','running','prepared'):
        assert next_action({'status':state},{'best_success':False,'best_failed':False})=='halt'
    assert next_action({'status':'failed','error':'CUDA out of memory'},{})=='halt'
    assert next_action({'status':'completed'},{})=='halt'


@pytest.mark.parametrize('outcome,expected_started',[('miss',2),('hit',1),('stopped',1)])
def test_prepared_sequence_runs_fallback_only_after_a_finished_miss(tmp_path,outcome,expected_started):
    from tools.run_mppi_trials import run
    sequence=tmp_path/'sequence';trials=[]
    script=tmp_path/'fake_trial.py'
    script.write_text('import json,sys\nfrom pathlib import Path\nj=Path(sys.argv[1]);s=sys.argv[2]\n(j/"ran").touch()\n(j/"status.json").write_text(json.dumps({"status":"stopped" if s=="stopped" else "completed"}))\n(j/"result.json").write_text(json.dumps({"best_success":s=="hit","best_failed":False}))\n')
    for i,horizon in enumerate((1.,1.5)):
        job=tmp_path/f'job{i}'
        atomic_json(job/'status.json',{'status':'prepared'});atomic_json(job/'settings.json',{'mppi':{'samples':512,'horizon_s':horizon}})
        trials.append({'job':str(job),'command':[sys.executable,str(script),str(job),outcome if i==0 else 'miss']})
    atomic_json(sequence/'plan.json',{'root':str(tmp_path),'trials':trials});run(sequence)
    assert sum((tmp_path/f'job{i}'/'ran').exists() for i in range(2))==expected_started
    result=json.loads((sequence/'status.json').read_text())
    assert result['status']==('stopped' if outcome=='stopped' else 'completed')
    assert json.loads((tmp_path/'config/pva/mppi.json').read_text())['mppi']['horizon_s']==(1.5 if outcome=='miss' else 1.)
    assert not (sequence/'worker.lock').exists()
