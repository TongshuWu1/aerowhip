from pathlib import Path
import json, shutil, subprocess, sys
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experimental_data.io import atomic_json,sha256_file
from planning.pva_job import prepare

def main():
    source=ROOT/'runs/rehearsals_pva/M0-brake-1p3s'
    audit=ROOT/'runs/audits/M0-mppi-accuracy-20260914'
    audit.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,audit/'run_comparison.py')
    base=json.loads((source/'settings.json').read_text())
    base['model_path']=str(source/'model.json')
    base['visualization']['live_mppi']=False
    cases=[('original_reward_512x40',512,40,False),
           ('target_reward_512x40',512,40,True),
           ('target_reward_512x80',512,80,True),
           ('target_reward_1024x40',1024,40,True)]
    rows=[]
    for name,samples,iterations,stronger in cases:
        cfg=deepcopy(base)
        cfg['mppi'].update(samples=samples,iterations=iterations,minimum_iterations=iterations,
                           patience=iterations+1,seed=657)
        if stronger:cfg['trajectory_objective'].update(contact=900.,miss=1200.,proximity_scale_m=.1)
        job,command=prepare(ROOT,cfg,name,
            development_review='User-requested offline M0 reward and search-budget comparison; no flight activation.')
        rows.append(dict(name=name,job=str(job),samples=samples,iterations=iterations,
                         stronger_target_reward=stronger,random_candidates=samples*(iterations+1),
                         total_candidates=(samples+cfg['mppi']['proposal_count']+3)*(iterations+1)))
    atomic_json(audit/'manifest.json',dict(source=str(source),source_csv_sha256=sha256_file(source/'fullstate_30hz.csv'),
        initialization='Same existing config/pva/m0_spline seed in all searches; not a continuation of saved optimizer state.',
        planning_target_radius_m=base['task']['target_radius_m'],cases=rows,
        notes='Single seed exploratory comparison. Budgets exclude independent batch-one verification. No active-flight changes.'))
    for row in rows:
        job=Path(row['job'])
        print('START '+row['name']+' '+str(job),flush=True)
        with (job/'comparison_console.log').open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-u',str(ROOT/'tools/run_pva.py'),'--job',str(job)],
                                  cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        row['returncode']=result.returncode
        if (job/'result.json').exists():row['result']=json.loads((job/'result.json').read_text())
        else:row['status']=json.loads((job/'status.json').read_text())
        atomic_json(audit/'results.json',rows)
        print('DONE '+json.dumps(row),flush=True)
    print(str(audit/'results.json'),flush=True)

if __name__=='__main__':main()
