from pathlib import Path
import sys,os,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experimental_data.current_adaptation import read,save
from experimental_data.current_adaptation_fit import load_engine,cable_windows,note
from experimental_data.preliminary_prepare import PreliminaryTrial
from experimental_data.preliminary_fit import publish
job=Path(sys.argv[1]).resolve()
try:
    torch.set_num_threads(4)
    save(job/'status.json',dict(status='running',stage='postfit_review',pid=os.getpid()))
    note(job,'rebuilding frozen cable windows for final review')
    model=read(job/'source_candidate/model.json');trials=[PreliminaryTrial(job,w,model) for w in read(job/'windows.json')]
    engine=load_engine(job);records,rejected=cable_windows(trials,engine.physics,[0.,1.],1.)
    mapping={t.name:t for t in trials}
    for r in records:r['take']=mapping[r['name']].take;r['role']=mapping[r['name']].role
    assert len(records)==175 and len(rejected)==15
    publish(job,trials,records)
except BaseException as exc:
    save(job/'status.json',dict(status='failed',error=str(exc)));raise
