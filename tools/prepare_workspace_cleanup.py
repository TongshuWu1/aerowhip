"""Prepare a reversible artifact cleanup using the active scientific dependency graph."""
from pathlib import Path
from collections import deque
import json,os,re

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'delete/cleanup-20260913'
SKIP={'source_snapshot','__pycache__','.git','assets','checkpoints','qa','before'}


def strings(x):
    if isinstance(x,str):yield x
    elif isinstance(x,dict):
        for k,v in x.items():yield k;yield from strings(v)
    elif isinstance(x,list):
        for v in x:yield from strings(v)


def main():
    assert not (DEST/'plan.json').exists(), 'Cleanup plan already exists'
    DEST.mkdir(parents=True,exist_ok=True)
    units={}
    for category in ('adaptation','audits','data_review','development','evaluation','flight_packages',
                     'mppi_pva','ppo_pva','pva_jobs','reference_tracking','rehearsals_pva','retired'):
        for p in (ROOT/'runs'/category).iterdir():
            units[p.relative_to(ROOT).as_posix()]=p
    for category in ('data/flight_batches','data/raw_takes','exports'):
        for p in (ROOT/category).iterdir():
            if p.is_dir():units[p.relative_to(ROOT).as_posix()]=p
    aliases={'rehearsal_csv_and_result_in_real_flight/':'data/flight_batches/','policies/':'exports/'}
    reasons={};queue=deque()
    def protect(name,why):
        if name in units and name not in reasons:reasons[name]=why;queue.append(name)
    def reference(value,why):
        value=value.replace('\\','/').strip()
        prefix=ROOT.as_posix()+'/'
        if value.lower().startswith(prefix.lower()):value=value[len(prefix):]
        for old,new in aliases.items():
            if value.startswith(old):value=new+value[len(old):]
        for name in units:
            if value==name or value.startswith(name+'/'):protect(name,why);break
    catalog=json.loads((ROOT/'config/evaluation/campaign.json').read_text(encoding='utf-8'))
    active_models=[m for m in catalog['models'] if m['id'] in ('M0','M1','M2-selected')]
    for value in strings(active_models):reference(value,'active model catalog')
    for path in (ROOT/'config').rglob('*.json'):
        if 'history' in path.parts or 'flight_selection_history' in path.parts or path.parent.name=='evaluation':continue
        for value in strings(json.loads(path.read_text(encoding='utf-8'))):reference(value,'active config '+path.relative_to(ROOT).as_posix())
    for name in ('M0_Bspline_slower_brake_1s','M1_local_fixed_tip_reference','M2_selected_fixed_tip_reference'):
        protect('exports/'+name,'current flown/next CSV and raw takes')
    for name in ('M0_paper_slower_brake_20260913','M1_paper_local_correction_20260913','preliminary1'):
        protect('data/flight_batches/'+name,'current original measurements')
    for name in units:
        if name.startswith('data/raw_takes/'):protect(name,'retained preliminary raw measurements')
    for name in ('runs/reference_tracking/M0-paper-fixed-reference',
                 'runs/data_review/M0-paper-20260913','runs/data_review/M1-paper-20260913',
                 'runs/evaluation/M1-paper-20260913-common-1p5s',
                 'runs/audits/M1-paper-20260913-gradients','runs/audits/M2-drone-stage-diagnosis-20260913',
                 'runs/audits/icra2027-related-work-20260911'):
        protect(name,'current research results or literature')
    # Read dependency metadata, never raw tracking contents or protected recordings.
    while queue:
        name=queue.popleft();p=units[name]
        paths=[]
        if p.is_dir():
            for base,dirs,files in os.walk(p,followlinks=False):
                dirs[:]=[d for d in dirs if d not in SKIP and not (Path(base)/d).is_symlink()]
                paths.extend(Path(base)/f for f in files if f.endswith('.json'))
        elif p.suffix=='.json':paths=[p]
        for path in paths:
            if 'fig8vertical_002' in str(path).lower():continue
            try:value=json.loads(path.read_text(encoding='utf-8'))
            except (UnicodeError,json.JSONDecodeError):continue
            for text in strings(value):reference(text,path.relative_to(ROOT).as_posix())
    moves=[]
    for name,p in units.items():
        if name not in reasons:
            # Keep recent standalone development checks and wrappers; obsolete
            # evidence and unused exported commands are staged by full directory.
            if name.startswith('runs/development/'):continue
            moves.append(dict(source=name,reason='Not referenced by the active model/configuration/measurement dependency closure'))
    # Keep current manuscript, bibliography, and the latest measurement figure.
    # The live manuscript resides in Dropbox and is outside this cleanup.
    for p in (ROOT/'paper/figures').iterdir():
        if p.name.startswith('M1_take004_whip_comparison') or p.name in ('AeroWhip_Figure1.png','AeroWhip_Figure1a.png','AeroWhip_Figure1b.png'):continue
        moves.append(dict(source=p.relative_to(ROOT).as_posix(),reason='Superseded paper figure or illustration attempt'))
    for p in (ROOT/'tmp').iterdir():
        moves.append(dict(source=p.relative_to(ROOT).as_posix(),reason='Completed temporary manuscript revision, preview, backup or scratch file'))
    for name in ('docs/history','docs/development','paper/REVIEW_2026-09-12.md',
                 'config/evaluation/system_review.json','config/pva/flight_selection_history'):
        if (ROOT/name).exists():moves.append(dict(source=name,reason='Superseded development documentation or legacy study selection'))
    keep_paper={'PAPER_WRITING_HANDOFF.md','PAPER_EXPERIMENT_PROTOCOL.md','FRAMEWORK_EDITORIAL_REVISION_20260913.md',
        'MODEL_IMPROVEMENT_FRAMING_20260913.md','OVERLEAF_FRAMEWORK_REVISION_20260913.md',
        'ICRA_2027_RELATED_WORK_AND_FRAMING.md','RELATED_PAPERS_DETAILED_REVIEW.md','related_papers.bib',
        'UAV_Cable_Whip_Literature_Review.pdf'}
    for p in (ROOT/'docs/paper').iterdir():
        if p.name not in keep_paper:moves.append(dict(source=p.relative_to(ROOT).as_posix(),reason='Superseded manuscript review or study proposal'))
    # All executable code, tests, dependencies, current source, calibration and
    # frozen source snapshots remain; their hashes bind the active fitting jobs.
    for item in moves:
        p=ROOT/item['source']
        assert p.exists() and p.resolve().is_relative_to(ROOT)
        assert not p.is_symlink() and not (p.stat().st_file_attributes & 0x400),str(p)
        if p.is_dir():
            for base,dirs,files in os.walk(p,followlinks=False):
                for n in dirs+files:
                    q=Path(base)/n
                    if 'fig8vertical_002' in n.lower():raise ValueError('Protected recording unit: '+str(p))
                    if q.stat().st_file_attributes & 0x400:raise ValueError('Reparse point in move: '+str(q))
    plan=dict(schema='reversible_workspace_cleanup_v1',root=str(ROOT),destination=str(DEST),
        active_models=[m['id'] for m in active_models],retained_dependencies=reasons,moves=moves,
        deletion_performed=False,external_paths_untouched=True)
    (DEST/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print('Retained artifact units:',len(reasons),'Proposed moves:',len(moves))
    print('Retained data:',[n for n in reasons if n.startswith('data/')])
    print('Retained fit jobs:',[n for n in reasons if n.startswith('runs/adaptation/')])
    print('Move categories:',{category:sum(x['source'].startswith(category+'/') for x in moves) for category in ('runs','data','exports','docs','paper','tmp','config')})


if __name__=='__main__':main()
