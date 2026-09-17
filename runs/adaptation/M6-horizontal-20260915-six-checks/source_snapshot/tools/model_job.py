"""Explicit UI entry point for the existing, bounded M0 adaptation protocol."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(stage,job,batch=None):
    import json
    import numpy as np
    import torch
    from experimental_data.current_adaptation import prepare,SOURCE,save
    from experimental_data.adaptation_check import flight_names,load_comparison,sha256
    if stage=='all':
        run('prepare',job,batch)
        for step in ['drone','cable_batched','attitude_refine','baseline','validate']:
            run(step,job)
        return
    job=Path(job).resolve()
    if not job.is_relative_to(ROOT/'runs/adaptation'):
        raise ValueError('Model jobs must be inside runs/adaptation.')
    if stage=='prepare':
        if job.exists():raise ValueError('Use a new job name; existing inputs are immutable.')
        batch=Path(batch).resolve()
        from simulator.workflow import read_json
        if read_json(batch/'protocol.json',{}).get('schema')=='prospective_whip_adaptation_v1':
            raise ValueError('Use tools/adapt_whip.py for the reviewed raw-coordinate whip workflow; this legacy fitter normalizes height and trains neural models.')
        names=flight_names(batch)
        if len(names)!=5:raise ValueError('The current fitter requires exactly five paired flights. Collect/review the M0 repeat first.')
        original=ROOT/'runs/rehearsals/20260908-203914-039721/fullstate_30hz.csv'
        if sha256(batch/'simulation_csv/fullstate_30hz.csv')!=sha256(original):
            raise ValueError('This fitter is limited to the original M0 flight CSV. Review a new protocol before fitting a different maneuver.')
        for name in names:
            data=load_comparison(ROOT,batch,name)
            if not np.isclose(data['metadata']['whip_end_s'],1):raise ValueError('Expected a one-second whip.')
            print('Checked paired OptiTrack / commands:',name,flush=True)
        from experimental_data.hover_calibration import calibrate_batch
        height=calibrate_batch(ROOT,batch)
        if not height['checks_passed']:
            raise ValueError('Review batch hover calibration before fitting: '+'; '.join(height['warnings']))
        print('Batch vertical hover bias [m]:',height['bias_z_m'],flush=True)
        prepare(job,batch,SOURCE)
        hardware=ROOT/'config/current_vehicle.json'
        if hardware.exists():
            save(job/'recorded_hardware.json',json.loads(hardware.read_text()))
        save(job/'ui_prepare_completed.json',dict(batch=str(batch),source=str(SOURCE),takes=names))
        return
    # Only jobs created and reviewed through this protocol are eligible here.
    if not (job/'ui_prepare_completed.json').exists():
        raise ValueError('Use a job prepared by this UI; historical jobs remain read-only.')
    if not json.loads((job/'protocol.json').read_text()).get('vertical_calibration'):
        raise ValueError('Prepare a new job with batch hover normalization; old unnormalized jobs are historical only.')
    if (job/'STOP').exists():raise ValueError('Job has a stop request. Preserve it and prepare a new job.')
    done=job/('ui_'+stage+'_completed.json')
    if done.exists():raise ValueError('This stage is completed; its saved results will not be overwritten.')
    dependencies={'drone':[], 'cable_batched':[], 'attitude_refine':['drone'],
                  'baseline':[], 'validate':['drone','cable_batched','attitude_refine','baseline']}
    for prerequisite in dependencies[stage]:
        if not (job/('ui_'+prerequisite+'_completed.json')).exists():
            raise ValueError('Complete '+prerequisite+' first.')
    from experimental_data.current_adaptation_fit import drone_run
    from experimental_data.adaptation_ensemble import run as cable_run
    from experimental_data.adaptation_attitude import run as attitude_run
    from experimental_data.current_adaptation_validation import baseline_run,validation_run
    torch.set_num_threads(1)
    functions={'drone':drone_run,'cable_batched':cable_run,'attitude_refine':attitude_run,'baseline':baseline_run,'validate':validation_run}
    lock=job/'ui_stage.lock'
    with lock.open('x') as f:f.write(stage)
    try:
        functions[stage](job)
        save(done,dict(stage=stage,model_selected=False,policy_training=False))
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','prepare','drone','cable_batched','attitude_refine','baseline','validate'])
    p.add_argument('--job',required=True,type=Path);p.add_argument('--batch',type=Path)
    args=p.parse_args();run(args.stage,args.job,args.batch)
