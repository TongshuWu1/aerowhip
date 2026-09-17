"""Execute an explicitly prepared, finite MPPI trial list; never create runs.

A manual stop cancels the list. Only a completed miss/infeasibility or the
planner's explicit no-feasible-candidate outcome may advance to the next trial.
Infrastructure/programming failures halt for inspection, rather than retrying.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,subprocess,time,os
from experimental_data.io import atomic_json


def next_action(status,result):
    if status.get('status')=='completed':
        if not result or 'best_success' not in result or 'best_failed' not in result:return 'halt'
        return 'success' if result['best_success'] and not result['best_failed'] else 'next'
    if status.get('status')=='failed' and status.get('error','').startswith('No feasible MPPI lookahead candidate;'):
        return 'next'
    return 'halt'


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def run(directory):
    directory=Path(directory).resolve();plan=read(directory/'plan.json');trials=plan['trials']
    if not trials:raise ValueError('An explicitly prepared trial list is required')
    if (directory/'status.json').exists():raise ValueError('Trial list already started; inspect its status')
    lock=directory/'worker.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    state=dict(status='running',trials=[],pid=os.getpid(),active_job=None)
    try:
        for trial in trials:
            job=Path(trial['job'])
            if (directory/'STOP').exists() or (job/'STOP').exists():state['status']='stopped';break
            if read(job/'status.json').get('status')!='prepared':raise ValueError('Trial is no longer prepared: '+str(job))
            cfg=read(job/'settings.json')
            # Adopt settings only when their authorized trial actually starts.
            atomic_json(Path(plan['root'])/'config/pva/mppi.json',cfg)
            state.update(active_job=str(job),stage='optimizing',horizon_s=cfg['mppi']['horizon_s'],samples=cfg['mppi']['samples'])
            atomic_json(directory/'status.json',state)
            with (job/'console.log').open('ab') as out,(job/'stderr.log').open('ab') as err:
                worker=subprocess.Popen(trial['command'],cwd=plan['root'],stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                while worker.poll() is None:
                    if (directory/'STOP').exists():(job/'STOP').touch(exist_ok=True)
                    time.sleep(.5)
            status=read(job/'status.json');result=read(job/'result.json');action=next_action(status,result)
            state['trials'].append(dict(job=str(job),horizon_s=cfg['mppi']['horizon_s'],samples=cfg['mppi']['samples'],status=status,result=result,decision=action,exit_code=worker.returncode))
            if (directory/'STOP').exists() or status.get('status')=='stopped':state['status']='stopped';break
            if action=='success':state.update(status='completed',outcome='modeled_strike',successful_job=str(job));break
            if action=='halt':state.update(status='needs_attention',outcome='execution_error');break
            atomic_json(directory/'status.json',state)
        else:state.update(status='completed',outcome='no_valid_modeled_strike')
        state['stage']='finished';atomic_json(directory/'status.json',state)
    except BaseException as exc:
        state.update(status='needs_attention',error=str(exc));atomic_json(directory/'status.json',state);raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',type=Path,required=True)
    run(parser.parse_args().sequence)
