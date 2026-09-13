"""Review the completed single trial; never optimize or regenerate a forecast."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from matplotlib.figure import Figure
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from tools.review_preferred_whip import evaluate
from tools.check_preferred_fold_objective import AUDIT,PREFERRED,CURRENT,load_case,evaluate as rank


def main():
    status=read_json(AUDIT/'status.json');assert status['status']=='completed'
    new=Path(status['rehearsal']);report={};views={}
    import torch
    reference=torch.tensor(np.load(AUDIT/'wave_reference.npz')['shape'],dtype=torch.float64)
    for key,path in [('preferred_old',PREFERRED),('previous_sweep',CURRENT),('new_trial',new)]:
        with np.load(path/'rehearsal.npz') as z:arrays={k:z[k] for k in z.files}
        meta=read_json(path/'rehearsal.json');cfg=read_json(path/'settings.json')
        metrics,view=evaluate(arrays,meta,cfg)
        metrics['new_objective_fixed_diagnostic']=rank(load_case(path),reference)
        report[key]=metrics;views[key]=view
    atomic_json(AUDIT/'motion_review.json',report)
    fig=Figure(figsize=(14,8),layout='constrained');axes=fig.subplots(2,3)
    for col,key in enumerate(views):
        a=views[key];t=a['t'];q=a['q'];x0=a['origin0'][0]
        for fraction,color in zip((.4,.6,.8,1.),('#cbd5e1','#94a3b8','#f59e0b','#dc2626')):
            i=np.argmin(abs(t-fraction*t[-1]));axes[0,col].plot(q[i,:,0]-x0,q[i,:,2],'-o',ms=3,color=color,label=f'{t[i]:.2f}s')
            axes[0,col].plot(a['origin'][i,0]-x0,a['origin'][i,2],'s',color=color,ms=5)
        axes[0,col].plot(a['target'][0]-x0,a['target'][2],'bx',ms=10)
        axes[0,col].set(title=key.replace('_',' '),xlabel='X relative to initial drone [m]',ylabel='Z [m]',xlim=(-.2,1.5),ylim=(.1,2.3))
        axes[0,col].set_aspect('equal');axes[0,col].legend(fontsize=8);axes[0,col].grid(alpha=.2)
        axes[1,col].plot(t,a['total_turn'],label='Total cable turning [rad]')
        axes[1,col].plot(t,a['v'][:,-1,0],label='Tip forward speed [m/s]')
        axes[1,col].plot(t,a['drone_v'],label='Drone forward speed [m/s]')
        axes[1,col].set(xlabel='Time [s]',ylim=(-3,7));axes[1,col].grid(alpha=.2);axes[1,col].legend(fontsize=8)
    fig.suptitle('Preferred-fold trial: same new model, revised objective\nOriginal old forecast shown as a style reference; each curve stops at first tip encounter.',fontsize=14)
    fig.savefig(AUDIT/'motion_comparison.png',dpi=160)
    frozen=read_json(AUDIT/'frozen_forecast_manifest.json')
    protected=read_json(AUDIT/'preflight.json')['protected_hashes']
    assert all(sha256_file(Path(p))==digest for p,digest in {**frozen,**protected}.items())
    atomic_json(AUDIT/'review_verification.json',dict(status='passed',frozen_files_verified=len(frozen),protected_files_unchanged=len(protected),
        physical_validation=False,optimization_restarted=False))
    print(json.dumps(report['new_trial'],indent=2))

if __name__=='__main__':main()
