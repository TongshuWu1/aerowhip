"""Read-only figures of all logged attempts in the failed, pre-continuation PPO run."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT / 'runs/ppo/20260908-080833-956336-seed655'
files = sorted((RUN/'attempts').glob('*.npz'))
columns = {k: [] for k in ('episode', 'reward', 'success', 'batch_end')}
hashes = {}
for path in files:
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as data:
        for key in columns:
            columns[key].append(data[key].copy())
data = {k: np.concatenate(v) for k, v in columns.items()}
x, reward, success = data['episode'], data['reward'], data['success']
status = json.loads((RUN/'status.json').read_text())
assert status['status'] == 'FAILED'
assert np.array_equal(x, np.arange(1, status['episodes']+1)), 'Missing or duplicated attempts'
assert np.isfinite(reward).all()
assert success.dtype == bool
assert int(success.sum()) == status['successes']
with (RUN/'training_log.csv').open(newline='') as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == len(files)
for row in rows:
    subset = data['batch_end'] == int(row['episodes'])
    assert np.isclose(reward[subset].mean(), float(row['mean_episode_reward']), rtol=0, atol=1e-8)
    assert np.isclose(success[subset].mean(), float(row['batch_success_rate']), rtol=0, atol=1e-12)

validation = [json.loads(line) for line in (RUN/'validation_history.jsonl').read_text().splitlines()]
validation.sort(key=lambda row: row['training_episodes'])
vx = np.array([row['training_episodes'] for row in validation])
vr = np.array([row['mean_episode_reward'] for row in validation])
vs = np.array([row['success_rate'] for row in validation])
assert len(set(row['evaluation_id'] for row in validation)) == len(validation)
assert all(row['episodes'] == 256 and not row['evaluation_reused'] for row in validation)
assert vx[-1] == 166912

window = 5000
def trailing_mean(values):
    sums = np.r_[0., np.cumsum(values, dtype=float)]
    return (sums[window:] - sums[:-window])/window

mx = x[window-1:]
mr, ms = trailing_mean(reward), trailing_mean(success)
blue, amber, ink, grey = '#116B91', '#B96513', '#182D3D', '#8998A5'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
    'axes.titlesize':13, 'axes.titleweight':'bold', 'axes.labelcolor':ink,
    'text.color':ink, 'xtick.color':'#52616D', 'ytick.color':'#52616D',
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.edgecolor':'#C7D1D9', 'savefig.facecolor':'white'})
fig, axes = plt.subplots(2,2,figsize=(15.8,9.6),sharex=True)
fig.subplots_adjust(left=.065,right=.97,bottom=.14,top=.80,wspace=.18,hspace=.32)
fig.text(.065,.955,'PPO learning before the crash',fontsize=24,weight='bold')
fig.text(.065,.917,'167,936 recorded attempts  |  164 batches of 1,024  |  Native 30 Hz, both residuals',
         fontsize=12,color='#52616D')
fig.text(.065,.855,'EXPLORATORY TRAINING',fontsize=12,weight='bold',color=blue)
fig.text(.553,.855,'DETERMINISTIC SIMULATED VALIDATION',fontsize=12,weight='bold',color=amber)

ax=axes[0,0]
ax.scatter(x,reward,s=1.,c=grey,alpha=.12,linewidths=0,rasterized=True,label='Each training attempt')
ax.plot(mx,mr,color=blue,lw=2.2,label='Trailing 5,000-attempt mean')
ax.set_title('Reward per attempt',loc='left',pad=12)
ax.set_ylabel('Total episode reward')
ax.set_ylim(-115,270)
ax.legend(loc='upper left',frameon=True,facecolor='white',edgecolor='white',fontsize=9)
ax.annotate(f'{mr[-1]:.1f}',(mx[-1],mr[-1]),xytext=(-8,12),textcoords='offset points',
            ha='right',color=blue,weight='bold')

ax=axes[0,1]
ax.plot(vx,vr,color=amber,lw=1.8,marker='.',ms=3)
ax.set_title('Mean reward of 256 validation trials',loc='left',pad=12)
ax.set_ylabel('Mean episode reward')
ax.set_ylim(-115,270)
ax.annotate(f'{vr[-1]:.1f}',(vx[-1],vr[-1]),xytext=(-5,10),textcoords='offset points',
            ha='right',color=amber,weight='bold')

ax=axes[1,0]
ax.scatter(x,success.astype(float),s=1.1,c=grey,alpha=.08,linewidths=0,rasterized=True,
           label='Each attempt: miss = 0%, hit = 100%')
ax.plot(mx,ms,color=blue,lw=2.2,label='Trailing 5,000-attempt success rate')
ax.set_title('Success per attempt and rolling rate',loc='left',pad=12)
ax.set_ylabel('Success')
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_ylim(-.035,1.05)
ax.legend(loc='upper left',bbox_to_anchor=(0,.9),frameon=True,facecolor='white',edgecolor='white',fontsize=9)
ax.annotate(f'{100*ms[-1]:.1f}%',(mx[-1],ms[-1]),xytext=(-8,10),textcoords='offset points',
            ha='right',color=blue,weight='bold')

ax=axes[1,1]
ax.plot(vx,vs,color=amber,lw=1.8,marker='.',ms=3)
ax.set_title('Success rate on the fixed validation scenarios',loc='left',pad=12)
ax.set_ylabel('Trials that hit successfully')
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_ylim(-.035,1.05)
ax.annotate(f'{100*vs[-1]:.1f}%',(vx[-1],vs[-1]),xytext=(-5,10),textcoords='offset points',
            ha='right',color=amber,weight='bold')

for ax in axes.flat:
    ax.set_xlim(0,x[-1]*1.025)
    ax.set_xticks([0,40000,80000,120000,160000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'0' if value==0 else f'{value/1000:.0f}k'))
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#E4E9ED',lw=.8)
    ax.axvline(x[-1],color='#B65C62',ls=':',lw=1.1)
for ax in axes[1]:
    ax.set_xlabel('Training attempts completed')
fig.text(.065,.078,'Grey points are individual training attempts; blue lines average the previous 5,000 attempts. '
         'Training uses exploration; validation uses the deterministic policy.',fontsize=9.5,color='#52616D')
fig.text(.065,.053,'Attempts within each parallel batch have arbitrary order. Dotted line: crash at 167,936. '
         'Last completed validation: 166,912. Fixed simulation development set.',fontsize=9.5,color='#52616D')
fig.savefig(OUT/'learning_before_crash.png',dpi=180)
fig.savefig(OUT/'learning_before_crash.pdf',dpi=180)
plt.close(fig)

summary = dict(run=str(RUN),attempts=len(x),batches=len(files),all_attempts_present=True,
    batch_metrics_match_log=True,successes=int(success.sum()),overall_training_success=float(success.mean()),
    last_5000_training_reward=float(mr[-1]),last_5000_training_success=float(ms[-1]),
    last_batch_reward=float(rows[-1]['mean_episode_reward']),last_batch_success=float(rows[-1]['batch_success_rate']),
    last_validation_attempts=int(vx[-1]),last_validation_reward=float(vr[-1]),last_validation_success=float(vs[-1]),
    validation_trials_per_checkpoint=256,smoothing_window=window,input_sha256=hashes,
    note='All original training attempts included. Per-attempt order within parallel batches is arbitrary. '
         'Binary success is shown at 0% or 100%; the blue line is a rolling frequency. No continuation data included.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},indent=2))
