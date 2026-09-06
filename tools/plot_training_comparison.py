"""Export raw data and trailing averages for an immutable two-algorithm study."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rolling(values,window):
    values=np.asarray(values,dtype=float);valid=np.isfinite(values)
    sums=np.r_[0.,np.cumsum(np.where(valid,values,0.))]
    counts=np.r_[0,np.cumsum(valid)]
    ends=np.arange(1,len(values)+1);starts=np.maximum(0,ends-window)
    den=counts[ends]-counts[starts]
    return np.divide(sums[ends]-sums[starts],den,out=np.full(len(values),np.nan),where=den>0)


def read_attempts(run):
    shards=[]
    for path in sorted((run/'attempts').glob('*.npz')):
        with np.load(path) as saved:shards.append({key:saved[key] for key in saved.files})
    if not shards:return {}
    data={key:np.concatenate([s[key] for s in shards]) for key in shards[0]}
    if not np.array_equal(data['episode'],np.arange(1,len(data['episode'])+1)):
        raise ValueError('Attempt records must cover contiguous complete batches from episode one.')
    return data


def export(study):
    study=Path(study);protocol=json.loads((study/'protocol.json').read_text())
    folder=study/'plots';folder.mkdir(exist_ok=True)
    raw=study/'raw';raw.mkdir(exist_ok=True)
    window=protocol['smoothing_attempts']; colors={'PPO':'#2864a5','SAC':'#d16928'}
    attempts={};validation={}
    for label,item in protocol['runs'].items():
        run=Path(item['directory']);data=read_attempts(run);attempts[label]=data
        path=run/'validation_history.jsonl'
        rows=[]
        if path.exists():
            for line in path.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:continue
        validation[label]=rows
        if data:
            with (raw/f'{label.lower()}_attempts.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(data)
                writer.writerows(zip(*data.values()))
        (raw/f'{label.lower()}_validation.json').write_text(json.dumps(rows,indent=2))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    def finish(fig,axes,name,title):
        for ax in np.atleast_1d(axes).flat:
            ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            handles,_=ax.get_legend_handles_labels()
            if handles:ax.legend(frameon=False)
        fig.suptitle(title+'\nOne training seed · shared searched initialization',fontsize=11)
        for ext in ('png','pdf'):fig.savefig(folder/f'{name}.{ext}',dpi=220)
        plt.close(fig)
    for key,ylabel,name in [('reward','Task return per attempt','training_reward'),
                            ('success','Valid-hit rate [%]','training_success')]:
        fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
        for label,data in attempts.items():
            if not data:continue
            scale=100 if key=='success' else 1
            x=data['episode'];y=data[key].astype(float)*scale
            ends=np.flatnonzero(np.r_[np.diff(data['batch_end'])!=0,True]);starts=np.r_[0,ends[:-1]+1]
            means=[np.mean(y[a:b+1]) for a,b in zip(starts,ends)]
            ax.plot(x[ends],means,'o',color=colors[label],alpha=.35,ms=3)
            ix=np.unique(np.r_[np.arange(0,len(x),100),len(x)-1])
            ax.plot(x[ix],rolling(y,window)[ix],label=label,color=colors[label])
        ax.set(xlabel='Training attempts',ylabel=ylabel,xlim=(0,protocol['attempts_per_algorithm']))
        if key=='success':ax.set_ylim(0,100)
        if not any(attempts.values()):ax.text(.5,.5,'Waiting for the first completed training batch',ha='center',transform=ax.transAxes)
        finish(fig,ax,name,f'Training · trailing {window:,}-attempt mean; dots = raw batch means')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for label,rows in validation.items():
        if not rows:continue
        x=[r['training_episodes'] for r in rows]
        axes[0].plot(x,[100*r['success_rate'] for r in rows],'-o',color=colors[label],label=label,ms=3)
        axes[1].plot(x,[r['mean_episode_reward'] for r in rows],'-o',color=colors[label],label=label,ms=3)
    axes[0].set(ylabel='Deterministic validation hit rate [%]',ylim=(0,100))
    axes[1].set(ylabel='Deterministic validation mean task return')
    for ax in axes:ax.set(xlabel='Training attempts',xlim=(0,protocol['attempts_per_algorithm']))
    finish(fig,axes,'validation',f'Validation · {protocol["validation_episodes"]} fixed scenarios · raw points')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    specifications=[('joint_success','Hit + settled recovery [%]',100),('impact_speed_m_s','Successful impact speed [m/s]',1),
                    ('hit_time_s','Successful hit time [s]',1),('maximum_execution_drone_displacement_m','Peak drone travel, including recovery [m]',1)]
    for ax,(key,label_y,scale) in zip(axes.flat,specifications):
        for label,data in attempts.items():
            if not data or key not in data:continue
            y=data[key].astype(float)*scale
            if key in ('impact_speed_m_s','hit_time_s'):y=np.where(data['success'],y,np.nan)
            ix=np.unique(np.r_[np.arange(0,len(y),100),len(y)-1])
            ax.plot(data['episode'][ix],rolling(y,window)[ix],label=label,color=colors[label])
        ax.set(xlabel='Training attempts',ylabel=label_y,xlim=(0,protocol['attempts_per_algorithm']))
        if scale==100:ax.set_ylim(0,100)
    finish(fig,axes,'impact_and_recovery',f'Training diagnostics · trailing {window:,}-attempt means')
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for label,data in attempts.items():
        if not data:continue
        ends=np.flatnonzero(np.r_[np.diff(data['batch_end'])!=0,True])
        ax.plot(data['elapsed_s'][ends]/3600,data['episode'][ends],'-o',label=label,color=colors[label],ms=3)
    ax.set(xlabel='Active training wall time [hours]',ylabel='Completed training attempts')
    finish(fig,ax,'throughput','Throughput · includes training and periodic validation; excludes queue wait')
    (study/'plot_status.json').write_text(json.dumps({k:len(v.get('episode',[])) for k,v in attempts.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('study',type=Path)
    export(parser.parse_args().study)
