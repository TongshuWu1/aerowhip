from pathlib import Path
import sys
import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
import torch
from tools.diagnose_preliminary_cable import load, METHODS, SOURCE, metrics
from experimental_data.preliminary_fit import CableForward
from experimental_data.current_adaptation import read,save
from experimental_data.io import sha256_file
from simulator.research_execution import ResearchExecutionModel

torch.set_num_threads(4)
out=Path(__file__).parent
model,selected,states=load(out)
e=ResearchExecutionModel.from_mapping(model,root=SOURCE/'candidate',device='cuda')
data=states[METHODS[1]]
params=[model['cable']['EI_n_m2'],model['cable']['Cb_n_m2_s']]
q,v=CableForward(e,data)(params)
rows=metrics(q,data,selected,list(e.cable.marker_node_indices[1:]))
save(out/'existing_residual_comparison.json',dict(scope='Published residual frozen; same weighted causal initialization, no learning',metrics=rows))
np.savez_compressed(out/'published_M0_with_residual__causal_weighted_1s.npz',q=q.cpu().numpy())
evaluation=read(out/'evaluation.json')
evaluation['published_M0_with_residual__causal_weighted_1s']=rows
summary={}
for label,rows in evaluation.items():
    summary[label]={}
    for role in ('training','validation'):
        subset=[r for r in rows if r['role']==role]
        summary[label][role]={}
        for horizon in ('0.25','0.5','1.0','2.0'):
            summary[label][role][horizon]={metric:float(np.sqrt(np.mean([r['horizons'][horizon][metric]**2 for r in subset]))) for metric in ('tip_rmse_m','all_marker_rmse_m')}
save(out/'summary.json',summary)
# Diagnostic visualization; original motion/prediction arrays remain unchanged.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
labels=[('published_M0_physics__causal_uniform_1s','Uniform 1 s / M0 physics'),
        ('published_M0_physics__causal_weighted_1s','Weighted 1 s / M0 physics'),
        ('pilot_physics__causal_weighted_1s','Weighted 1 s / pilot physics'),
        ('published_M0_with_residual__causal_weighted_1s','Weighted 1 s / M0 + residual')]
for ax,role,title in zip(axes[:2],('training','validation'),('12 training windows','3 separate-take windows')):
    for key,label in labels:
        ax.plot([.25,.5,1,2],[100*summary[key][role][h]['tip_rmse_m'] for h in ('0.25','0.5','1.0','2.0')],'-o',label=label)
    ax.set(xlabel='Prediction duration (s)',ylabel='Tip position RMS (cm)',title=title)
    ax.grid(alpha=.25)
axes[0].legend(fontsize=7)
grid=read(out/'grid.json')
losses=np.array(grid['objectives'][:49]).reshape(7,7)
im=axes[2].imshow(losses,origin='lower',aspect='auto',extent=[-9.5,-2.5,-9.4167,-3.5833])
axes[2].set(xlabel='log10 bending damping',ylabel='log10 bending stiffness',title='Training-only physical loss surface')
fig.colorbar(im,ax=axes[2],label='Robust trajectory objective')
fig.suptitle('Cable-only pilot: measured attachment, no rollout resets; candidate is not selected',fontsize=12)
fig.savefig(out/'comparison.png',dpi=170)
plt.close(fig)
# Figure-eight midpoint trajectory: the same predetermined window as prior audits.
index=next(i for i,w in enumerate(selected) if w['name']=='figure8_001-02441')
truth=data['truth'][index,:,-1].cpu().numpy()
fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained')
t=np.arange(len(truth))/150
curves={key:np.load(out/(key+'.npz'))['q'][index,:,-1] for key in (labels[1][0],labels[2][0],labels[3][0])}
for d,ax in enumerate(axes):
    ax.plot(t,truth[:,d],color='black',label='Measured tip',linewidth=2)
    for key,label in labels[1:]:
        ax.plot(t,curves[key][:,d],label=label,alpha=.8)
    ax.set_ylabel('XYZ'[d]+' (m)');ax.grid(alpha=.25)
axes[0].legend(fontsize=8)
axes[-1].set_xlabel('Time from initialization (s)')
fig.suptitle('Training figure-eight midpoint: frozen initial state and measured attachment')
fig.savefig(out/'figure8_midpoint.png',dpi=170)
plt.close(fig)
before=read(out/'protected_before.json')
changed=[p for p,h in before.items() if sha256_file(p)!=h]
save(out/'integrity.json',dict(protected_files=len(before),changed=changed))
assert not changed
print('SUMMARY',summary)
