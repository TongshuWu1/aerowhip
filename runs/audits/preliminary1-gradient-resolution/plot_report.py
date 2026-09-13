from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).parent
read=lambda p:json.loads(p.read_text())
coupled=read(out/'coupled_diagnostics.json')['takes'];boundary=read(out/'boundary_comparison.json')['takes']
names=list(coupled);x=np.arange(len(names))
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
for offset,label,title,color in [(-.18,'original_M0_weighted','Existing M0','#a94d32'),(.18,'development_M0_weighted','Development variant','#167d8d')]:
    axes[0].bar(x+offset,[100*coupled[n]['models'][label]['tip']['rmse_m'] for n in names],.36,label=title,color=color)
axes[0].set(title='Same command-driven windows',ylabel='2 s tip RMS (cm)')
for offset,key,label in [(-.18,'measured_attachment_tip','Measured attachment'),(.18,'command_driven_tip','Predicted attachment')]:
    axes[1].bar(x+offset,[100*boundary[n][key]['rmse_m'] for n in names],.36,label=label)
axes[1].set(title='Development variant: remaining error',ylabel='2 s tip RMS (cm)')
for ax in axes[:2]:
    ax.set_xticks(x,['Fig8 train','Fig8 separate','Osc 1','Osc 2','Vertical'],rotation=25,ha='right')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
for key,label in [('original','Original smoothing'),('smoother-frame','Wider smoothing / 1 s'),('smoother-two-seconds','Wider smoothing / 2 s')]:
    row=read(out/key/'result.json')
    axes[2].loglog([r['epsilon'] for r in row['finite_differences']],
                  [r['relative_error'] for r in row['finite_differences']],'-o',label=label)
axes[2].set(title='Gradient check across perturbation sizes',xlabel='Parameter perturbation',ylabel='Relative derivative discrepancy')
axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
fig.suptitle('Preliminary M0 development: improved numerics and damping; prospective whip validation still pending')
fig.savefig(out/'comparison.png',dpi=160)
