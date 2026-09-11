"""One-page scientific illustration from an original forecast and measured take.

No rollout, fitting, time re-alignment, measurement recentering or data mutation.
The best-agreement selection is explicit, not an aggregate validation claim.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulator.workflow import read_json
from experimental_data.io import atomic_json,sha256_file
from experimental_data.adaptation_check import load_comparison
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.system_comparison import load_review

OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
TMP=ROOT/'tmp/pdfs/M1_take004_whip';TMP.mkdir(parents=True,exist_ok=True)
review=load_review(ROOT)
# Select globally among reviewed flights by the original, prospective tip RMS.
choices=[(v['tip']['rmse_m'],mid,name,v) for mid,f in review['flights'].items() for name,v in f['takes'].items()
         if v['tip']['coverage']==1 and v['encounter']['contiguous_segment_coverage']==1]
_,mid,take,metrics=min(choices,key=lambda row:row[0])
assert (mid,take)==('M1-full','whip_m1_004')
flight=review['flights'][mid];batch=Path(flight['batch']);rehearsal=Path(flight['rehearsal'])
d=load_comparison(ROOT,batch,take)
assert d['rehearsal']==rehearsal
protected=dict(d['hashes'])
protected.update({str(rehearsal/'rehearsal.npz'):flight['forecast_sha256'],str(rehearsal/'fullstate_30hz.csv'):flight['command_sha256']})
verify_hashes(protected)
t=d['time'];end=d['metadata']['whip_end_s'];use=(t>=0)&(t<=end+1e-10)
q=d['measured_cable'].copy();pred=d['predicted_cable'];origin=d['measured_origin'];po=d['predicted_origin']
valid=np.isfinite(q).all(-1)
jumps=np.linalg.norm(np.diff(q,axis=0),axis=-1)>.15
valid[:-1]&=~jumps;valid[1:]&=~jumps
model=read_json(rehearsal/'model.json')
long=np.linalg.norm(np.diff(q,axis=1),axis=-1)>np.asarray(model['cable']['marker_interval_lengths_m'])+.015
valid[:,1:]&=~long;valid[:,:-1]&=~long
q[~valid]=np.nan
residual=q[:,-1]-pred[:,-1]
norm=np.linalg.norm(residual,axis=1);drone_norm=np.linalg.norm(origin-po,axis=1)
assert valid[use,-1].all()
np.testing.assert_allclose(np.sqrt(np.mean(norm[use]**2)),metrics['tip']['rmse_m'],atol=1e-12)
np.testing.assert_allclose(np.sqrt(np.mean(drone_norm[use]**2)),metrics['drone']['rmse_m'],atol=1e-12)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':10,
    'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
blue='#2764b4';orange='#d96728';ink='#213146';muted='#687589'
fig=plt.figure(figsize=(11.69,6.45),facecolor='white')
ax=fig.add_axes([.025,.45,.47,.52],projection='3d',computed_zorder=False)
ax.plot(*q[use,-1].T,color=orange,lw=2.2,label='Measured tip')
ax.plot(*pred[use,-1].T,color=blue,lw=1.9,ls='--',label='Predicted tip')
ax.plot(*origin[use].T,color=orange,lw=1,alpha=.85,label='Measured drone')
ax.plot(*po[use].T,color=blue,lw=1,ls='--',alpha=.85,label='Predicted drone')
target=d['target']
u,v=np.meshgrid(np.linspace(0,2*np.pi,19),np.linspace(0,np.pi,11))
ax.plot_wireframe(target[0]+.05*np.cos(u)*np.sin(v),target[1]+.05*np.sin(u)*np.sin(v),target[2]+.05*np.cos(v),
    color='#a65161',lw=.35,alpha=.45)
ax.scatter(*target,color='#923448',marker='+',s=50)
ax.text(target[0]-.06,target[1]+.04,target[2]+.10,'Target',fontsize=8,color='#923448')
ax.text(.08,.01,1.91,'Drone paths',fontsize=8,color=muted)
ax.text(.40,-.07,.53,'Tip paths',fontsize=8,color=muted)
ax.set(xlim=(-.1,1.42),ylim=(-.28,.22),zlim=(.2,2.0),xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]')
ax.set_box_aspect((1.52,.50,1.8));ax.view_init(elev=24,azim=-69)
ax.xaxis.set_major_locator(MultipleLocator(.5));ax.yaxis.set_major_locator(MultipleLocator(.2));ax.zaxis.set_major_locator(MultipleLocator(.5))
ax.tick_params(pad=1);ax.xaxis.labelpad=0;ax.yaxis.labelpad=0;ax.zaxis.labelpad=0
for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
    axis.pane.fill=False;axis._axinfo['grid']['color']=(.85,.88,.91,.45)
fig.text(.055,.97,'A   Drone and cable-tip trajectories',color=ink,fontsize=10.5,weight='bold')
fig.text(.055,.40,'Raw world coordinates. Thin paths: drone; thick paths: tip.',fontsize=7.5,color=muted)

signed=fig.add_axes([.58,.675,.365,.255]);mag=fig.add_axes([.58,.43,.365,.165],sharex=signed)
for i,(label,color) in enumerate(zip('XYZ',['#1d7193','#bd5a56','#6c64a8'])):
    signed.plot(t[use],100*residual[use,i],lw=1.45,color=color,label=label)
signed.axhline(0,color='#7d8795',lw=.7);signed.set_ylabel('Measured - predicted [cm]')
signed.set_title('B   Tip position residual',loc='left',color=ink,fontweight='bold',pad=9)
signed.legend(ncol=3,loc='upper left',frameon=False,fontsize=8,borderaxespad=.1)
mag.plot(t[use],100*norm[use],color=orange,lw=1.6,label='Tip')
mag.plot(t[use],100*drone_norm[use],color='#526777',lw=1.35,label='Drone')
mag.set(xlabel='Time from command onset [s]',ylabel='3D error [cm]',xlim=(0,end),ylim=(0,None))
mag.legend(ncol=2,loc='upper left',frameon=False,fontsize=8)
for a in (signed,mag):
    a.grid(axis='y',alpha=.18);a.set_axisbelow(True);a.xaxis.set_major_locator(MultipleLocator(.2))
signed.tick_params(labelbottom=False)

fig.text(.055,.325,'C   Cable shape at matched times (side view, XZ)',color=ink,fontsize=10.5,weight='bold')
snapshots=[]
for j,desired in enumerate([.30,.78,1.18]):
    idx=int(np.argmin(abs(t-desired)));snapshots.append(dict(time_s=float(t[idx]),masked_sites=np.flatnonzero(~valid[idx]).tolist()))
    a=fig.add_axes([.06+j*.31,.065,.245,.22])
    a.plot(pred[idx,:,0],pred[idx,:,2],color=blue,lw=1.25,ls='--')
    a.plot(q[idx,:,0],q[idx,:,2],color=orange,lw=1.3,marker='o',ms=2.4)
    a.scatter(po[idx,0],po[idx,2],c=blue,marker='s',s=19,zorder=5)
    a.scatter(origin[idx,0],origin[idx,2],c=orange,marker='s',s=19,zorder=6)
    circle=plt.Circle((target[0],target[2]),.05,facecolor='none',edgecolor='#923448',lw=.8);a.add_patch(circle)
    a.set(xlim=(-.1,1.4),ylim=(.2,2.02),xlabel='X [m]',ylabel='Z [m]')
    a.set_aspect('equal');a.grid(alpha=.15);a.set_title(f't = {t[idx]:.3f} s',pad=4,fontsize=9)
    a.xaxis.set_major_locator(MultipleLocator(.5));a.yaxis.set_major_locator(MultipleLocator(.5))
fig.savefig(TMP/'plots.pdf',transparent=True)
verify_hashes(protected)
atomic_json(TMP/'figure_data.json',dict(model_id=mid,take=take,rehearsal=str(rehearsal),whip_end_s=end,
    samples=int(use.sum()),tip_rms_cm=100*metrics['tip']['rmse_m'],drone_rms_cm=100*metrics['drone']['rmse_m'],
    nearest_cm=100*metrics['encounter']['nearest']['distance_m'],nearest_time_s=metrics['encounter']['nearest']['time_s'],
    snapshots=snapshots,residual='measured minus original preflight predicted position; not neural-network output',
    selection='Lowest original-forecast tip RMS of 13 reviewed development takes; full tip and approach coverage required.',
    protected_hashes=protected,builder_sha256=sha256_file(Path(__file__))))
print(mid,take,'tip RMS',100*metrics['tip']['rmse_m'],'cm; original evidence unchanged.')
