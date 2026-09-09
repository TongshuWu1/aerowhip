"""Improve reference plot aspect ratio without changing the CSV."""
from pathlib import Path
import csv,json,hashlib,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[3]
folder=root/'policies/PPO-fullstate-gentle-recovery-20260907'
meta=json.loads((folder/'fullstate.json').read_text())
assert hashlib.sha256((folder/'fullstate_30hz.csv').read_bytes()).hexdigest()==meta['csv_sha256']
rows=list(csv.DictReader((folder/'fullstate_30hz.csv').open()))
p=np.array([[float(r[k]) for k in ('px_m','py_m','pz_m')] for r in rows])
n=meta['original_whip_rows']
fig=plt.figure(figsize=(9,6));ax=fig.add_subplot(111,projection='3d')
ax.plot(*p[:n].T,color='#f59e0b',label='Unchanged whip')
ax.plot(*p[n-1:].T,color='#3b82f6',label='Gentle recovery reference')
ax.scatter(*meta['hover_position_m'],color='green',label='Hover')
span=np.maximum(np.ptp(p,axis=0),.4);center=(p.max(0)+p.min(0))/2
ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Exported drone reference (not measured flight)',
       xlim=(center[0]-span[0]/2,center[0]+span[0]/2),
       ylim=(center[1]-span[1]/2,center[1]+span[1]/2),
       zlim=(center[2]-span[2]/2,center[2]+span[2]/2))
ax.set_box_aspect(span);ax.legend();fig.tight_layout()
fig.savefig(folder/'recovery_reference.png',dpi=140);plt.close(fig)
meta['files']['recovery_reference.png']=hashlib.sha256((folder/'recovery_reference.png').read_bytes()).hexdigest()
(folder/'fullstate.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
with zipfile.ZipFile(folder.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in folder.iterdir():
        if path.is_file():archive.write(path,path.name)
print('CSV unchanged; plot and bundle hashes updated.')
