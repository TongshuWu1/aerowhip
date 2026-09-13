from pathlib import Path
import numpy as np,re,json,hashlib,shutil
root=Path(__file__).resolve().parent
live=Path(r'C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA')
job=Path('runs/reference_tracking/M1-local-20260913-032256-816506')
rehearsal=Path('runs/rehearsals_pva/20260913-032256-816506-M1-local-fixed-tip-reference')
files=['main.tex','references.bib','ieeeconf.cls','ieeetrans.cls','figures/aerowhip_tikz_styles.tex','figures/command_correction.tex','figures/problem_whipping_motion.tex','figures/problem_dder_model.tex']
for sub in ['before','qa']:
    for name in files:
        dest=root/sub/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(live/name,dest)
shutil.copy2('output/pdf/AeroWhip_framework_revision_20260913.pdf',root/'before/preview.pdf')
ref=np.load(job/'reference.npz');base=np.load(job/'baseline_prediction.npz');aft=np.load(rehearsal/'rehearsal.npz')
result=json.loads((job/'result.json').read_text());meta=json.loads((rehearsal/'rehearsal.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (job/'model.json').read_bytes()==(rehearsal/'model.json').read_bytes()
assert sha(job/'reference.npz')==meta['reference_sha256']
assert (job/'reference.npz').read_bytes()==Path('runs/reference_tracking/M0-paper-fixed-reference/reference.npz').read_bytes()
assert meta['csv_sha256']==result['csv_sha256']
t=ref['time_s'];n=len(t);r=ref['tip_position_m'];b=base['cable_positions_m'][:,-1];a=aft['cable_positions_m'][:n,-1]
assert np.array_equal(t,base['time_s'])
assert np.allclose(t,aft['prediction_time_s'][:n],atol=1e-12,rtol=0)
assert np.array_equal(r,aft['reference_tip_positions_m'])
rmse=lambda v:float(np.sqrt(np.mean(np.sum((v-r)**2,axis=1))))
assert abs(rmse(b)-result['original_reference_tip_rmse_m'])<1e-10
assert abs(rmse(a)-result['corrected_reference_tip_rmse_m'])<1e-10
assert all(np.isfinite(v).all() for v in [r,b,a])
assert all((v[:,0]>=-.1).all() and (v[:,0]<=1.5).all() and (v[:,2]>=.3).all() and (v[:,2]<=1.4).all() for v in [r,b,a])
def coord(p):return (.10+1.9*(p[0]+.1),.20+1.9*(p[2]-.3))
def xy(p):return f'({p[0]:.6f},{p[1]:.6f})'
indices=[round((n-1)*q) for q in [.25,.5,.75,1.]]
f=(root/'qa/figures/command_correction.tex').read_text(encoding='utf-8')
f=f.replace('Saved M2-selected predictions','Saved M0-to-M1 correction predictions')
f=f.replace('Executed M1','M0').replace(r'{Corrected\\command}',r'{Corrected M1\\command}')
f=f.replace('Same refined model (M2-selected, fixed)','Refined M1 model (fixed in both panels)')
f=f.replace('Fixed reference','M0 reference').replace('{Prediction}','{M1 prediction}')
scopes=re.findall(r'\\begin\{scope\}.*?\\end\{scope\}',f,re.S)
assert len(scopes)==2
for original,pred in zip(scopes,[b,a]):
    lines=[line for line in original.splitlines() if not ((line.startswith((r'\draw[refpath]',r'\draw[predpath]')) and len(line)>300) or line.startswith(r'\node[sample'))]
    lines.pop()
    for style,data in [('refpath',r),('predpath',pred)]:lines.append('\\draw['+style+'] '+'--'.join(xy(coord(v)) for v in data)+';')
    for i in indices:
        lines.append(r'\node[sample,draw=black!65] at '+xy(coord(r[i]))+' {};')
        lines.append(r'\node[sample,draw=awCable,fill=awCable] at '+xy(coord(pred[i]))+' {};')
    lines.append(r'\end{scope}')
    f=f.replace(original,'\n'.join(lines))
assert 'errpath' not in f and 'Same-time error' not in f and 'M2' not in f
(root/'qa/figures/command_correction.tex').write_text(f,encoding='utf-8')
s=(root/'qa/main.tex').read_text(encoding='utf-8')
old=r'''\caption{Saved tip predictions before and after command correction
    with the fixed M2-selected model, shown in the world $x$--$z$ plane.
    Both panels use the original MPPI reference; markers pair equal
    timestamps. Curves are simulation predictions, not measurements.}'''
new=r'''\caption{M0-to-M1 command correction, shown in the world $x$--$z$
    plane. The fixed M1 model predicts motion under (a) the original
    M0 commands and (b) the corrected M1 commands. Dashed curves show
    the original M0 predicted tip reference; markers pair equal
    timestamps. All curves are saved simulation predictions.}'''
assert s.count(old)==1;s=s.replace(old,new)
old=r'''Figure~\ref{fig:command_correction} shows saved tip predictions
before and after command correction under the same refined model.'''
new=r'''Figure~\ref{fig:command_correction} shows the M0-to-M1 correction:
the fixed M1 model predicts motion under the original M0 commands
and the corrected M1 commands.'''
assert s.count(old)==1;s=s.replace(old,new)
(root/'qa/main.tex').write_text(s,encoding='utf-8')
(root/'qa/qa.tex').write_text('\\RequirePackage[OT1]{fontenc}\n\\input{main.tex}\n')
shutil.copy2('tmp/correction_data_figure_20260913/qa/figure_preview.tex',root/'qa/figure_preview.tex')
sources=[job/'reference.npz',job/'baseline_prediction.npz',job/'result.json',job/'model.json',rehearsal/'rehearsal.npz',rehearsal/'rehearsal.json',rehearsal/'model.json']
manifest=dict(sources={str(p.resolve()):sha(p) for p in sources},before_command='M0',after_command='M1',predictor='M1 in both panels',reference='Original M0 forecast',
    samples=n,interval_s=t[[0,-1]].tolist(),marker_indices=indices,marker_times_s=t[indices].tolist(),
    projection='world x-z; identical limits and equal scaling',red_error_connectors=False,
    smoothing=False,retiming=False,simulation_rerun=False,physical_measurements=False,
    before_tip_reference_rmse_3d_m=rmse(b),after_tip_reference_rmse_3d_m=rmse(a))
(root/'data_provenance.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='sources'},indent=2))
