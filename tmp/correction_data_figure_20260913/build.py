from pathlib import Path
import numpy as np, json, hashlib, shutil, re
root=Path(__file__).resolve().parent
live=Path(r'C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA')
job=Path('runs/reference_tracking/M2-selected-local-20260913-042637-826368')
rehearsal=Path('runs/rehearsals_pva/20260913-042637-826368-M2-selected-local-fixed-tip-reference')
files=['main.tex','references.bib','ieeeconf.cls','ieeetrans.cls','figures/aerowhip_tikz_styles.tex','figures/problem_whipping_motion.tex','figures/problem_dder_model.tex','figures/command_correction.tex']
for sub in ['before','qa']:
    for name in files:
        dest=root/sub/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(live/name,dest)
shutil.copy2('output/pdf/AeroWhip_framework_revision_20260913.pdf',root/'before/preview.pdf')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ref=np.load(job/'reference.npz');base=np.load(job/'baseline_prediction.npz');aft=np.load(rehearsal/'rehearsal.npz')
result=json.loads((job/'result.json').read_text());meta=json.loads((rehearsal/'rehearsal.json').read_text())
assert (job/'model.json').read_bytes()==(rehearsal/'model.json').read_bytes()
assert sha(job/'reference.npz')==meta['reference_sha256']
assert (job/'reference.npz').read_bytes()==Path('runs/reference_tracking/M0-paper-fixed-reference/reference.npz').read_bytes()
assert meta['csv_sha256']==result['csv_sha256']
t=ref['time_s'];n=len(t);r=ref['tip_position_m'];b=base['cable_positions_m'][:,-1];a=aft['cable_positions_m'][:n,-1]
assert np.array_equal(t,base['time_s'])
assert np.allclose(t,aft['prediction_time_s'][:n],atol=1e-12,rtol=0)
assert np.array_equal(r,aft['reference_tip_positions_m'])
assert np.array_equal(t,aft['reference_time_s'])
rmse=lambda v:float(np.sqrt(np.mean(np.sum((v-r)**2,axis=1))))
assert abs(rmse(b)-result['original_reference_tip_rmse_m'])<1e-10
assert abs(rmse(a)-result['corrected_reference_tip_rmse_m'])<1e-10
assert all(np.isfinite(v).all() for v in [r,b,a])
# Exact saved positions: x-z projection, common limits and equal spatial scale.
scale=1.9
def coord(p):return (.10+scale*(p[0]+.1), .20+scale*(p[2]-.3))
def xy(p):return f'({p[0]:.6f},{p[1]:.6f})'
old=(root/'before/figures/command_correction.tex').read_text()
prefix=old[:old.index(r'\begin{scope}')]
prefix=prefix.replace('% Schematic local command correction; not simulated or measured data.','% Saved M2-selected predictions. No simulation rerun or artificial trajectory changes.')
prefix=prefix.replace('% Curves are tip paths, not cable centerlines. Equal parameters mark equal times.','% World-frame x-z projection. All 171 saved samples; common axes and equal spatial scale.')
prefix=prefix.replace('Latest executed','Executed M1')
prefix=prefix.replace('Same refined vehicle and cable model (fixed)','Same refined model (M2-selected, fixed)')
prefix='\n'.join(line for line in prefix.splitlines() if 'Tip paths: schematic spatial view' not in line)+'\n'
prefix=prefix.replace('(0,-.13) rectangle','(0,-.60) rectangle')
tail=old[old.index(r'\draw[refpath] (.32,-.04)'):].replace(',-.04)',',-.40)')
lines=[prefix]
indices=[round((n-1)*q) for q in [.25,.5,.75,1.]]
for shift,pred in [(.43,b),(4.87,a)]:
    lines.append(f'\\begin{{scope}}[shift={{({shift},.43)}}]')
    lines += [r'\draw[black!45,line width=.45pt] (.1,.20)--(3.14,.20);',r'\draw[black!45,line width=.45pt] (.1,.20)--(.1,2.29);']
    for x in [0,.5,1.,1.5]:
        xx=coord([x,0,.3])[0]
        lines.append(f'\\draw[black!45,line width=.4pt] ({xx:.4f},.20)--({xx:.4f},.15);')
        lines.append(f'\\node[font=\\fontsize{{7}}{{8}}\\selectfont,anchor=north,inner sep=1pt] at ({xx:.4f},.13) {{{x:g}}};')
    for z in [.4,.8,1.2]:
        yy=coord([-.1,0,z])[1]
        lines.append(f'\\draw[black!45,line width=.4pt] (.1,{yy:.4f})--(.05,{yy:.4f});')
        lines.append(f'\\node[font=\\fontsize{{7}}{{8}}\\selectfont,anchor=east,inner sep=1pt] at (.01,{yy:.4f}) {{{z:g}}};')
    lines += [r'\node[font=\fontsize{7.5}{8.5}\selectfont,anchor=north,inner sep=1pt] at (1.62,-.13) {$x$ (m)};',
              r'\node[font=\fontsize{7.5}{8.5}\selectfont,anchor=south,inner sep=1pt] at (.1,2.30) {$z$ (m)};']
    for i in indices:lines.append(r'\draw[errpath] '+xy(coord(r[i]))+'--'+xy(coord(pred[i]))+';')
    for style,data in [('refpath',r),('predpath',pred)]:
        lines.append('\\draw['+style+'] '+'--'.join(xy(coord(v)) for v in data)+';')
    for i in indices:
        lines.append(r'\node[sample,draw=black!65] at '+xy(coord(r[i]))+' {};')
        lines.append(r'\node[sample,draw=awCable,fill=awCable] at '+xy(coord(pred[i]))+' {};')
    lines.append(r'\end{scope}')
lines.append(tail)
(root/'qa/figures/command_correction.tex').write_text('\n'.join(lines),encoding='utf-8')
s=(root/'qa/main.tex').read_text(encoding='utf-8')
oldcap=r'''\caption{Local command correction (schematic).
    The refined model is fixed while commands change toward the
    original tip-motion reference. Paired markers denote matching
    times; red segments show position errors.}'''
newcap=r'''\caption{Saved tip predictions before and after command correction
    with the fixed M2-selected model, shown in the world $x$--$z$ plane.
    Both panels use the original MPPI reference; markers pair equal
    timestamps. Curves are simulation predictions, not measurements.}'''
assert s.count(oldcap)==1;s=s.replace(oldcap,newcap)
oldtext=r'''Figure~\ref{fig:command_correction} illustrates local correction
under a fixed refined model: commands change while the desired
tip trajectory and timing remain unchanged.'''
newtext=r'''Figure~\ref{fig:command_correction} shows saved tip predictions
before and after command correction under the same refined model.
The desired tip trajectory and timing remain unchanged.'''
assert s.count(oldtext)==1;s=s.replace(oldtext,newtext)
(root/'qa/main.tex').write_text(s,encoding='utf-8')
(root/'qa/qa.tex').write_text('\\RequirePackage[OT1]{fontenc}\n\\input{main.tex}\n')
shutil.copy2('tmp/correction_figure_20260913/qa/figure_preview.tex',root/'qa/figure_preview.tex')
sources=[job/'reference.npz',job/'baseline_prediction.npz',job/'result.json',job/'model.json',rehearsal/'rehearsal.npz',rehearsal/'rehearsal.json',rehearsal/'model.json']
manifest=dict(sources={str(p.resolve()):sha(p) for p in sources},samples=n,interval_s=t[[0,-1]].tolist(),
    marker_indices=indices,marker_times_s=t[indices].tolist(),projection='world x-z; identical limits and equal axis scaling',
    smoothing=False,retiming=False,simulation_rerun=False,physical_measurements=False,
    before_tip_reference_rmse_3d_m=rmse(b),after_tip_reference_rmse_3d_m=rmse(a))
(root/'data_provenance.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='sources'},indent=2))
