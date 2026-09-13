from pathlib import Path
import shutil,re,json
root=Path(__file__).resolve().parent
live=Path(r'C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA')
for folder in ['before','qa']:
    for name in ['main.tex','references.bib','ieeeconf.cls','ieeetrans.cls','figures/aerowhip_tikz_styles.tex','figures/command_correction.tex','figures/problem_whipping_motion.tex','figures/problem_dder_model.tex']:
        p=root/folder/name;p.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(live/name,p)
shutil.copy2('output/pdf/AeroWhip_framework_revision_20260913.pdf',root/'before/preview.pdf')
p=root/'qa/figures/command_correction.tex';s=p.read_text(encoding='utf-8');before=s
s='\n'.join(l for l in s.splitlines() if 'errpath' not in l and 'Same-time error' not in l)+'\n'
s=s.replace(r'(.32,-.40)--(.79,-.40)',r'(1.48,-.40)--(1.95,-.40)')
s=s.replace(r'(.85,-.40)',r'(2.01,-.40)')
s=s.replace(r'(3.13,-.40)--(3.60,-.40)',r'(4.82,-.40)--(5.29,-.40)')
s=s.replace(r'(3.66,-.40)',r'(5.35,-.40)')
# Numerical trajectory coordinates and marker locations stay byte-identical.
paths=lambda t:[l for l in t.splitlines() if l.startswith((r'\draw[refpath]',r'\draw[predpath]')) and len(l)>300]
assert paths(s)==paths(before)
markers=lambda t:[l for l in t.splitlines() if l.startswith(r'\node[sample')]
assert markers(s)==markers(before)
p.write_text(s,encoding='utf-8')
(root/'qa/qa.tex').write_text('\\RequirePackage[OT1]{fontenc}\n\\input{main.tex}\n')
shutil.copy2('tmp/correction_data_figure_20260913/qa/figure_preview.tex',root/'qa/figure_preview.tex')
print('Removed red connectors and legend; saved trajectory paths and markers unchanged.')
