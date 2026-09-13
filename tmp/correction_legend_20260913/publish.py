from pathlib import Path
import shutil,json,re
root=Path(__file__).resolve().parent
live=Path(r'C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA')
for p in (root/'before').rglob('*'):
    if p.is_file() and p.name!='preview.pdf':
        rel=p.relative_to(root/'before');assert (live/rel).read_bytes()==p.read_bytes()
        if p.name!='command_correction.tex':assert (root/'qa'/rel).read_bytes()==p.read_bytes()
log=(root/'qa/qa.log').read_text(encoding='utf-8')
assert not re.search('Overfull|undefined|multiply defined',log,re.I)
out=Path('output/pdf')
assert (out/'AeroWhip_framework_revision_20260913.pdf').read_bytes()==(root/'before/preview.pdf').read_bytes()
shutil.copy2(root/'qa/figures/command_correction.tex',live/'figures/command_correction.tex')
shutil.copy2(root/'qa/qa.pdf',out/'AeroWhip_framework_revision_20260913.pdf')
shutil.copy2(root/'qa/figure_preview.png',out/'AeroWhip_command_correction.png')
(root/'publication.json').write_text(json.dumps(dict(removed='red same-time error connectors and legend',
    numerical_paths_unchanged=True,markers_unchanged=True,manuscript_text_unchanged=True,
    checked='Tectonic compile and visual inspection of artwork and affected page',cloud_sync_verified=False),indent=2))
print('Published figure and rebuilt manuscript; numerical trajectories unchanged.')
