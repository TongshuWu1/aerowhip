from pathlib import Path
import re,json,hashlib,shutil
from pypdf import PdfReader
root=Path(__file__).resolve().parent
live=Path(r'C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA')
b=(root/'before/main.tex').read_text(encoding='utf-8');a=(root/'qa/main.tex').read_text(encoding='utf-8')
eq=lambda s:re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',s,re.S)
head=lambda s:re.findall(r'\\(?:section|subsection|subsubsection)\{[^}]+\}',s)
log=(root/'qa/qa.log').read_text(encoding='utf-8')
checks=dict(equations_preserved=eq(a)==eq(b),equation_count=len(eq(a)),headings_preserved=head(a)==head(b),
    citations_preserved=re.findall(r'\\cite\{[^}]*\}',a)==re.findall(r'\\cite\{[^}]*\}',b),
    overfull_boxes=re.findall(r'Overfull[^\n]*',log),underfull_boxes=re.findall(r'Underfull[^\n]*',log),
    unresolved_in_log=bool(re.search('undefined|multiply defined',log,re.I)),pages=len(PdfReader(root/'qa/qa.pdf').pages))
assert checks['equations_preserved'] and checks['headings_preserved'] and checks['citations_preserved']
assert checks['equation_count']==17 and not checks['overfull_boxes'] and not checks['unresolved_in_log']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
provenance=json.loads((root/'data_provenance.json').read_text())
assert all(sha(Path(p))==h for p,h in provenance['sources'].items())
for p in (root/'before').rglob('*'):
    if p.is_file() and p.name!='preview.pdf':
        rel=p.relative_to(root/'before');assert (live/rel).read_bytes()==p.read_bytes(),f'Concurrent edit: {rel}'
        if p.name not in ['main.tex','command_correction.tex']:assert (root/'qa'/rel).read_bytes()==p.read_bytes()
out=Path('output/pdf')
assert (out/'AeroWhip_framework_revision_20260913.pdf').read_bytes()==(root/'before/preview.pdf').read_bytes()
for f in ['main.tex','figures/command_correction.tex']:shutil.copy2(root/'qa'/f,live/f)
shutil.copy2(root/'qa/qa.pdf',out/'AeroWhip_framework_revision_20260913.pdf')
shutil.copy2(root/'qa/figure_preview.png',out/'AeroWhip_command_correction.png')
shutil.copy2(root/'data_provenance.json','docs/paper/COMMAND_CORRECTION_FIGURE_DATA_20260913.json')
section=a[a.index(r'\section{AeroWhip Framework}'):a.index(r'\bibliographystyle')]
Path('output/paper/AeroWhip_Framework.tex').write_text(section,encoding='utf-8')
(root/'publication.json').write_text(json.dumps(dict(checks=checks,data=provenance,
    source_sha256=sha(live/'main.tex'),figure_sha256=sha(live/'figures/command_correction.tex'),
    cloud_sync_verified=False,simulation_rerun=False),indent=2))
print(json.dumps(checks,indent=2))
