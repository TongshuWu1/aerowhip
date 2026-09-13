from pathlib import Path
import re,json,shutil,hashlib
from pypdf import PdfReader
root=Path(__file__).resolve().parent
live=Path(r'C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA')
b=(root/'before/main.tex').read_text(encoding='utf-8');a=(root/'qa/main.tex').read_text(encoding='utf-8')
eq=lambda s:re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',s,re.S)
head=lambda s:re.findall(r'\\(?:section|subsection|subsubsection)\{[^}]+\}',s)
notes=lambda s:re.findall(r'\\(?:Tony|david|edward)\{[^}]*\}',s)
log=(root/'qa/qa.log').read_text(encoding='utf-8')
labels=re.findall(r'\\label\{([^}]+)\}',a);refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',a)
checks=dict(equations_preserved=eq(a)==eq(b),equation_count=len(eq(a)),headings_preserved=head(a)==head(b),
    author_notes_preserved=notes(a)==notes(b),citations_preserved=re.findall(r'\\cite\{[^}]*\}',a)==re.findall(r'\\cite\{[^}]*\}',b),
    undefined_refs=sorted(set(refs)-set(labels)),duplicate_labels=len(labels)!=len(set(labels)),
    overfull_boxes=re.findall(r'Overfull[^\n]*',log),underfull_boxes=re.findall(r'Underfull[^\n]*',log),
    unresolved_in_log=bool(re.search('undefined|multiply defined',log,re.I)),pages=len(PdfReader(root/'qa/qa.pdf').pages))
assert all(checks[k] for k in ['equations_preserved','headings_preserved','author_notes_preserved','citations_preserved'])
assert checks['equation_count']==17 and not any(checks[k] for k in ['undefined_refs','duplicate_labels','overfull_boxes','unresolved_in_log'])
assert not any(ord(c)<32 and c not in '\n\r\t' for c in a)
for p in (root/'before').rglob('*'):
    if p.is_file() and p.name!='preview.pdf':
        rel=p.relative_to(root/'before');assert (live/rel).read_bytes()==p.read_bytes(),f'Concurrent source edit: {rel}'
        if p.name!='main.tex':assert (root/'qa'/rel).read_bytes()==p.read_bytes()
assert not (live/'figures/command_correction.tex').exists()
out=Path('output/pdf');out.mkdir(exist_ok=True,parents=True)
shutil.copy2(root/'qa/figures/command_correction.tex',live/'figures/command_correction.tex')
shutil.copy2(root/'qa/main.tex',live/'main.tex')
shutil.copy2(root/'qa/qa.pdf',out/'AeroWhip_framework_revision_20260913.pdf')
shutil.copy2(root/'qa/figure_preview.png',out/'AeroWhip_command_correction.png')
section=a[a.index(r'\section{AeroWhip Framework}'):a.index(r'\bibliographystyle')]
Path('output/paper/AeroWhip_Framework.tex').write_text(section,encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(checks=checks,source=str(live/'main.tex'),source_sha256=sha(live/'main.tex'),
    figure=str(live/'figures/command_correction.tex'),figure_sha256=sha(live/'figures/command_correction.tex'),
    pdf=str((out/'AeroWhip_framework_revision_20260913.pdf').absolute()),
    backup=str(root/'before'),figure_status='schematic; no measured or simulated data',
    current_figure_number=2,planned_number_after_framework_overview=3,cloud_sync_verified=False)
(root/'publication.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
