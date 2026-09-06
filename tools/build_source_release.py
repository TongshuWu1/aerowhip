"""Build a reviewed source-only snapshot without changing the live workspace.

Explicit allowlists exclude recordings, checkpoints, historical reports, local
state and Git history. Output is a release candidate, not a licensing decision.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGES=('simulator','learning','experimental_data','tools','tests')
GUIDES=('INSTALL.md','ARCHITECTURE.md','REPRODUCIBILITY.md','PUBLICATION.md')
CONFIGS=('model','task','ppo','sac','cable_fit')


def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode('utf-8')
def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return digest(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())


def portable_configs(root):
    result={name:json.loads((root/f'config/{name}.json').read_text(encoding='utf-8')) for name in CONFIGS}
    original={name:canonical(value) for name,value in result.items()}
    model=result['model']
    if model.get('motion_residual',{}).get('enabled'):
        raise ValueError('An enabled residual requires a separately reviewed checkpoint export')
    model['cable'].pop('previous_parameter_source',None)
    model['cable']['parameter_source']='bundled physical values; original fit data not distributed'
    for name in ('ppo','sac'):
        prior=result[name].get('bootstrap')
        if prior and prior.get('enabled'):
            if not prior.get('actions'):raise ValueError('Prior is not self-contained')
            prior['source']='embedded CEM action prior; preceding search used 8192 attempts'
        result[name]['training']['device']='auto'
        result[name]['training']['collection_batch']=64
    result['ppo']['curriculum']['meaning']='Disabled; exact task with the embedded searched force prior'
    # GPU comparison settings remain documented in the original config hashes.
    # Smaller defaults avoid allocating a workstation-size buffer on first use.
    result['sac']['sac']['replay_capacity']=65536
    result['sac']['sac']['updates_per_collection']=16
    result['baseline']=dict(version='bundled-physical-baseline',model_sha256=canonical(model))
    return result,original


def build(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Choose a new output folder; existing releases are immutable')
    archive=Path(str(output)+'.zip')
    if archive.exists():raise ValueError('Release archive already exists')
    files={}
    def add(relative):
        path=root/relative
        if path.is_symlink():raise ValueError(f'Linked release source: {relative}')
        if not path.resolve().is_relative_to(root):raise ValueError('Source outside workspace')
        files[Path(relative).as_posix()]=path.read_bytes()
    for folder in PACKAGES:
        for path in sorted((root/folder).rglob('*')):
            if '__pycache__' in path.parts:continue
            if path.is_file() and path.suffix in ('.py','.svg'):add(path.relative_to(root))
    for path in sorted((root/'requirements').glob('*')):
        if path.suffix in ('.txt','.json'):add(path.relative_to(root))
    for name in ('run_simulation.py','run_ppo.py','run_sac.py','run_tests.py',
                 'requirements.txt','pytest.ini','.editorconfig','.github/workflows/smoke.yml','tests/README.md'):
        add(name)
    for name in GUIDES:add('docs/'+name)
    readme=root/'docs/release/README.md'
    files['README.md']=(readme if readme.exists() else root/'README.md').read_bytes()
    files['docs/FLIGHT_ADAPTATION_QUICKSTART.md']=(root/'docs/FLIGHT_ADAPTATION_QUICKSTART.md').read_bytes()
    files['.gitignore']=b'__pycache__/\n*.py[cod]\n.venv/\n.pytest_cache/\n.idea/\n/runs/\n/results/\n/dist/\n/archive/\n/data/**\n!/data/README.md\n!/data/dataset_manifest.json\n'
    files['.gitattributes']=b'* text=auto\n*.py text eol=lf\n*.json text eol=lf\n*.md text eol=lf\n'
    configs,original=portable_configs(root)
    for name,value in configs.items():files[f'config/{name}.json']=encoded(value)
    files['data/dataset_manifest.json']=encoded(dict(schema='aerial_cable_dataset_manifest_v1',takes={}))
    files['data/README.md']=b'No real recordings, derived datasets, or checkpoints are included. Import your own reviewed flight logs. See ../docs/REPRODUCIBILITY.md.\n'
    # Record limits explicitly, rather than inserting invented authors/license/DOI.
    files['PUBLICATION_METADATA.json']=encoded(dict(status='review_candidate',license=None,authors=[],paper_doi=None,
        scope='source only; no claim of real-flight validation',required_before_public_release=[
            'Confirm authors, ownership, license and third-party attribution',
            'Decide whether to release a separately reviewed data/checkpoint artifact',
            'Review claims and run final release tests on supported environments']))
    findings=[]
    patterns={'personal_machine_path':r'[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_-]+[\\/]',
              'private_key_header':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
              'github_token':r'gh[pousr]_[A-Za-z0-9]{30,}'}
    for name,raw in files.items():
        text=raw.decode('utf-8')
        for kind,pattern in patterns.items():
            if re.search(pattern,text):findings.append(dict(path=name,kind=kind))
        if len(raw)>2_000_000:findings.append(dict(path=name,kind='unexpected_large_file'))
    if findings:raise ValueError(f'Release requires review: {findings}')
    manifest=dict(schema='source_release_manifest_v1',created_at=datetime.now(timezone.utc).isoformat(),
        files={name:dict(sha256=digest(raw),bytes=len(raw)) for name,raw in sorted(files.items())},
        original_config_sha256=original,bundled_config_sha256={name:canonical(value) for name,value in configs.items()},
        transformations=['Removed local provenance paths; preserved physical/reward/action values',
            'Embedded prior retained; bootstrap.source is provenance, not a required file',
            'Device auto and collection batch64; SAC replay65536 and16 updates/collection for smaller development runs',
            'Empty dataset manifest; no raw data, fit reports or checkpoints; no Git history'],
        byte_count=sum(len(raw) for raw in files.values()),scan_findings=[],
        scan_limits='Selected patterns only; not a complete secret, license, or Git-history audit')
    files['RELEASE_MANIFEST.json']=encoded(manifest)
    output.mkdir(parents=True)
    for name,raw in files.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as stream:
        for name,raw in sorted(files.items()):stream.writestr(name,raw)
    archive.with_suffix('.zip.sha256').write_text(digest(archive.read_bytes())+'  '+archive.name+'\n')
    return dict(directory=str(output),archive=str(archive),files=len(files),bytes=manifest['byte_count'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(ROOT,args.output),indent=2))
