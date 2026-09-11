"""Build a reviewed source-only snapshot without changing the live workspace.

Explicit allowlists exclude recordings, checkpoints, historical reports, local
state and Git history. Output is a release candidate, not a licensing decision.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PureWindowsPath
import posixpath
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGES=('simulator','learning','planning','experimental_data','tools','tests','deployment')
GUIDES=('README.md','setup/INSTALL.md','ARCHITECTURE.md','setup/REPRODUCIBILITY.md','setup/PUBLICATION.md',
        'setup/CONTROLLER_INTERFACE_REVIEW.md','setup/LAB_SETUP.md','paper/PAPER_WRITING_HANDOFF.md',
        'paper/PAPER_READINESS_REVIEW.md','methods/SIM_REAL_EVALUATION.md','development/M0_TO_M1_ADAPTATION.md',
        'methods/DIRECT_PVA_WORKFLOW.md','methods/FUTURE_ADAPTATION_FITTING.md','development/M2_PPO_MPPI_MATCH.md',
        'paper/PAPER_EXPERIMENT_PROTOCOL.md','methods/FROZEN_SYSTEM_IDENTIFICATION.md',
        'methods/DRONE_COMMAND_CHAIN_AUDIT.md','methods/DRONE_RESPONSE_ADAPTATION.md')
CONFIGS=('model','task','ppo','sac','cable_fit')


def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode('utf-8')
def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return digest(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())


MARKDOWN_LINK=re.compile(r'(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)')


def local_guide_target(root,document,link):
    """Map a guide link to a repository path without reading research assets."""
    link=link.strip('<>')
    if link.startswith(('https:','http:','mailto:','#')):return None
    path,separator,anchor=link.partition('#')
    if not path:return None
    if re.match(r'^[A-Za-z]:[\\/]',path):
        # Retained literature notes can name the old Windows checkout. Convert
        # only this repository's prefix, including when exporting from Linux.
        parts=PureWindowsPath(path).parts
        markers={root.name,'particle_filter_cable_project','aerowhip'}
        indices=[i for i,part in enumerate(parts) if part in markers]
        if not indices:return None
        target=root.joinpath(*parts[indices[-1]+1:])
    else:
        target=Path(path)
        if not target.is_absolute():target=(root/document).parent/target
    target=target.resolve()
    if not target.is_relative_to(root):return None
    return target.relative_to(root).as_posix(),('#'+anchor if separator else '')


def guide_paths(root):
    """Include the selected guides and their existing local documentation links."""
    pending=['docs/'+name for name in GUIDES];selected=set()
    while pending:
        document=pending.pop()
        if document in selected:continue
        path=root/document
        if not path.is_file():raise FileNotFoundError(f'Required release guide: {document}')
        selected.add(document)
        if path.suffix!='.md':continue
        for _,link in MARKDOWN_LINK.findall(path.read_text(encoding='utf-8')):
            target=local_guide_target(root,document,link)
            if target is None:continue
            relative,_=target
            if relative.startswith('docs/') and Path(relative).suffix in ('.md','.bib') and (root/relative).is_file():
                pending.append(relative)
    return sorted(selected)


def portable_guide_links(root,files):
    """Keep shipped links relative; label separately held evidence without a dead link."""
    for document,raw in list(files.items()):
        if not document.endswith('.md'):continue
        def replace(match):
            target=local_guide_target(root,document,match.group(2))
            if target is None:return match.group(0)
            relative,anchor=target
            if relative not in files:
                return f'{match.group(1)} (`{relative}`, not included in this source-only release)'
            portable=posixpath.relpath(relative,posixpath.dirname(document) or '.')+anchor
            if ' ' in portable:portable='<'+portable+'>'
            return f'[{match.group(1)}]({portable})'
        files[document]=MARKDOWN_LINK.sub(replace,raw.decode('utf-8')).encode('utf-8')


def portable_configs(root):
    result={name:json.loads((root/f'config/{name}.json').read_text(encoding='utf-8')) for name in CONFIGS}
    original={name:canonical(value) for name,value in result.items()}
    model=result['model']
    if model.get('motion_residual',{}).get('enabled'):
        raise ValueError('An enabled residual requires a separately reviewed checkpoint export')
    model['cable'].pop('previous_parameter_source',None)
    model['cable']['parameter_source']='bundled physical values; original fit data not distributed'
    for name in ('ppo','sac'):
        # Compatibility configurations are retained for independent legacy tests.
        # Never distribute a local selected prior as a fresh-system default.
        result[name]['bootstrap']={'enabled':False}
        result[name]['training']['device']='auto'
        result[name]['training']['collection_batch']=64
    result['ppo']['curriculum']['meaning']='Disabled; compatibility configuration, not the selected PVA experiment'
    # GPU comparison settings remain documented in the original config hashes.
    # Smaller defaults avoid allocating a workstation-size buffer on first use.
    result['sac']['sac']['replay_capacity']=65536
    result['sac']['sac']['updates_per_collection']=16
    result['baseline']=dict(version='bundled-physical-baseline',model_sha256=canonical(model))
    for name in ('research_30hz/model','research_30hz/task','research_30hz/ppo','current_vehicle','pva/ppo','pva/mppi'):
        value=json.loads((root/f'config/{name}.json').read_text(encoding='utf-8'))
        original[name]=canonical(value);result[name]=value
    structural=result['research_30hz/model']
    if structural.get('motion_residual',{}).get('enabled') or structural.get('fullstate_execution',{}).get('enabled'):
        raise ValueError('Source release requires an unfitted structural template; fitted models need separate reviewed assets')
    for method in ('ppo','mppi'):
        result['pva/'+method]['device']='auto'
        result['pva/'+method]['model_path']=''
        result['pva/'+method].pop('development_model_review',None)
    objective=result['pva/ppo'].get('ppo_objective')
    if objective:
        # The reference is a separately reviewed research asset, like the model.
        # Keep its checksum but never distribute a workstation path/authorization.
        objective['reference_source']='config/pva/wave_reference.npz'
        objective['source_mppi_run']='separately held selected MPPI run; see source settings checksum'
    result['research_workspace']=dict(schema='research_workspace_v1',config_directory='config/research_30hz',
        bundle=None,model_label='Unfitted structural template',selected_by_user=False)
    result['experiment']=dict(schema='unseen_system_experiment_v1',preliminary_batch=None,fit_job=None,
        status='Collect and review data before fitting; no flight or model selected')
    result['evaluation/campaign']=dict(schema='model_evolution_v1',models=[],flights=[])
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
            if folder=='deployment' and 'reference' in path.relative_to(root/folder).parts:continue
            if path.is_file() and path.suffix in ('.py','.svg'):add(path.relative_to(root))
    for path in sorted((root/'requirements').glob('*')):
        if path.suffix in ('.txt','.json'):add(path.relative_to(root))
    for name in ('run_simulation.py','run_ppo.py','run_sac.py','run_tests.py',
                 'requirements.txt','pytest.ini','.editorconfig','.github/workflows/smoke.yml','tests/README.md',
                 'HANDOFF.md'):
        add(name)
    for name in ('exports/README.md','paper/README.md','third_party/README.md'):
        if (root/name).is_file():add(name)
    for name in guide_paths(root):add(name)
    files['README.md']=b'''# AeroWhip research source\n\nThis source-only candidate contains the current PVA planner, fitting, comparison\nand desktop UI. No fitted model, flight command, forecast or recording is bundled.\nInstall using docs/setup/INSTALL.md, then run `python run_simulation.py`. Review your\ndata and prepare a model before planning. No job starts automatically.\n\nRead docs/paper/PAPER_WRITING_HANDOFF.md and docs/paper/PAPER_READINESS_REVIEW.md for the method\nand evidence limits. Experiment paths in these guides refer to separately held\nresearch artifacts. Legacy numerical backends remain for compatibility tests;\nthey are not the selected experiment. See PUBLICATION_METADATA.json for release\nstatus. This package does not reproduce reported trajectories without their\nseparately reviewed model, source snapshot and exact command assets.\n'''
    files['docs/setup/FLIGHT_ADAPTATION_QUICKSTART.md']=(root/'docs/setup/FLIGHT_ADAPTATION_QUICKSTART.md').read_bytes()
    files['.gitignore']=b'__pycache__/\n*.py[cod]\n.venv/\n.pytest_cache/\n.idea/\n/runs/\n/results/\n/dist/\n/archive/\n/data/**\n!/data/README.md\n!/data/dataset_manifest.json\n'
    files['.gitattributes']=b'* text=auto\n*.py text eol=lf\n*.json text eol=lf\n*.md text eol=lf\n'
    configs,original=portable_configs(root)
    for name,value in configs.items():files[f'config/{name}.json']=encoded(value)
    files['data/dataset_manifest.json']=encoded(dict(schema='aerial_cable_dataset_manifest_v1',takes={}))
    files['data/README.md']=b'No real recordings, derived datasets, or checkpoints are included. Import your own reviewed flight logs. See ../docs/setup/REPRODUCIBILITY.md.\n'
    # Record limits explicitly, rather than inserting invented authors/license/DOI.
    files['PUBLICATION_METADATA.json']=encoded(dict(status='review_candidate',license=None,authors=[],paper_doi=None,
        scope='source only; no claim of real-flight validation',required_before_public_release=[
            'Confirm authors, ownership, license and third-party attribution',
            'Decide whether to release a separately reviewed data/checkpoint artifact',
            'Review claims and run final release tests on supported environments']))
    portable_guide_links(root,files)
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
            'Compatibility force priors disabled; current PVA and structural configurations included',
            'No selected model, replay, flight package, evaluation candidate or fit job is distributed',
            'Device auto and collection batch64; SAC replay65536 and16 updates/collection for smaller development runs',
            'Empty dataset manifest; no raw data, fit reports or checkpoints; no Git history',
            'Included linked current guides; portable documentation links; absent research evidence labeled separately held'],
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
