"""Capture training source and environment for experiment provenance."""
from pathlib import Path
from simulator.workflow import atomic_json


def archive_sources(directory):
    import zipfile,hashlib,importlib.metadata
    root=Path.cwd()
    files=[root/name for name in ('run_ppo.py','run_sac.py')]
    for folder in ('learning','simulator','experimental_data','tools'):
        files.extend((root/folder).rglob('*.py'))
    with zipfile.ZipFile(directory/'training_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.relative_to(root).as_posix())
    atomic_json(directory/'training_source_manifest.json',dict(
        sha256=hashlib.sha256((directory/'training_source.zip').read_bytes()).hexdigest(),
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}))
