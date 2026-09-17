"""Read-only generation/session index, using explicit forecast identities."""
from pathlib import Path
from .adaptation_check import discover_batches,flight_names
from .model_evaluation import load_catalog,model_identity
from .io import sha256_file
from simulator.workflow import read_json


def flight_library(root):
    root=Path(root);catalog=load_catalog(root)
    models={m['id']:dict(m) for m in catalog['models']}
    bindings={}
    for f in catalog.get('flights',[]):
        key=(f.get('command_sha256'),f.get('forecast_sha256'))
        if all(key) and f.get('model') in models:bindings.setdefault(key,set()).add(f['model'])
    sessions=[]
    for batch in discover_batches(root):
        if (batch/'ARCHIVED').exists():continue
        try:
            p=read_json(batch/'protocol.json',{})
            matches=bindings.get((p.get('command_sha256'),p.get('forecast_sha256')),set())
            # A folder called M1, or a bare generation integer, is not model identity.
            generation=next(iter(matches)) if len(matches)==1 else 'unassigned'
            rehearsal=Path(p['rehearsal']) if p.get('rehearsal') else None
            if rehearsal is not None and not rehearsal.is_absolute():rehearsal=root/rehearsal
            # A new generation's prepared inbox can exist before a comparison
            # report is registered. Bind it through actual frozen model content.
            if not matches and rehearsal is not None:
                if (sha256_file(batch/'simulation_csv/fullstate_30hz.csv')==p.get('command_sha256')
                        and sha256_file(rehearsal/'rehearsal.npz')==p.get('forecast_sha256')):
                    signature,_=model_identity(rehearsal/'model.json')
                    owners=[key for key,m in models.items() if m.get('signature')==signature]
                    if len(owners)==1:generation=owners[0]
            meta=read_json(rehearsal/'rehearsal.json',{}) if rehearsal else {}
            names=flight_names(batch);roles=p.get('planned_roles',{})
            planner=str(meta.get('planner','PVA')).upper()
            if planner.startswith('MPPI'):planner='MPPI'
            elif planner.startswith('PPO'):planner='PPO'
            sessions.append(dict(batch=str(batch.resolve()),generation=generation,
                planner=planner,rehearsal=str(rehearsal) if rehearsal else None,
                takes=[dict(name=n,role=roles.get(n,'unassigned')) for n in names]))
        except (OSError,ValueError,KeyError,TypeError):
            sessions.append(dict(batch=str(batch.resolve()),generation='unassigned',planner='PVA',
                rehearsal=None,takes=[dict(name=n,role='unassigned') for n in flight_names(batch)]))
    return dict(models=models,sessions=sessions)
