"""Cable-only recovery must preserve an already selected quadrotor fit."""
from types import SimpleNamespace
import pytest
from experimental_data.io import atomic_json
from experimental_data import whip_full_continuation as continuation


def test_reuse_does_not_restart_translation_or_attitude_optimization(tmp_path,monkeypatch):
    source=tmp_path/'source';job=tmp_path/'next';job.mkdir()
    result=dict(stop_reason='practical_plateau',numerically_verified=True,selected_update=1010)
    for folder in ('drone_nominal','drone_residual','attitude_refinement','adapted_drone','cable_physics'):
        atomic_json(source/folder/'result.json',result if folder=='drone_residual' else {})
    for stage in ('drone_nominal','drone_residual'):
        atomic_json(source/'stages'/stage/'model.json',dict(stage=stage))
    chosen=dict(parameters=[1e-8,1e-4,.4],training_loss=2.,update=1)
    contract=dict(seed=1,replay_weight=.5,continuation=dict(source=str(source),hashes={},
        selected_physical=chosen,reuse_completed_drone=True))
    engine=SimpleNamespace(drone=None);nominal=object();selected=object();saved=[]
    monkeypatch.setattr(continuation.data,'load',lambda *a:({},dict(full_update=contract),engine))
    monkeypatch.setattr(continuation.data,'drone_trials',lambda *a:[])
    monkeypatch.setattr(continuation.data,'cable_rows',lambda *a:[])
    monkeypatch.setattr(continuation.data,'cable_data',lambda *a:{})
    class AtCableStage(Exception):pass
    def mapping(model,**kwargs):
        if 'stage' not in model:raise AtCableStage
        return SimpleNamespace(drone=nominal if model['stage']=='drone_nominal' else selected)
    monkeypatch.setattr(continuation.ResearchExecutionModel,'from_mapping',mapping)
    def save(model,e,folder,job,**kwargs):
        saved.append((folder.name,e.drone));return {}
    monkeypatch.setattr(continuation.full,'save_model',save)
    def forbidden(*a,**k):raise AssertionError('Completed quadrotor fitting must not restart')
    monkeypatch.setattr(continuation,'train_to_plateau',forbidden)
    monkeypatch.setattr(continuation,'fit_attitude',forbidden)
    with pytest.raises(AtCableStage):continuation.run(job)
    assert saved==[('drone_nominal',nominal),('drone_residual',selected),('cable_physics',selected)]
    assert (job/'drone_residual/result.json').read_bytes()==(source/'drone_residual/result.json').read_bytes()
