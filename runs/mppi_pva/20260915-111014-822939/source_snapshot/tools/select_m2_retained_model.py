"""Bounded selection of saved M2 components; no fitting or flight-performance claim."""
from pathlib import Path
from copy import deepcopy
import gc
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data import whip_full_data as data
from experimental_data.whip_adaptation_fit import records, evaluate
from experimental_data.whip_full_fit import immutable_identity, validate_candidate
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.model_evaluation import model_identity
from experimental_data.io import atomic_json, sha256_file
from planning.pva_job import freeze_model_assets
from simulator.workflow import read_json


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source = ROOT / 'runs/adaptation/M2-paper-20260913'
    out = ROOT / 'runs/adaptation/M2-selected-20260913'
    out.mkdir(parents=True, exist_ok=False)
    parent, protocol, engine = data.load(source)
    verify_hashes(read_json(source / 'fit/result.json')['candidate_hashes'])
    names = ['M1_003', 'M1_005']
    assert all(protocol['takes'][n]['role'] == 'validation' for n in names)
    rows = records(source, names, parent, engine, 'cuda')
    nominal_path = source / 'stages/drone_nominal/model.json'
    full_path = source / 'candidate/model.json'
    nominal, full = read_json(nominal_path), read_json(full_path)
    physics = deepcopy(nominal)
    physics['cable'] = deepcopy(full['cable'])
    cable = deepcopy(physics)
    cable['motion_residual'] = deepcopy(full['motion_residual'])
    candidates = {'M1_baseline': (parent, source / 'source_candidate'),
                  'retained_drone_M1_cable': (nominal, nominal_path.parent),
                  'retained_drone_M2_physics': (physics, nominal_path.parent),
                  'retained_drone_M2_cable': (cable, nominal_path.parent),
                  'M2_full_reference': (full, full_path.parent)}
    contract = dict(schema='saved_component_selection_v1', source_job=str(source),
        candidates=list(candidates), selection_takes=names,
        rule='Lowest equal-take mean combined tip RMSE among saved candidates whose equal-take mean quadrotor RMSE is no worse than M1. M1 is the fallback.',
        data_use='Previously inspected M1 validation recordings now explicitly used for development model selection; not independent final evidence.',
        authorization='User requested retaining the best saved model and regenerating commands after the residual regression diagnosis.',
        training_restarted=False, prospective_flight_performed=False,
        shared_causal_initial_state=True, fixed_original_masks=True,
        script_sha256=sha256_file(__file__))
    atomic_json(out / 'selection_contract.json', contract)
    base_identity = immutable_identity(source / 'source_candidate/model.json')
    parent_weights = {k: v.detach().clone() for k, v in engine.drone.residual.state_dict().items()}
    source_hashes = {}
    for path in [source / 'source_candidate/model.json', nominal_path, full_path]:
        source_hashes.update(model_identity(path)[1])
    for name in ('prepared_hashes.json', 'source_hashes.json', 'code_hashes.json', 'protocol.json'):
        source_hashes[str(source / name)] = sha256_file(source / name)
    atomic_json(out / 'source_hashes.json', source_hashes)
    metrics = {}
    for name, (model, root) in candidates.items():
        print('Evaluating saved candidate:', name, flush=True)
        folder = out / 'variants' / name
        folder.mkdir(parents=True, exist_ok=False)
        saved = freeze_model_assets(model, folder, source_root=root, portable=True)
        atomic_json(folder / 'model.json', saved)
        assert immutable_identity(folder / 'model.json') == base_identity
        candidate_engine = validate_candidate(folder / 'model.json', protocol['full_update'])
        if name.startswith('retained_drone'):
            assert all(torch.equal(v, parent_weights[k]) for k, v in candidate_engine.drone.residual.state_dict().items())
        per_take = evaluate(rows, candidate_engine, saved['cable']['external_drag_s_inv'], out / 'evaluation' / name)
        metrics[name] = dict(
            tip_rmse_mean_m=float(np.mean([r['command_driven']['tip']['rmse_m'] for r in per_take.values()])),
            quadrotor_rmse_mean_m=float(np.mean([r['drone']['rmse_m'] for r in per_take.values()])),
            per_take=per_take, model=str(folder / 'model.json'))
        atomic_json(out / 'comparison_progress.json', metrics)
        del candidate_engine
        gc.collect(); torch.cuda.empty_cache()
    ceiling = metrics['M1_baseline']['quadrotor_rmse_mean_m']
    eligible = [n for n, r in metrics.items() if r['quadrotor_rmse_mean_m'] <= ceiling + 1e-12]
    selected = min(eligible, key=lambda n: metrics[n]['tip_rmse_mean_m'])
    selected_path = Path(metrics[selected]['model'])
    folder = out / 'candidate'; folder.mkdir()
    saved = freeze_model_assets(read_json(selected_path), folder, source_root=selected_path.parent, portable=True)
    saved['provenance'] = dict(label='M2-selected', generation_index=2, parent_model_id='M1',
        source_job=str(out), fit_job=str(source), fit_complete=True, selected_model=True,
        flight_ready=False, prospective_flight_evidence=False, candidate_variant=selected,
        selection_contract=str(out / 'selection_contract.json'), selection_takes=names,
        data_use=contract['data_use'], training_restarted=False,
        training_takes=['M1_001', 'M1_002', 'M1_004'],
        retained_geometry_and_numerics=True)
    atomic_json(folder / 'model.json', saved)
    assert model_identity(folder / 'model.json')[0] == model_identity(selected_path)[0]
    assert immutable_identity(folder / 'model.json') == base_identity
    verify_hashes(source_hashes)
    report = dict(**contract, selected=selected, eligible=eligible, metrics=metrics,
                  retained_M1_drone_residual=selected != 'M2_full_reference',
                  geometry_and_numerics_unchanged=True)
    atomic_json(out / 'comparison.json', report)
    atomic_json(out / 'fit/selection_frozen.json', dict(selected=selected,
        selection_contract_sha256=sha256_file(out / 'selection_contract.json'),
        comparison_sha256=sha256_file(out / 'comparison.json'),
        selected_signature=model_identity(folder / 'model.json')[0],
        selection_uses_development_recordings=True, physical_performance_pending=True))
    hashes = model_identity(folder / 'model.json')[1]
    for path in [out / 'fit/selection_frozen.json', out / 'comparison.json', out / 'selection_contract.json']:
        hashes[str(path)] = sha256_file(path)
    atomic_json(out / 'fit/result.json', dict(status='completed', operation='saved_component_selection',
        training_restarted=False, model_selected=True, selected=selected, candidate_hashes=hashes))
    atomic_json(out / 'status.json', dict(status='completed', selected=selected))
    print('SELECTED', selected, flush=True)
    print({n: {k: r[k] for k in ('tip_rmse_mean_m', 'quadrotor_rmse_mean_m')} for n, r in metrics.items()}, flush=True)


if __name__ == '__main__':
    main()
