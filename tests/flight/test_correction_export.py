"""Actual CUDA correction/export with synthetic fit metadata in a temp folder."""
from pathlib import Path
import numpy as np
import pytest
import torch
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import model_identity
from planning.pva_job import freeze_model_assets
from simulator.workflow import read_json


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_generic_export_recovers_from_a_synthetic_command_perturbation(tmp_path):
    from planning.correction_job import run
    root=Path(__file__).resolve().parents[2]
    rehearsal=root/'runs/rehearsals_pva/20260913-012740-484590-M0-slower-brake-1s'
    reference=root/'runs/reference_tracking/M0-paper-fixed-reference'
    if not rehearsal.exists() or not reference.exists():pytest.skip('Local frozen M0 fixture is unavailable')
    csv=root/'exports/M0_Bspline_slower_brake_1s/fullstate_30hz.csv';before=sha256_file(csv)
    model_before=sha256_file(rehearsal/'model.json')
    # This is the unchanged M0 model, wrapped only for export API testing.
    # No fitting is performed and these temporary fixtures are not registered.
    source=tmp_path/'synthetic-fit';candidate=source/'candidate';candidate.mkdir(parents=True)
    model=freeze_model_assets(read_json(rehearsal/'model.json'),candidate,source_root=rehearsal,portable=True)
    atomic_json(candidate/'model.json',model)
    _,hashes=model_identity(candidate/'model.json')
    atomic_json(source/'fit/selection_frozen.json',dict(test_fixture=True))
    atomic_json(source/'fit/result.json',dict(status='completed',candidate_hashes=hashes,test_fixture=True))
    meta=read_json(reference/'reference.json');previous=tmp_path/'synthetic-previous';previous.mkdir()
    atomic_json(previous/'settings.json',dict(correction=dict(reference_sha256=meta['reference_sha256'])))
    with np.load(reference/'reference.npz') as z:controls=z['original_position_control_points_m'].copy()
    origin=np.asarray(meta['launch_origin_m'])
    np.savez_compressed(previous/'plan.npz',position_control_points_m=origin+.995*(controls-origin))
    job=tmp_path/'correction';output=tmp_path/'rehearsal';export=tmp_path/'export'
    run(source,reference,job,output,export,previous_rehearsal=previous,iterations=2)
    result=read_json(job/'result.json')
    assert result['corrected_cost_m2']<result['baseline_cost_m2']
    assert result['standard_fast_cable_max_difference_m']<=1e-5
    assert result['independent_cost_difference_m2']<=1e-6
    assert sha256_file(export/'fullstate_30hz.csv')==result['csv_sha256']
    assert sha256_file(csv)==before and sha256_file(rehearsal/'model.json')==model_before
    history=read_json(job/'history.json')
    assert all(attempt['strike'] is not None for r in history for attempt in r['attempts'] if 'cost_m2' in attempt)
