from copy import deepcopy
import numpy as np
import pytest
from tools.evaluate_vertical_initial_filter import gate, LIMITS, initial_metrics
from simulator.cable import CableConfiguration
from simulator.workflow import read_json
from pathlib import Path


def test_gate_inclusive_limits_all_conditions_and_nonfinite():
    assert gate(dict(LIMITS)) == (True, [])
    for key in LIMITS:
        values=dict(LIMITS); values[key]=np.nextafter(values[key],np.inf)
        assert gate(values)==(False,[key])
        values[key]=np.nan
        assert gate(values)==(False,[key])


def test_filter_ignores_future_measurements_and_prediction_outcomes():
    root=Path(__file__).resolve().parents[2]
    job=root/'runs/adaptation/M4-collection-20260915/whip_inputs'
    if not job.exists():
        pytest.skip('Local immutable flight fixture unavailable')
    p=read_json(job/'protocol.json'); name=next(iter(p['takes']))
    with np.load(job/'inputs'/name/'data.npz') as z:
        data={k:z[k].copy() for k in z.files}
    cable=CableConfiguration.from_mapping(read_json(job/'source_candidate/model.json')['cable'])
    rd=root/'runs/reference_tracking/M0-2cm-brake-1p3s-fixed-reference'
    with np.load(rd/'reference.npz') as z: nominal=z['cable_position_m'][0].copy()
    origin=np.asarray(read_json(rd/'reference.json')['launch_origin_m'])
    before=initial_metrics(data,p,cable,nominal,origin)
    changed=deepcopy(data); future=changed['time']>=0
    for key in ('position','sites'):
        changed[key][future]=1e6
    for key in ('pose_valid','marker_valid'):
        changed[key][future]=False
    after=initial_metrics(changed,p,cable,nominal,origin)
    assert before==after and gate(before)==gate(after)
    assert before['last_observation_time_s']<0
