import json
import numpy as np
import pytest

from deployment.fullstate import sample_fullstate, write_fullstate
from deployment.fullstate_recovery import complete_reference, recovery_path, evaluate
from deployment.fullstate_playback import load_reference, play_reference


def source():
    t = np.arange(82)*.01
    p = np.column_stack((t**3, t*0, 1.5+.2*t))
    v = np.column_stack((3*t*t, t*0, t*0+.2))
    return dict(time_s=t, positions_m=p[:,None], velocities_m_s=v[:,None])


def test_full_reference_preserves_whip_and_has_continuous_recovery():
    trajectory = source()
    ref, sample, details = complete_reference(trajectory,[0,0,1.5])
    st,sp,sv,sa = sample_fullstate(trajectory['time_s'],trajectory['positions_m'][:,0],trajectory['velocities_m_s'][:,0])
    for actual, expected in zip((ref['time_s'],ref['position_m'],ref['velocity_m_s'],ref['acceleration_m_s2']),(st,sp,sv,sa)):
        np.testing.assert_array_equal(actual[:len(st)],expected)
    for got,expected in zip(sample(np.array([0.])),(sp[-1],sv[-1],sa[-1])):
        np.testing.assert_allclose(got[0],expected,atol=1e-12)
    # Analytic boundary limits match both p/v/a, not just sampled positions.
    brake,back = [np.array(c) for c in details['coefficients']]
    for left,right in zip(evaluate(brake,[details['brake_s']]),evaluate(back,[0.])):
        np.testing.assert_allclose(left,right,atol=1e-10)
    p,v,a = evaluate(back,[details['return_s']])
    np.testing.assert_allclose(p,[[0,0,1.5]],atol=1e-10)
    np.testing.assert_allclose(v,0,atol=1e-10)
    np.testing.assert_allclose(a,0,atol=1e-10)
    assert np.all(np.diff(ref['time_s'])>0)
    assert ref['time_s'][-1] == pytest.approx(7.31)
    assert set(ref['phase']) == {'whip','recovery_brake','recovery_return','hover_hold'}
    np.testing.assert_array_equal(ref['position_m'][-1],[0,0,1.5])
    np.testing.assert_array_equal(ref['velocity_m_s'][-1],np.zeros(3))
    np.testing.assert_array_equal(ref['acceleration_m_s2'][-1],np.zeros(3))


def test_complete_export_and_deadline_player(tmp_path):
    trajectory = source()
    reference,_,details = complete_reference(trajectory,[0,0,1.5])
    details['final_max_cable_speed_m_s'] = 0.
    trajectory.update(reference=reference,recovery=details,preview=source())
    (tmp_path/'plan.npz').write_bytes(b'immutable source')
    meta = write_fullstate(trajectory,tmp_path,dict(cutoff_s=.81))
    rows,loaded = load_reference(tmp_path)
    assert len(rows)>200
    assert meta['total_duration_s']==loaded['total_duration_s']
    clock=[0.]
    sent=[]
    def sleep(dt):
        clock[0]+=dt
    play_reference(rows,lambda row:sent.append((row['sample_index'],clock[0])),lambda:clock[0],sleep)
    assert len(sent)==len(rows)
    assert clock[0]==pytest.approx(meta['total_duration_s'])  # no extra terminal hold
    np.testing.assert_allclose([t for _,t in sent],[r['time_s'] for r in rows],atol=1e-10)
    with pytest.raises(InterruptedError):
        play_reference(rows,lambda _:None,lambda:0.,lambda _:None,cancel_requested=lambda:True)
    clock[0]=0.
    with pytest.raises(RuntimeError,match='deadline'):
        play_reference(rows,lambda _:sleep(.3),lambda:clock[0],sleep)
    path=tmp_path/'fullstate_30hz.csv'
    path.write_text('\n'.join(path.read_text().splitlines()[:21]))
    with pytest.raises(ValueError,match='hash mismatch'):
        load_reference(tmp_path)


@pytest.mark.parametrize('bad',[0,-1,float('nan'),float('inf')])
def test_bad_recovery_rejected(bad):
    with pytest.raises(ValueError):
        recovery_path([0,0,1.5],[0,0,0],[0,0,0],[0,0,1.5],dict(brake_s=bad))
