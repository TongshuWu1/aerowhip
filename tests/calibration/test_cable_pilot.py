import numpy as np
import pytest
from tools.diagnose_preliminary_cable import centered_velocity
from tools.diagnose_preliminary_cable import main


def test_offline_reference_recovers_velocity_in_global_coordinates():
    t = 73.2 + np.arange(-5,6)*.01
    x = t-73.2
    q = np.stack([3+2*x+4*x*x,-2-3*x,1+x*x],-1)
    np.testing.assert_allclose(centered_velocity(t,q,73.2),[2,-3,0],atol=1e-10)


def test_offline_reference_rejects_a_missing_frame_or_marker():
    t = np.arange(11)*.01
    q = np.zeros((11,4,3))
    with pytest.raises(ValueError,match='contiguous'):
        centered_velocity(np.delete(t,4),np.delete(q,4,axis=0),.05)
    q[3,2,1] = np.nan
    with pytest.raises(ValueError,match='missing'):
        centered_velocity(t,q,.05)


@pytest.mark.parametrize('stage',['checks','search'])
def test_duplicate_run_refusal_preserves_original_status(tmp_path,monkeypatch,stage):
    status=tmp_path/'status.json'
    status.write_bytes(b'{"status":"completed"}')
    (tmp_path/'forward_checks.json').write_text('{}')
    (tmp_path/'evaluations.json').write_text('[]')
    monkeypatch.setattr('sys.argv',['diagnostic','--output',str(tmp_path),'--stage',stage])
    with pytest.raises(FileExistsError):
        main()
    assert status.read_bytes()==b'{"status":"completed"}'
