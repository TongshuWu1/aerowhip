from types import SimpleNamespace
import numpy as np
import torch
from learning.attempt_records import save_attempts
from tools.plot_training_comparison import rolling,read_attempts


def test_exact_episode_records_and_trailing_missing_data(tmp_path):
    score=SimpleNamespace(episode_reward=torch.tensor([2.,6.,4.]),episode_success=torch.tensor([True,False,True]),
        episode_impact_speed=torch.tensor([5.,float('nan'),6.]),episode_hit_time_s=torch.tensor([.8,float('nan'),.9]),
        failed=torch.zeros(3,dtype=torch.bool),deployment={})
    save_attempts(tmp_path,score,3,2.)
    result=read_attempts(tmp_path)
    assert result['episode'].tolist()==[1,2,3]
    np.testing.assert_allclose(rolling(result['reward'],2),[2,4,5])
    np.testing.assert_allclose(rolling(result['impact_speed_m_s'],2),[5,5,6])
    assert not list((tmp_path/'attempts').glob('*.tmp'))
