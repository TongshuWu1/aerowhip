import torch
import pytest
from experimental_data import whip_bounded_training as bounded


@pytest.mark.parametrize('updates,seconds,expected,reason',[(3,100,3,'update_budget'),(100,0,1,'wall_time_budget')])
def test_residual_budget_stops_and_preserves_best(tmp_path,monkeypatch,updates,seconds,expected,reason):
    net=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
    with torch.no_grad():net.weight.fill_(1.)
    # Test the stopping/saving layer independently of the CUDA gradient audit.
    monkeypatch.setattr(bounded,'verify_selected_residual',lambda net,obj,folder,result:result)
    contract=dict(residual_stopping=dict(minimum=40,patience=6,relative=.005,check_every=5),
        stage_budgets=dict(cable_residual=dict(maximum_updates=updates,maximum_seconds=seconds)))
    result=bounded.train_bounded(net,lambda:net.weight.square().sum(),contract,tmp_path/'fit',tmp_path,'cable_residual')
    assert result['updates']==expected and result['stop_reason']==reason
    assert (tmp_path/'fit/state.pt').exists() and list((tmp_path/'fit').glob('best-*.pt'))
    saved=torch.load(tmp_path/'fit/state.pt',weights_only=True)
    assert saved['update']==expected
    torch.testing.assert_close(net.weight,saved['best']['weight'])
