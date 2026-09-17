import torch
import pytest
from learning.pva_success import criterion,contact_success,TIP_CONTACT,LEGACY
from learning.pva_env import defaults
from planning.pva_job import validate_settings


def test_missing_version_preserves_historical_success_and_new_mppi_uses_tip_contact():
    assert criterion({})==LEGACY
    assert criterion(defaults('mppi')['task'])==TIP_CONTACT


def test_geometric_hit_ignores_legacy_motion_gates_but_not_invalidity_or_misses():
    running=torch.tensor([True,True,False,True])
    entry=torch.tensor([.2,0.,.1,float('inf')])
    legacy=torch.zeros(4,dtype=torch.bool)
    assert contact_success({'success_criterion':TIP_CONTACT},running,entry,legacy).tolist()==[True,True,False,False]
    assert not contact_success({},running,entry,legacy).any()


def test_legacy_result_unchanged_and_unknown_version_rejected():
    legacy=torch.tensor([True,False])
    torch.testing.assert_close(contact_success({},torch.ones(2,dtype=torch.bool),torch.tensor([.2,.3]),legacy),legacy)
    cfg=defaults('mppi');cfg['task']['success_criterion']='ambiguous'
    with pytest.raises(ValueError,match='Unknown PVA success'):validate_settings(cfg)
