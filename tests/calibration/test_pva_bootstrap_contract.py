import torch
import pytest
from experimental_data.plateau import Plateau
from simulator.drone_pose_residual import DronePoseResidual,save_residual,load_residual
from experimental_data.io import sha256_file


def test_plateau_keeps_small_best_improvements_without_resetting_patience():
    p=Plateau(minimum=30,patience=2,relative=.01)
    assert p.observe(0,1.)==(True,False)
    assert p.observe(10,.999)==(True,False)
    assert p.observe(20,.998)==(True,False)
    assert p.observe(30,.997)==(True,True)
    assert p.best==.997
    with pytest.raises(FloatingPointError):p.observe(40,float('nan'))


def test_new_residual_preserves_hover_but_historical_spec_is_unchanged(tmp_path):
    old=DronePoseResidual().double();new=DronePoseResidual(hover_gate=True).double()
    assert 'hover_gate' not in old.specification()
    for net in (old,new):
        with torch.no_grad():net.net[-1].bias.fill_(1.)
    p=torch.tensor([[1.,2.,1.2]],dtype=torch.float64);v=torch.zeros_like(p);b=torch.ones_like(p)
    command=torch.zeros(1,11,dtype=p.dtype);command[:,:3]=p+.1
    assert torch.count_nonzero(new(p,v,b,command))==0
    assert torch.count_nonzero(old(p,v,b,command))==3
    command[:,6]=3.
    assert 0<new(p,v,b,command).abs().max()<=.5
    path=tmp_path/'residual.pt';save_residual(path,new)
    loaded=load_residual(path,sha256_file(path),'cpu')
    torch.testing.assert_close(loaded(p,v,b,command),new(p,v,b,command))
