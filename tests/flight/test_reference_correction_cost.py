import torch
from planning.reference_correction import tracking_cost


def test_fixed_times_penalize_same_path_in_wrong_order():
    tip=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]],dtype=torch.float64)
    quad=torch.zeros_like(tip);commands=torch.zeros(1,3,11,dtype=torch.float64)
    ref=dict(tip=tip[0],quadrotor=quad[0],command=commands[0,:,:3])
    weights=dict(tip=1.,quadrotor=.1,command=.01)
    exact,_=tracking_cost(tip,quad,commands,ref,weights)
    wrong,_=tracking_cost(tip.flip(1),quad,commands,ref,weights)
    assert exact.item()==0 and wrong.item()>0


def test_terms_are_squared_3d_distance_with_declared_priority():
    zero=torch.zeros(1,4,3,dtype=torch.float64)
    commands=torch.zeros(1,4,11,dtype=torch.float64)
    ref=dict(tip=zero[0],quadrotor=zero[0],command=zero[0])
    tip=zero+torch.tensor([3.,4.,0.]);quad=zero+torch.tensor([0.,0.,2.])
    commands[:,:,:3]=1.
    cost,terms=tracking_cost(tip,quad,commands,ref,dict(tip=1.,quadrotor=.1,command=.01))
    assert terms['tip'].item()==25 and terms['quadrotor'].item()==4
    assert terms['command'].item()==3
    torch.testing.assert_close(cost,torch.tensor([25.43],dtype=torch.float64))
