"""Corotational damping must not change with floating-point epsilon."""
import math
import pytest
import torch
from simulator.cable.dder import _curvature_rate_jacobian_impl, _curvature_binormals_impl


@pytest.mark.parametrize('angle', [0., .0001, .0005, .001, .002, .005, .1, 1.])
def test_small_bend_jacobian_agrees_between_float32_and_float64(angle):
    q = torch.tensor([[[0., 0., 0.], [.1, 0., 0.],
                       [.1 + .1 * math.cos(angle), .1 * math.sin(angle), 0.]]], dtype=torch.float64)
    lengths = torch.tensor([.1, .1], dtype=torch.float64)
    reference, _, _ = _curvature_rate_jacobian_impl(q, lengths, None, None)
    actual, _, _ = _curvature_rate_jacobian_impl(q.float(), lengths.float(), None, None)
    torch.testing.assert_close(actual.double(), reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('angle', [.001, .02, .4, 1.])
def test_stable_jacobian_matches_independent_rigid_spin_removal(angle):
    torch.manual_seed(16)
    q = torch.tensor([[[0., 0., 0.], [.1, 0., 0.],
                       [.1 + .1 * math.cos(angle), .1 * math.sin(angle), 0.]]], dtype=torch.float64)
    v = torch.randn_like(q)
    edges = q[:, 1:] - q[:, :-1]
    lengths = edges.norm(dim=-1, keepdim=True)
    tangent = edges / lengths
    rate = v[:, 1:] - v[:, :-1]
    rate = (rate - tangent * (tangent * rate).sum(-1, keepdim=True)) / lengths
    matrix = 2 * torch.eye(3, dtype=q.dtype) - torch.einsum('bni,bnj->bij', tangent, tangent)
    rhs = torch.linalg.cross(tangent, rate).sum(dim=1)
    omega = (torch.linalg.pinv(matrix, rtol=1e-12) @ rhs[..., None])[..., 0]
    curvature = lambda positions: _curvature_binormals_impl(positions, validate=False)
    _, world_rate = torch.autograd.functional.jvp(curvature, q, v)
    expected = world_rate - torch.linalg.cross(omega[:, None], curvature(q))
    jacobian, _, _ = _curvature_rate_jacobian_impl(q, lengths[0, :, 0], None, None)
    actual = (jacobian @ v.reshape(1, -1, 1)).reshape(1, 1, 3)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)


def test_bent_cable_rigid_rotation_has_no_bending_dissipation():
    q = torch.tensor([[[0., 0., 0.], [.1, 0., 0.], [.18, .06, 0.]]], dtype=torch.float64)
    omega = q.new_tensor([.3, -.7, .8]).expand_as(q)
    velocity = torch.linalg.cross(omega, q) + q.new_tensor([1., -.3, 2.])
    jacobian, _, _ = _curvature_rate_jacobian_impl(q, q.new_tensor([.1, .1]), None, None)
    rate = jacobian @ velocity.reshape(1, -1, 1)
    torch.testing.assert_close(rate, torch.zeros_like(rate), atol=1e-12, rtol=0)
