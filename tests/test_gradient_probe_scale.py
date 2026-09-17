"""A smooth, sensitive objective needs smaller probes, not a looser tolerance."""
import torch
from experimental_data.whip_full_cable import gradient_check


def test_sensitive_smooth_gradient_converges_at_two_smaller_steps(tmp_path):
    net = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        net.weight.zero_()
    result = gradient_check(
        net, lambda: torch.atan(3e6 * net.weight).sum() / 3e6,
        tmp_path / 'gradient.json',
    )
    assert result['passed']
    assert result['relative_tolerance'] == .02
    assert result['absolute_tolerance'] == 1e-6
    direction = result['directions'][0]
    assert direction['epsilon'] < 5e-8
    assert all(c['agrees'] for c in direction['checks'][-2:])
    assert not next(c for c in direction['checks'] if c['epsilon'] == 1e-7)['agrees']
    assert torch.equal(net.weight, torch.zeros_like(net.weight))
