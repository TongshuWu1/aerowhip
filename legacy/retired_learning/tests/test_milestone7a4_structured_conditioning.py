from __future__ import annotations

import torch

from learning.action_diffusion import (
    ExponentialMovingAverage,
    FiLMResidualMlpBlock,
    StructuredConditionalActionDiffusion,
    cosine_alpha_bar_schedule,
    ddim_timestep_schedule,
    diffusion_epsilon_loss,
    sample_ddim,
)
from learning.policy_action import decode_policy_action
from planning.cem_task import load_variable_duration_task


def _model() -> StructuredConditionalActionDiffusion:
    torch.manual_seed(7404)
    return StructuredConditionalActionDiffusion()


def test_structured_diffusion_dimensions_condition_paths_and_backward() -> None:
    model = _model()
    context = torch.randn(4, 83)
    noisy = torch.randn(4, 49)
    timestep = torch.tensor([0, 17, 53, 95])
    output = model(noisy, context, timestep)
    assert output.shape == (4, 49)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    for name in (
        "uav_encoder.0.weight",
        "cable_node_encoder.0.weight",
        "goal_encoder.0.weight",
        "theta_encoder.0.weight",
    ):
        parameter = dict(model.context_encoder.named_parameters())[name]
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0
    assert len(model.blocks) == 4
    assert all(isinstance(block, FiLMResidualMlpBlock) for block in model.blocks)
    assert all(block.modulation.out_features == 512 for block in model.blocks)


def test_structured_cable_encoder_is_node_order_sensitive() -> None:
    model = _model().eval()
    context = torch.randn(2, 83)
    swapped = context.clone()
    positions = swapped[:, 10:40].reshape(2, 10, 3).clone()
    velocities = swapped[:, 40:70].reshape(2, 10, 3).clone()
    positions[:, [2, 7]] = positions[:, [7, 2]]
    velocities[:, [2, 7]] = velocities[:, [7, 2]]
    swapped[:, 10:40] = positions.reshape(2, 30)
    swapped[:, 40:70] = velocities.reshape(2, 30)
    first = model.context_encoder(context)
    second = model.context_encoder(swapped)
    assert not torch.allclose(first, second)


def test_structured_diffusion_ema_ddim_and_production_action_compatibility() -> None:
    model = _model()
    context = torch.randn(8, 83)
    action = torch.empty(8, 49).uniform_(-1.0, 1.0)
    alpha_bar = cosine_alpha_bar_schedule()
    result = diffusion_epsilon_loss(
        model,
        context,
        action,
        alpha_bar=alpha_bar,
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.isfinite(result.loss)
    result.loss.backward()

    ema = ExponentialMovingAverage(model, decay=0.999)
    restored = _model()
    ema.copy_to(restored)
    for key, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)

    noise = torch.randn(32, 49, generator=torch.Generator().manual_seed(11))
    schedule = ddim_timestep_schedule()
    first = sample_ddim(
        restored, context[:1], noise, alpha_bar=alpha_bar, timestep_schedule=schedule
    )
    second = sample_ddim(
        restored, context[:1], noise, alpha_bar=alpha_bar, timestep_schedule=schedule
    )
    assert torch.equal(first.bounded_action, second.bounded_action)
    task = load_variable_duration_task(
        "config/tasks/canonical_whip_variable_duration_tuned_reward_v1.json"
    )
    decoded = decode_policy_action(first.bounded_action, task, duration_max_s=1.8)
    assert decoded.normalized_action.shape == (32, 49)
    assert bool((decoded.duration_s >= 0.45).all())
    assert bool((decoded.duration_s <= 1.80).all())
