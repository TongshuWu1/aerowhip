from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from learning.context_sampling import build_context_from_specification, sample_context_specification
from learning.normalization import FixedContextNormalizer
from learning.replay import TerminalReplayBuffer
from learning.sac import (
    OneShotActor,
    OneShotCritic,
    TerminalSacAgent,
    radial_squash,
    squash_raw_action,
)
from learning.state_bank import generate_initial_state_bank
from learning.training import collect_training_batch, load_training_checkpoint, save_training_checkpoint
from planning.cem_task import load_variable_duration_task
from simulator.parameters import SimulatorSettings
from simulator.production import active_model_paths, build_production_simulator, load_active_model_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = PROJECT_ROOT / "config" / "tasks" / "canonical_whip_variable_duration_tuned_reward_v1.json"


def test_radial_squash_bound_and_analytic_log_jacobian():
    vectors = (
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([1.0e-9, -2.0e-9, 3.0e-9], dtype=torch.float64),
        torch.tensor([0.1, -0.3, 0.2], dtype=torch.float64),
        torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64),
        torch.tensor([6.0, -1.0, 2.0], dtype=torch.float64),
    )
    for vector in vectors:
        value = vector.clone().requires_grad_(True)
        squashed, analytic = radial_squash(value[None])
        jacobian = torch.autograd.functional.jacobian(
            lambda item: radial_squash(item[None])[0][0], value
        )
        autograd_log_det = torch.linalg.slogdet(jacobian).logabsdet
        assert torch.linalg.vector_norm(squashed[0]) < 1.0 or torch.equal(value, torch.zeros_like(value))
        assert analytic[0].item() == pytest.approx(autograd_log_det.item(), abs=2.0e-10)


def test_duration_transform_actor_and_critic_are_finite():
    torch.manual_seed(7)
    raw = torch.randn(32, 49)
    action, log_det = squash_raw_action(raw)
    assert action.shape == (32, 49)
    assert log_det.shape == (32, 1)
    assert torch.all(torch.linalg.vector_norm(action[:, :48].reshape(-1, 16, 3), dim=-1) < 1.0)
    assert torch.all(torch.abs(action[:, -1]) < 1.0)
    assert torch.isfinite(log_det).all()
    actor = OneShotActor()
    critic = OneShotCritic()
    context = torch.randn(32, 83)
    sample = actor(context)
    assert sample.normalized_action.shape == (32, 49)
    assert sample.log_prob.shape == (32, 1)
    assert sample.deterministic_mean_action.shape == (32, 49)
    value = critic(context, sample.normalized_action)
    value.mean().backward()
    assert value.shape == (32, 1)
    assert torch.isfinite(value).all() and torch.isfinite(sample.log_prob).all()


def test_terminal_sac_update_uses_reward_only_and_is_finite():
    signature = inspect.signature(TerminalSacAgent.update)
    assert "next_state" not in signature.parameters
    assert "gamma" not in signature.parameters
    torch.manual_seed(11)
    agent = TerminalSacAgent.create(device="cpu")
    metrics = agent.update(
        torch.randn(64, 83),
        torch.randn(64, 49).tanh(),
        torch.linspace(-2.0, 2.0, 64)[:, None],
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["alpha"] > 0.0


def test_fixed_normalizer_and_replay_checkpoint(tmp_path: Path):
    samples = torch.randn(500, 83)
    samples[:, 76:83] = 1.0
    normalizer = FixedContextNormalizer.fit(samples)
    normalized = normalizer.normalize(samples)
    assert torch.isfinite(normalized).all()
    assert torch.equal(normalizer.standard_deviation[76:83], torch.ones(7))
    path = tmp_path / "normalizer.json"
    normalizer.save(path)
    restored = FixedContextNormalizer.load(path)
    assert torch.equal(normalizer.mean, restored.mean)

    replay = TerminalReplayBuffer(128)
    rows = 32
    replay.add(
        context=normalized[:rows],
        action=torch.zeros(rows, 49),
        scaled_reward=torch.ones(rows),
        raw_reward=torch.full((rows,), 100.0),
        success=torch.zeros(rows, dtype=torch.bool),
        feasible=torch.ones(rows, dtype=torch.bool),
        tip_distance_m=torch.ones(rows),
        directed_speed_m_s=torch.zeros(rows),
        direction_error_deg=torch.full((rows,), 180.0),
    )
    replay_path = tmp_path / "replay.pt"
    replay.save(replay_path)
    loaded = TerminalReplayBuffer.load(replay_path)
    batch = loaded.sample(16, device="cpu", generator=torch.Generator().manual_seed(1))
    assert batch.context.shape == (16, 83)
    assert torch.equal(batch.scaled_reward, torch.ones(16, 1))


@pytest.fixture(scope="module")
def cuda_learning_case():
    if not torch.cuda.is_available():
        pytest.skip("Milestone 5B production collection requires CUDA.")
    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    task = load_variable_duration_task(TASK_PATH)
    bank = generate_initial_state_bank(
        simulator, task, count=8, seed=510, logical_batch_size=8
    )
    return simulator, task, bank


def test_state_bank_context_collection_and_validation_exclusion(cuda_learning_case):
    simulator, task, bank = cuda_learning_case
    specification = sample_context_specification(
        bank,
        count=8,
        generator=torch.Generator().manual_seed(8),
        split="training",
    )
    context = build_context_from_specification(simulator, bank, specification)
    selected = bank.select(specification.state_indices, device="cuda")
    assert torch.equal(context.initial_state_world.uav.position_m, selected.state.uav.position_m)
    assert torch.equal(
        context.initial_state_world.uav.residual_history.features,
        selected.state.uav.residual_history.features,
    )
    normalizer = FixedContextNormalizer.fit(context.to_tensor().detach().cpu())
    agent = TerminalSacAgent.create(device="cuda")
    replay = TerminalReplayBuffer(64)
    metrics = collect_training_batch(
        agent,
        normalizer,
        replay,
        simulator,
        task,
        bank,
        batch_size=8,
        context_generator=torch.Generator().manual_seed(9),
        canonical_fraction=0.2,
        reward_scale_divisor=100.0,
    )
    assert len(replay) == 8
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    # The collection API accepts only the explicitly supplied training bank;
    # fixed validation specifications have no route into replay.
    assert "validation" not in inspect.signature(collect_training_batch).parameters


def test_checkpoint_save_resume(cuda_learning_case, tmp_path: Path):
    simulator, _, _ = cuda_learning_case
    agent = TerminalSacAgent.create(device=simulator.device)
    generator = torch.Generator().manual_seed(99)
    expected = {name: value.detach().clone() for name, value in agent.actor.state_dict().items()}
    checkpoint = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        checkpoint,
        agent=agent,
        episodes=123,
        gradient_updates=7,
        training_history=[{"episodes": 123}],
        evaluation_history=[],
        context_generator=generator,
        replay_metadata={"size": 123},
    )
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.add_(1.0)
    payload = load_training_checkpoint(checkpoint, agent=agent, context_generator=generator)
    assert payload["episodes"] == 123
    for name, value in agent.actor.state_dict().items():
        assert torch.equal(value, expected[name])
