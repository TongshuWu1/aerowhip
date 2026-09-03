from __future__ import annotations

import torch

from learning.simple_ppo import (
    SIMPLE_PPO_ACTION_DIM,
    BoundedGaussianPolicy,
    PPORollout,
    SimplePPOAgent,
    generalized_advantage_estimate,
)
from learning.ppo_validation import (
    TRAINING_SUCCESS_ROLLING_WINDOW_EPISODES,
    rolling_episode_mean,
    rolling_success_rate,
)
from learning.ppo_initial_states import MixedPPOInitialStateSampler
from learning.state_bank import InitialStateBank
from run_simple_ppo import DEFAULT_CONFIG, _build_agent, _load_config


ROOT = DEFAULT_CONFIG.parents[2]


def test_selected_target_aligned_sagittal_policy_is_fresh_and_batch_aligned() -> None:
    config = _load_config(DEFAULT_CONFIG)
    assert config["episode_duration_s"] == 7.0
    assert config["episode_duration_s"] / config["control_dt_s"] == 70
    assert config["episode_duration_s"] / config["physics_dt_s"] == 700
    assert config["requested_episodes"] == 1_000_000
    assert config["initialization"]["uses_previous_policy_checkpoint"] is False
    assert "policy_checkpoint" not in config["initialization"]
    assert config["action"]["mode"] == "target_aligned_sagittal_3d"
    assert config["action"]["dimensions"] == 3
    assert config["action"]["lateral_acceleration_available"] is False
    assert config["reported_success"]["episode_continues_after_success"] is False
    assert config["reported_success"]["successful_transition_is_terminal"] is True
    assert config["training_initial_states"]["mode"] == "mixed_state_bank"
    assert config["training_initial_states"]["canonical_fraction"] == 0.25
    assert config["validation"]["episodes"] == 512
    assert config["validation"]["fixed_numerical_batch_size"] == 2048
    assert config["early_stopping"]["enabled"] is True
    assert config["early_stopping"]["minimum_episodes"] == 250_000
    assert config["early_stopping"]["validation_patience_evaluations"] == 30
    assert config["reward"]["objective_type"] == "single_scalar_reward_maximization"
    assert config["reward"]["progress_weight"] == 20.0
    assert config["reward"]["maximum_displacement_weight"] == 0.0
    assert config["reward"]["terminal_displacement_weight"] == 40.0
    assert config["reward"]["displacement_integral_weight"] == 2.0
    assert config["reward"]["displacement_cost_scale_m"] == 0.35
    assert config["reward"]["time_to_success_weight_per_s"] == 1.0
    assert config["reward"]["directed_speed_near_target_weight"] == 0.0
    assert config["reward"]["direction_near_target_weight"] == 0.0
    assert config["reward"]["strike_quality_improvement_weight"] == 60.0
    assert config["reward"]["directed_speed_reward_cap_m_s"] == 4.0
    assert config["reward"]["success_bonus"] == 100.0
    assert config["reward"]["directed_speed_shaping_reference"] == "attachment_relative"
    assert config["reward"]["success_compactness_bonus"] == 0.0
    assert config["reward"]["success_compactness_scale_m"] == 0.25
    assert config["reward"]["uav_speed_integral_weight"] == 0.0
    assert config["reward"]["acceleration_effort_weight"] == 0.0
    assert config["reward"]["body_rate_effort_weight"] == 0.0
    assert config["reward"]["action_smoothness_weight"] == 0.0
    assert config["reward"]["non_tip_first_penalty"] == 25.0
    assert (
        config["validation"]["every_episodes"] % config["collection_batch"] == 0
    )
    assert config["validation"]["maximum_distance_quantile"] == 1.0

    agent = _build_agent(config, torch.device("cpu"))
    assert agent.policy.log_std.shape == (3,)
    assert agent.policy.mean_network[-1].out_features == 3


def test_mixed_initial_state_sampler_is_disjoint_balanced_and_resumable() -> None:
    training_path = ROOT / "results" / "common" / "training_state_bank.npz"
    training_manifest = (
        ROOT / "results" / "common" / "training_state_bank_manifest.json"
    )
    validation_path = ROOT / "results" / "common" / "validation_state_bank.npz"
    validation_manifest = (
        ROOT / "results" / "common" / "validation_state_bank_manifest.json"
    )
    training = InitialStateBank.load(training_path, training_manifest)
    canonical = training.select(torch.tensor([0]), device="cpu").state
    sampler = MixedPPOInitialStateSampler.create(
        training_bank_path=training_path,
        training_manifest_path=training_manifest,
        validation_bank_path=validation_path,
        validation_manifest_path=validation_manifest,
        canonical_state=canonical,
        command_yaw_world_rad=0.0,
        batch_size=16,
        canonical_fraction=0.25,
        seed=91,
    )
    assert sampler.manifest["training_bank_count"] == 4_096
    assert sampler.manifest["validation_bank_count"] == 512
    assert sampler.manifest["exact_state_overlap_count"] == 0
    assert sampler.canonical_count == 4

    generator_state = sampler.get_state()
    first = sampler.sample(device="cpu")
    sampler.set_state(generator_state)
    replay = sampler.sample(device="cpu")
    assert torch.equal(first.source_indices, replay.source_indices)
    assert int((first.source_indices == len(training)).sum()) == 4
    varied = first.source_indices[first.source_indices != len(training)]
    assert varied.unique().numel() == 12
    assert torch.equal(
        first.state.cable.positions_m,
        replay.state.cable.positions_m,
    )


def test_displacement_return_finetune_warm_starts_with_compact_selection() -> None:
    config = _load_config(
        ROOT / "config" / "learning" / "whip_ppo_displacement_return_finetune_v1.json"
    )
    assert config["initialization"]["uses_previous_policy_checkpoint"] is True
    assert config["initialization"]["source_validation_success_rate"] == 0.94921875
    assert config["reward"]["terminal_displacement_weight"] == 80.0
    assert config["reward"]["displacement_integral_weight"] == 4.0
    assert config["ppo"]["learning_rate"] == 0.0001
    assert config["early_stopping"]["minimum_validation_success_rate"] == 0.90
    assert (
        config["early_stopping"]["selection_metric"]
        == "lower_mean_uav_displacement_subject_to_success_floor"
    )


def test_forward_return_100_pilot_changes_only_the_intended_reward_pressure() -> None:
    config = _load_config(
        ROOT / "config" / "learning" / "whip_ppo_forward_return_100_pilot_v1.json"
    )
    assert config["requested_episodes"] == 100_000
    assert config["initialization"]["uses_previous_policy_checkpoint"] is True
    assert config["initialization"]["load_value_network"] is True
    assert config["reward"]["success_forward_return_bonus_weight"] == 100.0
    assert config["reward"]["success_release_bonus_weight"] == 25.0
    assert config["reward"]["terminal_displacement_weight"] == 50.0
    assert config["reward"]["terminal_displacement_success_only"] is True
    assert config["reward"]["displacement_integral_weight"] == 0.0
    assert config["reward"]["time_to_success_weight_per_s"] == 1.0
    assert config["early_stopping"]["minimum_validation_success_rate"] == 0.90
    assert (
        config["early_stopping"]["selection_metric"]
        == "higher_return_quality_subject_to_success_floor"
    )


def test_attachment_compensated_progress_pilot_keeps_world_success_contract() -> None:
    config = _load_config(
        ROOT
        / "config"
        / "learning"
        / "whip_ppo_attachment_compensated_progress_v1.json"
    )
    assert config["requested_episodes"] == 100_000
    assert config["reward"]["progress_shaping_reference"] == (
        "attachment_compensated_tip"
    )
    assert config["reward"]["success_forward_return_bonus_weight"] == 50.0
    assert config["reward"]["success_release_bonus_weight"] == 25.0
    assert config["reported_success"]["tip_target_distance_m"] == 0.05
    assert config["reported_success"]["minimum_directed_tip_speed_m_s"] == 4.0
    assert config["reported_success"]["maximum_direction_error_deg"] == 30.0


def test_blended_progress_run_is_a_genuinely_fresh_policy() -> None:
    config = _load_config(
        ROOT / "config" / "learning" / "whip_ppo_blended_progress_fresh_v1.json"
    )
    assert config["requested_episodes"] == 1_000_000
    assert config["initialization"]["uses_previous_policy_checkpoint"] is False
    assert "policy_checkpoint" not in config["initialization"]
    assert config["reward"]["progress_shaping_reference"] == (
        "blended_world_attachment"
    )
    assert config["reward"]["progress_attachment_compensation_fraction"] == 0.5
    assert config["reward"]["success_forward_return_bonus_weight"] == 50.0
    assert config["ppo"]["learning_rate"] == 3.0e-4
    assert config["ppo"]["update_epochs"] == 4


def test_attachment_progress_fresh_run_uses_one_progress_definition() -> None:
    config = _load_config(
        ROOT / "config" / "learning" / "whip_ppo_attachment_progress_fresh_v1.json"
    )
    assert config["requested_episodes"] == 1_000_000
    assert config["initialization"]["uses_previous_policy_checkpoint"] is False
    assert "policy_checkpoint" not in config["initialization"]
    assert config["reward"]["progress_shaping_reference"] == (
        "attachment_compensated_tip"
    )
    assert config["reward"]["progress_attachment_compensation_fraction"] == 1.0
    assert config["reward"]["success_forward_return_bonus_weight"] == 50.0


def test_forward_reverse_d60_pilot_changes_only_terminal_displacement_pressure() -> None:
    config = _load_config(
        ROOT
        / "config"
        / "learning"
        / "whip_ppo_forward_reverse_release_d60_pilot_v1.json"
    )
    assert config["requested_episodes"] == 100_000
    assert config["initialization"]["uses_previous_policy_checkpoint"] is True
    assert "whip_ppo_forward_reverse_release_v1" in config["initialization"][
        "policy_checkpoint"
    ]
    assert config["reward"]["progress_shaping_reference"] == "world_tip"
    assert config["reward"]["terminal_displacement_weight"] == 60.0
    assert config["reward"]["terminal_displacement_success_only"] is True
    assert config["reward"]["displacement_integral_weight"] == 0.0
    assert config["reward"]["success_forward_return_bonus_weight"] == 50.0
    assert config["reward"]["success_release_bonus_weight"] == 25.0
    assert config["early_stopping"]["minimum_validation_success_rate"] == 0.90


def test_dense_return_release_pilot_preserves_d50_and_times_release_at_strike() -> None:
    config = _load_config(
        ROOT
        / "config"
        / "learning"
        / "whip_ppo_dense_return_release_pilot_v1.json"
    )
    reward = config["reward"]
    assert config["requested_episodes"] == 100_000
    assert "whip_ppo_forward_reverse_release_v1" in config["initialization"][
        "policy_checkpoint"
    ]
    assert reward["progress_shaping_reference"] == "world_tip"
    assert reward["terminal_displacement_weight"] == 50.0
    assert reward["displacement_integral_weight"] == 0.0
    assert reward["return_release_improvement_weight"] == 50.0
    assert reward["success_release_at_strike"] is True
    assert reward["success_forward_return_bonus_weight"] == 50.0
    assert reward["success_release_bonus_weight"] == 25.0


def test_dense_return_release_100_continuation_changes_only_dense_return_pressure() -> None:
    config = _load_config(
        ROOT
        / "config"
        / "learning"
        / "whip_ppo_dense_return_release_100_continuation_v1.json"
    )
    reward = config["reward"]
    assert config["requested_episodes"] == 500_000
    assert "whip_ppo_dense_return_release_pilot_v1" in config["initialization"][
        "policy_checkpoint"
    ]
    assert reward["return_release_improvement_weight"] == 100.0
    assert reward["terminal_displacement_weight"] == 50.0
    assert reward["displacement_integral_weight"] == 0.0
    assert reward["success_release_at_strike"] is True
    assert config["early_stopping"]["minimum_validation_success_rate"] == 0.90


def test_gradual_displacement_return_changes_only_terminal_cost() -> None:
    config = _load_config(
        ROOT / "config" / "learning" / "whip_ppo_displacement_return_gradual_v1.json"
    )
    assert config["initialization"]["uses_previous_policy_checkpoint"] is True
    assert config["initialization"]["source_validation_success_rate"] == 0.94921875
    assert config["requested_episodes"] == 500_000
    assert config["reward"]["terminal_displacement_weight"] == 50.0
    assert config["reward"]["displacement_integral_weight"] == 2.0
    assert config["ppo"]["learning_rate"] == 0.0001
    assert config["early_stopping"]["minimum_validation_success_rate"] == 0.90
    assert config["early_stopping"]["minimum_episodes"] == 500_000
    assert (
        config["early_stopping"]["selection_metric"]
        == "lower_mean_uav_displacement_subject_to_success_floor"
    )


def test_success_conditioned_return_finetune_is_guarded() -> None:
    config = _load_config(
        ROOT
        / "config"
        / "learning"
        / "whip_ppo_success_conditioned_return_finetune_v1.json"
    )
    assert config["initialization"]["load_value_network"] is True
    assert config["requested_episodes"] == 50_000
    assert config["reward"]["terminal_displacement_weight"] == 45.0
    assert config["reward"]["terminal_displacement_success_only"] is True
    assert config["reward"]["displacement_integral_weight"] == 0.0
    assert config["reward"]["body_rate_effort_weight"] == 0.0
    assert config["ppo"]["learning_rate"] == 0.00002
    assert config["early_stopping"]["minimum_validation_success_rate"] == 0.90
    assert config["early_stopping"]["abort_below_validation_success_rate"] == 0.80
    assert config["early_stopping"]["abort_below_patience_evaluations"] == 2
    assert config["early_stopping"]["minimum_episodes"] == 50_000


def test_success_conditioned_return_integral_stage_is_incremental() -> None:
    config = _load_config(
        ROOT
        / "config"
        / "learning"
        / "whip_ppo_success_conditioned_return_integral_0p5_v1.json"
    )
    assert config["initialization"]["load_value_network"] is True
    assert config["requested_episodes"] == 500_000
    assert "whip_ppo_success_conditioned_return_finetune_v1" in config[
        "initialization"
    ]["policy_checkpoint"]
    assert config["reward"]["terminal_displacement_weight"] == 45.0
    assert config["reward"]["terminal_displacement_success_only"] is True
    assert config["reward"]["displacement_integral_weight"] == 0.5
    assert config["early_stopping"]["abort_below_validation_success_rate"] == 0.80
    assert config["early_stopping"]["minimum_episodes"] == 500_000


def test_terminal_50_stage_changes_only_success_compactness_strength() -> None:
    config = _load_config(
        ROOT
        / "config"
        / "learning"
        / "whip_ppo_success_conditioned_return_terminal_50_v1.json"
    )
    assert config["requested_episodes"] == 500_000
    assert config["initialization"]["load_value_network"] is True
    assert config["initialization"]["source_checkpoint_episodes"] == 471_040
    assert config["reward"]["terminal_displacement_weight"] == 50.0
    assert config["reward"]["terminal_displacement_success_only"] is True
    assert config["reward"]["displacement_integral_weight"] == 0.5
    assert config["reward"]["body_rate_effort_weight"] == 0.0
    assert config["ppo"]["learning_rate"] == 0.00002
    assert config["early_stopping"]["abort_below_validation_success_rate"] == 0.80


def test_forward_reverse_release_stage_uses_success_conditioned_mechanism() -> None:
    config = _load_config(
        ROOT / "config" / "learning" / "whip_ppo_forward_reverse_release_v1.json"
    )
    reward = config["reward"]
    assert config["requested_episodes"] == 500_000
    assert config["initialization"]["source_checkpoint_episodes"] == 419_840
    assert reward["terminal_displacement_weight"] == 50.0
    assert reward["terminal_displacement_success_only"] is True
    assert reward["displacement_integral_weight"] == 0.0
    assert reward["success_forward_return_bonus_weight"] == 50.0
    assert reward["success_release_bonus_weight"] == 25.0
    assert reward["body_rate_effort_weight"] == 0.0
    assert (
        config["early_stopping"]["selection_metric"]
        == "higher_return_quality_subject_to_success_floor"
    )


def test_bounded_policy_has_finite_actions_and_consistent_log_probabilities() -> None:
    torch.manual_seed(3)
    policy = BoundedGaussianPolicy(84, SIMPLE_PPO_ACTION_DIM, hidden_dim=32)
    observation = torch.randn(16, 84)
    action, sampled_log_probability, entropy = policy.sample(observation)
    evaluated_log_probability, evaluated_entropy = policy.evaluate(observation, action)
    assert action.shape == (16, 6)
    assert float(action.detach().abs().max()) < 1.0
    assert torch.allclose(sampled_log_probability, evaluated_log_probability, atol=2e-5)
    assert torch.allclose(entropy, evaluated_entropy)
    assert bool(torch.isfinite(action).all() and torch.isfinite(entropy).all())


def test_gae_excludes_post_failure_transitions() -> None:
    rewards = torch.tensor([[[1.0]], [[2.0]], [[50.0]], [[50.0]]])
    dones = torch.tensor([[[0.0]], [[1.0]], [[0.0]], [[1.0]]])
    masks = torch.tensor([[[1.0]], [[1.0]], [[0.0]], [[0.0]]])
    values = torch.zeros_like(rewards)
    advantages, returns = generalized_advantage_estimate(
        rewards, dones, masks, values, gamma=1.0, gae_lambda=1.0
    )
    assert advantages[:, 0, 0].tolist() == [3.0, 2.0, 0.0, 0.0]
    assert torch.equal(advantages, returns)


def test_ppo_update_is_finite_and_changes_policy_parameters() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")
    agent = SimplePPOAgent(84, device=device, hidden_dim=32)
    rollout = PPORollout.allocate(8, 16, 84, 6, device=device)
    rollout.observations.normal_()
    with torch.no_grad():
        for step in range(8):
            action, log_probability, value = agent.act(rollout.observations[step])
            rollout.actions[step].copy_(action)
            rollout.log_probabilities[step].copy_(log_probability)
            rollout.values[step].copy_(value)
    rollout.rewards.normal_()
    rollout.dones.zero_()
    rollout.dones[-1].fill_(1.0)
    rollout.masks.fill_(1.0)
    before = [parameter.detach().clone() for parameter in agent.policy.parameters()]
    metrics = agent.update(
        rollout,
        minibatch_size=32,
        epochs=2,
        generator=torch.Generator(device="cpu").manual_seed(11),
    )
    after = list(agent.policy.parameters())
    assert metrics.valid_transitions == 128
    assert metrics.epochs_completed >= 1
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in (
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.approximate_kl,
            metrics.clip_fraction,
            metrics.gradient_norm,
        )
    )
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_training_success_rolling_rate_uses_episode_counts() -> None:
    assert TRAINING_SUCCESS_ROLLING_WINDOW_EPISODES == 5_000
    episodes = torch.tensor([2_048, 4_096, 6_144, 8_192]).numpy()
    successes = torch.tensor([0, 1, 1, 3]).numpy()
    rolling = rolling_success_rate(episodes, successes, window_episodes=4_096)
    assert rolling.tolist() == [0.0, 1.0 / 4_096.0, 1.0 / 4_096.0, 2.0 / 4_096.0]


def test_training_reward_rolling_mean_uses_episode_counts() -> None:
    episodes = torch.tensor([2_048, 4_096, 6_144]).numpy()
    batch_means = torch.tensor([10.0, 20.0, 30.0]).numpy()
    rolling = rolling_episode_mean(
        episodes,
        batch_means,
        window_episodes=4_096,
    )
    assert rolling.tolist() == [10.0, 15.0, 25.0]
