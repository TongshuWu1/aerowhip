from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from drone_mpc.model import load_cable_model
from drone_mpc.oracle import ORACLE_SCHEMA
from drone_mpc.rl_env import TaskDistribution, VectorWhipEnvironment
from drone_mpc.sac import (
    DemonstrationReplay,
    DemonstrationTransitions,
    ReplayBuffer,
    RunningMeanVariance,
    SAC_CHECKPOINT_SCHEMA,
    SacAgent,
    SacSettings,
    SacTrainingUpdate,
    SacValidationPreview,
    SquashedGaussianActor,
    TASK_DISTRIBUTION_LABEL,
    _diagnostic_policy_path,
    _evaluation_rank,
    _latest_policy_path,
    _resolved_evaluation_seeds,
    _mixed_replay_batch,
    _random_actions,
    evaluate_policy,
    load_verified_demonstrations,
    save_policy,
)
from drone_mpc.simulator import SimulationSettings, TensorRollout, WhipSimulator
from optitrack_offline.fitting import MODEL_SCHEMA


class DroneSacTests(unittest.TestCase):
    def test_default_sac_timing_is_time_consistent_at_50_hz(self) -> None:
        settings = SacSettings()
        self.assertAlmostEqual(settings.physics_dt_s, 0.01)
        self.assertAlmostEqual(settings.control_interval_s, 0.02)
        self.assertAlmostEqual(1.0 / settings.control_interval_s, 50.0)
        self.assertAlmostEqual(
            settings.discount ** (1.0 / settings.control_interval_s),
            0.99 ** 10,
            places=3,
        )
        self.assertEqual(settings.warmup_transitions, 100_000)
        self.assertEqual(settings.total_transitions, 2_500_000)

        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=1,
            episode_horizon_s=0.04,
            task=TaskDistribution(),
            seed=1,
        )
        self.assertAlmostEqual(environment.per_step_reward_scale, 0.2)
        self.assertAlmostEqual(environment.action_change_reward_scale, 5.0)

    def test_actor_is_bounded_and_log_probability_is_finite(self) -> None:
        actor = SquashedGaussianActor(17, 3, 32)
        observation = torch.randn((8, 17))
        action, log_probability = actor.sample(observation)
        self.assertEqual(action.shape, (8, 3))
        self.assertEqual(log_probability.shape, (8, 1))
        self.assertTrue(
            bool(torch.all(torch.linalg.vector_norm(action, dim=1) < 1.0))
        )
        self.assertTrue(bool(torch.all(torch.isfinite(log_probability))))
        self.assertLess(
            float(torch.linalg.vector_norm(action, dim=1).mean().detach()), 0.5
        )

    def test_standard_sac_warmup_samples_the_full_action_ball(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(9)
        action = _random_actions(
            1024,
            torch.device("cpu"),
            generator,
            1.0,
            3,
        )
        self.assertEqual(action.shape, (1024, 3))
        self.assertTrue(bool(torch.all(torch.linalg.vector_norm(action, dim=1) <= 1.0)))
        self.assertTrue(bool(torch.all(torch.std(action, dim=0) > 0.1)))

    def test_replay_buffer_wraps_and_samples_tensor_batches(self) -> None:
        buffer = ReplayBuffer(8, 5, 3, device=torch.device("cpu"), seed=4)
        for offset in (0.0, 10.0):
            observation = torch.full((6, 5), offset)
            buffer.add(
                observation,
                torch.zeros((6, 3)),
                torch.ones(6),
                observation + 1.0,
                torch.zeros(6, dtype=torch.bool),
            )
        self.assertEqual(buffer.size, 8)
        sample = buffer.sample(4)
        self.assertEqual(sample[0].shape, (4, 5))
        self.assertEqual(sample[1].shape, (4, 3))

    def test_prior_replay_batch_is_exactly_half_demonstration(self) -> None:
        online = self._constant_replay(1.0, seed=1)
        prior = self._constant_prior(9.0, seed=2)
        batch = _mixed_replay_batch(online, prior, 8, 0.5)
        observations = batch[0][:, 0]
        self.assertEqual(int(torch.count_nonzero(observations == 1.0)), 4)
        self.assertEqual(int(torch.count_nonzero(observations == 9.0)), 4)

    def test_prior_replay_without_demonstrations_is_online_only(self) -> None:
        online = self._constant_replay(3.0, seed=3)
        batch = _mixed_replay_batch(online, None, 8, 0.5)
        torch.testing.assert_close(batch[0], torch.full((8, 5), 3.0))

    def test_demonstration_loader_strictly_rejects_incompatible_artifacts(self) -> None:
        simulator = self._simulator()
        settings = SacSettings(
            total_transitions=8,
            environment_count=2,
            replay_capacity=8,
            warmup_transitions=2,
            batch_size=2,
            hidden_size=8,
            controller_node_count=simulator.snapshot.node_count,
            episode_horizon_s=0.04,
            physics_dt_s=0.01,
            control_interval_s=0.02,
            attachment_drop_m=0.10,
            maximum_acceleration_m_s2=3.0,
            maximum_speed_m_s=2.0,
            log_interval_transitions=2,
            evaluation_episodes=1,
        )
        task = TaskDistribution(
            horizontal_distance_min_m=0.30,
            horizontal_distance_max_m=0.30,
            target_height_offset_min_m=-0.10,
            target_height_offset_max_m=-0.10,
            minimum_impact_speed_min_m_s=0.2,
            minimum_impact_speed_max_m_s=0.2,
            drone_keepout_radius_m=0.05,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                {"schema": "not-an-oracle-schema"},
                {"target_position_m": (0.31, 0.0, -0.10)},
                {"source_model_sha256": "0" * 64},
            )
            for index, override in enumerate(cases):
                with self.subTest(override=override):
                    path = self._write_demonstration_artifact(
                        root / f"invalid_{index}.npz",
                        simulator,
                        settings,
                        task,
                        **override,
                    )
                    with self.assertRaises(ValueError):
                        load_verified_demonstrations(
                            (path,),
                            simulator,
                            settings,
                            task,
                            torch.device("cpu"),
                        )

    def test_demonstration_loader_accepts_multiple_in_range_goal_directions(self) -> None:
        simulator = self._simulator()
        settings = self._small_settings(simulator, environment_count=2)
        task = TaskDistribution(
            horizontal_distance_min_m=0.55,
            horizontal_distance_max_m=1.20,
            target_azimuth_min_deg=-180.0,
            target_azimuth_max_deg=180.0,
            target_height_offset_min_m=-0.20,
            target_height_offset_max_m=0.00,
            minimum_impact_speed_min_m_s=1.5,
            minimum_impact_speed_max_m_s=1.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forward = self._write_demonstration_artifact(
                root / "forward.npz",
                simulator,
                settings,
                task,
                target_position_m=(0.60, 0.0, -0.10),
                impact_direction=(1.0, 0.0, 0.0),
            )
            lateral = self._write_demonstration_artifact(
                root / "lateral.npz",
                simulator,
                settings,
                task,
                target_position_m=(0.0, 0.80, -0.05),
                impact_direction=(0.0, 1.0, 0.0),
            )
            with patch(
                "drone_mpc.sac.VectorWhipEnvironment",
                self._successful_demonstration_environment,
            ):
                demonstrations = load_verified_demonstrations(
                    (forward, lateral),
                    simulator,
                    settings,
                    task,
                    torch.device("cpu"),
                )
            self.assertEqual(demonstrations.count, 2)
            self.assertEqual(len(demonstrations.provenance), 2)
            self.assertAlmostEqual(
                float(demonstrations.provenance[0]["target_azimuth_deg"]),
                0.0,
            )
            self.assertAlmostEqual(
                float(demonstrations.provenance[1]["target_azimuth_deg"]),
                90.0,
            )

    def test_demonstration_loader_rejects_out_of_range_and_wrong_direction(self) -> None:
        simulator = self._simulator()
        settings = self._small_settings(simulator, environment_count=2)
        task = TaskDistribution()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                {
                    "target_position_m": (1.30, 0.0, -0.10),
                    "impact_direction": (1.0, 0.0, 0.0),
                },
                {
                    "target_position_m": (0.0, 0.80, -0.10),
                    "impact_direction": (1.0, 0.0, 0.0),
                },
            )
            for index, override in enumerate(cases):
                with self.subTest(override=override):
                    path = self._write_demonstration_artifact(
                        root / f"invalid_goal_{index}.npz",
                        simulator,
                        settings,
                        task,
                        **override,
                    )
                    with self.assertRaises(ValueError):
                        load_verified_demonstrations(
                            (path,),
                            simulator,
                            settings,
                            task,
                            torch.device("cpu"),
                        )

    def test_one_sac_update_is_finite(self) -> None:
        settings = SacSettings(
            total_transitions=32,
            environment_count=4,
            replay_capacity=64,
            warmup_transitions=4,
            batch_size=8,
            hidden_size=32,
            controller_node_count=3,
            evaluation_episodes=4,
            log_interval_transitions=8,
        )
        agent = SacAgent(11, 3, settings, device=torch.device("cpu"))
        batch = (
            torch.randn((8, 11)),
            torch.tanh(torch.randn((8, 3))),
            torch.randn((8, 1)),
            torch.randn((8, 11)),
            torch.zeros((8, 1)),
        )
        metrics = agent.update(batch)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_observation_statistics_normalize_and_round_trip(self) -> None:
        statistics = RunningMeanVariance((2,), device=torch.device("cpu"))
        statistics.update(torch.tensor(((1.0, 10.0), (3.0, 14.0))))
        agent = SacAgent(
            2,
            3,
            SacSettings(
                total_transitions=8,
                environment_count=2,
                replay_capacity=8,
                warmup_transitions=2,
                batch_size=2,
                hidden_size=8,
                controller_node_count=3,
                log_interval_transitions=2,
                evaluation_episodes=1,
            ),
            device=torch.device("cpu"),
        )
        agent.observation_statistics.load_state_dict(statistics.state_dict())
        normalized = agent._normalize_observation(torch.tensor(((2.0, 12.0),)))
        torch.testing.assert_close(normalized, torch.zeros_like(normalized), atol=1.0e-3, rtol=0.0)

    def test_best_policy_rank_prioritizes_balanced_reach_success(self) -> None:
        miss = {
            "success_rate": 0.0,
            "within_initial_reach_success_rate": 0.0,
            "beyond_initial_reach_success_rate": 0.0,
            "mean_minimum_error_m": 0.01,
            "mean_directional_speed_m_s": 3.0,
            "mean_hit_drone_displacement_m": float("nan"),
        }
        hit = {
            "success_rate": 1.0,
            "within_initial_reach_success_rate": 1.0,
            "beyond_initial_reach_success_rate": 1.0,
            "mean_minimum_error_m": 0.04,
            "mean_directional_speed_m_s": 1.5,
            "mean_hit_drone_displacement_m": 0.2,
        }
        more_accurate_hit = dict(hit, mean_minimum_error_m=0.02)
        self.assertGreater(_evaluation_rank(hit), _evaluation_rank(miss))
        self.assertGreater(_evaluation_rank(more_accurate_hit), _evaluation_rank(hit))

        pooled_but_no_beyond_reach_success = dict(
            hit,
            success_rate=0.9,
            within_initial_reach_success_rate=1.0,
            beyond_initial_reach_success_rate=0.0,
        )
        balanced = dict(
            hit,
            success_rate=0.6,
            within_initial_reach_success_rate=0.6,
            beyond_initial_reach_success_rate=0.6,
        )
        self.assertGreater(
            _evaluation_rank(balanced),
            _evaluation_rank(pooled_but_no_beyond_reach_success),
        )

    def test_evaluation_reports_within_and_beyond_initial_reach_success(self) -> None:
        simulator = self._simulator()
        settings = self._small_settings(simulator, environment_count=4)
        observation_size = 6 * simulator.snapshot.node_count + 22

        class ConstantAgent:
            @staticmethod
            def act(observation: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
                assert deterministic
                return torch.ones((len(observation), 3))

        reset_calls = 0
        step_calls = 0
        reach_queries = 0

        def environment_factory(
            _simulator: WhipSimulator,
            environment_count: int,
            _episode_horizon_s: float,
            _task: TaskDistribution,
            *,
            seed: int,
        ) -> SimpleNamespace:
            del seed
            self.assertEqual(environment_count, 4)
            observation = torch.zeros((4, observation_size))
            success = torch.tensor((True, False, True, True))
            beyond = torch.tensor((False, False, True, True))
            termination_step = torch.tensor((1, 2, 3, 4))

            def step(action: torch.Tensor) -> SimpleNamespace:
                nonlocal step_calls
                step_calls += 1
                previously_finished = termination_step < step_calls
                if bool(torch.any(previously_finished)):
                    torch.testing.assert_close(
                        action[previously_finished],
                        torch.zeros_like(action[previously_finished]),
                    )
                torch.testing.assert_close(
                    action[~previously_finished],
                    torch.ones_like(action[~previously_finished]),
                )
                return SimpleNamespace(
                    transition_observation=observation + step_calls,
                    reward=torch.zeros(4),
                    done=termination_step <= step_calls,
                    success=success,
                    strike_attempt=torch.tensor((True, True, True, True)),
                    unsafe=torch.tensor((False, True, False, False)),
                    minimum_tip_error_m=torch.tensor((0.01, 0.08, 0.02, 0.03)),
                    directional_tip_speed_m_s=torch.tensor((2.0, 0.5, 2.1, 2.2)),
                    relative_cable_kinetic_energy_j=torch.zeros(4),
                    impact_drone_displacement_m=torch.tensor(
                        (0.1, float("nan"), 0.2, 0.3)
                    ),
                )

            def reset_done(_mask: torch.Tensor) -> torch.Tensor:
                nonlocal reset_calls
                reset_calls += 1
                raise AssertionError(
                    "Evaluation must not resample a target after early termination."
                )

            def target_beyond_initial_cable_reach() -> torch.Tensor:
                nonlocal reach_queries
                reach_queries += 1
                return beyond

            return SimpleNamespace(
                observation=lambda: observation,
                step=step,
                reset_done=reset_done,
                target_beyond_initial_cable_reach=target_beyond_initial_cable_reach,
                peak_relative_cable_energy_j=torch.zeros(4),
                maximum_steps=4,
            )

        with patch("drone_mpc.sac.VectorWhipEnvironment", environment_factory):
            metrics = evaluate_policy(
                ConstantAgent(),
                simulator,
                settings,
                TaskDistribution(),
                episodes=4,
                seed=123,
            )
        self.assertEqual(metrics["episodes"], 4.0)
        self.assertEqual(metrics["mean_episode_reward"], 0.0)
        self.assertEqual(metrics["strike_attempt_rate"], 1.0)
        self.assertEqual(metrics["within_initial_reach_episodes"], 2.0)
        self.assertEqual(metrics["beyond_initial_reach_episodes"], 2.0)
        self.assertEqual(
            metrics["within_initial_reach_episodes"]
            + metrics["beyond_initial_reach_episodes"],
            metrics["episodes"],
        )
        self.assertEqual(metrics["within_initial_reach_success_rate"], 0.5)
        self.assertEqual(metrics["beyond_initial_reach_success_rate"], 1.0)
        self.assertEqual(metrics["unsafe_rate"], 0.25)
        self.assertEqual(step_calls, 4)
        self.assertEqual(reach_queries, 4)
        self.assertEqual(reset_calls, 0)

    def test_policy_checkpoint_records_goal_conditioned_task_metadata(self) -> None:
        simulator = self._simulator()
        settings = self._small_settings(simulator, environment_count=2)
        task = TaskDistribution()
        observation_size = 6 * simulator.snapshot.node_count + 22
        agent = SacAgent(observation_size, 3, settings, device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_policy(
                path,
                agent,
                simulator.snapshot,
                simulator.snapshot,
                settings,
                task,
                observation_size,
                3,
                8,
                checkpoint_role="test",
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(
            SAC_CHECKPOINT_SCHEMA,
            "drone_whip_goal_conditioned_her_sac_3d_v10",
        )
        self.assertEqual(payload["schema"], SAC_CHECKPOINT_SCHEMA)
        self.assertEqual(payload["task_distribution"], TASK_DISTRIBUTION_LABEL)
        self.assertEqual(payload["task"]["horizontal_distance_min_m"], 0.55)
        self.assertEqual(payload["task"]["horizontal_distance_max_m"], 1.20)
        self.assertEqual(payload["task"]["target_azimuth_min_deg"], -180.0)
        self.assertEqual(payload["task"]["target_azimuth_max_deg"], 180.0)
        self.assertEqual(payload["task"]["strike_event_reward"], 0.0)
        self.assertEqual(payload["task"]["progress_reward"], 5.0)
        self.assertEqual(payload["task"]["success_reward"], 100.0)

    def test_evaluation_target_seeds_can_be_frozen_across_training_seeds(self) -> None:
        first = self._small_settings(self._simulator(), environment_count=2)
        second = replace(first, seed=first.seed + 1)
        self.assertEqual(
            _resolved_evaluation_seeds(first, 1042, 10042),
            _resolved_evaluation_seeds(second, 1042, 10042),
        )
        self.assertEqual(
            _resolved_evaluation_seeds(second, None, None),
            (second.seed + 1_000, second.seed + 10_000),
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _resolved_evaluation_seeds(second, -1, 10042)

    def test_training_update_is_strict_json_serializable_and_explicit(self) -> None:
        update = SacTrainingUpdate(
            run_status="running",
            transitions=12_800,
            transition_limit=None,
            endless=True,
            elapsed_s=4.0,
            transitions_per_s=3_200.0,
            completed_episodes=320,
            recent_training_success_rate=None,
            recent_training_mean_episode_reward=4.25,
            checkpoint_evaluation_episodes=64,
            validation_success_rate=0.625,
            validation_mean_episode_reward=31.5,
            validation_mean_minimum_error_m=0.031,
            validation_mean_directional_speed_m_s=2.4,
            validation_mean_peak_relative_cable_energy_j=0.12,
            validation_mean_hit_drone_displacement_m=None,
            validation_unsafe_rate=0.125,
            validation_within_initial_reach_success_rate=0.75,
            validation_beyond_initial_reach_success_rate=0.50,
            actor_loss=-1.25,
            critic_loss=0.75,
            alpha=0.21,
            best_validation_success_rate=0.625,
            best_transitions=12_800,
            best_checkpoint_updated=True,
            best_policy_path="C:/experiment/sac_policy.pt",
            latest_policy_path="C:/experiment/sac_policy.latest.pt",
        )

        encoded = json.dumps(asdict(update), allow_nan=False)
        restored = json.loads(encoded)

        self.assertEqual(restored["run_status"], "running")
        self.assertEqual(restored["transitions"], 12_800)
        self.assertIsNone(restored["transition_limit"])
        self.assertTrue(restored["endless"])
        self.assertIsNone(restored["recent_training_success_rate"])
        self.assertEqual(restored["recent_training_mean_episode_reward"], 4.25)
        self.assertEqual(restored["validation_mean_episode_reward"], 31.5)
        self.assertIsNone(restored["validation_mean_hit_drone_displacement_m"])
        self.assertEqual(restored["validation_unsafe_rate"], 0.125)
        self.assertTrue(restored["best_checkpoint_updated"])
        self.assertEqual(
            restored["latest_policy_path"],
            "C:/experiment/sac_policy.latest.pt",
        )

    def test_evaluation_emits_one_strict_validation_preview(self) -> None:
        simulator = self._simulator()
        settings = self._small_settings(simulator, environment_count=2)

        class ZeroAgent:
            @staticmethod
            def act(observation: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
                assert deterministic
                return torch.zeros((len(observation), 3), dtype=observation.dtype)

        previews: list[SacValidationPreview] = []
        metrics = evaluate_policy(
            ZeroAgent(),
            simulator,
            settings,
            TaskDistribution(),
            episodes=2,
            seed=1042,
            preview_progress=previews.append,
            preview_checkpoint_transitions=50_000,
            preview_completed_training_episodes=300,
        )

        self.assertIsNotNone(metrics)
        self.assertEqual(len(previews), 1)
        preview = previews[0]
        self.assertEqual(preview.event_type, "validation_preview")
        self.assertEqual(preview.checkpoint_transitions, 50_000)
        self.assertEqual(preview.completed_training_episodes, 300)
        self.assertEqual(preview.validation_seed, 1042)
        self.assertEqual(len(preview.time_s), settings.episode_horizon_s / settings.control_interval_s + 1)
        self.assertEqual(np.asarray(preview.drone_positions_m).shape, (3, 3))
        self.assertEqual(
            np.asarray(preview.cable_positions_m).shape,
            (3, simulator.snapshot.node_count, 3),
        )
        json.dumps(asdict(preview), allow_nan=False)

    def test_evaluation_honors_stop_before_first_policy_step(self) -> None:
        simulator = self._simulator()
        settings = self._small_settings(simulator, environment_count=2)

        class EvaluationEnvironment:
            maximum_steps = 4

            @staticmethod
            def observation() -> torch.Tensor:
                return torch.zeros((2, 3), dtype=torch.float32)

        cancellation_queries = 0

        def cancelled() -> bool:
            nonlocal cancellation_queries
            cancellation_queries += 1
            return cancellation_queries >= 2

        class UnusedAgent:
            @staticmethod
            def act(_observation: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
                raise AssertionError("Cancellation must precede the first policy step.")

        with patch(
            "drone_mpc.sac.VectorWhipEnvironment",
            return_value=EvaluationEnvironment(),
        ) as environment_factory:
            result = evaluate_policy(
                UnusedAgent(),  # type: ignore[arg-type]
                simulator,
                settings,
                TaskDistribution(),
                episodes=2,
                seed=1042,
                cancelled=cancelled,
            )

        self.assertIsNone(result)
        self.assertEqual(cancellation_queries, 2)
        environment_factory.assert_called_once()

    def test_sac_checkpoint_paths_keep_best_latest_and_final_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            best = (Path(directory) / "goal.policy.pt").resolve()
            latest = _latest_policy_path(best)
            final = _diagnostic_policy_path(best)

        self.assertEqual(latest.name, "goal.policy.latest.pt")
        self.assertEqual(final.name, "goal.policy.final.pt")
        self.assertEqual(len({best, latest, final}), 3)

    def test_vector_environment_is_goal_conditioned_and_resets(self) -> None:
        simulator = self._simulator()
        environment = VectorWhipEnvironment(
            simulator,
            environment_count=3,
            episode_horizon_s=0.04,
            task=TaskDistribution(
                horizontal_distance_min_m=0.30,
                horizontal_distance_max_m=0.40,
                target_height_offset_min_m=-0.45,
                target_height_offset_max_m=-0.35,
                minimum_impact_speed_min_m_s=0.2,
                minimum_impact_speed_max_m_s=0.5,
                drone_keepout_radius_m=0.05,
            ),
            seed=8,
        )
        observation = environment.observation()
        self.assertEqual(observation.shape, (3, environment.observation_size))
        self.assertEqual(
            environment.observation_size,
            6 * simulator.snapshot.node_count + 22,
        )
        self.assertEqual(environment.action_size, 3)
        self.assertFalse(torch.equal(environment.target_position_m[0], environment.target_position_m[1]))
        easy_target = environment.target_position_m.clone()
        environment.reset_done(torch.ones(3, dtype=torch.bool))
        self.assertFalse(torch.equal(easy_target, environment.target_position_m))
        step = environment.step(torch.zeros((3, 3)))
        self.assertEqual(step.reward.shape, (3,))
        self.assertEqual(step.transition_observation.shape, observation.shape)
        self.assertTrue(
            bool(torch.all(torch.isfinite(step.relative_cable_kinetic_energy_j)))
        )
        self.assertTrue(bool(torch.all(torch.isfinite(step.reward))))
        step = environment.step(torch.zeros((3, 3)))
        self.assertTrue(bool(torch.all(step.done)))
        old_target = environment.target_position_m.clone()
        environment.reset_done(step.done)
        self.assertFalse(torch.equal(old_target, environment.target_position_m))

    def test_default_goal_distribution_samples_azimuth_and_beyond_static_reach(self) -> None:
        simulator = self._simulator()
        task = TaskDistribution()
        self.assertEqual(task.horizontal_distance_min_m, 0.55)
        self.assertEqual(task.horizontal_distance_max_m, 1.20)
        environment = VectorWhipEnvironment(
            simulator,
            environment_count=512,
            episode_horizon_s=0.04,
            task=task,
            seed=29,
        )
        relative = environment.target_position_m - environment.start_drone_position_m
        horizontal_radius = torch.linalg.vector_norm(relative[:, :2], dim=1)
        azimuth = torch.atan2(relative[:, 1], relative[:, 0])
        self.assertGreaterEqual(float(torch.min(horizontal_radius)), 0.55)
        self.assertLessEqual(float(torch.max(horizontal_radius)), 1.20)
        self.assertLess(float(torch.min(azimuth)), -2.0)
        self.assertGreater(float(torch.max(azimuth)), 2.0)
        attachment = environment.start_drone_position_m.clone()
        attachment[:, 2] -= simulator.settings.attachment_drop_m
        initial_reach_distance = torch.linalg.vector_norm(
            environment.target_position_m - attachment,
            dim=1,
        )
        self.assertTrue(
            bool(
                torch.any(
                    initial_reach_distance > simulator.snapshot.cable_length_m
                )
            )
        )

    def test_policy_observation_contains_the_full_3d_cable_state(self) -> None:
        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=2,
            episode_horizon_s=0.04,
            seed=2,
        )
        before = environment.observation().clone()
        positions = environment.state.cable.positions_m.clone()
        velocities = environment.state.cable.velocities_m_s.clone()
        positions[:, 1:-1] += 0.2 * torch.randn_like(positions[:, 1:-1])
        velocities[:, 1:-1] += torch.randn_like(velocities[:, 1:-1])
        environment.state = type(environment.state)(
            environment.state.drone_position_m,
            environment.state.drone_velocity_m_s,
            type(environment.state.cable)(positions, velocities),
        )
        self.assertFalse(torch.equal(environment.observation(), before))

    def test_fixed_target_configuration_matches_drone_centred_coordinates(self) -> None:
        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=8,
            episode_horizon_s=0.04,
            task=TaskDistribution(
                horizontal_distance_min_m=0.65,
                horizontal_distance_max_m=0.65,
                target_azimuth_min_deg=0.0,
                target_azimuth_max_deg=0.0,
                target_height_offset_min_m=-0.10,
                target_height_offset_max_m=-0.10,
            ),
            seed=3,
        )
        environment.reset_done(torch.ones(8, dtype=torch.bool))
        height_offset = (
            environment.target_position_m[:, 2]
            - environment.start_drone_position_m[:, 2]
        )
        torch.testing.assert_close(
            height_offset,
            torch.full_like(height_offset, -0.10),
        )
        torch.testing.assert_close(
            environment.target_position_m[:, 1],
            torch.zeros_like(environment.target_position_m[:, 1]),
        )
        torch.testing.assert_close(
            environment.target_position_m[:, 0],
            torch.full_like(environment.target_position_m[:, 0], 0.65),
        )
        torch.testing.assert_close(
            environment.minimum_impact_speed_m_s,
            torch.full_like(environment.minimum_impact_speed_m_s, 1.5),
        )

    def test_drone_control_uses_all_three_axes(self) -> None:
        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=2,
            episode_horizon_s=0.04,
            task=TaskDistribution(
                target_azimuth_min_deg=0.0,
                target_azimuth_max_deg=0.0,
            ),
            seed=5,
        )
        initial = environment.state.drone_position_m.clone()
        environment.step(
            torch.tensor(((0.4, 0.3, 0.2), (0.4, -0.3, -0.2)))
        )
        moved = environment.state.drone_position_m
        self.assertTrue(bool(torch.all(moved[:, 0] > initial[:, 0])))
        self.assertGreater(float(moved[0, 1]), float(initial[0, 1]))
        self.assertLess(float(moved[1, 1]), float(initial[1, 1]))
        self.assertGreater(float(moved[0, 2]), float(initial[0, 2]))
        self.assertLess(float(moved[1, 2]), float(initial[1, 2]))

    def test_sac_has_no_prescribed_motion_phase(self) -> None:
        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=2,
            episode_horizon_s=0.04,
            seed=6,
        )
        self.assertFalse(hasattr(environment, "injection_steps"))
        self.assertEqual(
            environment.observation().shape,
            (2, 6 * environment.simulator.snapshot.node_count + 22),
        )

    def test_motion_without_a_target_plane_crossing_has_no_positive_reward(self) -> None:
        task = TaskDistribution(
            safety_penalty=0.0,
            safety_margin_penalty=0.0,
            time_penalty=0.0,
            action_effort_penalty=0.0,
            action_change_penalty=0.0,
            impact_drone_displacement_penalty=0.0,
        )
        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=1,
            episode_horizon_s=0.04,
            task=task,
            seed=7,
        )
        frame_count = 2
        drone = environment.state.drone_position_m[:, None].repeat(
            1, frame_count, 1
        )
        cable = environment.state.cable.positions_m[:, None].repeat(
            1, frame_count, 1, 1
        )
        cable_velocity = torch.zeros_like(cable)
        cable_velocity[..., 2] = 2.0
        rollout = TensorRollout(
            time_s=torch.tensor((0.01, 0.02), dtype=environment.dtype),
            drone_positions_m=drone,
            drone_velocities_m_s=torch.zeros_like(drone),
            attachment_positions_m=drone,
            cable_positions_m=cable,
            cable_velocities_m_s=cable_velocity,
            accelerations_m_s2=torch.zeros_like(drone),
        )
        scored = environment._score_rollout(rollout, torch.zeros((1, 3)))
        self.assertAlmostEqual(float(scored[0][0]), 0.0, places=7)
        self.assertFalse(bool(scored[1][0]))
        self.assertFalse(bool(scored[2][0]))

    def test_release_reward_scores_the_complete_impact_velocity_vector(self) -> None:
        task = TaskDistribution(
            horizontal_distance_min_m=0.30,
            horizontal_distance_max_m=0.30,
            target_azimuth_min_deg=0.0,
            target_azimuth_max_deg=0.0,
            target_height_offset_min_m=-0.10,
            target_height_offset_max_m=-0.10,
            minimum_impact_speed_min_m_s=0.5,
            minimum_impact_speed_max_m_s=0.5,
            strike_event_reward=1.0,
            progress_reward=0.0,
            success_reward=100.0,
            safety_penalty=0.0,
            safety_margin_penalty=0.0,
            time_penalty=0.0,
            action_effort_penalty=0.0,
            action_change_penalty=0.0,
            impact_drone_displacement_penalty=0.0,
        )
        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=1,
            episode_horizon_s=0.04,
            task=task,
            seed=9,
        )

        def impact_rollout(lateral_speed_m_s: float) -> TensorRollout:
            frame_count = 2
            drone = environment.state.drone_position_m[:, None].repeat(
                1, frame_count, 1
            )
            cable = environment.state.cable.positions_m[:, None].repeat(
                1, frame_count, 1, 1
            )
            cable[:, :, -1] = environment.target_position_m[:, None]
            velocity = torch.zeros_like(cable)
            velocity[:, :, -1, 0] = 1.0
            velocity[:, :, -1, 1] = lateral_speed_m_s
            return TensorRollout(
                time_s=torch.tensor((0.01, 0.02), dtype=environment.dtype),
                drone_positions_m=drone,
                drone_velocities_m_s=torch.zeros_like(drone),
                attachment_positions_m=drone,
                cable_positions_m=cable,
                cable_velocities_m_s=velocity,
                accelerations_m_s2=torch.zeros_like(drone),
            )

        aligned = environment._score_rollout(
            impact_rollout(0.0), torch.zeros((1, 3))
        )
        transverse = environment._score_rollout(
            impact_rollout(1.0), torch.zeros((1, 3))
        )
        self.assertGreater(float(aligned[0][0]), float(transverse[0][0]))
        self.assertGreater(float(aligned[9][0]), float(transverse[9][0]))
        self.assertGreater(float(aligned[0][0]), task.strike_event_reward)
        self.assertLessEqual(float(transverse[0][0]), task.strike_event_reward)

    def test_success_penalizes_drone_displacement_at_the_hit_event(self) -> None:
        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=1,
            episode_horizon_s=0.04,
            task=TaskDistribution(
                horizontal_distance_min_m=0.30,
                horizontal_distance_max_m=0.30,
                target_azimuth_min_deg=0.0,
                target_azimuth_max_deg=0.0,
                target_height_offset_min_m=-0.10,
                target_height_offset_max_m=-0.10,
                minimum_impact_speed_min_m_s=0.2,
                minimum_impact_speed_max_m_s=0.2,
                drone_keepout_radius_m=0.05,
                impact_drone_displacement_penalty=2.0,
            ),
            seed=11,
        )
        environment.reset_done(torch.ones(1, dtype=torch.bool))
        def hit_rollout(drone_offset_y_m: float) -> TensorRollout:
            frame_count = 2
            drone = environment.start_drone_position_m[:, None].repeat(
                1, frame_count, 1
            )
            drone[:, :, 1] += drone_offset_y_m
            drone_velocity = torch.zeros_like(drone)
            cable = environment.state.cable.positions_m[:, None].repeat(
                1, frame_count, 1, 1
            )
            cable[:, :, -1] = environment.target_position_m[:, None]
            cable_velocity = torch.zeros_like(cable)
            cable_velocity[:, :, -1, 0] = 1.0
            return TensorRollout(
                time_s=torch.tensor((0.01, 0.02), dtype=environment.dtype),
                drone_positions_m=drone,
                drone_velocities_m_s=drone_velocity,
                attachment_positions_m=drone,
                cable_positions_m=cable,
                cable_velocities_m_s=cable_velocity,
                accelerations_m_s2=torch.zeros_like(drone),
            )

        centered = environment._score_rollout(
            hit_rollout(0.0), torch.zeros((1, 3))
        )
        displaced = environment._score_rollout(
            hit_rollout(0.20), torch.zeros((1, 3))
        )
        self.assertTrue(bool(centered[2][0]))
        self.assertTrue(bool(displaced[2][0]))
        self.assertEqual(int(environment.step_index[0]), 0)
        self.assertAlmostEqual(float(centered[7][0]), 0.0, places=7)
        self.assertAlmostEqual(float(displaced[7][0]), 0.20, places=6)
        expected_penalty = 2.0 * (
            0.20 / environment.simulator.snapshot.cable_length_m
        ) ** 2
        self.assertAlmostEqual(
            float(centered[0][0] - displaced[0][0]),
            expected_penalty,
            places=5,
        )

    def test_hit_contract_does_not_require_a_straight_cable(self) -> None:
        task = TaskDistribution(
            horizontal_distance_min_m=0.10,
            horizontal_distance_max_m=0.10,
            target_azimuth_min_deg=0.0,
            target_azimuth_max_deg=0.0,
            target_height_offset_min_m=-0.10,
            target_height_offset_max_m=-0.10,
            minimum_impact_speed_min_m_s=0.2,
            minimum_impact_speed_max_m_s=0.2,
            minimum_extension_ratio=0.85,
            drone_keepout_radius_m=0.05,
        )
        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=1,
            episode_horizon_s=0.04,
            task=task,
            seed=13,
        )
        frame_count = 2
        drone = environment.start_drone_position_m[:, None].repeat(
            1, frame_count, 1
        )
        cable = environment.state.cable.positions_m[:, None].repeat(
            1, frame_count, 1, 1
        )
        cable[:, :, -1] = environment.target_position_m[:, None]
        cable_velocity = torch.zeros_like(cable)
        cable_velocity[:, :, -1, 0] = 1.0
        rollout = TensorRollout(
            time_s=torch.tensor((0.01, 0.02), dtype=environment.dtype),
            drone_positions_m=drone,
            drone_velocities_m_s=torch.zeros_like(drone),
            attachment_positions_m=drone,
            cable_positions_m=cable,
            cable_velocities_m_s=cable_velocity,
            accelerations_m_s2=torch.zeros_like(drone),
        )

        scored = environment._score_rollout(rollout, torch.zeros((1, 3)))
        extension_ratio = torch.linalg.vector_norm(
            cable[0, 0, -1] - cable[0, 0, 0]
        ) / environment.simulator.snapshot.cable_length_m
        self.assertLess(float(extension_ratio), task.minimum_extension_ratio)
        self.assertTrue(bool(scored[2][0]))

        legacy_environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=1,
            episode_horizon_s=0.04,
            task=replace(task, minimum_extension_ratio=0.0),
            seed=13,
        )
        legacy_environment.target_position_m.copy_(environment.target_position_m)
        legacy_scored = legacy_environment._score_rollout(
            rollout, torch.zeros((1, 3))
        )
        self.assertTrue(bool(legacy_scored[2][0]))

    def test_impact_velocity_vector_is_explicitly_goal_conditioned(self) -> None:
        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=4,
            episode_horizon_s=0.04,
            task=TaskDistribution(
                desired_impact_azimuth_offset_min_deg=90.0,
                desired_impact_azimuth_offset_max_deg=90.0,
                desired_impact_elevation_deg=30.0,
            ),
            seed=14,
        )
        expected = torch.tensor(
            (0.0, np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))),
            dtype=environment.dtype,
        )[None].repeat(4, 1)
        torch.testing.assert_close(
            environment.desired_impact_direction_frame,
            expected,
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_future_goal_hindsight_relabeling_creates_successful_strikes(self) -> None:
        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=1,
            episode_horizon_s=0.06,
            task=TaskDistribution(
                safety_penalty=0.0,
                safety_margin_penalty=0.0,
                action_effort_penalty=0.0,
                impact_drone_displacement_penalty=0.0,
            ),
            seed=15,
        )
        observations = [environment.observation()[0].clone()]
        actions = []
        for _ in range(3):
            action = torch.tensor(((0.5, 0.0, 0.0),), dtype=environment.dtype)
            step = environment.step(action)
            actions.append(action[0])
            observations.append(step.transition_observation[0].clone())
        observation_tensor = torch.stack(observations)
        layout = environment._observation_slices()
        position = observation_tensor[-1, layout["cable_position"]].reshape(
            environment.simulator.snapshot.node_count, 3
        )
        position[-1, 0] += (
            0.20 / environment.simulator.snapshot.cable_length_m
        )
        velocity = observation_tensor[-1, layout["cable_velocity"]].reshape(
            environment.simulator.snapshot.node_count, 3
        )
        velocity[-1] = torch.tensor((1.0, 0.0, 0.0), dtype=environment.dtype)
        relabeled = environment.hindsight_relabel_episode(
            observation_tensor,
            torch.stack(actions),
            torch.zeros(3, dtype=torch.bool),
            generator=torch.Generator(device="cpu").manual_seed(4),
            goals_per_episode=1,
            minimum_achieved_speed_m_s=0.1,
        )
        self.assertIsNotNone(relabeled)
        assert relabeled is not None
        self.assertTrue(bool(relabeled[4][-1]))
        self.assertGreater(float(torch.amax(relabeled[2])), 90.0)

    def test_physical_hit_terminates_the_strike_attempt(self) -> None:
        environment = VectorWhipEnvironment(
            self._simulator(),
            environment_count=1,
            episode_horizon_s=0.04,
            task=TaskDistribution(
                horizontal_distance_min_m=0.30,
                horizontal_distance_max_m=0.30,
                target_azimuth_min_deg=0.0,
                target_azimuth_max_deg=0.0,
                target_height_offset_min_m=-0.10,
                target_height_offset_max_m=-0.10,
                minimum_impact_speed_min_m_s=0.2,
                minimum_impact_speed_max_m_s=0.2,
                drone_keepout_radius_m=0.05,
            ),
            seed=12,
        )
        frame_count = 2
        drone = environment.start_drone_position_m[:, None].repeat(
            1, frame_count, 1
        )
        cable = environment.state.cable.positions_m[:, None].repeat(
            1, frame_count, 1, 1
        )
        cable[:, 0, -1] = environment.target_position_m
        cable_velocity = torch.zeros_like(cable)
        cable_velocity[:, 0, -1, 0] = 1.0
        rollout = TensorRollout(
            time_s=torch.tensor((0.01, 0.02), dtype=environment.dtype),
            drone_positions_m=drone,
            drone_velocities_m_s=torch.zeros_like(drone),
            attachment_positions_m=drone,
            cable_positions_m=cable,
            cable_velocities_m_s=cable_velocity,
            accelerations_m_s2=torch.zeros_like(drone),
        )
        scored = environment._score_rollout(rollout, torch.zeros((1, 3)))
        self.assertEqual(int(environment.step_index[0]), 0)
        self.assertTrue(bool(scored[2][0]))
        self.assertTrue(bool(scored[1][0]))

    @staticmethod
    def _artifact() -> dict[str, object]:
        rest = [0.03] * 10
        coordinates = np.r_[0.0, np.cumsum(rest)].tolist()
        masses = [0.002] * 11
        return {
            "schema": MODEL_SCHEMA,
            "measured": {
                "marker_count": 11,
                "node_count": 11,
                "rod_segments_per_marker_interval": 1,
                "marker_node_indices": list(range(11)),
                "marker_interval_lengths_m": [0.03] * 10,
                "rest_lengths_m": rest,
                "marker_material_coordinates_m": np.linspace(0.0, 0.3, 11).tolist(),
                "rod_material_coordinates_m": coordinates,
                "length_m": 0.3,
                "bare_cable_mass_kg": 0.012,
                "moving_marker_count": 10,
                "moving_marker_masses_kg": [0.001] * 10,
                "vertex_masses_kg": masses,
                "total_dynamic_mass_kg": sum(masses),
                "diameter_m": 0.0035,
            },
            "optimized": {
                "bending_stiffness_n_m2": 1.0e-7,
                "bending_damping_n_m2_s": 1.0e-7,
            },
            "fit": {"status": "completed"},
            "solver": {
                "gravity_m_s2": [0.0, 0.0, -9.80665],
                "substeps": 2,
                "constraint_iterations": 4,
            },
            "boundary_condition": {
                "prescribed_vertices": [0],
                "attachment": "prescribed position",
                "distal_terminal": "dynamic and free",
            },
        }

    def _simulator(self) -> WhipSimulator:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(self._artifact()), encoding="utf-8")
            snapshot = load_cable_model(path)
        return WhipSimulator(
            snapshot,
            SimulationSettings(
                horizon_s=0.02,
                simulation_dt_s=0.01,
                control_interval_s=0.02,
                attachment_drop_m=0.10,
                maximum_acceleration_m_s2=3.0,
                maximum_speed_m_s=2.0,
            ),
            device="cpu",
        )

    @staticmethod
    def _constant_replay(value: float, *, seed: int) -> ReplayBuffer:
        replay = ReplayBuffer(16, 5, 3, device=torch.device("cpu"), seed=seed)
        observation = torch.full((16, 5), value)
        replay.add(
            observation,
            torch.full((16, 3), value),
            torch.full((16,), value),
            observation + 0.5,
            torch.zeros(16, dtype=torch.bool),
        )
        return replay

    @staticmethod
    def _constant_prior(value: float, *, seed: int) -> DemonstrationReplay:
        observation = torch.full((16, 5), value)
        transitions = DemonstrationTransitions(
            observation=observation,
            action=torch.full((16, 3), value),
            reward=torch.full((16,), value),
            next_observation=observation + 0.5,
            done=torch.zeros(16, dtype=torch.bool),
            provenance=(),
        )
        return DemonstrationReplay(transitions, seed=seed)

    @staticmethod
    def _small_settings(
        simulator: WhipSimulator,
        *,
        environment_count: int,
    ) -> SacSettings:
        return SacSettings(
            total_transitions=8,
            environment_count=environment_count,
            replay_capacity=8,
            warmup_transitions=2,
            batch_size=2,
            hidden_size=8,
            controller_node_count=simulator.snapshot.node_count,
            episode_horizon_s=0.04,
            physics_dt_s=0.01,
            control_interval_s=0.02,
            attachment_drop_m=0.10,
            maximum_acceleration_m_s2=3.0,
            maximum_speed_m_s=2.0,
            log_interval_transitions=2,
            evaluation_episodes=environment_count,
        )

    @staticmethod
    def _successful_demonstration_environment(
        simulator: WhipSimulator,
        environment_count: int,
        episode_horizon_s: float,
        task: TaskDistribution,
        *,
        initial_drone_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        seed: int,
    ) -> SimpleNamespace:
        del episode_horizon_s, initial_drone_position_m, seed
        assert environment_count == 1
        azimuth = torch.deg2rad(
            torch.tensor(
                task.target_azimuth_min_deg,
                dtype=simulator.dtype,
                device=simulator.device,
            )
        )
        zero = torch.zeros((), dtype=simulator.dtype, device=simulator.device)
        one = torch.ones((), dtype=simulator.dtype, device=simulator.device)
        forward = torch.stack((torch.cos(azimuth), torch.sin(azimuth), zero))
        lateral = torch.stack((-torch.sin(azimuth), torch.cos(azimuth), zero))
        up = torch.stack((zero, zero, one))
        target_frame = torch.stack((forward, lateral, up))[None]
        observation = torch.zeros(
            (1, 6 * simulator.snapshot.node_count + 19),
            dtype=simulator.dtype,
            device=simulator.device,
        )

        def step(_action: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(
                reward=torch.ones(1),
                transition_observation=observation + 1.0,
                done=torch.ones(1, dtype=torch.bool),
                success=torch.ones(1, dtype=torch.bool),
                unsafe=torch.zeros(1, dtype=torch.bool),
                minimum_tip_error_m=torch.full((1,), 0.01),
                directional_tip_speed_m_s=torch.full((1,), 2.0),
            )

        return SimpleNamespace(
            target_frame=target_frame,
            observation=lambda: observation.clone(),
            step=step,
        )

    @staticmethod
    def _write_demonstration_artifact(
        path: Path,
        simulator: WhipSimulator,
        settings: SacSettings,
        task: TaskDistribution,
        *,
        schema: str = ORACLE_SCHEMA,
        target_position_m: tuple[float, float, float] = (0.30, 0.0, -0.10),
        impact_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
        source_model_sha256: str | None = None,
    ) -> Path:
        oracle_settings = {
            "horizon_s": settings.episode_horizon_s,
            "physics_dt_s": settings.physics_dt_s,
            "control_interval_s": settings.control_interval_s,
            "attachment_drop_m": settings.attachment_drop_m,
            "maximum_acceleration_m_s2": settings.maximum_acceleration_m_s2,
            "maximum_speed_m_s": settings.maximum_speed_m_s,
            "seed": settings.seed,
        }
        problem = {
            "target_position_m": list(target_position_m),
            "impact_direction": list(impact_direction),
            "minimum_impact_speed_m_s": task.minimum_impact_speed_min_m_s,
            "drone_keepout_radius_m": task.drone_keepout_radius_m,
            "maximum_drone_excursion_m": 10.0,
            "minimum_forward_stroke_m": 0.05,
            "minimum_recoil_stroke_m": 0.05,
            "drone_workspace_center_m": None,
            "maximum_tip_error_m": task.hit_tolerance_m,
            "planning_tip_error_margin_m": 0.0,
            "maximum_impact_angle_deg": task.impact_angle_deg,
            "planning_impact_angle_margin_deg": 0.0,
        }
        np.savez_compressed(
            path,
            schema=np.asarray(schema),
            created_utc=np.asarray("2026-08-22T00:00:00+00:00"),
            feasible=np.asarray(True),
            controls_m_s2=np.zeros((2, 3), dtype=np.float32),
            drone_positions_m=np.zeros((3, 3), dtype=np.float32),
            source_model_sha256=np.asarray(
                source_model_sha256 or simulator.snapshot.sha256
            ),
            source_node_count=np.asarray(simulator.snapshot.node_count),
            target_position_m=np.asarray(target_position_m),
            impact_direction=np.asarray(impact_direction),
            term_names=np.asarray(("feasible", "constraint_violation")),
            term_values=np.asarray((1.0, 0.0)),
            settings_json=np.asarray(json.dumps(oracle_settings, sort_keys=True)),
            problem_json=np.asarray(json.dumps(problem, sort_keys=True)),
        )
        return path


if __name__ == "__main__":
    unittest.main()
