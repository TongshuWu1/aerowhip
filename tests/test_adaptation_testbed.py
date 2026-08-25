from __future__ import annotations

from dataclasses import fields
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from drone_mpc.adaptation_testbed import (
    NominalSacController,
    OptitrackMeasurement,
    TestbedSettings,
    load_adaptation_testbed,
    save_episode,
)
from drone_mpc.model import load_cable_model
from drone_mpc.reduced import stable_controller_model
from drone_mpc.rl_env import TaskDistribution, VectorWhipEnvironment
from drone_mpc.sac import SacAgent, SacSettings, save_policy
from drone_mpc.simulator import SimulationSettings, WhipSimulator
from optitrack_offline.fitting import MODEL_SCHEMA


class AdaptationTestbedTests(unittest.TestCase):
    def test_optitrack_contract_contains_ordered_markers_but_no_hidden_state(self) -> None:
        names = tuple(field.name for field in fields(OptitrackMeasurement))
        self.assertEqual(
            names,
            (
                "time_s",
                "drone_position_m",
                "drone_velocity_m_s",
                "attachment_position_m",
                "marker_material_coordinates_m",
                "marker_positions_m",
                "marker_velocities_m_s",
                "marker_valid",
                "applied_acceleration_m_s2",
            ),
        )
        self.assertFalse(any("hidden" in name or "node" in name for name in names))
        measurement = OptitrackMeasurement(
            time_s=0.0,
            drone_position_m=np.zeros(3),
            drone_velocity_m_s=np.zeros(3),
            attachment_position_m=np.zeros(3),
            marker_material_coordinates_m=np.array((0.0, 0.1, 0.2)),
            marker_positions_m=np.zeros((3, 3)),
            marker_velocities_m_s=np.zeros((3, 3)),
            marker_valid=np.ones(3, dtype=np.bool_),
            applied_acceleration_m_s2=np.zeros(3),
        )
        self.assertFalse(measurement.marker_positions_m.flags.writeable)
        self.assertFalse(measurement.marker_velocities_m_s.flags.writeable)
        self.assertFalse(measurement.marker_valid.flags.writeable)
        np.testing.assert_array_equal(
            measurement.free_tip_position_m,
            measurement.marker_positions_m[-1],
        )
        self.assertEqual(
            tuple(inspect.signature(NominalSacController.select_action).parameters),
            ("self", "measurement"),
        )

    def test_wrong_nominal_model_is_rejected_by_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            wrong_payload = self._artifact()
            wrong_payload["unrelated_provenance_change"] = True
            wrong = Path(directory) / "wrong_nominal.json"
            wrong.write_text(json.dumps(wrong_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different cable model"):
                load_adaptation_testbed(
                    wrong,
                    paths["policy"],
                    hidden_model_path=paths["model"],
                    settings=self._settings(),
                    device="cpu",
                )

    def test_compatible_unseen_hidden_artifact_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            hidden_payload = self._artifact()
            hidden_payload["optimized"]["bending_stiffness_n_m2"] = 1.2e-7
            hidden = root / "hidden.json"
            hidden.write_text(json.dumps(hidden_payload), encoding="utf-8")
            session = load_adaptation_testbed(
                paths["model"],
                paths["policy"],
                hidden_model_path=hidden,
                settings=self._settings(),
                device="cpu",
            )
            self.assertNotEqual(
                session.hidden_source.sha256,
                session.nominal_source.sha256,
            )
            self.assertNotEqual(
                session.hidden_plant.sha256,
                session.nominal_controller.sha256,
            )
            self.assertEqual(
                session.hidden_plant.node_count,
                session.nominal_controller.node_count,
            )

    def test_controller_action_is_independent_of_hidden_plant_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            hidden_payload = self._artifact()
            hidden_payload["optimized"]["bending_stiffness_n_m2"] = 1.4e-7
            hidden = root / "different_hidden.json"
            hidden.write_text(json.dumps(hidden_payload), encoding="utf-8")
            nominal_session = load_adaptation_testbed(
                paths["model"],
                paths["policy"],
                hidden_model_path=paths["model"],
                settings=self._settings(),
                device="cpu",
            )
            mismatched_session = load_adaptation_testbed(
                paths["model"],
                paths["policy"],
                hidden_model_path=hidden,
                settings=self._settings(),
                device="cpu",
            )
            first = nominal_session._controller()
            second = mismatched_session._controller()
            material_coordinates = nominal_session.marker_material_coordinates_m
            positions = np.column_stack(
                (
                    np.zeros(len(material_coordinates)),
                    np.zeros(len(material_coordinates)),
                    -0.1 - material_coordinates,
                )
            )
            measurement = OptitrackMeasurement(
                time_s=0.0,
                drone_position_m=np.zeros(3),
                drone_velocity_m_s=np.zeros(3),
                attachment_position_m=np.array((0.0, 0.0, -0.1)),
                marker_material_coordinates_m=material_coordinates,
                marker_positions_m=positions,
                marker_velocities_m_s=np.zeros_like(positions),
                marker_valid=np.ones(len(material_coordinates), dtype=np.bool_),
                applied_acceleration_m_s2=np.zeros(3),
            )
            action_before = first.select_action(measurement)
            # The controller has the same nominal artifact and measurement but
            # the environment owns a different hidden cable artifact.
            action_after = second.select_action(measurement)
            np.testing.assert_array_equal(
                action_before.normalized_action,
                action_after.normalized_action,
            )
            np.testing.assert_array_equal(
                action_before.acceleration_m_s2,
                action_after.acceleration_m_s2,
            )

    def test_matched_episode_is_deterministic_and_records_50hz_10hz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            session = load_adaptation_testbed(
                paths["model"],
                paths["policy"],
                hidden_model_path=paths["model"],
                settings=self._settings(),
                device="cpu",
            )
            self.assertEqual(
                session.hidden_plant.sha256,
                session.nominal_controller.sha256,
            )
            first = session.run_episode()
            second = session.run_episode()
            self.assertEqual(first.frame_count, 11)
            self.assertEqual(first.action_count, 2)
            self.assertEqual(first.physics_rate_hz, 50.0)
            self.assertEqual(first.control_rate_hz, 10.0)
            self.assertEqual(
                len(first.marker_material_coordinates_m),
                session.nominal_controller.node_count,
            )
            np.testing.assert_allclose(
                [frame.time_s for frame in first.frames],
                np.linspace(0.0, 0.2, 11),
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_array_equal(
                np.stack([frame.drone_position_m for frame in first.frames]),
                np.stack([frame.drone_position_m for frame in second.frames]),
            )
            np.testing.assert_array_equal(
                np.stack([frame.measured_marker_positions_m for frame in first.frames]),
                np.stack([frame.measured_marker_positions_m for frame in second.frames]),
            )
            # Every velocity exposed to the controller is causal: BDF1 on the
            # first interval and BDF2 thereafter.  The pinned marker and drone
            # attachment share exactly the same measured velocity.
            np.testing.assert_array_equal(
                first.frames[0].drone_velocity_m_s,
                np.zeros(3),
            )
            for index, frame in enumerate(first.frames):
                np.testing.assert_allclose(
                    frame.measured_marker_velocities_m_s[0],
                    frame.drone_velocity_m_s,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                if index == 0:
                    np.testing.assert_array_equal(
                        frame.measured_marker_velocities_m_s,
                        np.zeros_like(frame.measured_marker_velocities_m_s),
                    )
                    continue
                previous = first.frames[index - 1]
                dt = frame.time_s - previous.time_s
                if index == 1:
                    expected_drone_velocity = (
                        frame.drone_position_m - previous.drone_position_m
                    ) / dt
                    expected_marker_velocities = (
                        frame.measured_marker_positions_m
                        - previous.measured_marker_positions_m
                    ) / dt
                else:
                    second_previous = first.frames[index - 2]
                    previous_dt = previous.time_s - second_previous.time_s
                    current_weight = (2.0 * dt + previous_dt) / (
                        dt * (dt + previous_dt)
                    )
                    previous_weight = -(dt + previous_dt) / (dt * previous_dt)
                    second_previous_weight = dt / (
                        previous_dt * (dt + previous_dt)
                    )
                    expected_drone_velocity = (
                        current_weight * frame.drone_position_m
                        + previous_weight * previous.drone_position_m
                        + second_previous_weight * second_previous.drone_position_m
                    )
                    expected_marker_velocities = (
                        current_weight * frame.measured_marker_positions_m
                        + previous_weight * previous.measured_marker_positions_m
                        + second_previous_weight
                        * second_previous.measured_marker_positions_m
                    )
                np.testing.assert_allclose(
                    frame.drone_velocity_m_s,
                    expected_drone_velocity,
                    rtol=0.0,
                    atol=1.0e-10,
                )
                np.testing.assert_allclose(
                    frame.measured_marker_velocities_m_s,
                    expected_marker_velocities,
                    rtol=0.0,
                    atol=1.0e-10,
                )
            # With matched models, the causal predict/position-correct observer
            # preserves the policy's training-state semantics for the entire
            # closed-loop episode, not merely the first open-loop block.
            for frame in first.frames:
                np.testing.assert_allclose(
                    frame.hidden_cable_positions_m,
                    frame.belief_cable_positions_m,
                    atol=2.0e-6,
                    rtol=0.0,
                )
            output = save_episode(first, Path(directory) / "episode.npz")
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(archive["time_s"].shape, (11,))
                self.assertEqual(
                    archive["measured_marker_positions_m"].shape,
                    (11, 15, 3),
                )
                self.assertEqual(
                    archive["measured_marker_velocities_m_s"].shape,
                    (11, 15, 3),
                )
                self.assertEqual(
                    archive["measurement_applied_accelerations_m_s2"].shape,
                    (11, 3),
                )
                np.testing.assert_array_equal(
                    archive["measurement_applied_accelerations_m_s2"][0],
                    np.zeros(3),
                )
                np.testing.assert_allclose(
                    archive["measurement_applied_accelerations_m_s2"][5],
                    archive["commanded_accelerations_m_s2"][0],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                self.assertEqual(archive["action_updated"].sum(), 2)
                metadata = json.loads(str(archive["metadata_json"]))
                self.assertEqual(metadata["sac_settings"]["controller_node_count"], 15)
                self.assertEqual(metadata["task"]["hit_tolerance_m"], 0.05)
                self.assertIn("nominal DDER predicted", metadata["controller_state_observer"])

    def test_target_outside_checkpoint_training_support_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "outside the SAC checkpoint"):
                load_adaptation_testbed(
                    paths["model"],
                    paths["policy"],
                    hidden_model_path=paths["model"],
                    settings=TestbedSettings(
                        target_position_m=(0.40, 0.0, -0.1),
                        impact_direction=(1.0, 0.0, 0.0),
                        episode_horizon_s=0.2,
                    ),
                    device="cpu",
                )

    @classmethod
    def _fixture(cls, root: Path) -> dict[str, Path]:
        model_path = root / "model.json"
        model_path.write_text(json.dumps(cls._artifact()), encoding="utf-8")
        source = load_cable_model(model_path)
        settings = cls._sac_settings()
        task = cls._task()
        controller = stable_controller_model(
            source,
            simulation_dt_s=settings.physics_dt_s,
            node_count=settings.controller_node_count,
            constraint_iterations=4,
        )
        simulator = WhipSimulator(
            controller,
            SimulationSettings(
                horizon_s=settings.control_interval_s,
                simulation_dt_s=settings.physics_dt_s,
                control_interval_s=settings.control_interval_s,
                attachment_drop_m=settings.attachment_drop_m,
                maximum_acceleration_m_s2=settings.maximum_acceleration_m_s2,
                maximum_speed_m_s=settings.maximum_speed_m_s,
            ),
            device="cpu",
        )
        environment = VectorWhipEnvironment(
            simulator,
            1,
            settings.episode_horizon_s,
            task,
            seed=settings.seed,
        )
        torch.manual_seed(3)
        agent = SacAgent(
            environment.observation_size,
            environment.action_size,
            settings,
            device=torch.device("cpu"),
        )
        policy_path = root / "policy.pt"
        save_policy(
            policy_path,
            agent,
            source,
            controller,
            settings,
            task,
            environment.observation_size,
            environment.action_size,
            1,
            checkpoint_role="test",
        )
        return {"model": model_path, "policy": policy_path}

    @staticmethod
    def _settings() -> TestbedSettings:
        return TestbedSettings(
            initial_drone_position_m=(0.0, 0.0, 0.0),
            target_position_m=(0.55, 0.0, -0.1),
            impact_direction=(1.0, 0.0, 0.0),
            episode_horizon_s=0.2,
            seed=11,
        )

    @staticmethod
    def _sac_settings() -> SacSettings:
        return SacSettings(
            total_transitions=2,
            environment_count=1,
            replay_capacity=2,
            warmup_transitions=1,
            batch_size=1,
            hidden_size=16,
            controller_node_count=15,
            episode_horizon_s=0.2,
            physics_dt_s=0.02,
            control_interval_s=0.1,
            attachment_drop_m=0.1,
            maximum_acceleration_m_s2=2.0,
            maximum_speed_m_s=3.0,
            log_interval_transitions=1,
            evaluation_episodes=1,
        )

    @staticmethod
    def _task() -> TaskDistribution:
        return TaskDistribution(
            horizontal_distance_min_m=0.55,
            horizontal_distance_max_m=0.55,
            target_height_offset_min_m=-0.1,
            target_height_offset_max_m=-0.1,
            minimum_impact_speed_min_m_s=0.2,
            minimum_impact_speed_max_m_s=0.2,
            drone_keepout_radius_m=0.05,
        )

    @staticmethod
    def _artifact() -> dict[str, object]:
        marker_intervals = [0.03] * 10
        rest = [0.015] * 20
        coordinates = np.r_[0.0, np.cumsum(rest)].tolist()
        marker_coordinates = np.r_[0.0, np.cumsum(marker_intervals)].tolist()
        masses = [0.001] * 21
        return {
            "schema": MODEL_SCHEMA,
            "measured": {
                "marker_count": 11,
                "node_count": 21,
                "rod_segments_per_marker_interval": 2,
                "marker_node_indices": list(range(0, 21, 2)),
                "marker_interval_lengths_m": marker_intervals,
                "rest_lengths_m": rest,
                "marker_material_coordinates_m": marker_coordinates,
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


if __name__ == "__main__":
    unittest.main()
