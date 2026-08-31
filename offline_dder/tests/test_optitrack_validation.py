from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from cable_twin.shared.dder import DderState, START_PINNED_FREE_END
from optitrack_offline.config import CableSpecification, DEFAULT_CONFIG
from optitrack_offline.data import MotiveCableTake
from optitrack_offline.fitting import (
    MODEL_SCHEMA,
    WindowBatch,
    _make_model,
    _rollout_metrics,
    marker_positions_to_feasible_rod,
)
from optitrack_offline.validation import (
    RESULT_SCHEMA,
    _held_attachment_inputs,
    _scheduled_correction_indices,
    evaluate_continuous_validation_take,
    load_fitted_optitrack_model,
)


class _DifferentiableFreeTipProbe:
    """Minimal stepper that exposes whether the loss includes the free tip."""

    def __init__(self) -> None:
        self.pinned_masks: list[tuple[bool, bool]] = []

    def step(
        self,
        state: DderState,
        _attachment: torch.Tensor,
        _dt: torch.Tensor,
        *,
        bending_stiffness_n_m2: torch.Tensor,
        bending_damping_n_m2_s: torch.Tensor,
        pinned_endpoints: tuple[bool, bool],
        **_kwargs: object,
    ) -> DderState:
        self.pinned_masks.append(pinned_endpoints)
        spatial_mask = torch.zeros_like(state.positions_m)
        spatial_mask[:, -1, 0] = 1.0
        response = (
            1000.0 * bending_stiffness_n_m2
            + 5000.0 * bending_damping_n_m2_s
        )
        positions = state.positions_m + spatial_mask * response
        return DderState(positions, torch.zeros_like(positions))


class _RecordingFreeTipStep:
    """Deterministic validation step used to audit observation plumbing."""

    boundaries: list[np.ndarray] = []

    def __init__(self, _model: object, _reference: torch.Tensor) -> None:
        type(self).boundaries = []

    def __call__(
        self,
        state: DderState,
        boundary: torch.Tensor,
        dt: torch.Tensor,
    ) -> DderState:
        type(self).boundaries.append(boundary.detach().cpu().numpy().copy())
        attachment_delta = boundary - state.positions_m[:, :1]
        positions = state.positions_m + attachment_delta
        free_tip_increment = torch.zeros_like(positions)
        free_tip_increment[:, -1, 2] = 0.001
        positions = positions + free_tip_increment
        velocities = (positions - state.positions_m) / dt[:, None, None]
        return DderState(positions, velocities)


class OptitrackValidationTests(unittest.TestCase):
    def test_model_and_rollout_contract_are_one_attached_and_gj_free(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            fit=replace(DEFAULT_CONFIG.fit, device="cpu", substeps=1),
        )
        model = _make_model(config, ei=2.0e-5, cb=2.0e-6)

        self.assertEqual(START_PINNED_FREE_END, (True, False))
        self.assertEqual(model.parameters.torsional_stiffness_n_m2, 0.0)
        self.assertFalse(hasattr(config.fit, "gj_min_n_m2"))
        self.assertFalse(hasattr(config.fit, "gj_max_n_m2"))

    def test_free_tip_contributes_to_rollout_loss_and_ei_cb_gradients(self) -> None:
        node_count = 21
        marker_indices = torch.arange(0, node_count, 2, dtype=torch.long)
        initial = torch.zeros((1, node_count, 3), dtype=torch.float64)
        observations = torch.zeros((1, 2, 11, 3), dtype=torch.float64)
        # Every moving marker except c10 is an exact prediction. Therefore a
        # non-zero objective and gradient can only come from the free tip.
        observations[:, 1, -1, 0] = 0.05
        batch = WindowBatch(
            observations_m=observations,
            initial_positions_m=initial,
            initial_velocities_m_s=torch.zeros_like(initial),
            timestamps_s=torch.tensor(((0.0, 0.01),), dtype=torch.float64),
            initialization_rmse_m=0.0,
            marker_node_indices=marker_indices,
        )
        log_ei = torch.tensor(
            np.log(2.0e-5), dtype=torch.float64, requires_grad=True
        )
        log_cb = torch.tensor(
            np.log(3.0e-6), dtype=torch.float64, requires_grad=True
        )
        probe = _DifferentiableFreeTipProbe()
        loss, squared = _rollout_metrics(
            probe,  # type: ignore[arg-type]
            batch,
            ei=torch.exp(log_ei),
            cb=torch.exp(log_cb),
            robust_scale_m=0.01,
            create_graph=True,
        )
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(float(squared.detach()), 0.0)
        self.assertNotEqual(float(log_ei.grad), 0.0)
        self.assertNotEqual(float(log_cb.grad), 0.0)
        self.assertEqual(probe.pinned_masks, [START_PINNED_FREE_END])

    def test_artifact_loader_accepts_only_new_one_attachment_schema(self) -> None:
        cable = self._cable()
        payload = self._artifact_payload(cable)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current_path = root / "one_attachment.json"
            current_path.write_text(json.dumps(payload), encoding="utf-8")

            runtime_model = load_fitted_optitrack_model(current_path)
            self.assertEqual(payload["schema"], MODEL_SCHEMA)
            self.assertNotIn("torsional_stiffness_n_m2", payload["optimized"])
            self.assertEqual(
                runtime_model.parameters.torsional_stiffness_n_m2,
                0.0,
            )
            self.assertAlmostEqual(
                runtime_model.parameters.cable_mass_kg,
                cable.total_dynamic_mass_kg,
            )

            legacy = copy.deepcopy(payload)
            legacy["schema"] = "optitrack_twist_aware_rod_v5"
            legacy["optimized"]["torsional_stiffness_n_m2"] = 1.0e-5
            legacy_path = root / "legacy_two_holder.json"
            legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "direct OptiTrack cable model"):
                load_fitted_optitrack_model(legacy_path)

    def test_continuous_validation_uses_only_attachment_and_holds_occlusion(self) -> None:
        cable = self._cable()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            take_path = root / "held_out.csv"
            take_path.write_text("synthetic held-out take\n", encoding="utf-8")
            take = self._take(take_path, cable, missing_attachment_frame=5)
            model_path = root / "model.json"
            result_path = root / "result.json"
            model_path.write_text(
                json.dumps(self._artifact_payload(cable)), encoding="utf-8"
            )

            with (
                mock.patch(
                    "optitrack_offline.validation.load_motive_cable_csv",
                    return_value=take,
                ),
                mock.patch(
                    "optitrack_offline.validation._RuntimeRolloutStep",
                    _RecordingFreeTipStep,
                ),
            ):
                result = evaluate_continuous_validation_take(
                    take_path,
                    model_path,
                    history_frames=3,
                    device="cpu",
                    result_path=result_path,
                )

            self.assertEqual(result.start_index, 2)
            self.assertTrue(result.attachment_hold_mask[3, 0])
            self.assertTrue(_RecordingFreeTipStep.boundaries)
            self.assertTrue(
                all(value.shape == (1, 1, 3) for value in _RecordingFreeTipStep.boundaries)
            )
            held_boundary = _RecordingFreeTipStep.boundaries[2][0, 0]
            np.testing.assert_allclose(held_boundary, take.positions_m[4, 0])
            self.assertGreater(
                np.linalg.norm(result.predictions_m[-1, -1] - result.predictions_m[0, -1]),
                0.0,
            )
            self.assertGreater(result.free_tip_rmse_m, 0.0)
            self.assertGreater(result.overall_rmse_m, 0.0)

            saved = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema"], RESULT_SCHEMA)
            self.assertEqual(saved["protocol"]["observation_mode"], "attachment_only")
            self.assertEqual(
                saved["protocol"]["attachment_occlusion_policy"],
                "zero_order_hold_last_trustworthy_position",
            )
            self.assertIn("free_tip_rmse_m", saved["metrics"])

    def test_periodic_full_position_correction_resets_dynamic_free_tip(self) -> None:
        cable = self._cable()
        timestamps = np.arange(10, dtype=np.float64) * 0.01
        np.testing.assert_array_equal(
            _scheduled_correction_indices(timestamps, 2, 0.03),
            np.asarray((3, 6), dtype=np.int64),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            take_path = root / "periodic.csv"
            take_path.write_text("synthetic held-out take\n", encoding="utf-8")
            take = self._take(take_path, cable)
            model_path = root / "model.json"
            result_path = root / "periodic.json"
            model_path.write_text(
                json.dumps(self._artifact_payload(cable)), encoding="utf-8"
            )

            with (
                mock.patch(
                    "optitrack_offline.validation.load_motive_cable_csv",
                    return_value=take,
                ),
                mock.patch(
                    "optitrack_offline.validation._RuntimeRolloutStep",
                    _RecordingFreeTipStep,
                ),
            ):
                result = evaluate_continuous_validation_take(
                    take_path,
                    model_path,
                    history_frames=3,
                    observation_interval_s=0.03,
                    device="cpu",
                    result_path=result_path,
                )

            np.testing.assert_array_equal(
                np.flatnonzero(result.correction_mask),
                np.asarray((0, 3, 6), dtype=np.int64),
            )
            for local_index in (3, 6):
                absolute_index = result.start_index + local_index
                feasible, _ = marker_positions_to_feasible_rod(
                    take.positions_m[absolute_index : absolute_index + 1], cable
                )
                np.testing.assert_allclose(
                    result.predictions_m[local_index],
                    feasible[0],
                    rtol=0.0,
                    atol=1.0e-12,
                )

    def test_attachment_reacquisition_cannot_inject_a_one_frame_jump(self) -> None:
        cable = self._cable()
        with tempfile.TemporaryDirectory() as directory:
            take = self._take(Path(directory) / "reacquisition.csv", cable)
        positions = take.positions_m.copy()
        observed = take.observed.copy()
        attachment_observed = take.attachment_observed.copy()
        attachment_error = take.attachment_error_m.copy()
        observed[3:8, 0] = False
        attachment_observed[3:8] = False
        positions[3:8, 0] = np.nan
        attachment_error[3:8] = np.nan
        # Over the entire five-frame gap this displacement is below 5 m/s,
        # but applying it in the reacquisition frame would be a 20 m/s impulse.
        positions[8, 0, 1] = positions[2, 0, 1] + 0.20
        take = replace(
            take,
            positions_m=positions,
            observed=observed,
            attachment_observed=attachment_observed,
            attachment_error_m=attachment_error,
        )
        held_positions, held = _held_attachment_inputs(
            take,
            2,
            {
                "maximum_rigid_body_error_m": 0.005,
                "maximum_attachment_speed_m_s": 5.0,
            },
        )

        self.assertTrue(held[6, 0])
        np.testing.assert_allclose(held_positions[6], held_positions[5])

    @staticmethod
    def _cable() -> CableSpecification:
        return CableSpecification(
            rest_lengths_m=(0.05,) * 10,
            bare_cable_mass_kg=0.006,
            moving_marker_masses_kg=(0.0005,) * 10,
            diameter_m=0.0035,
            rod_segments_per_marker_interval=2,
        )

    @staticmethod
    def _artifact_payload(cable: CableSpecification) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "measured": {
                "marker_count": cable.marker_count,
                "node_count": cable.node_count,
                "rod_segments_per_marker_interval": (
                    cable.rod_segments_per_marker_interval
                ),
                "marker_node_indices": list(cable.marker_node_indices),
                "marker_interval_lengths_m": list(cable.rest_lengths_m),
                "rest_lengths_m": list(cable.rod_rest_lengths_m),
                "marker_material_coordinates_m": list(
                    cable.marker_material_coordinates_m
                ),
                "rod_material_coordinates_m": list(cable.rod_material_coordinates_m),
                "length_m": cable.length_m,
                "bare_cable_mass_kg": cable.bare_cable_mass_kg,
                "moving_marker_count": cable.moving_marker_count,
                "moving_marker_masses_kg": list(cable.moving_marker_masses_kg),
                "vertex_masses_kg": list(cable.vertex_masses_kg),
                "total_dynamic_mass_kg": cable.total_dynamic_mass_kg,
                "diameter_m": cable.diameter_m,
            },
            "optimized": {
                "bending_stiffness_n_m2": 1.0e-6,
                "bending_damping_n_m2_s": 1.0e-6,
            },
            "solver": {
                "gravity_m_s2": [0.0, 0.0, -9.80665],
                "substeps": 2,
                "constraint_iterations": 4,
            },
            "fit": {"status": "completed"},
            "measurement_quality": {
                "maximum_rigid_body_error_m": 0.005,
                "maximum_attachment_speed_m_s": 5.0,
                "maximum_marker_speed_m_s": 5.0,
                "maximum_chord_excess_m": 0.005,
                "maximum_initialization_rmse_m": 0.005,
            },
            "boundary_condition": {
                "prescribed_vertices": [0],
                "distal_terminal": "dynamic, force-free and moment-free",
            },
            "sources": [],
        }

    @staticmethod
    def _take(
        path: Path,
        cable: CableSpecification,
        *,
        missing_attachment_frame: int | None = None,
    ) -> MotiveCableTake:
        frame_count = 10
        base = np.zeros((cable.marker_count, 3), dtype=np.float64)
        base[:, 0] = np.asarray(cable.marker_material_coordinates_m)
        base[:, 2] = 0.8
        positions = np.repeat(base[None], frame_count, axis=0)
        positions[:, :, 1] += 0.0001 * np.arange(frame_count)[:, None]
        observed = np.ones((frame_count, cable.marker_count), dtype=bool)
        attachment_observed = np.ones(frame_count, dtype=bool)
        attachment_error = np.full(frame_count, 0.0002, dtype=np.float64)
        if missing_attachment_frame is not None:
            positions[missing_attachment_frame, 0] = np.nan
            observed[missing_attachment_frame, 0] = False
            attachment_observed[missing_attachment_frame] = False
            attachment_error[missing_attachment_frame] = np.nan
        return MotiveCableTake(
            source_path=path.resolve(),
            take_name="synthetic",
            asset_name="cable1",
            capture_rate_hz=100.0,
            export_rate_hz=100.0,
            coordinate_space="Global",
            source_axis_convention="Motive Y-up",
            axis_convention="Project Z-up",
            frame_numbers=np.arange(frame_count, dtype=np.int64),
            timestamps_s=np.arange(frame_count, dtype=np.float64) * 0.01,
            marker_names=("Attachment", *(f"c{index}" for index in range(1, 11))),
            positions_m=positions,
            observed=observed,
            attachment_rigid_body_name="drone_attachment",
            attachment_observed=attachment_observed,
            attachment_error_m=attachment_error,
        )


if __name__ == "__main__":
    unittest.main()
