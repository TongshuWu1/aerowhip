from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from experimental_data.io import classify_csv, sha256_file
from experimental_data.logger import parse_logger
from experimental_data.motive import parse_motive_take
from experimental_data.processing import process_take

from ._common import PROJECT_ROOT, SETTINGS


RAW = PROJECT_ROOT / "data" / "raw_takes" / "osc_001"
LOGGER = RAW / "experiment_osc_001.csv"
MOTIVE = RAW / "osc_001.csv"
FIG8_001_RAW = PROJECT_ROOT / "data" / "raw_takes" / "fig8_001"
FIG8_001_LOGGER = FIG8_001_RAW / "experiment_fig8_001.csv"
FIG8_001_MOTIVE = FIG8_001_RAW / "fig8_001.csv"


@pytest.mark.skipif(not LOGGER.exists() or not MOTIVE.exists(), reason="supplied real take unavailable")
def test_raw_file_types_are_detected_by_content_and_hashes_are_frozen() -> None:
    assert classify_csv(LOGGER) == "logger"
    assert classify_csv(MOTIVE) == "motive"
    assert sha256_file(LOGGER) == "ea63f4333c011e6628eae243ec7981349dd8696361ccf50ccd7ff13f86a8721b"
    assert sha256_file(MOTIVE) == "be813abc8742ae3ac0ab251e4388123469878ffa858c0810ca14865cff827d43"


@pytest.mark.skipif(not LOGGER.exists() or not MOTIVE.exists(), reason="supplied real take unavailable")
def test_motive_parser_uses_persistent_labels_not_lexical_or_logger_ids() -> None:
    motive = parse_motive_take(
        MOTIVE,
        uav_label="cf_7",
        cable_labels=[f"cable:c{index}" for index in range(1, 11)],
    )
    logger = parse_logger(LOGGER)
    assert motive.frame.shape == (1931,)
    assert motive.uav_position_m.shape == (1931, 3)
    assert motive.cable_marker_positions_m.shape == (1931, 10, 3)
    assert motive.field_mapping["cable_position_columns_xyz"]["cable:c10"] == [724, 725, 726]
    assert motive.field_mapping["cable_position_columns_xyz"]["cable:c2"] == [727, 728, 729]
    assert int(np.count_nonzero(~motive.cable_marker_valid[:, 8])) == 1
    assert int(np.count_nonzero(~motive.cable_marker_valid[:, 9])) == 3
    assert logger.natnet_frame.shape == (3795,)
    # Live cable_XX fields deliberately do not enter LoggerTake.
    assert not hasattr(logger, "cable_marker_positions_m")


@pytest.mark.skipif(
    not FIG8_001_LOGGER.exists() or not FIG8_001_MOTIVE.exists(),
    reason="supplied reset-containing Figure-8 take unavailable",
)
def test_logger_natnet_counter_reset_is_preserved_and_audited(tmp_path: Path) -> None:
    logger = parse_logger(FIG8_001_LOGGER)
    reset = np.flatnonzero(np.diff(logger.natnet_frame) <= 0)
    assert reset.tolist() == [387]
    assert int(logger.natnet_frame[387]) == 26381
    assert int(logger.natnet_frame[388]) == 0
    assert len(np.unique(logger.natnet_frame)) == len(logger.natnet_frame)

    result = process_take(FIG8_001_RAW, processed_root=tmp_path)
    assert result["frames"] == 4285
    sync = json.loads((tmp_path / "fig8_001" / "sync_report.json").read_text())
    assert sync["frames"]["matched_count"] == 4284
    assert sync["frames"]["logger_frame_resets"] == [
        {
            "frame_after": 0,
            "frame_before": 26381,
            "motive_time_after_s": 263.82,
            "motive_time_before_s": 263.81,
            "row_before": 387,
        }
    ]
    assert sync["ros_to_logger_motive"]["rms_residual_s"] < 0.0002


@pytest.mark.skipif(not LOGGER.exists() or not MOTIVE.exists(), reason="supplied real take unavailable")
def test_processing_is_idempotent_deterministic_and_scientifically_gated(tmp_path: Path) -> None:
    before = {path.name: sha256_file(path) for path in (LOGGER, MOTIVE)}
    first = process_take(RAW, processed_root=tmp_path)
    take_path = tmp_path / "osc_001" / "take.npz"
    first_hash = sha256_file(take_path)
    second = process_take(RAW, processed_root=tmp_path)
    forced = process_take(RAW, processed_root=tmp_path, force=True)
    assert first["processed"] is True
    assert second["skipped_unchanged"] is True
    assert forced["processed"] is True
    assert sha256_file(take_path) == first_hash
    assert {path.name: sha256_file(path) for path in (LOGGER, MOTIVE)} == before

    metadata = json.loads((tmp_path / "osc_001" / "metadata.json").read_text())
    sync = json.loads((tmp_path / "osc_001" / "sync_report.json").read_text())
    assert metadata["schema"] == "aerial_cable_take_v1"
    assert metadata["fit_ready"] is True
    assert metadata["quality_status"] in {"READY", "WARNING"}
    assert metadata["command_angular_rate_frame"] == "body"
    assert metadata["command_angular_rate_units"] == "radians_per_second"
    assert metadata["command_semantics"]["yaw_units"] == "radians"
    assert metadata["command_semantics"]["fit_ready"] is True
    assert metadata["command_semantics"]["externally_commanded"] == [
        "position_world_m",
        "velocity_world_mps",
        "acceleration_world_mps2",
        "yaw_rad",
    ]
    assert metadata["command_semantics"]["externally_not_commanded"][:2] == [
        "roll", "pitch"
    ]
    assert metadata["scientific_timeline_source"] == "motive_manual_trim"
    assert metadata["motive_start_time_s"] == pytest.approx(12.74)
    assert metadata["motive_end_time_s"] == pytest.approx(32.04)
    assert metadata["logger_pre_history_available_s"] > 11.0
    assert sync["frames"]["matched_count"] == 1930
    assert sync["frames"]["motive_only_count"] == 1
    assert sync["commands"]["unique_command_count"] == 531
    assert sync["commands"]["command_event_rate_hz"] == pytest.approx(30.00048, rel=1e-5)
    assert sync["commands"]["command_coverage_fraction"] == pytest.approx(0.9249093734)
    assert sync["logger_motive_to_take"]["rms_residual_s"] < 1.0e-9
    assert sync["ros_to_logger_motive"]["rms_residual_s"] < 0.001
    assert sync["coordinate_audit"]["position_rms_m"] < 0.0016
    assert sync["coordinate_audit"]["orientation_absolute_dot_median"] > 0.999999

    with np.load(take_path, allow_pickle=False) as take:
        expected = {
            "time_s", "motive_source_time_s", "motive_frame", "command_position_m", "command_velocity_mps",
            "command_acceleration_mps2", "command_orientation_xyzw", "command_yaw",
            "command_angular_velocity", "command_source_ros_time", "command_age_s",
            "command_valid", "uav_position_m", "uav_orientation_xyzw", "uav_valid",
            "cable_marker_positions_m", "cable_marker_valid", "auto_frame_valid",
        }
        assert expected.issubset(take.files)
        assert not any("latent" in name or "fitted" in name or "smoothed" in name for name in take.files)
        assert take["time_s"][0] == 0.0
        assert take["time_s"][-1] == pytest.approx(19.3)
        assert len(np.unique(take["command_source_ros_time"][take["command_valid"]])) == 531
        assert np.isnan(take["cable_marker_positions_m"][~take["cable_marker_valid"]]).all()


def test_production_gui_has_training_status_page() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    import simulator.gui.main_window as main_module

    application = QApplication.instance() or QApplication([])
    window = main_module.SimulatorMainWindow(SETTINGS)
    assert [window.main_tabs.tabText(index) for index in range(window.main_tabs.count())] == [
        "Run & Replay",
        "PPO Training",
        "Production Model",
        "Experimental Data",
        "Planning Archive",
    ]
    assert window.main_tabs.tabBar().isHidden()
    assert len(window.navigation_buttons) == 5
    assert window.navigation_buttons[0].isChecked()
    assert window.shell_page_title.text() == "Run & Replay"
    window.main_tabs.setCurrentIndex(1)
    assert window.navigation_buttons[1].isChecked()
    assert window.shell_page_title.text() == "PPO Training"
    assert window.data_page.table.rowCount() >= 1
    protected = [
        window.data_page.table.item(row, 3).text()
        for row in range(window.data_page.table.rowCount())
    ]
    assert "PROTECTED — NOT EVALUATED" in protected
    assert window.model_page.ready_label.text() == "MODEL READY"
    assert "Verified / Frozen" in window.planning_page.model_label.text()
    assert window.planning_page.result_status.text() in {"PASS", "FAIL"}
    assert window.training_page.start_button.text() == "START NEW PPO RUN"
    assert window.training_page.save_config_button.text() == "SAVE CONFIG"
    assert window.training_page.load_config_button.text() == "LOAD CONFIG"
    assert window.training_page.timer.interval() == 500
    assert window.training_page.configuration_group.isCheckable()
    assert window.training_page.initialization_mode_input.currentData() == "fresh"
    assert (
        window.training_page.action_mode_input.currentData()
        == "target_aligned_sagittal_3d"
    )
    assert (
        window.training_page.speed_reference_input.currentData()
        == "attachment_relative"
    )
    launch_config = window.training_page._configuration_from_controls()
    assert launch_config["action"]["mode"] == "target_aligned_sagittal_3d"
    assert launch_config["action"]["dimensions"] == 3
    assert launch_config["action"]["lateral_acceleration_available"] is False
    assert (
        launch_config["reward"]["directed_speed_shaping_reference"]
        == "attachment_relative"
    )
    assert window.training_page.initialization_mode_input.findData("fresh") >= 0
    assert window.training_page.episode_budget_input.value() == 1_000_000
    assert window.training_page.episode_horizon_input.value() == 7.0
    assert window.training_page.rolling_window_input.value() == 5_000
    assert window.training_page.terminal_displacement_weight_input.value() == 40.0
    assert window.training_page.displacement_integral_weight_input.value() == 2.0
    assert window.training_page.learning_rate_input.value() == 0.0003
    assert window.training_page.update_epochs_input.value() == 4
    assert "rolling 5,000" in window.training_page.training_card.detail_label.text()
    assert window.training_page.run_validation_button.text() == "RUN CURRENT POLICY"
    assert window.training_page.validation_count_input.value() == 20
    assert (
        window.training_page.run_validation_button.objectName()
        == "runCurrentPolicyValidationButton"
    )
    assert window.training_page.curves.training_axis.get_ylabel() == "Training success (%)"
    assert window.training_page.curves.reward_axis.get_ylabel() == "Mean episodic reward"
    assert window.training_page.curves.validation_axis.get_ylabel() == "Validation success (%)"
    assert window.training_page.curves.tabs.count() == 3
    assert [
        window.training_page.curves.tabs.tabText(index)
        for index in range(window.training_page.curves.tabs.count())
    ] == ["TRAINING SUCCESS", "EPISODE REWARD", "VALIDATION"]
    window.simulator_page.load_latest_plan()
    assert window.simulator_page.current_result is not None
    assert not hasattr(window, "takes_widget")
    assert not hasattr(window, "fit_widget")
    window.close()
    application.processEvents()


def test_exclude_only_segment_preserves_unannotated_data() -> None:
    from fitting.dataset import ProcessedTake
    from fitting.segments import manual_use_mask

    take = ProcessedTake(
        take_id="synthetic",
        path=Path("synthetic.npz"),
        arrays={"time_s": np.asarray([0.0, 0.5, 1.0])},
        metadata={},
        sync_report={},
        role="training",
        enabled=True,
        note="",
        segments=({"start_s": 0.4, "end_s": 0.6, "use": False},),
    )
    np.testing.assert_array_equal(manual_use_mask(take), [True, False, True])
