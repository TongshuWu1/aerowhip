"""Deterministic logger + Motive processing into immutable take artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .io import (
    atomic_json,
    canonical_json_hash,
    deterministic_npz,
    resolve_raw_pair,
    sha256_file,
    utc_now,
)
from .logger import parse_logger
from .motive import parse_motive_take
from .quality import evaluate_quality
from .sync import build_clocks, frame_resets, missing_runs, reconstruct_commands


PROCESSING_SCHEMA = "aerial_cable_take_v1"
SCIENTIFIC_TIMELINE_SOURCE = "motive_manual_trim"
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "default_processing.json"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw_takes"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed_takes"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("processed_schema") != PROCESSING_SCHEMA:
        raise ValueError("Processing configuration has the wrong processed schema.")
    labels = payload.get("cable_labels")
    if labels != [f"cable:c{index}" for index in range(1, 11)]:
        raise ValueError("Cable labels must explicitly map cable:c1 through cable:c10.")
    return payload


def _transform_positions(values: np.ndarray, transform: dict[str, object]) -> np.ndarray:
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    translation = np.asarray(transform["translation_m"], dtype=np.float64)
    return np.asarray(values) @ rotation.T + translation


def _coordinate_audit(
    logger_position: np.ndarray,
    motive_position: np.ndarray,
    logger_orientation: np.ndarray,
    motive_orientation: np.ndarray,
    logger_index: np.ndarray,
    motive_index: np.ndarray,
) -> dict[str, object]:
    residual = logger_position[logger_index] - motive_position[motive_index]
    valid = np.isfinite(residual).all(axis=1)
    norm = np.linalg.norm(residual[valid], axis=1)
    logger_q = logger_orientation[logger_index]
    motive_q = motive_orientation[motive_index]
    logger_norm = np.linalg.norm(logger_q, axis=1)
    motive_norm = np.linalg.norm(motive_q, axis=1)
    q_valid = (
        np.isfinite(logger_q).all(axis=1)
        & np.isfinite(motive_q).all(axis=1)
        & (logger_norm > 1.0e-12)
        & (motive_norm > 1.0e-12)
    )
    absolute_dot = np.abs(
        np.sum(
            logger_q[q_valid] / logger_norm[q_valid, None]
            * motive_q[q_valid] / motive_norm[q_valid, None],
            axis=1,
        )
    )
    return {
        "configured_transform": "identity",
        "matched_pose_count": int(np.count_nonzero(valid)),
        "position_rms_m": float(np.sqrt(np.mean(np.square(norm)))),
        "position_max_m": float(np.max(norm)),
        "orientation_absolute_dot_median": float(np.median(absolute_dot)),
        "orientation_absolute_dot_minimum": float(np.min(absolute_dot)),
        "interpretation": "logger and Motive positions already share the same global metric frame",
    }


def _processing_fingerprint(
    source_hashes: dict[str, str], config: dict[str, object], model_config: dict[str, object]
) -> str:
    processor_sources = (
        "io.py", "logger.py", "motive.py", "sync.py", "quality.py", "processing.py"
    )
    processor_hashes = {
        name: sha256_file(PACKAGE_ROOT / name) for name in processor_sources
    }
    processor_hashes['../simulator/geometry.py'] = sha256_file(PROJECT_ROOT/'simulator/geometry.py')
    return canonical_json_hash(
        {
            "source_hashes": source_hashes,
            "processing_config": config,
            "processor_source_hashes": processor_hashes,
            "quality_geometry": {
                "offset_tracking_m": model_config['recorded_data']['optitrack_to_attachment_offset_body_m'],
                "cable_arc_lengths_m": model_config['cable']['marker_interval_lengths_m'],
            },
        }
    )


def _contiguous_ranges(values: list[int]) -> list[dict[str, int]]:
    if not values:
        return []
    ordered = sorted(set(values))
    result: list[dict[str, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            result.append({"first": start, "last": previous, "count": previous-start+1})
            start = value
        previous = value
    result.append({"first": start, "last": previous, "count": previous-start+1})
    return result


def ensure_manifest(take_ids: list[str]) -> None:
    DEFAULT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_MANIFEST_PATH.exists():
        manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {"schema": "aerial_cable_dataset_manifest_v1", "takes": {}}
    takes = manifest.setdefault("takes", {})
    for take_id in sorted(take_ids):
        takes.setdefault(
            take_id,
            {
                "role": "ignore",
                "enabled": False,
                "note": "",
                "segments": [],
                "episode_breaks_s": [],
            },
        )
    atomic_json(DEFAULT_MANIFEST_PATH, manifest)


def process_take(
    take_directory: str | Path,
    *,
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    force: bool = False,
) -> dict[str, object]:
    folder = Path(take_directory)
    take_id = folder.name
    logger_path, motive_path = resolve_raw_pair(folder)
    config = load_config(config_path)
    model_config = json.loads(
        (PROJECT_ROOT / 'config/model.json').read_text(encoding='utf-8'))
    source_hashes = {
        logger_path.name: sha256_file(logger_path),
        motive_path.name: sha256_file(motive_path),
    }
    fingerprint = _processing_fingerprint(source_hashes, config, model_config)
    output = Path(processed_root) / take_id
    metadata_path = output / "metadata.json"
    take_path = output / "take.npz"
    sync_path = output / "sync_report.json"
    if not force and metadata_path.exists() and take_path.exists() and sync_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("processing_fingerprint") == fingerprint:
            existing_sync = json.loads(sync_path.read_text(encoding="utf-8"))
            return {
                "take_id": take_id,
                "processed": False,
                "skipped_unchanged": True,
                "frames": int(existing["frames"]),
                "duration_s": float(existing["duration_s"]),
                "fit_ready": bool(existing["fit_ready"]),
                "quality_status": existing["quality_status"],
                "warnings": existing.get("warnings", []),
                "sync_rms_s": float(existing_sync["ros_to_logger_motive"]["rms_residual_s"]),
                "command_coverage": float(existing_sync["commands"]["command_coverage_fraction"]),
            }

    logger = parse_logger(logger_path)
    motive = parse_motive_take(
        motive_path,
        uav_label=str(config["uav_rigid_body_label"]),
        cable_labels=list(config["cable_labels"]),  # type: ignore[arg-type]
    )
    if motive.metadata.get("Length Units") != "Meters":
        raise ValueError("Only explicitly metric Motive exports are accepted.")
    if motive.metadata.get("Coordinate Space") != "Global":
        raise ValueError("Only Motive Global coordinate-space exports are accepted.")
    ros_clock, motive_clock, logger_index, motive_index = build_clocks(
        logger, motive, config
    )
    commands, command_report = reconstruct_commands(
        logger, motive, ros_clock, motive_clock
    )
    transform = config["source_to_simulator"]
    assert isinstance(transform, dict)
    time_s = motive.source_time_s - motive.source_time_s[0]
    logger_take_time = motive_clock.map(logger.motive_time_s)
    arrays: dict[str, np.ndarray] = {
        "time_s": time_s.astype(np.float64),
        "motive_source_time_s": motive.source_time_s.astype(np.float64),
        "motive_frame": motive.frame.astype(np.int64),
        "uav_position_m": _transform_positions(motive.uav_position_m, transform),
        "uav_orientation_xyzw": motive.uav_orientation_xyzw.astype(np.float64),
        "uav_valid": motive.uav_valid.astype(bool),
        "cable_marker_positions_m": _transform_positions(
            motive.cable_marker_positions_m, transform
        ),
        "cable_marker_valid": motive.cable_marker_valid.astype(bool),
    }
    arrays.update(commands)
    # The current calibrated transform is identity. Explicitly refuse to rotate
    # quaternions/commands under a future nonidentity transform until that
    # quaternion-frame operation receives its own tested calibration.
    if not np.array_equal(np.asarray(transform["rotation"]), np.eye(3)) or not np.array_equal(
        np.asarray(transform["translation_m"]), np.zeros(3)
    ):
        raise ValueError("Nonidentity source transforms require an explicit quaternion transform implementation.")
    from simulator.cable import CableConfiguration

    cable_configuration = CableConfiguration.from_mapping(model_config["cable"])
    flags, quality_report = evaluate_quality(
        arrays,
        quality_config=config["quality"],  # type: ignore[arg-type]
        attachment_offset_body_m=model_config["recorded_data"][
            "optitrack_to_attachment_offset_body_m"
        ],
        interval_lengths_m=cable_configuration.marker_interval_lengths_m,
    )
    arrays.update(flags)
    match_fraction = len(motive_index) / len(motive.frame)
    coordinate_audit = _coordinate_audit(
        logger.uav_position_m,
        motive.uav_position_m,
        logger.uav_orientation_xyzw,
        motive.uav_orientation_xyzw,
        logger_index,
        motive_index,
    )
    command_semantics = config["command_semantics"]
    assert isinstance(command_semantics, dict)
    fit_ready = bool(command_semantics.get("fit_ready", False))
    warnings: list[str] = []
    if not fit_ready:
        warnings.append(
            "Fitting disabled: the logger mapping is audited, but the FullState publisher, Crazyflie controller/estimator configuration, and cmd_omega frame remain unavailable."
        )
    if match_fraction < float(config["quality"]["minimum_frame_match_fraction"]):  # type: ignore[index]
        warnings.append(f"Motive frame match fraction is {match_fraction:.6f}.")
        fit_ready = False
    if ros_clock.rms_residual_s > float(config["quality"]["maximum_clock_rms_s"]):  # type: ignore[index]
        warnings.append("ROS-to-logger-Motive clock residual exceeds configured tolerance.")
        fit_ready = False
    if motive_clock.rms_residual_s > float(config["quality"]["maximum_clock_rms_s"]):  # type: ignore[index]
        warnings.append("Logger-Motive-to-Take clock residual exceeds configured tolerance.")
        fit_ready = False
    missing_counts = np.count_nonzero(~motive.cable_marker_valid, axis=0)
    if int(np.sum(missing_counts)):
        warnings.append("Isolated Motive cable-marker dropout is retained as NaN plus validity masks.")
    quality_status = "READY" if fit_ready and not warnings else ("WARNING" if fit_ready else "NOT_FIT_READY")

    sync_report = {
        "schema": "aerial_cable_sync_report_v1",
        "take_id": take_id,
        "ros_to_logger_motive": ros_clock.as_dict(),
        "logger_motive_to_take": motive_clock.as_dict(),
        "frames": {
            "logger_count": int(len(logger.natnet_frame)),
            "motive_count": int(len(motive.frame)),
            "matched_count": int(len(motive_index)),
            "logger_only_count": int(len(logger.natnet_frame) - len(logger_index)),
            "motive_only_count": int(len(motive.frame) - len(motive_index)),
            "motive_match_fraction": float(match_fraction),
            "logger_missing_frame_runs": missing_runs(logger.natnet_frame),
            "logger_frame_resets": frame_resets(
                logger.natnet_frame, logger.motive_time_s
            ),
            "motive_missing_frame_runs": missing_runs(motive.frame),
            "motive_only_frames": [
                int(frame)
                for frame in motive.frame
                if int(frame) not in set(int(value) for value in logger.natnet_frame)
            ],
            "logger_only_frame_ranges": _contiguous_ranges(
                list(
                    set(int(value) for value in logger.natnet_frame)
                    - set(int(value) for value in motive.frame)
                )
            ),
        },
        "commands": command_report,
        "scientific_timeline": {
            "source": SCIENTIFIC_TIMELINE_SOURCE,
            "motive_start_time_s": float(motive.source_time_s[0]),
            "motive_end_time_s": float(motive.source_time_s[-1]),
            "duration_s": float(time_s[-1]),
            "logger_first_synchronized_time_s": float(logger_take_time[0]),
            "logger_last_synchronized_time_s": float(logger_take_time[-1]),
            "logger_pre_history_available_s": float(
                max(0.0, motive.source_time_s[0] - logger_take_time[0])
            ),
            "logger_post_history_excluded_s": float(
                max(0.0, logger_take_time[-1] - motive.source_time_s[-1])
            ),
            "scientific_samples_are_motive_frames_only": True,
        },
        "coordinate_audit": coordinate_audit,
        "missing_uav_samples": int(np.count_nonzero(~motive.uav_valid)),
        "missing_marker_samples": {
            f"c{index + 1}": int(value) for index, value in enumerate(missing_counts)
        },
        "quality": quality_report,
        "warnings": warnings,
        "processing_errors": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    deterministic_npz(take_path, arrays)
    (output / "processing_failure.json").unlink(missing_ok=True)
    take_hash = sha256_file(take_path)
    metadata = {
        "schema": PROCESSING_SCHEMA,
        "take_id": take_id,
        "source_files": {"logger": logger_path.name, "motive": motive_path.name},
        "source_sha256": source_hashes,
        "motive": {
            "take_name": motive.metadata.get("Take Name"),
            "capture_frame_rate_hz": float(motive.metadata["Capture Frame Rate"]),
            "export_frame_rate_hz": float(motive.metadata["Export Frame Rate"]),
            "coordinate_space": motive.metadata.get("Coordinate Space"),
            "length_units": motive.metadata.get("Length Units"),
            "rotation_type": motive.metadata.get("Rotation Type"),
            "original_metadata": motive.metadata,
        },
        "uav_rigid_body_label": config["uav_rigid_body_label"],
        "cable_label_mapping": {f"c{index+1}": label for index, label in enumerate(config["cable_labels"])},  # type: ignore[arg-type]
        "motive_field_mapping": motive.field_mapping,
        "coordinate_transform": transform,
        "quaternion_convention": "Motive header X,Y,Z,W; normalized sign-continuous active tracking-rigid-body-to-world",
        "reference_geometry": {
            "tracked_point": "cf_7 rigid-body origin at the top marker plane",
            "attachment_offset_tracking_m": model_config['recorded_data']['optitrack_to_attachment_offset_body_m'],
            "attachment_to_c1_arc_length_m": cable_configuration.marker_interval_lengths_m[0],
            "attachment_position": "tracked_position + R_tracking_to_world * offset_tracking",
            "firmware_body_frame_equivalence": "not established by this recording parser",
            "center_of_mass": "not inferred",
        },
        "command_semantics": command_semantics,
        "processed_command_fields": {
            "p_cmd_m": "command_position_m",
            "v_cmd_mps": "command_velocity_mps",
            "a_cmd_mps2": "command_acceleration_mps2",
            "yaw_cmd_rad": "command_yaw",
            "q_cmd_xyzw_provenance": "command_orientation_xyzw",
            "omega_cmd_body_rad_s_provenance": "command_angular_velocity",
        },
        "command_angular_rate_frame": command_semantics["angular_velocity_frame"],
        "command_angular_rate_units": command_semantics["angular_velocity_units"],
        "processing_config_schema": config["schema"],
        "processing_software_version": config["processing_version"],
        "processing_config_sha256": canonical_json_hash(config),
        "processor_source_sha256": {
            name: sha256_file(PACKAGE_ROOT / name)
            for name in ("io.py", "logger.py", "motive.py", "sync.py", "quality.py", "processing.py", "../simulator/geometry.py")
        },
        "processing_fingerprint": fingerprint,
        "processed_take_sha256": take_hash,
        "processing_timestamp_utc": utc_now(),
        "frames": int(len(time_s)),
        "duration_s": float(time_s[-1]),
        "scientific_timeline_source": SCIENTIFIC_TIMELINE_SOURCE,
        "motive_start_time_s": float(motive.source_time_s[0]),
        "motive_end_time_s": float(motive.source_time_s[-1]),
        "logger_source": logger_path.name,
        "logger_first_synchronized_time_s": float(logger_take_time[0]),
        "logger_last_synchronized_time_s": float(logger_take_time[-1]),
        "logger_command_coverage_within_take": float(
            command_report["command_coverage_fraction"]
        ),
        "logger_pre_history_available_s": float(
            max(0.0, motive.source_time_s[0] - logger_take_time[0])
        ),
        "logger_post_history_excluded_s": float(
            max(0.0, logger_take_time[-1] - motive.source_time_s[-1])
        ),
        "fit_ready": fit_ready,
        "quality_status": quality_status,
        "warnings": warnings,
    }
    atomic_json(sync_path, sync_report)
    atomic_json(metadata_path, metadata)
    return {
        "take_id": take_id,
        "processed": True,
        "skipped_unchanged": False,
        "frames": int(len(time_s)),
        "duration_s": float(time_s[-1]),
        "fit_ready": fit_ready,
        "quality_status": quality_status,
        "warnings": warnings,
        "command_coverage": command_report["command_coverage_fraction"],
        "sync_rms_s": max(ros_clock.rms_residual_s, motive_clock.rms_residual_s),
    }


def process_all(
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
    take_id: str | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    root = Path(raw_root)
    root.mkdir(parents=True, exist_ok=True)
    directories = [item for item in sorted(root.iterdir()) if item.is_dir()]
    if take_id is not None:
        directories = [item for item in directories if item.name == take_id]
        if not directories:
            raise FileNotFoundError(f"No raw take directory named {take_id!r}.")
    results: list[dict[str, object]] = []
    for directory in directories:
        try:
            result = process_take(
                directory, processed_root=processed_root, force=force
            )
        except Exception as error:
            failure_dir = Path(processed_root) / directory.name
            failure_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                failure_dir / "processing_failure.json",
                {
                    "schema": "aerial_cable_processing_failure_v1",
                    "take_id": directory.name,
                    "quality_status": "PROCESSING_FAILED",
                    "error": f"{type(error).__name__}: {error}",
                    "timestamp_utc": utc_now(),
                },
            )
            result = {
                "take_id": directory.name,
                "processed": False,
                "fit_ready": False,
                "quality_status": "PROCESSING_FAILED",
                "warnings": [str(error)],
            }
        results.append(result)
        if progress is not None:
            progress(json.dumps(result, sort_keys=True))
    ensure_manifest([directory.name for directory in directories])
    return results
