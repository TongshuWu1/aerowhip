"""Scientific command/timeline contract audit and metadata-only migration.

The processed numerical arrays are immutable evidence.  This module may enrich
metadata only after proving that those arrays are the exact manually trimmed
Motive frames. It audits provenance without altering processed numeric evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import atomic_json, resolve_raw_pair, sha256_file
from .logger import parse_logger
from .motive import parse_motive_take
from .processing import (
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_RAW_ROOT,
    SCIENTIFIC_TIMELINE_SOURCE,
    _processing_fingerprint,
    load_config,
)
from .sync import build_clocks, reconstruct_commands


def audit_take_contract(
    take_id: str,
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
) -> dict[str, object]:
    """Prove the stored scientific samples are exactly the Motive export."""

    config = load_config()
    logger_path, motive_path = resolve_raw_pair(Path(raw_root) / take_id)
    logger = parse_logger(logger_path)
    motive = parse_motive_take(
        motive_path,
        uav_label=str(config["uav_rigid_body_label"]),
        cable_labels=list(config["cable_labels"]),  # type: ignore[arg-type]
    )
    ros_clock, motive_clock, _, _ = build_clocks(logger, motive, config)
    _, command_report = reconstruct_commands(
        logger, motive, ros_clock, motive_clock
    )
    take_path = Path(processed_root) / take_id / "take.npz"
    with np.load(take_path, allow_pickle=False) as archive:
        processed_time = np.asarray(archive["time_s"])
        processed_source_time = np.asarray(archive["motive_source_time_s"])
        processed_frame = np.asarray(archive["motive_frame"])
        command_valid = np.asarray(archive["command_valid"])
    expected_time = motive.source_time_s - motive.source_time_s[0]
    logger_take_time = motive_clock.map(logger.motive_time_s)
    exact = bool(
        np.array_equal(processed_time, expected_time)
        and np.array_equal(processed_source_time, motive.source_time_s)
        and np.array_equal(processed_frame, motive.frame)
    )
    command_diagnostics = command_report["raw_command_diagnostics_no_unit_inference"]
    return {
        "take_id": take_id,
        "timeline_contract": "PASS" if exact else "NEEDS_CORRECTION",
        "processed_take_sha256": sha256_file(take_path),
        "raw_logger_file": logger_path.name,
        "raw_motive_file": motive_path.name,
        "raw_source_sha256": {
            logger_path.name: sha256_file(logger_path),
            motive_path.name: sha256_file(motive_path),
        },
        "motive_first_frame": int(motive.frame[0]),
        "motive_last_frame": int(motive.frame[-1]),
        "motive_frames": int(len(motive.frame)),
        "motive_start_time_s": float(motive.source_time_s[0]),
        "motive_end_time_s": float(motive.source_time_s[-1]),
        "motive_duration_s": float(expected_time[-1]),
        "logger_first_synchronized_time_s": float(logger_take_time[0]),
        "logger_last_synchronized_time_s": float(logger_take_time[-1]),
        "logger_pre_history_available_s": float(
            max(0.0, motive.source_time_s[0] - logger_take_time[0])
        ),
        "logger_post_history_excluded_s": float(
            max(0.0, logger_take_time[-1] - motive.source_time_s[-1])
        ),
        "processed_start_s": float(processed_time[0]),
        "processed_end_s": float(processed_time[-1]),
        "processed_duration_s": float(processed_time[-1]),
        "processed_frames": int(len(processed_time)),
        "exact_time_array_match": bool(np.array_equal(processed_time, expected_time)),
        "exact_source_time_array_match": bool(
            np.array_equal(processed_source_time, motive.source_time_s)
        ),
        "exact_frame_array_match": bool(np.array_equal(processed_frame, motive.frame)),
        "maximum_time_difference_s": float(
            np.max(np.abs(processed_time - expected_time))
        ),
        "logger_command_coverage_within_take": float(np.mean(command_valid)),
        "q_cmd_xy_norm_max": command_diagnostics["command_quaternion_xy_norm_max"],
        "omega_cmd_absolute_max": command_diagnostics["command_omega_raw_absolute_max"],
        "horizontal_acceleration_max_mps2": command_diagnostics[
            "horizontal_acceleration_max"
        ],
    }


def enrich_processed_metadata(
    take_id: str,
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
) -> dict[str, object]:
    """Add contract provenance without changing ``take.npz`` or its hash."""

    audit = audit_take_contract(
        take_id, raw_root=raw_root, processed_root=processed_root
    )
    if audit["timeline_contract"] != "PASS":
        raise RuntimeError(
            f"{take_id}: processed arrays are not the exact Motive manual trim."
        )
    output = Path(processed_root) / take_id
    metadata_path = output / "metadata.json"
    sync_path = output / "sync_report.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sync = json.loads(sync_path.read_text(encoding="utf-8"))
    old_take_hash = str(metadata["processed_take_sha256"])
    if old_take_hash != audit["processed_take_sha256"]:
        raise RuntimeError(f"{take_id}: stored processed hash is inconsistent.")
    config = load_config()
    source_hashes = dict(audit["raw_source_sha256"])
    metadata.update(
        {
            "command_semantics": config["command_semantics"],
            "command_angular_rate_frame": "body",
            "command_angular_rate_units": "radians_per_second",
            "processed_command_fields": {
                "p_cmd_m": "command_position_m",
                "v_cmd_mps": "command_velocity_mps",
                "a_cmd_mps2": "command_acceleration_mps2",
                "yaw_cmd_rad": "command_yaw",
                "q_cmd_xyzw_provenance": "command_orientation_xyzw",
                "omega_cmd_body_rad_s_provenance": "command_angular_velocity",
            },
            "scientific_timeline_source": SCIENTIFIC_TIMELINE_SOURCE,
            "motive_start_time_s": audit["motive_start_time_s"],
            "motive_end_time_s": audit["motive_end_time_s"],
            "duration_s": audit["motive_duration_s"],
            "logger_source": audit["raw_logger_file"],
            "logger_first_synchronized_time_s": audit[
                "logger_first_synchronized_time_s"
            ],
            "logger_last_synchronized_time_s": audit[
                "logger_last_synchronized_time_s"
            ],
            "logger_command_coverage_within_take": audit[
                "logger_command_coverage_within_take"
            ],
            "logger_pre_history_available_s": audit[
                "logger_pre_history_available_s"
            ],
            "logger_post_history_excluded_s": audit[
                "logger_post_history_excluded_s"
            ],
            "processing_fingerprint": _processing_fingerprint(
                source_hashes, config
            ),
            "fit_ready": True,
        }
    )
    unresolved = "Fitting disabled: the logger mapping is audited"
    warnings = [
        str(value)
        for value in metadata.get("warnings", [])
        if not str(value).startswith(unresolved)
    ]
    metadata["warnings"] = warnings
    metadata["quality_status"] = "READY" if not warnings else "WARNING"
    metadata["processed_take_sha256"] = old_take_hash
    sync["scientific_timeline"] = {
        "source": SCIENTIFIC_TIMELINE_SOURCE,
        "motive_start_time_s": audit["motive_start_time_s"],
        "motive_end_time_s": audit["motive_end_time_s"],
        "duration_s": audit["motive_duration_s"],
        "logger_first_synchronized_time_s": audit[
            "logger_first_synchronized_time_s"
        ],
        "logger_last_synchronized_time_s": audit[
            "logger_last_synchronized_time_s"
        ],
        "logger_pre_history_available_s": audit[
            "logger_pre_history_available_s"
        ],
        "logger_post_history_excluded_s": audit[
            "logger_post_history_excluded_s"
        ],
        "scientific_samples_are_motive_frames_only": True,
    }
    atomic_json(metadata_path, metadata)
    atomic_json(sync_path, sync)
    if sha256_file(output / "take.npz") != old_take_hash:
        raise RuntimeError(f"{take_id}: metadata migration changed take.npz.")
    return audit

