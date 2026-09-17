"""Split repeated PVA recordings into reviewed, event-log-backed flight batches.

The source CSVs are never changed. Each output take retains native timestamps and
copies original data rows byte-for-byte; only the Motive metadata header is
updated with the segment name and exported-frame count.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experimental_data.adaptation_check import command_onset, load_comparison
from experimental_data.adaptation_rounds import (
    COMMAND_COLUMNS,
    align,
    read_controller,
    read_optitrack,
)
from experimental_data.current_adaptation import (
    causal_history_indices,
    observed_cable_history,
    observed_endpoint_velocity,
)
from experimental_data.io import atomic_json, sha256_file
from simulator.cable import CableConfiguration
from simulator.geometry import attachment_positions, normalized_rotations_xyzw


def _reference_values(reference: np.ndarray) -> np.ndarray:
    return np.column_stack([reference[name] for name in reference.dtype.names[1:]])


def _command_values(commands: np.ndarray) -> np.ndarray:
    return np.column_stack([commands[name] for name in COMMAND_COLUMNS])


def _source_pairs(source: Path) -> list[tuple[str, Path, Path, Path, Path]]:
    pairs = []
    for logger in sorted(source.glob("experiment_*.csv")):
        if logger.name.endswith(".commands.csv"):
            continue
        recording = logger.stem.removeprefix("experiment_")
        tracking = source / f"{recording}.csv"
        events = source / f"experiment_{recording}.commands.csv"
        metadata = source / f"experiment_{recording}.metadata.json"
        missing = [path.name for path in (tracking, events, metadata) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{recording}: missing {missing}")
        pairs.append((recording, tracking, logger, events, metadata))
    if not pairs:
        raise ValueError(f"No complete recording pairs found in {source}")
    return pairs


def _detect_executions(events: np.ndarray, reference: np.ndarray) -> list[dict[str, object]]:
    ref_values = _reference_values(reference)
    event_values = _command_values(events)
    dynamic = np.linalg.norm(ref_values[:, 3:9], axis=1) > 1.0e-5
    dynamic_indices = np.flatnonzero(dynamic)
    if not len(dynamic_indices):
        raise ValueError("The reference has no dynamic P/V/A rows.")
    first = int(dynamic_indices[0])
    if np.count_nonzero(np.all(np.isclose(ref_values, ref_values[first], rtol=0, atol=1.0e-12), axis=1)) != 1:
        raise ValueError("The first dynamic reference row is not unique.")
    candidates = np.flatnonzero(
        np.all(np.isclose(event_values, ref_values[first], rtol=0, atol=1.0e-10), axis=1)
    )
    executions = []
    expected = ref_values[first:]
    for hit in candidates:
        stop = int(hit) + len(expected)
        if stop > len(events):
            continue
        if not np.allclose(event_values[hit:stop], expected, rtol=0, atol=1.0e-10):
            continue
        onset, packets, spread = command_onset(events[hit:stop], reference)
        executions.append(
            {
                "event_start": int(hit),
                "event_stop": stop,
                "onset_s": onset,
                "moving_packets": packets,
                "onset_spread_ms": 1000.0 * spread,
                "complete_moving_sequence": True,
            }
        )
    if not executions:
        raise ValueError("No complete execution of the supplied reference was found.")
    onsets = np.array([item["onset_s"] for item in executions], dtype=float)
    if len(onsets) > 1 and np.min(np.diff(onsets)) <= float(reference["time_s"][-1]):
        raise ValueError("Detected command executions overlap.")
    return executions


def _snapshot_event_consistency(logger: Path, events: Path) -> bool:
    snapshots = np.genfromtxt(logger, delimiter=",", names=True, encoding="utf-8-sig")
    commands = np.genfromtxt(events, delimiter=",", names=True, encoding="utf-8-sig")
    event_index = {int(sequence): index for index, sequence in enumerate(commands["cmd_sequence"])}
    mapped = np.array([event_index.get(int(sequence), -1) for sequence in snapshots["cmd_sequence"]])
    valid = (snapshots["cmd_valid"] > 0.5) & (snapshots["cmd_sequence"] > 0) & (mapped >= 0)
    if not np.any(valid):
        return False
    indices = mapped[valid]
    if not np.array_equal(
        snapshots["cmd_header_stamp_ns"][valid], commands["cmd_header_stamp_ns"][indices]
    ):
        return False
    if not np.array_equal(
        snapshots["cmd_receive_monotonic_ns"][valid], commands["receive_monotonic_ns"][indices]
    ):
        return False
    received = snapshots["time_s"][valid] - snapshots["cmd_age"][valid]
    if not np.allclose(received, commands["time_s"][indices], rtol=0, atol=1.0e-10):
        return False
    for name in COMMAND_COLUMNS:
        if not np.array_equal(snapshots[name][valid], commands[name][indices]):
            return False
    return True


def _gap_aware_alignment(
    motive_time: np.ndarray,
    motive_position: np.ndarray,
    logger: np.ndarray,
    initial_offset: float,
) -> dict[str, float | int]:
    logger_time = logger["time_s"]
    logger_position = np.column_stack([logger[name] for name in ("x", "y", "z")])

    def score(offset: float, report: bool = False):
        query = motive_time + offset
        right = np.searchsorted(logger_time, query)
        inside = (right > 0) & (right < len(logger_time))
        right = np.clip(right, 1, len(logger_time) - 1)
        left = right - 1
        valid = inside & np.isfinite(motive_position).all(axis=1)
        valid &= np.isfinite(logger_position[left]).all(axis=1)
        valid &= np.isfinite(logger_position[right]).all(axis=1)
        if "pose_valid" in logger.dtype.names:
            valid &= (logger["pose_valid"][left] > 0.5) & (logger["pose_valid"][right] > 0.5)
        if "tracking_gap" in logger.dtype.names:
            valid &= logger["tracking_gap"][right] < 0.5
        if np.count_nonzero(valid) < 30:
            return (float("inf"), 0) if report else float("inf")
        weight = (query - logger_time[left]) / (logger_time[right] - logger_time[left])
        fitted = logger_position[left] + weight[:, None] * (
            logger_position[right] - logger_position[left]
        )
        squared = np.sum(np.square(fitted[valid] - motive_position[valid]), axis=1)
        mse = float(np.mean(squared))
        return (mse**0.5, int(np.count_nonzero(valid))) if report else mse

    grid = np.linspace(initial_offset - 0.03, initial_offset + 0.03, 1201)
    values = np.asarray([score(offset) for offset in grid])
    best = int(np.argmin(values))
    lower = grid[max(0, best - 2)]
    upper = grid[min(len(grid) - 1, best + 2)]
    result = minimize_scalar(
        score, bounds=(lower, upper), method="bounded", options={"xatol": 1.0e-14}
    )
    rms, count = score(float(result.x), report=True)
    return {"offset_s": float(result.x), "rms_m": float(rms), "matched_frames": count}


def _line_time(line: bytes, column: int) -> float:
    return float(line.rstrip(b"\r\n").split(b",")[column])


def _write_plain_subset(source: Path, destination: Path, start: float, end: float) -> int:
    lines = source.read_bytes().splitlines(keepends=True)
    selected = [line for line in lines[1:] if start <= _line_time(line, 0) <= end]
    if not selected:
        raise ValueError(f"No rows selected from {source}")
    destination.write_bytes(lines[0] + b"".join(selected))
    return len(selected)


def _write_motive_subset(
    source: Path,
    destination: Path,
    name: str,
    start: float,
    end: float,
) -> int:
    lines = source.read_bytes().splitlines(keepends=True)
    if len(lines) < 8:
        raise ValueError(f"Unsupported Motive file: {source}")
    selected = [line for line in lines[7:] if start <= _line_time(line, 1) <= end]
    if not selected:
        raise ValueError(f"No Motive rows selected from {source}")
    ending = b"\r\n" if lines[0].endswith(b"\r\n") else b"\n"
    header = lines[0].rstrip(b"\r\n").decode("utf-8-sig").split(",")
    for key, value in (("Take Name", name), ("Total Exported Frames", str(len(selected)))):
        index = header.index(key)
        header[index + 1] = value
    destination.write_bytes(",".join(header).encode("utf-8") + ending + b"".join(lines[1:7]) + b"".join(selected))
    return len(selected)


def _marker_quality(
    motive: dict[str, object], model: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    drone = np.asarray(motive["drone"])
    quaternion = np.asarray(motive["quaternion"])
    cable = np.asarray(motive["cable"])
    _, rotation_valid = normalized_rotations_xyzw(quaternion)
    attachment, attachment_valid = attachment_positions(
        drone,
        quaternion,
        model["recorded_data"]["optitrack_to_attachment_offset_body_m"],
    )
    pose_valid = np.isfinite(drone).all(axis=1) & rotation_valid & attachment_valid
    marker_valid = np.isfinite(cable).all(axis=2)
    jumps = np.linalg.norm(np.diff(cable, axis=0), axis=-1) > 0.15
    marker_valid[:-1] &= ~jumps
    marker_valid[1:] &= ~jumps
    sites = np.concatenate([attachment[:, None], cable], axis=1)
    maximum = np.asarray(model["cable"]["marker_interval_lengths_m"]) + 0.015
    too_long = np.linalg.norm(np.diff(sites, axis=1), axis=-1) > maximum
    marker_valid &= ~too_long
    marker_valid[:, :-1] &= ~too_long[:, 1:]
    return pose_valid, marker_valid


def _initialization_check(
    motive: dict[str, object],
    relative_time: np.ndarray,
    model: dict[str, object],
    pose_valid: np.ndarray,
    marker_valid: np.ndarray,
) -> str | None:
    try:
        pre = causal_history_indices(relative_time, 0.0, 1.0)
        attachment, _ = attachment_positions(
            np.asarray(motive["drone"]),
            np.asarray(motive["quaternion"]),
            model["recorded_data"]["optitrack_to_attachment_offset_body_m"],
        )
        sites = np.concatenate([attachment[:, None], np.asarray(motive["cable"])], axis=1)
        nodes, observed = observed_cable_history(
            {"sites": sites, "pose_valid": pose_valid, "marker_valid": marker_valid},
            pre,
            CableConfiguration.from_mapping(model["cable"]),
            None,
        )
        velocity = observed_endpoint_velocity(relative_time[pre], nodes, observed, 0.02)
        if not np.isfinite(velocity).all():
            raise ValueError("Nonfinite initial cable velocity")
    except (ValueError, IndexError) as error:
        return str(error)
    return None


def _session_number(recording: str, fallback: int) -> int:
    try:
        return int(recording.rsplit("_", 1)[-1])
    except ValueError:
        return fallback


def process(args: argparse.Namespace) -> dict[str, object]:
    export = args.export.resolve()
    source = (args.source or export / "flight_take" / args.model_id).resolve()
    reference_path = export / "fullstate_30hz.csv"
    rehearsal = args.rehearsal.resolve()
    model_path = rehearsal / "model.json"
    prediction_path = rehearsal / "rehearsal.npz"
    metadata_path = rehearsal / "rehearsal.json"
    for path in (reference_path, model_path, prediction_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = np.genfromtxt(reference_path, delimiter=",", names=True)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    rehearsal_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(reference_path) != rehearsal_metadata["csv_sha256"]:
        raise ValueError("Exported command does not match the frozen rehearsal.")
    pairs = _source_pairs(source)
    planned_outputs = []
    for fallback, pair in enumerate(pairs, 1):
        session = _session_number(pair[0], fallback)
        batch = ROOT / "data" / "flight_batches" / f"{args.label}-session{session}-four-whips"
        review = ROOT / "runs" / "data_review" / f"{args.label}-session{session}-four-whips"
        planned_outputs.extend([batch, review])
    complete_output = ROOT / "runs" / "data_review" / f"{args.label}-complete-collection"
    planned_outputs.append(complete_output)
    existing = [path for path in planned_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to replace existing outputs: {existing}")

    original_hashes = {
        str(path.resolve()): sha256_file(path)
        for pair in pairs
        for path in pair[1:]
    }
    original_hashes.update(
        {
            str(reference_path.resolve()): sha256_file(reference_path),
            str(model_path.resolve()): sha256_file(model_path),
            str(prediction_path.resolve()): sha256_file(prediction_path),
        }
    )
    session_reports = []
    all_takes: dict[str, dict[str, object]] = {}
    verification_rows = []
    for fallback, (recording, tracking_path, logger_path, events_path, source_metadata_path) in enumerate(pairs, 1):
        session = _session_number(recording, fallback)
        batch = ROOT / "data" / "flight_batches" / f"{args.label}-session{session}-four-whips"
        review_output = ROOT / "runs" / "data_review" / f"{args.label}-session{session}-four-whips"
        raw_tracking = read_optitrack(tracking_path)
        raw_logger = read_controller(logger_path)
        event_commands = read_controller(
            logger_path, commands_only=True, command_source="event_log"
        )
        executions = _detect_executions(event_commands, reference)
        if len(executions) != 4:
            raise ValueError(f"{recording}: expected four complete executions, found {len(executions)}")
        snapshot_consistent = _snapshot_event_consistency(logger_path, events_path)
        if not snapshot_consistent:
            raise ValueError(f"{recording}: command snapshots disagree with the event log")
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata.get("status") != "complete":
            raise ValueError(f"{recording}: logger metadata does not mark the recording complete")
        global_alignment = align(raw_tracking, raw_logger)

        flight_take = batch / "flight_take"
        simulation = batch / "simulation_csv"
        flight_take.mkdir(parents=True)
        simulation.mkdir()
        review_output.mkdir(parents=True)
        shutil.copy2(reference_path, simulation / "fullstate_30hz.csv")
        alignments = {}
        recording_groups = {}
        repetitions = []
        motive_time = np.asarray(raw_tracking["time"])
        motive_position = np.asarray(raw_tracking["drone"])
        logger_raw = np.genfromtxt(logger_path, delimiter=",", names=True, encoding="utf-8-sig")
        event_raw = np.genfromtxt(events_path, delimiter=",", names=True, encoding="utf-8-sig")
        window_end = float(reference["time_s"][-1]) + 0.5
        for repetition, execution in enumerate(executions, 1):
            onset = float(execution["onset_s"])
            controller_start = onset - 5.0
            controller_end = onset + window_end
            motive_start = controller_start - float(global_alignment["offset_s"])
            motive_end = controller_end - float(global_alignment["offset_s"])
            tracking_mask = (motive_time >= motive_start) & (motive_time <= motive_end)
            if np.count_nonzero(tracking_mask) < 100:
                raise ValueError(f"{recording} repetition {repetition}: insufficient Motive coverage")
            segment_alignment = _gap_aware_alignment(
                motive_time[tracking_mask],
                motive_position[tracking_mask],
                logger_raw,
                float(global_alignment["offset_s"]),
            )
            name = f"{args.model_id}_side_{session:03d}_whip_{repetition:03d}"
            tracking_out = flight_take / f"{name}.csv"
            logger_out = flight_take / f"experiment_{name}.csv"
            events_out = flight_take / f"experiment_{name}.commands.csv"
            tracking_frames = _write_motive_subset(
                tracking_path, tracking_out, name, motive_start, motive_end
            )
            logger_rows = _write_plain_subset(logger_path, logger_out, controller_start, controller_end)
            command_events = _write_plain_subset(events_path, events_out, controller_start, controller_end)
            motive = read_optitrack(tracking_out)
            relative_time = (
                np.asarray(motive["time"]) + float(segment_alignment["offset_s"]) - onset
            )
            pose_valid, marker_valid = _marker_quality(motive, model)
            whip = (relative_time >= 0.0) & (
                relative_time <= float(rehearsal_metadata["whip_end_s"])
            )
            initialization_error = _initialization_check(
                motive, relative_time, model, pose_valid, marker_valid
            )
            logger_mask = (logger_raw["time_s"] >= controller_start) & (
                logger_raw["time_s"] <= controller_end
            )
            gap_times = (
                logger_raw["time_s"][logger_mask & (logger_raw["tracking_gap"] > 0.5)] - onset
            ).tolist()
            invalid = np.count_nonzero(~marker_valid[whip], axis=0)
            valid_fraction = np.mean(marker_valid[whip], axis=0)
            segment = {
                **execution,
                "repetition": repetition,
                "name": name,
                "recording_group": recording,
                "source_tracking": str(tracking_path.resolve()),
                "source_logger": str(logger_path.resolve()),
                "source_commands": str(events_path.resolve()),
                "offset_s": float(segment_alignment["offset_s"]),
                "alignment_rms_m": float(segment_alignment["rms_m"]),
                "tracking_frames": tracking_frames,
                "logger_rows": logger_rows,
                "command_events": command_events,
                "tracking_relative_span_s": [float(relative_time[0]), float(relative_time[-1])],
                "prehover_available_s": float(max(0.0, -relative_time[0])),
                "full_csv_tracking_complete": bool(
                    relative_time[0] <= 0.0 and relative_time[-1] >= float(reference["time_s"][-1])
                ),
                "one_second_initialization_passed": initialization_error is None,
                "initialization_error": initialization_error,
                "whip_pose_valid_fraction": float(np.mean(pose_valid[whip])),
                "whip_marker_valid_fractions": valid_fraction.tolist(),
                "whip_invalid_marker_samples": invalid.tolist(),
                "whip_tip_valid_fraction": float(valid_fraction[-1]),
                "whip_tf_gaps": int(
                    np.count_nonzero(
                        logger_mask
                        & (logger_raw["tracking_gap"] > 0.5)
                        & (logger_raw["time_s"] >= onset)
                        & (
                            logger_raw["time_s"]
                            <= onset + float(rehearsal_metadata["whip_end_s"])
                        )
                    )
                ),
                "segment_tf_gap_times_s": gap_times,
                "native_timestamps_preserved": True,
                "original_data_rows_preserved": True,
                "clock_verified": False,
                "fit_role": "unassigned",
                "physical_target_present": None,
                "cable_touched_or_reset": None,
            }
            segment.pop("event_start")
            segment.pop("event_stop")
            atomic_json(flight_take / f"{name}.segment.json", segment)
            atomic_json(
                flight_take / f"{name}.tracking.json",
                {"drone": raw_tracking["drone_label"], "optitrack_sha256": sha256_file(tracking_out)},
            )
            atomic_json(
                flight_take / f"experiment_{name}.metadata.json",
                {
                    "schema": "segmented_optitrack_tf_pva_v2",
                    "source_metadata_path": str(source_metadata_path.resolve()),
                    "source_metadata": source_metadata,
                    "segment": segment,
                    "files": {
                        "tracking": tracking_out.name,
                        "samples": logger_out.name,
                        "commands": events_out.name,
                    },
                },
            )
            alignments[name] = {
                "offset_s": float(segment_alignment["offset_s"]),
                "rms_m": float(segment_alignment["rms_m"]),
                "clock_verified": False,
                "source": "Estimated per repetition from measured native-rate TF XYZ and native Motive XYZ, excluding interpolation across logger gaps; no command/forecast or spatial fitting. Includes unknown transport latency; original timestamps retained.",
                "optitrack_sha256": sha256_file(tracking_out),
                "controller_sha256": sha256_file(logger_out),
                "source_report": str((review_output / "report.json").resolve()),
            }
            recording_groups[name] = recording
            repetitions.append(segment)
            all_takes[name] = segment

        protocol = {
            "schema": "segmented_flight_review_v1",
            "rehearsal": str(rehearsal),
            "command_sha256": sha256_file(reference_path),
            "forecast_sha256": sha256_file(prediction_path),
            "command_source": "event_log",
            "frame": "raw_global_xyz",
            "normalization": False,
            "fitting_performed": False,
            "purpose": f"Individual playback and original {args.model_id} curved-side forecast comparison; four complete whips.",
            "planned_roles": {name: "unassigned" for name in recording_groups},
            "recording_groups": recording_groups,
            "user_context": {
                "physical_target_present": None,
                "cable_touched_or_reset": None,
                "review_status": "operator confirmation pending",
            },
        }
        atomic_json(batch / "protocol.json", protocol)
        atomic_json(batch / "time_alignment.json", alignments)
        session_hashes = {
            key: value
            for key, value in original_hashes.items()
            if key in {
                str(tracking_path.resolve()),
                str(logger_path.resolve()),
                str(events_path.resolve()),
                str(source_metadata_path.resolve()),
                str(reference_path.resolve()),
                str(model_path.resolve()),
                str(prediction_path.resolve()),
            }
        }
        session_report = {
            "recording": recording,
            "source_directory": str(source),
            "source_hashes": session_hashes,
            "original_files_unchanged": True,
            "original_logger_sample_count": int(len(logger_raw)),
            "uploaded_logger_sample_count": int(source_metadata["counts"]["samples"]),
            "user_trimmed_sample_rows": int(source_metadata["counts"]["samples"] - len(logger_raw)),
            "native_optitrack_rate_hz": float(1.0 / np.median(np.diff(motive_time))),
            "new_logger_observed_rate_hz": float(1.0 / np.median(np.diff(logger_raw["time_s"]))),
            "global_alignment_diagnostic": global_alignment,
            "alignment_policy": "Independent constant offset per recording segment from the two measured position streams; global estimate used only to locate windows. No target or forecast alignment.",
            "alignment_note": "Separate recording clocks have different origins; offsets are not physical drone delays. Small offset variation between repetitions remains explicit.",
            "command_snapshot_event_timestamps_consistent": snapshot_consistent,
            "all_command_executions": [
                {
                    key: item[key]
                    for key in (
                        "repetition",
                        "onset_s",
                        "onset_spread_ms",
                        "moving_packets",
                        "complete_moving_sequence",
                    )
                }
                for item in repetitions
            ],
            "processed_repetitions": list(range(1, len(repetitions) + 1)),
            "excluded_repetitions": [],
            "exclusion_reason": None,
            "repetitions": repetitions,
            "fitting_performed": False,
            "user_context": protocol["user_context"],
            "processor_sha256": sha256_file(Path(__file__)),
        }
        atomic_json(batch / "split_manifest.json", session_report)
        atomic_json(review_output / "report.json", session_report)
        readme = (
            f"# {args.model_id} curved-side session {session}: four complete whips\n\n"
            f"All four complete {len(reference)}-row command executions match the saved CSV. "
            "Playback and later evaluation use the command-event log. Native timestamps and "
            "source data rows are preserved; fitting was not performed. Operator confirmation "
            "of cable resets/interventions and physical target conditions remains pending.\n"
        )
        (batch / "README.md").write_text(readme, encoding="utf-8")
        comparisons = []
        for name in recording_groups:
            loaded = load_comparison(ROOT, batch, name, rehearsal)
            comparisons.append(
                {
                    "take": name,
                    "comparison_frames": int(len(loaded["time"])),
                    "packets": int(loaded["packet_count"]),
                    "command_source": "event_log",
                }
            )
        verification = {
            "original_hashes_unchanged": all(
                Path(path).is_file() and sha256_file(path) == digest
                for path, digest in session_hashes.items()
            ),
            "raw_rows_byte_preserved": True,
            "command_event_consistency_passed": snapshot_consistent,
            "command_source": "event_log",
            "comparison_loads": comparisons,
        }
        atomic_json(batch / "verification.json", verification)
        verification_rows.extend(comparisons)
        session_reports.append(session_report)

    complete_output.mkdir(parents=True)
    gaps = [
        {"take": take, "relative_s": value}
        for take, item in all_takes.items()
        for value in item["segment_tf_gap_times_s"]
        if 0.0 <= value <= float(rehearsal_metadata["whip_end_s"])
    ]
    complete_report = {
        "model": args.model_id,
        "experiment": args.label,
        "processed_whips": len(all_takes),
        "recording_sessions": len(session_reports),
        "command_source": "event_log",
        "fitting_performed": False,
        "original_files_unchanged": all(
            Path(path).is_file() and sha256_file(path) == digest
            for path, digest in original_hashes.items()
        ),
        "all_one_second_initializations_passed": all(
            item["one_second_initialization_passed"] for item in all_takes.values()
        ),
        "all_whip_tracking_intervals_continuous": all(
            item["whip_pose_valid_fraction"] == 1.0 for item in all_takes.values()
        ),
        "invalid_marker_samples": int(
            sum(sum(item["whip_invalid_marker_samples"]) for item in all_takes.values())
        ),
        "logger_gaps_during_whip": gaps,
        "partial_recovery_tails": [
            name for name, item in all_takes.items() if not item["full_csv_tracking_complete"]
        ],
        "excluded_whips": [],
        "operator_context_required": True,
        "operator_context": {
            "physical_target_present": None,
            "cable_touched_or_reset_between_whips": None,
            "manual_intervention": None,
        },
        "source_hashes": original_hashes,
        "session_reports": session_reports,
        "takes": all_takes,
        "verification": verification_rows,
    }
    atomic_json(complete_output / "report.json", complete_report)
    (complete_output / "README.md").write_text(
        f"# {args.model_id} curved-side collection\n\n"
        f"Processed {len(all_takes)} complete whips from {len(session_reports)} recording sessions. "
        "The output is segmented and validated against the frozen command and forecast without fitting. "
        "Operator-dependent experimental conditions remain explicitly unconfirmed.\n",
        encoding="utf-8",
    )
    return complete_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--model-id", default="M5")
    parser.add_argument("--label", default="M5-curved-side")
    args = parser.parse_args()
    report = process(args)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "model",
                    "experiment",
                    "processed_whips",
                    "recording_sessions",
                    "all_one_second_initializations_passed",
                    "all_whip_tracking_intervals_continuous",
                    "invalid_marker_samples",
                    "logger_gaps_during_whip",
                    "partial_recovery_tails",
                    "operator_context_required",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
