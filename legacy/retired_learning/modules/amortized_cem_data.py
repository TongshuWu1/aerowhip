"""Permanent sharded data contract for production-CEM policy amortization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .context_sampling import target_direction_from_local_target
from .policy_context import POLICY_CONTEXT_DIM
from .state_bank import InitialStateBank


ACTION_DIM = 49
CONTINUOUS_OUTCOME_NAMES = (
    "legacy_optimizer_reward",
    "log_tip_distance",
    "directed_tip_speed",
    "cos_direction_error",
    "max_uav_displacement",
    "max_uav_speed",
    "max_command_acceleration",
)
BINARY_OUTCOME_NAMES = ("tip_first", "finite", "scientific_success")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AmortizedCemContextRecord:
    context_index: int
    context_id: str
    split: str
    bank: str
    state_id: str
    state_index: int
    target_number: int
    target_local_m: tuple[float, float, float]
    direction_local: tuple[float, float, float]
    theta_nominal: tuple[float, float, float, float, float, float, float]


def stable_context_id(
    *, bank: str, state_index: int, target_local_m: Iterable[float], split: str
) -> str:
    payload = {
        "bank": bank,
        "state_index": int(state_index),
        "target_local_m": [round(float(value), 9) for value in target_local_m],
        "split": split,
        "schema": "amortized_cem_context_id_v1",
    }
    suffix = hashlib.sha256(_json_bytes(payload)).hexdigest()[:16]
    return f"{split.lower()}_{bank}_{state_index:04d}_{suffix}"


def state_descriptor(bank: InitialStateBank) -> np.ndarray:
    """Physical coverage descriptor including UAV and distributed cable state."""

    root = bank.cable_positions_m[:, 1:2]
    relative = bank.cable_positions_m[:, 2:12] - root
    curvature = relative[:, 2:] - 2.0 * relative[:, 1:-1] + relative[:, :-2]
    tensors = (
        bank.uav_velocity_m_s,
        bank.uav_orientation_xyzw[:, :3],
        bank.uav_angular_velocity_world_rad_s,
        bank.cable_velocities_m_s[:, 11],
        relative.reshape(len(bank), -1),
        curvature.reshape(len(bank), -1),
        bank.cable_velocities_m_s[:, 2:12].reshape(len(bank), -1),
    )
    value = torch.cat(tensors, dim=-1).numpy().astype(np.float64)
    scale = value.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    return (value - value.mean(axis=0)) / scale


def farthest_point_indices(
    descriptor: np.ndarray, count: int, *, seed: int
) -> np.ndarray:
    values = np.asarray(descriptor, dtype=np.float64)
    if values.ndim != 2 or count < 1 or count > values.shape[0]:
        raise ValueError("Invalid farthest-point selection request.")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(values.shape[0]))
    selected = [first]
    distance = np.sum((values - values[first]) ** 2, axis=1)
    distance[first] = -1.0
    for _ in range(1, count):
        index = int(np.argmax(distance))
        selected.append(index)
        candidate = np.sum((values - values[index]) ** 2, axis=1)
        distance = np.minimum(distance, candidate)
        distance[np.asarray(selected, dtype=np.int64)] = -1.0
    return np.asarray(selected, dtype=np.int64)


def sobol_target_pairs(
    state_count: int,
    *,
    seed: int,
    lower: tuple[float, float, float] = (0.85, -0.15, -0.12),
    upper: tuple[float, float, float] = (1.05, 0.15, 0.02),
) -> np.ndarray:
    engine = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed)
    base = engine.draw(state_count).numpy().astype(np.float64)
    paired = np.mod(base + 0.5, 1.0)
    unit = np.stack((base, paired), axis=1)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    return (low + unit * (high - low)).astype(np.float32)


def build_initial_context_records(
    training_bank: InitialStateBank,
    heldout_bank: InitialStateBank,
    *,
    theta_nominal: Iterable[float],
    seed: int = 42,
) -> tuple[list[AmortizedCemContextRecord], dict[str, Any], dict[str, Any]]:
    train_indices = farthest_point_indices(state_descriptor(training_bank), 256, seed=seed + 1)
    heldout_selected = farthest_point_indices(
        state_descriptor(heldout_bank), 192, seed=seed + 2
    )
    rng = np.random.default_rng(seed + 3)
    heldout_selected = heldout_selected[rng.permutation(heldout_selected.size)]
    validation_indices = heldout_selected[:64]
    test_indices = heldout_selected[64:128]
    edge_indices = heldout_selected[128:192]
    theta = tuple(float(value) for value in theta_nominal)
    if len(theta) != 7:
        raise ValueError("The permanent context schema requires seven theta values.")

    records: list[AmortizedCemContextRecord] = []
    target_rows: list[dict[str, Any]] = []
    groups = (
        ("TRAIN", "training", train_indices, seed + 101),
        ("VALIDATION", "heldout", validation_indices, seed + 202),
        ("TEST", "heldout", test_indices, seed + 303),
    )
    for split, bank_name, indices, target_seed in groups:
        targets = sobol_target_pairs(len(indices), seed=target_seed)
        directions = target_direction_from_local_target(
            torch.from_numpy(targets.reshape(-1, 3))
        ).reshape(len(indices), 2, 3).numpy()
        for state_number, state_index in enumerate(indices.tolist()):
            for target_number in range(2):
                target = tuple(float(value) for value in targets[state_number, target_number])
                direction = tuple(float(value) for value in directions[state_number, target_number])
                context_id = stable_context_id(
                    bank=bank_name,
                    state_index=state_index,
                    target_local_m=target,
                    split=split,
                )
                record = AmortizedCemContextRecord(
                    len(records),
                    context_id,
                    split,
                    bank_name,
                    f"{bank_name}:{state_index:04d}",
                    int(state_index),
                    target_number,
                    target,
                    direction,
                    theta,
                )
                records.append(record)
                target_rows.append(
                    {
                        "context_id": context_id,
                        "target_local_m": list(target),
                        "direction_local": list(direction),
                        "sobol_seed": target_seed,
                        "paired_target_number": target_number,
                    }
                )

    validation_set = {int(value) for value in validation_indices}
    test_set = {int(value) for value in test_indices}
    edge_set = {int(value) for value in edge_indices}
    if validation_set & test_set or validation_set & edge_set or test_set & edge_set:
        raise RuntimeError("Held-out state split leakage detected.")
    manifest = {
        "schema": "amortized_cem_state_split_v1",
        "seed": seed,
        "state_level_split": True,
        "counts": {"TRAIN": 256, "VALIDATION": 64, "TEST": 64},
        "context_counts": {"TRAIN": 512, "VALIDATION": 128, "TEST": 128},
        "training_state_ids": [f"training:{i:04d}" for i in train_indices.tolist()],
        "validation_state_ids": [f"heldout:{i:04d}" for i in validation_indices.tolist()],
        "test_state_ids": [f"heldout:{i:04d}" for i in test_indices.tolist()],
        "edge_reserved_test_state_ids": [f"heldout:{i:04d}" for i in edge_indices.tolist()],
        "aggregation_train_pool_state_ids": [
            f"training:{i:04d}" for i in range(len(training_bank)) if i not in set(train_indices.tolist())
        ],
        "records": [asdict(record) for record in records],
        "disjoint": True,
        "protected_data_used": False,
    }
    target_manifest = {
        "schema": "amortized_cem_sobol_targets_v1",
        "domain_local_m": {
            "lower": [0.85, -0.15, -0.12],
            "upper": [1.05, 0.15, 0.02],
        },
        "targets_per_state": 2,
        "construction": "scrambled Sobol plus half-period antithetic pair",
        "rows": target_rows,
    }
    return records, manifest, target_manifest


def build_sealed_evaluation_records(
    heldout_bank: InitialStateBank,
    split_manifest: dict[str, Any],
    *,
    theta_nominal: Iterable[float],
    seed: int = 42,
) -> dict[str, list[AmortizedCemContextRecord]]:
    """Create fixed neural-only conditioning/joint/edge evaluation manifests."""

    theta = tuple(float(value) for value in theta_nominal)
    occupied = {
        int(state_id.split(":")[1])
        for name in (
            "validation_state_ids",
            "test_state_ids",
            "edge_reserved_test_state_ids",
        )
        for state_id in split_manifest[name]
    }
    available = np.asarray(
        [index for index in range(len(heldout_bank)) if index not in occupied], dtype=np.int64
    )
    descriptor = state_descriptor(heldout_bank)
    local_selection = farthest_point_indices(descriptor[available], 128, seed=seed + 404)
    joint_states = available[local_selection]
    edge_states = np.asarray(
        [int(value.split(":")[1]) for value in split_manifest["edge_reserved_test_state_ids"]],
        dtype=np.int64,
    )

    def make_record(
        split: str,
        context_index: int,
        state_index: int,
        target: np.ndarray,
        target_number: int,
    ) -> AmortizedCemContextRecord:
        direction = target_direction_from_local_target(
            torch.as_tensor(target, dtype=torch.float32).reshape(1, 3)
        )[0]
        target_tuple = tuple(float(value) for value in target)
        return AmortizedCemContextRecord(
            context_index,
            stable_context_id(
                bank="heldout",
                state_index=int(state_index),
                target_local_m=target_tuple,
                split=split,
            ),
            split,
            "heldout",
            f"heldout:{int(state_index):04d}",
            int(state_index),
            target_number,
            target_tuple,
            tuple(float(value) for value in direction.tolist()),
            theta,
        )

    groups: dict[str, list[AmortizedCemContextRecord]] = {}
    joint_targets = sobol_target_pairs(128, seed=seed + 505)
    groups["JOINT_TEST"] = [
        make_record("JOINT_TEST", 2 * row + target_number, state_index, joint_targets[row, target_number], target_number)
        for row, state_index in enumerate(joint_states.tolist())
        for target_number in range(2)
    ]

    base_targets = np.asarray(
        (
            (0.86, -0.13, -0.11),
            (1.04, 0.13, 0.01),
            (0.88, 0.12, -0.01),
            (1.02, -0.12, -0.09),
        ),
        dtype=np.float32,
    )
    jitter = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed + 606).draw(32).numpy()
    jitter = (jitter - 0.5) * np.asarray((0.01, 0.01, 0.006), dtype=np.float32)
    target_records = []
    for row, state_index in enumerate(joint_states[:32].tolist()):
        for target_number in range(4):
            target = base_targets[target_number] + jitter[row]
            target_records.append(
                make_record(
                    "TARGET_CONDITIONING",
                    4 * row + target_number,
                    state_index,
                    target,
                    target_number,
                )
            )
    groups["TARGET_CONDITIONING"] = target_records

    state_targets = sobol_target_pairs(16, seed=seed + 707).reshape(32, 3)
    state_records = []
    for target_number, target in enumerate(state_targets):
        for state_number in range(4):
            state_index = int(joint_states[(target_number * 4 + state_number) % joint_states.size])
            state_records.append(
                make_record(
                    "STATE_CONDITIONING",
                    4 * target_number + state_number,
                    state_index,
                    target,
                    target_number,
                )
            )
    groups["STATE_CONDITIONING"] = state_records

    rng = np.random.default_rng(seed + 808)
    edge_records = []
    for row, state_index in enumerate(edge_states.tolist()):
        for target_number in range(2):
            bits = rng.integers(0, 2, size=3)
            target = np.where(
                bits,
                np.asarray((1.05, 0.15, 0.02)),
                np.asarray((0.85, -0.15, -0.12)),
            ).astype(np.float32)
            edge_records.append(
                make_record(
                    "EDGE_TEST",
                    2 * row + target_number,
                    state_index,
                    target,
                    target_number,
                )
            )
    groups["EDGE_TEST"] = edge_records
    return groups


def save_context_table(
    path: str | Path,
    records: list[AmortizedCemContextRecord],
    context_tensors: np.ndarray,
) -> None:
    contexts = np.asarray(context_tensors, dtype=np.float32)
    if contexts.shape != (len(records), POLICY_CONTEXT_DIM):
        raise ValueError("Context table must align records with 83-D tensors.")
    np.savez_compressed(
        Path(path),
        schema=np.asarray("amortized_cem_context_table_v1"),
        contexts=contexts,
        context_indices=np.asarray([record.context_index for record in records], dtype=np.int64),
        context_ids=np.asarray([record.context_id for record in records]),
        state_ids=np.asarray([record.state_id for record in records]),
        splits=np.asarray([record.split for record in records]),
        theta=np.asarray([record.theta_nominal for record in records], dtype=np.float32),
    )


def compact_outcome(metrics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    tip = float(metrics.get("best_event_tip_distance_m", metrics.get("minimum_tip_target_distance_m", 10.0)))
    if not math.isfinite(tip):
        tip = 10.0
    directed = float(metrics.get("best_event_directed_speed_m_s", 0.0))
    angle = float(metrics.get("best_event_direction_angle_deg", 180.0))
    if not math.isfinite(angle):
        angle = 180.0
    task_cost = float(metrics.get("task_cost", metrics.get("cost", 1.0e6)))
    continuous = np.asarray(
        (
            -task_cost,
            math.log(max(tip, 0.0) + 1.0e-4),
            directed if math.isfinite(directed) else 0.0,
            math.cos(math.radians(angle)),
            float(metrics.get("maximum_uav_displacement_m", 1.0e6)),
            float(metrics.get("maximum_uav_speed_m_s", 1.0e6)),
            float(metrics.get("maximum_command_acceleration_m_s2", 1.0e6)),
        ),
        dtype=np.float32,
    )
    continuous = np.nan_to_num(continuous, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    binary = np.asarray(
        (
            float(metrics.get("first_entry_marker") == 10),
            float(bool(metrics.get("finite", False))),
            float(bool(metrics.get("success", False))),
        ),
        dtype=np.float32,
    )
    return continuous, binary


def deduplicate_actions(actions: np.ndarray, *, tolerance: float = 1.0e-6) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM:
        raise ValueError("Actions must have shape Nx49.")
    keep: list[int] = []
    for index in range(value.shape[0]):
        if not keep or np.min(np.max(np.abs(value[keep] - value[index]), axis=1)) > tolerance:
            keep.append(index)
    return np.asarray(keep, dtype=np.int64)


def write_context_shards(
    teacher_directory: str | Path,
    scorer_directory: str | Path,
    *,
    record: AmortizedCemContextRecord,
    successful_actions: np.ndarray,
    successful_metrics: list[dict[str, Any]],
    successful_provenance: list[dict[str, Any]],
    scorer_actions: np.ndarray,
    scorer_metrics: list[dict[str, Any]],
    scorer_provenance: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    teacher_root, scorer_root = Path(teacher_directory), Path(scorer_directory)
    teacher_root.mkdir(parents=True, exist_ok=True)
    scorer_root.mkdir(parents=True, exist_ok=True)
    teacher_entry: dict[str, Any] | None = None
    success = np.asarray(successful_actions, dtype=np.float32).reshape(-1, ACTION_DIM)
    if success.shape[0]:
        teacher_path = teacher_root / f"context_{record.context_index:06d}.npz"
        np.savez_compressed(
            teacher_path,
            schema=np.asarray("amortized_cem_diffusion_teacher_shard_v1"),
            context_index=np.asarray(record.context_index, dtype=np.int64),
            normalized_actions=success,
            metrics_json=np.asarray([json.dumps(item, sort_keys=True) for item in successful_metrics]),
            provenance_json=np.asarray([json.dumps(item, sort_keys=True) for item in successful_provenance]),
        )
        teacher_entry = {
            "context_index": record.context_index,
            "context_id": record.context_id,
            "split": record.split,
            "row_count": int(success.shape[0]),
            "path": teacher_path.name,
            "sha256": sha256_file(teacher_path),
        }
    scorer = np.asarray(scorer_actions, dtype=np.float32).reshape(-1, ACTION_DIM)
    continuous, binary = zip(*(compact_outcome(item) for item in scorer_metrics), strict=True)
    scorer_path = scorer_root / f"context_{record.context_index:06d}.npz"
    np.savez_compressed(
        scorer_path,
        schema=np.asarray("amortized_cem_outcome_scorer_shard_v1"),
        context_index=np.asarray(record.context_index, dtype=np.int64),
        normalized_actions=scorer,
        continuous_targets=np.stack(continuous).astype(np.float32),
        binary_targets=np.stack(binary).astype(np.float32),
        continuous_target_names=np.asarray(CONTINUOUS_OUTCOME_NAMES),
        binary_target_names=np.asarray(BINARY_OUTCOME_NAMES),
        provenance_json=np.asarray([json.dumps(item, sort_keys=True) for item in scorer_provenance]),
    )
    scorer_entry = {
        "context_index": record.context_index,
        "context_id": record.context_id,
        "split": record.split,
        "row_count": int(scorer.shape[0]),
        "path": scorer_path.name,
        "sha256": sha256_file(scorer_path),
    }
    return teacher_entry, scorer_entry


def load_teacher_rows(
    context_table_path: str | Path,
    shard_directory: str | Path,
    manifest: dict[str, Any],
    *,
    split: str,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    with np.load(context_table_path, allow_pickle=False) as archive:
        contexts = np.asarray(archive["contexts"], dtype=np.float32)
    actions: dict[int, np.ndarray] = {}
    for entry in manifest["shards"]:
        if entry["split"] != split:
            continue
        with np.load(Path(shard_directory) / entry["path"], allow_pickle=False) as shard:
            actions[int(entry["context_index"])] = np.asarray(
                shard["normalized_actions"], dtype=np.float32
            )
    return contexts, actions


class ContextBalancedTeacherSampler:
    def __init__(
        self,
        contexts: np.ndarray,
        actions_by_context: dict[int, np.ndarray],
        *,
        seed: int,
    ) -> None:
        self.contexts = np.asarray(contexts, dtype=np.float32)
        self.actions = {
            int(key): np.asarray(value, dtype=np.float32) for key, value in actions_by_context.items()
        }
        self.context_indices = np.asarray(sorted(self.actions), dtype=np.int64)
        if not self.context_indices.size:
            raise ValueError("Teacher sampler requires at least one solved context.")
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_indices = self.rng.choice(self.context_indices, size=batch_size, replace=True)
        action_rows = [
            self.actions[int(index)][self.rng.integers(self.actions[int(index)].shape[0])]
            for index in context_indices
        ]
        return (
            torch.from_numpy(self.contexts[context_indices]),
            torch.from_numpy(np.stack(action_rows).astype(np.float32)),
            torch.from_numpy(context_indices.copy()),
        )


def pending_context_records(
    records: Iterable[AmortizedCemContextRecord],
    progress_rows: Iterable[dict[str, Any]],
    *,
    progress_id_key: str = "context_id",
) -> list[AmortizedCemContextRecord]:
    """Return only contexts absent from a durable generation manifest."""

    completed = {str(row[progress_id_key]) for row in progress_rows}
    return [record for record in records if record.context_id not in completed]


def load_scorer_rows(
    context_table_path: str | Path,
    shard_directory: str | Path,
    manifest: dict[str, Any],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(context_table_path, allow_pickle=False) as archive:
        context_table = np.asarray(archive["contexts"], dtype=np.float32)
    contexts, actions, continuous, binary = [], [], [], []
    for entry in manifest["shards"]:
        if entry["split"] != split:
            continue
        with np.load(Path(shard_directory) / entry["path"], allow_pickle=False) as shard:
            count = int(shard["normalized_actions"].shape[0])
            contexts.append(np.repeat(context_table[int(entry["context_index"])][None], count, axis=0))
            actions.append(np.asarray(shard["normalized_actions"], dtype=np.float32))
            continuous.append(np.asarray(shard["continuous_targets"], dtype=np.float32))
            binary.append(np.asarray(shard["binary_targets"], dtype=np.float32))
    if not actions:
        raise ValueError(f"No scorer rows found for split {split}.")
    return tuple(np.concatenate(items, axis=0) for items in (contexts, actions, continuous, binary))
