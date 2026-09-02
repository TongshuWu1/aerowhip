"""Deterministic, state-disjoint episode initialization for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch

from learning.state_bank import InitialStateBank, StateBankBatch, initial_state_bank_from_state
from simulator.state import SimulatorState


_BANK_FIELDS = (
    "uav_position_m",
    "uav_velocity_m_s",
    "uav_orientation_xyzw",
    "uav_angular_velocity_world_rad_s",
    "residual_history",
    "residual_acceleration_m_s2",
    "cable_positions_m",
    "cable_velocities_m_s",
    "command_position_world_m",
    "command_velocity_world_m_s",
    "command_yaw_world_rad",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_fingerprints(bank: InitialStateBank) -> set[str]:
    """Hash complete causal states to prove exact train/validation disjointness."""

    fingerprints: set[str] = set()
    for row in range(len(bank)):
        digest = hashlib.sha256()
        for name in _BANK_FIELDS:
            digest.update(getattr(bank, name)[row].contiguous().numpy().tobytes())
        fingerprints.add(digest.hexdigest())
    return fingerprints


def _append_canonical_state(
    training_bank: InitialStateBank,
    canonical_state: SimulatorState,
    *,
    command_yaw_world_rad: float,
) -> InitialStateBank:
    canonical = initial_state_bank_from_state(
        canonical_state,
        command_position_world_m=canonical_state.uav.position_m,
        command_velocity_world_m_s=canonical_state.uav.velocity_m_s,
        command_yaw_world_rad=command_yaw_world_rad,
    )
    values = {
        name: torch.cat(
            (getattr(training_bank, name), getattr(canonical, name)[:1]), dim=0
        )
        for name in _BANK_FIELDS
    }
    return InitialStateBank(
        **values,
        seed=training_bank.seed,
        generation={
            "method": "training_bank_plus_canonical_episode_source_v1",
            "training_bank_generation": training_bank.generation,
        },
    )


@dataclass(slots=True)
class MixedPPOInitialStateSampler:
    """Uniform TRAIN-bank sampling with a fixed canonical fraction per batch."""

    bank: InitialStateBank
    training_bank_count: int
    batch_size: int
    canonical_count: int
    generator: torch.Generator
    manifest: dict[str, Any]

    @property
    def varied_count(self) -> int:
        return self.batch_size - self.canonical_count

    @classmethod
    def create(
        cls,
        *,
        training_bank_path: str | Path,
        training_manifest_path: str | Path,
        validation_bank_path: str | Path,
        validation_manifest_path: str | Path,
        canonical_state: SimulatorState,
        command_yaw_world_rad: float,
        batch_size: int,
        canonical_fraction: float,
        seed: int,
    ) -> "MixedPPOInitialStateSampler":
        if batch_size < 1:
            raise ValueError("PPO initial-state batch size must be positive.")
        if not 0.0 <= canonical_fraction <= 1.0:
            raise ValueError("Canonical initial-state fraction must lie in [0,1].")
        training_path = Path(training_bank_path)
        training_manifest = Path(training_manifest_path)
        validation_path = Path(validation_bank_path)
        validation_manifest = Path(validation_manifest_path)
        if training_path.resolve() == validation_path.resolve():
            raise ValueError("Training and validation state banks must be distinct files.")
        training = InitialStateBank.load(training_path, training_manifest)
        validation = InitialStateBank.load(validation_path, validation_manifest)
        overlap = _row_fingerprints(training).intersection(_row_fingerprints(validation))
        if overlap:
            raise ValueError(
                f"Training and validation state banks contain {len(overlap)} identical states."
            )
        canonical_count = int(round(batch_size * canonical_fraction))
        combined = _append_canonical_state(
            training,
            canonical_state,
            command_yaw_world_rad=command_yaw_world_rad,
        )
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        manifest = {
            "schema": "mixed_ppo_initial_state_sampling_v1",
            "mode": "mixed_canonical_and_physically_propagated_train_bank",
            "batch_size": int(batch_size),
            "canonical_fraction_requested": float(canonical_fraction),
            "canonical_rows_per_batch": canonical_count,
            "training_bank_rows_per_batch": batch_size - canonical_count,
            "training_bank_count": len(training),
            "training_bank_seed": training.seed,
            "training_bank": str(training_path),
            "training_bank_manifest": str(training_manifest),
            "training_bank_sha256": _sha256(training_path),
            "validation_bank_count": len(validation),
            "validation_bank_seed": validation.seed,
            "validation_bank": str(validation_path),
            "validation_bank_manifest": str(validation_manifest),
            "validation_bank_sha256": _sha256(validation_path),
            "exact_state_overlap_count": 0,
            "state_ownership": "TRAIN gradients; VALIDATION checkpoint selection only",
            "training_sampling": "uniform without replacement when possible",
            "batch_shuffle": True,
            "sampler_seed": int(seed),
            "new_state_generation": False,
        }
        return cls(
            bank=combined,
            training_bank_count=len(training),
            batch_size=int(batch_size),
            canonical_count=canonical_count,
            generator=generator,
            manifest=manifest,
        )

    def sample(self, *, device: torch.device | str) -> StateBankBatch:
        if self.varied_count <= self.training_bank_count:
            varied = torch.randperm(
                self.training_bank_count, generator=self.generator
            )[: self.varied_count]
        else:
            varied = torch.randint(
                self.training_bank_count,
                (self.varied_count,),
                generator=self.generator,
            )
        canonical_index = self.training_bank_count
        canonical = torch.full(
            (self.canonical_count,), canonical_index, dtype=torch.int64
        )
        indices = torch.cat((canonical, varied))
        shuffle = torch.randperm(self.batch_size, generator=self.generator)
        return self.bank.select(indices[shuffle], device=device)

    def get_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state.detach().cpu())
