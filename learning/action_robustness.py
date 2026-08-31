"""Deterministic perturbation banks for open-loop action robustness audits."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import torch

from .policy_action import canonicalize_normalized_action


def stable_perturbation_seed(base_seed: int, context_id: str, family: str, level: float) -> int:
    payload = f"{int(base_seed)}:{context_id}:{family}:{float(level):.12g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _smooth_unit_noise(rng: np.random.Generator, count: int) -> np.ndarray:
    values = rng.standard_normal((count, 16, 3))
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    smoothed = np.empty_like(values)
    for sample in range(count):
        for axis in range(3):
            smoothed[sample, :, axis] = np.convolve(
                np.pad(values[sample, :, axis], (2, 2), mode="reflect"),
                kernel,
                mode="valid",
            )
    scale = np.sqrt(np.mean(smoothed**2, axis=(1, 2), keepdims=True))
    return smoothed / np.maximum(scale, 1.0e-12)


def build_action_perturbation_bank(
    teacher_action: np.ndarray | torch.Tensor,
    *,
    context_id: str,
    acceleration_sigmas: Iterable[float],
    duration_offsets_ms: Iterable[float],
    samples_per_level: int,
    base_seed: int,
    duration_range_s: float = 1.35,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Create exact, IID, smooth, and duration-only production actions."""

    teacher = torch.as_tensor(teacher_action, dtype=torch.float64).reshape(49)
    teacher = canonicalize_normalized_action(teacher)
    actions = [teacher.detach().cpu().numpy()]
    labels: list[dict[str, object]] = [
        {
            "family": "EXACT",
            "level": 0.0,
            "sample": 0,
            "requested_acceleration_sigma": 0.0,
            "requested_duration_offset_ms": 0.0,
        }
    ]
    if samples_per_level < 2:
        raise ValueError("At least two perturbations per level are required.")

    for family in ("IID_ACCELERATION", "SMOOTH_ACCELERATION"):
        for sigma in acceleration_sigmas:
            value = float(sigma)
            if value <= 0.0:
                raise ValueError("Acceleration perturbation scales must be positive.")
            rng = np.random.default_rng(
                stable_perturbation_seed(base_seed, context_id, family, value)
            )
            if family == "IID_ACCELERATION":
                noise = rng.standard_normal((samples_per_level, 16, 3))
            else:
                noise = _smooth_unit_noise(rng, samples_per_level)
            offsets = np.zeros((samples_per_level, 49), dtype=np.float64)
            offsets[:, :48] = value * noise.reshape(samples_per_level, 48)
            proposed = teacher[None] + torch.from_numpy(offsets)
            bounded = canonicalize_normalized_action(proposed).detach().cpu().numpy()
            actions.extend(bounded)
            labels.extend(
                {
                    "family": family,
                    "level": value,
                    "sample": sample,
                    "requested_acceleration_sigma": value,
                    "requested_duration_offset_ms": 0.0,
                }
                for sample in range(samples_per_level)
            )

    for milliseconds in duration_offsets_ms:
        value = float(milliseconds)
        if value <= 0.0:
            raise ValueError("Duration perturbation magnitudes must be positive.")
        rng = np.random.default_rng(
            stable_perturbation_seed(base_seed, context_id, "DURATION_ONLY", value)
        )
        signs = np.ones((samples_per_level,), dtype=np.float64)
        signs[: samples_per_level // 2] = -1.0
        rng.shuffle(signs)
        offsets = np.zeros((samples_per_level, 49), dtype=np.float64)
        offsets[:, 48] = signs * 2.0 * (value / 1000.0) / float(duration_range_s)
        proposed = teacher[None] + torch.from_numpy(offsets)
        bounded = canonicalize_normalized_action(proposed).detach().cpu().numpy()
        actions.extend(bounded)
        labels.extend(
            {
                "family": "DURATION_ONLY",
                "level": value,
                "sample": sample,
                "requested_acceleration_sigma": 0.0,
                "requested_duration_offset_ms": float(signs[sample] * value),
            }
            for sample in range(samples_per_level)
        )

    result = np.asarray(actions, dtype=np.float32)
    if result.shape != (len(labels), 49) or not np.isfinite(result).all():
        raise RuntimeError("Perturbation bank is invalid.")
    return result, labels
