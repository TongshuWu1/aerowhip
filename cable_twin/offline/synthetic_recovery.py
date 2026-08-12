"""Deterministic parameter-recovery check for the constrained cable model."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, OfflineSettings, load_settings
from .optimize import (
    FitResult,
    PreparedSequence,
    _fit,
    _make_model,
    _project_fixed_length,
)
from ..shared.dder import DderModel, DderState


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "data" / "offline_dder" / "synthetic_recovery.json"
)


def _initial_planar_curve(settings: OfflineSettings, device: torch.device) -> torch.Tensor:
    """Return an exactly inextensible, gently curved planar cable."""

    node_count = settings.cable.node_count
    segment_length = settings.cable.length_m / (node_count - 1)
    angle_step = 0.10
    radius = segment_length / (2.0 * math.sin(0.5 * angle_step))
    angle = torch.linspace(
        -0.5 * (node_count - 1) * angle_step,
        0.5 * (node_count - 1) * angle_step,
        node_count,
        dtype=torch.float64,
        device=device,
    )
    positions = torch.zeros((1, node_count, 3), dtype=torch.float64, device=device)
    positions[0, :, 0] = radius * torch.sin(angle)
    positions[0, :, 2] = -radius * torch.cos(angle)
    endpoint_height = positions[0, 0, 2].clone()
    positions[0, :, 2] -= endpoint_height
    return positions


def _smoothstep(value: float) -> float:
    clipped = min(max(value, 0.0), 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _pulse(time_s: float, start_s: float, rise_s: float, hold_s: float, fall_s: float) -> float:
    if time_s < start_s:
        return 0.0
    if time_s < start_s + rise_s:
        return _smoothstep((time_s - start_s) / rise_s)
    if time_s < start_s + rise_s + hold_s:
        return 1.0
    if time_s < start_s + rise_s + hold_s + fall_s:
        elapsed = time_s - start_s - rise_s - hold_s
        return 1.0 - _smoothstep(elapsed / fall_s)
    return 0.0


def _boundary_positions(base: torch.Tensor, time_s: float) -> torch.Tensor:
    """Prescribe slow shape changes and a faster change followed by settling."""

    boundary = base.clone()
    widening = 0.010 * _pulse(time_s, 0.25, 0.35, 0.12, 0.35)
    boundary[:, 0, 0] -= widening
    boundary[:, 1, 0] += widening
    tilt = 0.018 * _pulse(time_s, 1.15, 0.12, 0.18, 0.12)
    boundary[:, 0, 2] += tilt
    boundary[:, 1, 2] -= tilt
    return boundary


def _simulate(
    settings: OfflineSettings,
    *,
    true_ei: float,
    true_cb: float,
    frame_count: int,
    frame_rate_hz: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = _make_model(settings, ei=true_ei, cb=true_cb)
    position = _initial_planar_curve(settings, device)
    edges = position[:, 1:] - position[:, :-1]
    edge_tangent = edges / torch.linalg.vector_norm(edges, dim=-1, keepdim=True)
    node_tangent = torch.cat(
        (
            edge_tangent[:, :1],
            edge_tangent[:, :-1] + edge_tangent[:, 1:],
            edge_tangent[:, -1:],
        ),
        dim=1,
    )
    node_tangent /= torch.linalg.vector_norm(node_tangent, dim=-1, keepdim=True)
    node_normal = torch.stack(
        (
            -node_tangent[:, :, 2],
            torch.zeros_like(node_tangent[:, :, 1]),
            node_tangent[:, :, 0],
        ),
        dim=2,
    )
    phase = torch.linspace(
        0.0, math.pi, settings.cable.node_count, dtype=torch.float64, device=device
    )[None, :, None]
    initial_velocity = 0.25 * torch.sin(phase) * node_normal
    initial_velocity = model.project_velocities(
        position,
        initial_velocity,
        torch.zeros((1, 2, 3), dtype=torch.float64, device=device),
    )
    state = DderState(position, initial_velocity)
    base_boundary = position[:, (0, -1)].clone()
    dt_s = 1.0 / frame_rate_hz
    positions = [state.positions_m[0].detach().cpu().numpy()]
    velocities = [state.velocities_m_s[0].detach().cpu().numpy()]
    for frame in range(1, frame_count):
        boundary = _boundary_positions(base_boundary, frame * dt_s)
        state = model.step_unchecked(state, boundary, dt_s)
        positions.append(state.positions_m[0].detach().cpu().numpy())
        velocities.append(state.velocities_m_s[0].detach().cpu().numpy())
    timestamp_ns = 1_000_000_000 + np.rint(
        np.arange(frame_count, dtype=np.float64) * 1.0e9 / frame_rate_hz
    ).astype(np.int64)
    return np.asarray(positions), np.asarray(velocities), timestamp_ns


def _prepare_synthetic_sequence(
    settings: OfflineSettings,
    positions_m: np.ndarray,
    velocities_m_s: np.ndarray,
    timestamp_ns: np.ndarray,
    *,
    noise_std_m: float,
    seed: int,
    device: torch.device,
) -> PreparedSequence:
    observations = np.asarray(positions_m, dtype=np.float64).copy()
    if noise_std_m > 0.0:
        generator = np.random.default_rng(seed)
        observations += generator.normal(0.0, noise_std_m, observations.shape)
        observations[:, :, 1] = 0.0

    valid = np.ones(len(observations), dtype=bool)
    observed_velocity = np.asarray(velocities_m_s, dtype=np.float64).copy()
    valid &= np.all(np.isfinite(observations), axis=(1, 2))
    valid &= np.all(np.isfinite(observed_velocity), axis=(1, 2))
    separation = np.linalg.norm(
        observations[:, -1] - observations[:, 0], axis=1
    )
    valid &= separation <= settings.cable.length_m

    nominal_ei = math.sqrt(
        settings.optimization.ei_min_n_m2 * settings.optimization.ei_max_n_m2
    )
    nominal_cb = math.sqrt(
        settings.optimization.cb_min_n_m2_s
        * settings.optimization.cb_max_n_m2_s
    )
    nominal_model = _make_model(settings, ei=nominal_ei, cb=nominal_cb)
    initial_positions = _project_fixed_length(
        observations, valid, nominal_model, device
    )
    indices = np.flatnonzero(valid)
    q = torch.as_tensor(
        initial_positions[indices], dtype=torch.float64, device=device
    )
    v = torch.as_tensor(
        observed_velocity[indices], dtype=torch.float64, device=device
    )
    projected_velocity = nominal_model.project_velocities(
        q, v, v[:, (0, -1)]
    )
    initial_velocities = np.full_like(observed_velocity, np.nan)
    initial_velocities[indices] = projected_velocity.detach().cpu().numpy()
    initialization_shift = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.linalg.norm(
                        initial_positions[valid] - observations[valid], axis=2
                    )
                )
            )
        )
    )
    synthetic_path = Path("synthetic_parameter_recovery")
    return PreparedSequence(
        svo_path=synthetic_path.with_suffix(".svo2"),
        observation_path=synthetic_path.with_suffix(".npz"),
        timestamp_ns=timestamp_ns,
        observation_positions_m=observations,
        observation_velocities_m_s=observed_velocity,
        initial_positions_m=initial_positions,
        initial_velocities_m_s=initial_velocities,
        valid=valid,
        initialization_shift_m=initialization_shift,
        pixel_scale_m=max(noise_std_m, 1.0e-6),
    )


def _relative_error(estimate: float, reference: float) -> float:
    return abs(estimate - reference) / reference


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(arguments: argparse.Namespace) -> Path:
    settings = load_settings(arguments.config)
    settings = replace(
        settings,
        optimization=replace(
            settings.optimization,
            device=arguments.device or settings.optimization.device,
            optimizer_iterations=arguments.iterations,
            window_stride=settings.optimization.window_frames,
        ),
    )
    device = torch.device(settings.optimization.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the synthetic recovery check but is unavailable.")
    if settings.optimization.optimizer_iterations < 3:
        raise ValueError(
            "Synthetic recovery requires at least three optimizer iterations."
        )
    if arguments.frames < settings.optimization.window_frames:
        raise ValueError("Synthetic frame count must cover at least one fit window.")
    if arguments.frame_rate_hz <= 0.0:
        raise ValueError("Synthetic frame rate must be positive.")
    if arguments.noise_mm < 0.0:
        raise ValueError("Synthetic observation noise must be non-negative.")

    true_ei = float(arguments.true_ei)
    true_cb = float(arguments.true_cb)
    if not (
        settings.optimization.ei_min_n_m2 < true_ei < settings.optimization.ei_max_n_m2
    ):
        raise ValueError("True synthetic EI must lie inside the configured fit bounds.")
    if not (
        settings.optimization.cb_min_n_m2_s < true_cb < settings.optimization.cb_max_n_m2_s
    ):
        raise ValueError("True synthetic Cb must lie inside the configured fit bounds.")

    # Generation is a tiny sequential problem; CPU avoids one small CUDA launch
    # per frame. The batched production fitter below still uses the requested
    # device.
    positions, true_velocities, timestamp_ns = _simulate(
        settings,
        true_ei=true_ei,
        true_cb=true_cb,
        frame_count=arguments.frames,
        frame_rate_hz=arguments.frame_rate_hz,
        device=torch.device("cpu"),
    )
    sequence = _prepare_synthetic_sequence(
        settings,
        positions,
        true_velocities,
        timestamp_ns,
        noise_std_m=arguments.noise_mm * 1.0e-3,
        seed=arguments.seed,
        device=device,
    )
    result: FitResult = _fit((sequence,), settings)
    payload: dict[str, object] = {
        "method": "same_model_parameter_recovery_check",
        "interpretation": (
            "Uses exact simulated window-start velocities to isolate the dynamics and "
            "parameter optimizer. This is a necessary implementation sanity check, not "
            "an evaluation of RGB velocity estimation or independent physical validation."
        ),
        "seed": arguments.seed,
        "device": str(device),
        "frames": arguments.frames,
        "frame_rate_hz": arguments.frame_rate_hz,
        "observation_noise_std_m": arguments.noise_mm * 1.0e-3,
        "optimizer_iterations": settings.optimization.optimizer_iterations,
        "window_frames": settings.optimization.window_frames,
        "window_stride": settings.optimization.window_stride,
        "true": {
            "bending_stiffness_n_m2": true_ei,
            "bending_damping_n_m2_s": true_cb,
        },
        "recovered": {
            "bending_stiffness_n_m2": result.bending_stiffness_n_m2,
            "bending_damping_n_m2_s": result.bending_damping_n_m2_s,
        },
        "relative_error": {
            "bending_stiffness": _relative_error(
                result.bending_stiffness_n_m2, true_ei
            ),
            "bending_damping": _relative_error(
                result.bending_damping_n_m2_s, true_cb
            ),
        },
        "rollout_rmse_m": result.train_curve_residual_m,
        "initialization_constraint_shift_m": sequence.initialization_shift_m,
        "elapsed_s": result.elapsed_s,
    }
    output = Path(arguments.output).expanduser().resolve()
    _atomic_json(output, payload)
    print(
        f"Saved synthetic recovery report: {output} | "
        f"EI true/recovered={true_ei:.6g}/{result.bending_stiffness_n_m2:.6g}Nm2 | "
        f"Cb true/recovered={true_cb:.6g}/{result.bending_damping_n_m2_s:.6g}Nm2s | "
        f"RMSE={1000.0 * result.train_curve_residual_m:.3f}mm",
        flush=True,
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate known cable motion and recover EI/Cb with the production fitter."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--frame-rate-hz", type=float, default=30.0)
    parser.add_argument("--true-ei", type=float, default=8.0e-6)
    parser.add_argument("--true-cb", type=float, default=2.0e-8)
    parser.add_argument("--noise-mm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
