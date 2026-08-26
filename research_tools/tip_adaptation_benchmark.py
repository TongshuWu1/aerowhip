"""Run the first tip-only cable-parameter adaptation benchmark.

This is a synthetic, known-start identification check.  It does not claim a
full cable-state observer, real-flight adaptation, or physical validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import uuid

import numpy as np
import torch

import drone_mpc.adaptation as adaptation_module
from drone_mpc.adaptation import (
    SyntheticExcitationSettings,
    TipAdaptationResult,
    TipAdaptationSettings,
    estimate_tip_only_parameters,
    fixed_solver_model,
    make_synthetic_tip_measurements,
    predict_tip_positions,
)
from drone_mpc.model import CableModelSnapshot, load_cable_model
from optitrack_offline.config import DEFAULT_MODEL_PATH


BENCHMARK_SCHEMA = "known_start_tip_only_adaptation_benchmark_v1"
NODE_COUNT = 15
SUBSTEPS = 2
CONSTRAINT_ITERATIONS = 4
HELD_OUT_PHASE_RAD = 0.73

CASES: tuple[tuple[str, float, float, bool], ...] = (
    ("matched", 1.0, 1.0, False),
    ("ei_plus_20_percent", 1.2, 1.0, False),
    ("cb_minus_30_percent", 1.0, 0.7, False),
    ("coupled_ei_plus_20_cb_minus_20_percent", 1.2, 0.8, False),
    ("stationary", 1.0, 1.0, True),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run synthetic known-start, attachment/free-tip-only EI/Cb "
            "identification cases with one fixed DDER solver."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/drone_mpc/tip_adaptation_benchmark.json"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds per case.")
    parser.add_argument("--dt", type=float, default=0.02, help="DDER frame interval in seconds.")
    parser.add_argument(
        "--maximum-acceleration",
        type=float,
        default=6.0,
        help="Peak norm of the deterministic attachment excitation in m/s^2.",
    )
    parser.add_argument(
        "--tip-noise-mm",
        type=float,
        default=0.0,
        help="Optional independent synthetic tip-position noise standard deviation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=6)
    return parser


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _result_payload(
    result: TipAdaptationResult,
    *,
    hidden_model: CableModelSnapshot,
    true_ei_scale: float,
    true_cb_scale: float,
    measurement_sha256: str,
) -> dict[str, object]:
    return {
        "adaptation_schema": result.schema,
        "nominal_model_sha256": result.model_sha256,
        "provisional_model": result.provisional_model,
        "hidden_model_sha256": hidden_model.sha256,
        "measurement_sha256": measurement_sha256,
        "true": {
            "ei_scale": true_ei_scale,
            "cb_scale": true_cb_scale,
            "ei_n_m2": hidden_model.bending_stiffness_n_m2,
            "cb_n_m2_s": hidden_model.bending_damping_n_m2_s,
        },
        "estimate": {
            "update_applied": result.update_applied,
            "freeze_reason": result.freeze_reason,
            "ei_scale": result.estimated_ei_scale,
            "cb_scale": result.estimated_cb_scale,
            "ei_n_m2": result.estimated_ei_n_m2,
            "cb_n_m2_s": result.estimated_cb_n_m2_s,
            "ei_relative_error_percent": 100.0
            * (result.estimated_ei_scale / true_ei_scale - 1.0),
            "cb_relative_error_percent": 100.0
            * (result.estimated_cb_scale / true_cb_scale - 1.0),
        },
        "tip_error": {
            "initial_rmse_mm": 1000.0 * result.initial_tip_rmse_m,
            "final_rmse_mm": 1000.0 * result.final_tip_rmse_m,
        },
        "identifiability": {
            "normalized_sensitivity_singular_values": list(
                result.normalized_sensitivity_singular_values
            ),
            "relative_information_eigenvalues": list(
                result.relative_information_eigenvalues
            ),
            "condition": _finite_or_none(result.information_condition),
            "log_parameter_standard_deviations": list(
                result.log_parameter_standard_deviations
            ),
            "log_parameter_correlation": result.log_parameter_correlation,
            "ei_scale_95_interval": list(result.ei_scale_95_interval),
            "cb_scale_95_interval": list(result.cb_scale_95_interval),
        },
        "optimization": {
            "best_iteration": result.best_iteration,
            "evaluations": result.evaluations,
            "loss_history": list(result.loss_history),
            "runtime_s": result.runtime_s,
        },
        "record": {
            "frame_count": result.frame_count,
            "duration_s": result.duration_s,
        },
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable; pass --device cpu explicitly.")
    if arguments.tip_noise_mm < 0.0:
        raise ValueError("--tip-noise-mm must be non-negative.")
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative.")

    source_model = load_cable_model(arguments.model)
    nominal_model = fixed_solver_model(
        source_model,
        node_count=NODE_COUNT,
        substeps=SUBSTEPS,
        constraint_iterations=CONSTRAINT_ITERATIONS,
    )
    adaptation_settings = TipAdaptationSettings(
        optimizer_iterations=arguments.iterations,
    )

    print(
        f"Known-start tip-only benchmark | {NODE_COUNT} nodes | "
        f"{SUBSTEPS} fixed substeps | {arguments.device}"
    )
    if source_model.provisional:
        print(
            "WARNING: the loaded EI/Cb source is the provisional two-holder transfer; "
            "this run is algorithm development, not physical validation."
        )

    case_payloads: list[dict[str, object]] = []
    for case_index, (name, ei_scale, cb_scale, stationary) in enumerate(CASES):
        hidden_model = fixed_solver_model(
            source_model,
            node_count=NODE_COUNT,
            substeps=SUBSTEPS,
            constraint_iterations=CONSTRAINT_ITERATIONS,
            bending_stiffness_scale=ei_scale,
            bending_damping_scale=cb_scale,
        )
        excitation = SyntheticExcitationSettings(
            duration_s=arguments.duration,
            dt_s=arguments.dt,
            maximum_acceleration_m_s2=(
                0.0 if stationary else arguments.maximum_acceleration
            ),
            tip_noise_std_m=arguments.tip_noise_mm / 1000.0,
            noise_seed=arguments.seed + case_index,
        )
        measurements = make_synthetic_tip_measurements(
            hidden_model,
            excitation,
            device=arguments.device,
        )
        result = estimate_tip_only_parameters(
            nominal_model,
            measurements,
            adaptation_settings,
            device=arguments.device,
        )
        measurement_sha256 = _array_sha256(
            measurements.time_s,
            measurements.attachment_positions_m,
            measurements.free_tip_positions_m,
            measurements.commanded_accelerations_m_s2,
        )
        case_payload = _result_payload(
            result,
            hidden_model=hidden_model,
            true_ei_scale=ei_scale,
            true_cb_scale=cb_scale,
            measurement_sha256=measurement_sha256,
        )
        case_payload["case"] = name
        case_payload["excitation"] = asdict(excitation)

        held_out_excitation = SyntheticExcitationSettings(
            duration_s=arguments.duration,
            dt_s=arguments.dt,
            maximum_acceleration_m_s2=(
                0.0 if stationary else arguments.maximum_acceleration
            ),
            tip_noise_std_m=arguments.tip_noise_mm / 1000.0,
            noise_seed=arguments.seed + 10_000 + case_index,
            excitation_phase_rad=HELD_OUT_PHASE_RAD,
        )
        held_out = make_synthetic_tip_measurements(
            hidden_model,
            held_out_excitation,
            device=arguments.device,
        )
        nominal_held_out_tip = predict_tip_positions(
            nominal_model,
            held_out,
            device=arguments.device,
        )
        adapted_held_out_tip = predict_tip_positions(
            nominal_model,
            held_out,
            ei_scale=result.estimated_ei_scale,
            cb_scale=result.estimated_cb_scale,
            device=arguments.device,
        )
        held_out_nominal_rmse_mm = 1000.0 * float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            nominal_held_out_tip - held_out.free_tip_positions_m
                        ),
                        axis=1,
                    )
                )
            )
        )
        held_out_adapted_rmse_mm = 1000.0 * float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            adapted_held_out_tip - held_out.free_tip_positions_m
                        ),
                        axis=1,
                    )
                )
            )
        )
        case_payload["held_out_phase_shifted"] = {
            "excitation": asdict(held_out_excitation),
            "measurement_sha256": _array_sha256(
                held_out.time_s,
                held_out.attachment_positions_m,
                held_out.free_tip_positions_m,
                held_out.commanded_accelerations_m_s2,
            ),
            "nominal_tip_rmse_mm": held_out_nominal_rmse_mm,
            "adapted_tip_rmse_mm": held_out_adapted_rmse_mm,
        }
        case_payloads.append(case_payload)

        status = "updated" if result.update_applied else "frozen"
        condition = (
            f"{result.information_condition:.1f}"
            if math.isfinite(result.information_condition)
            else "inf"
        )
        print(
            f"{name}: {status} | EI {result.estimated_ei_scale:.3f} "
            f"(true {ei_scale:.3f}) | Cb {result.estimated_cb_scale:.3f} "
            f"(true {cb_scale:.3f}) | tip RMSE "
            f"{1000.0 * result.initial_tip_rmse_m:.2f}->"
            f"{1000.0 * result.final_tip_rmse_m:.2f} mm | held-out "
            f"{held_out_nominal_rmse_mm:.2f}->{held_out_adapted_rmse_mm:.2f} mm | "
            f"info cond={condition}"
        )

    payload: dict[str, object] = {
        "schema": BENCHMARK_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "claim": "synthetic known-start tip-only EI/Cb parameter identification",
            "measurement_contract": [
                "timestamps",
                "attachment positions",
                "free-tip positions",
                "commanded attachment accelerations",
            ],
            "not_claimed": [
                "full online moving-horizon cable-state estimation",
                "real-flight adaptation",
                "physical parameter validation",
                "learned-policy adaptation",
            ],
        },
        "source_model": {
            "path": str(source_model.source_path),
            "sha256": source_model.sha256,
            "schema": source_model.payload.get("schema"),
            "provisional": source_model.provisional,
            "provenance_note": source_model.provenance_note,
            "ei_n_m2": source_model.bending_stiffness_n_m2,
            "cb_n_m2_s": source_model.bending_damping_n_m2_s,
        },
        "nominal_solver_model": {
            "sha256": nominal_model.sha256,
            "node_count": NODE_COUNT,
            "substeps": SUBSTEPS,
            "constraint_iterations": CONSTRAINT_ITERATIONS,
            "dtype": "float64",
        },
        "settings": {
            "adaptation": asdict(adaptation_settings),
            "duration_s": arguments.duration,
            "dt_s": arguments.dt,
            "maximum_acceleration_m_s2": arguments.maximum_acceleration,
            "tip_noise_std_m": arguments.tip_noise_mm / 1000.0,
            "noise_seed": arguments.seed,
            "held_out_excitation_phase_rad": HELD_OUT_PHASE_RAD,
            "device": arguments.device,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if arguments.device == "cuda"
                else None
            ),
            "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        },
        "implementation": {
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": _file_sha256(Path(__file__).resolve()),
            "adaptation_path": str(Path(adaptation_module.__file__).resolve()),
            "adaptation_sha256": _file_sha256(
                Path(adaptation_module.__file__).resolve()
            ),
        },
        "cases": case_payloads,
    }
    output = _atomic_json(arguments.output, payload)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
