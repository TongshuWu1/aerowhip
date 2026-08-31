"""Explicit simulator settings and the only uncertain cable parameters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from .cable.config import CableConfiguration


Scalar = float | torch.Tensor


def _validate_positive_scalar_or_batch(name: str, value: Scalar) -> None:
    """Validate one scalar or one population vector of positive parameters."""

    tensor = torch.as_tensor(value)
    if tensor.ndim > 1 or tensor.numel() < 1:
        raise ValueError(f"{name} must be scalar or a one-dimensional batch.")
    detached = tensor.detach()
    if not bool(torch.isfinite(detached).all()) or not bool((detached > 0.0).all()):
        raise ValueError(f"{name} must contain finite positive values.")


def _uav_parameter_tensor(
    value: Scalar,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Return a scalar or Bx1 tensor suitable for Bx3 UAV arithmetic."""

    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 0:
        return tensor
    if tensor.shape != (reference.shape[0],):
        raise ValueError(
            f"Batched {name} must contain one value per UAV state row."
        )
    return tensor[:, None]


@dataclass(frozen=True, slots=True)
class CableParameters:
    """Minimal interpretable uncertain cable model."""

    EI: Scalar
    Cb: Scalar

    def __post_init__(self) -> None:
        for name, value in (("EI", self.EI), ("Cb", self.Cb)):
            _validate_positive_scalar_or_batch(name, value)

    def tensors(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(self.EI, dtype=reference.dtype, device=reference.device),
            torch.as_tensor(self.Cb, dtype=reference.dtype, device=reference.device),
        )


@dataclass(frozen=True, slots=True)
class UAVResponseParameters:
    """Effective FullState command-response parameters."""

    K_p: Scalar
    K_v: Scalar
    k_a: Scalar
    K_R: Scalar
    K_omega: Scalar

    def __post_init__(self) -> None:
        for name, value in (
            ("K_p", self.K_p),
            ("K_v", self.K_v),
            ("k_a", self.k_a),
            ("K_R", self.K_R),
            ("K_omega", self.K_omega),
        ):
            _validate_positive_scalar_or_batch(name, value)

    def tensors(
        self, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            _uav_parameter_tensor(value, reference, name)
            for name, value in (
                ("K_p", self.K_p),
                ("K_v", self.K_v),
                ("k_a", self.k_a),
                ("K_R", self.K_R),
                ("K_omega", self.K_omega),
            )
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SimulatorParameters:
    cable: CableParameters
    uav: UAVResponseParameters


@dataclass(frozen=True, slots=True)
class DebugTrajectorySettings:
    sinusoidal_amplitude_m: float
    sinusoidal_frequency_hz: float
    aggressive_amplitude_m: float
    aggressive_frequency_hz: float
    attitude_amplitude_deg: float
    attitude_frequency_hz: float


@dataclass(frozen=True, slots=True)
class SimulatorSettings:
    dt_s: float
    device: str
    initial_uav_position_m: tuple[float, float, float]
    initial_uav_orientation_xyzw: tuple[float, float, float, float]
    attachment_offset_body_m: tuple[float, float, float]
    attachment_tangent_body: tuple[float, float, float]
    cable_configuration: CableConfiguration
    parameters: SimulatorParameters
    debug: DebugTrajectorySettings

    @classmethod
    def load(cls, path: str | Path) -> "SimulatorSettings":
        source = Path(path).expanduser().resolve()
        payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
        simulation = payload["simulation"]
        parameters = payload["parameters"]
        cable_parameters = parameters.get("cable", parameters)
        uav_parameters = parameters["uav"]
        debug = payload["debug_trajectories"]
        return cls(
            dt_s=float(simulation["dt_s"]),
            device=str(simulation["device"]),
            initial_uav_position_m=tuple(
                simulation.get(
                    "initial_uav_position_m", simulation.get("initial_root_position_m")
                )
            ),
            initial_uav_orientation_xyzw=tuple(
                simulation.get("initial_uav_orientation_xyzw", [0.0, 0.0, 0.0, 1.0])
            ),
            attachment_offset_body_m=tuple(payload["attachment_offset_body_m"]),
            attachment_tangent_body=tuple(payload["attachment_tangent_body"]),
            cable_configuration=CableConfiguration.from_mapping(payload["cable"]),
            parameters=SimulatorParameters(
                cable=CableParameters(
                    EI=float(cable_parameters["EI"]),
                    Cb=float(cable_parameters["Cb"]),
                ),
                uav=UAVResponseParameters(
                    K_p=float(uav_parameters["K_p"]),
                    K_v=float(uav_parameters["K_v"]),
                    k_a=float(uav_parameters["k_a"]),
                    K_R=float(uav_parameters["K_R"]),
                    K_omega=float(uav_parameters["K_omega"]),
                ),
            ),
            debug=DebugTrajectorySettings(
                sinusoidal_amplitude_m=float(debug["sinusoidal_amplitude_m"]),
                sinusoidal_frequency_hz=float(debug["sinusoidal_frequency_hz"]),
                aggressive_amplitude_m=float(debug["aggressive_amplitude_m"]),
                aggressive_frequency_hz=float(debug["aggressive_frequency_hz"]),
                attitude_amplitude_deg=float(debug["attitude_amplitude_deg"]),
                attitude_frequency_hz=float(debug["attitude_frequency_hz"]),
            ),
        )

    @property
    def initial_root_position_m(self) -> tuple[float, float, float]:
        """Compatibility alias for the zero-offset Milestone-1 configuration."""

        return self.initial_uav_position_m

    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return device
