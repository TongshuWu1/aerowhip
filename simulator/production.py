"""Explicit construction of the single active UAV-residual-DDER predictor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from experimental_data.io import sha256_file

from .cable.cuda_fixed_pcg import VALIDATED_OPTIMIZED_DAMPING_BACKEND
from .parameters import SimulatorSettings
from .simulator import CoupledSimulator
from .uav.model import FullStateUAVModel
from .uav.residual import CausalTranslationalResidual


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODEL_MANIFEST = PROJECT_ROOT / "config" / "active_model.json"


def load_active_model_manifest(path: Path = ACTIVE_MODEL_MANIFEST) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "aerial_cable_active_model_v1":
        raise ValueError(f"Unsupported active-model manifest: {path}")
    if payload.get("selection_policy") != "explicit_manifest_only_never_newest_timestamp":
        raise ValueError("Active model must be selected by an explicit pinned manifest.")
    return payload


def _resolve(relative: str) -> Path:
    result = (PROJECT_ROOT / relative).resolve()
    if PROJECT_ROOT not in result.parents:
        raise ValueError(f"Active-model artifact escapes the project: {relative}")
    return result


def resolve_portable_artifact_reference(reference: str | Path) -> Path:
    """Resolve a frozen legacy path after the repository moves computers.

    Freeze manifests intentionally retain the absolute source path that existed
    when they were created.  If that path no longer exists, recover the same
    repository-relative ``data/...`` suffix.  This changes only path resolution;
    immutable artifact contents and hashes remain untouched.
    """

    source = Path(reference).expanduser()
    if source.exists():
        return source.resolve()
    parts = source.parts
    data_index = next(
        (index for index, part in enumerate(parts) if part.casefold() == "data"),
        None,
    )
    if data_index is not None:
        candidate = (PROJECT_ROOT / Path(*parts[data_index:])).resolve()
        if PROJECT_ROOT in candidate.parents and candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Frozen artifact is unavailable on this workstation: {reference}. "
        "Run verify_workstation.py and confirm the portable model assets were cloned."
    )


def active_model_paths(
    manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    selected = load_active_model_manifest() if manifest is None else manifest
    return {
        "configuration": _resolve(str(selected["simulator_configuration"])),
        "uav_residual_freeze": _resolve(str(selected["uav_residual_freeze"])),
        "development_cable_fit": _resolve(str(selected["development_cable_fit"])),
    }


def verify_active_geometry(
    settings: SimulatorSettings, manifest: dict[str, Any] | None = None
) -> None:
    selected = load_active_model_manifest() if manifest is None else manifest
    cable = settings.cable_configuration
    expected_rest = (
        0.0315,
        0.0315,
        0.0870,
        0.1000,
        0.1000,
        0.1000,
        0.1000,
        0.1000,
        0.1025,
        0.1000,
        0.1000,
    )
    if selected["geometry_version"] != "remeasured_0p9525m_12node_v1":
        raise ValueError("The active manifest does not select the latest geometry.")
    if cable.node_count != 12 or cable.marker_node_indices[1:] != tuple(range(2, 12)):
        raise ValueError("Production requires 12 nodes with c1...c10 at nodes 2...11.")
    if any(abs(actual - expected) > 1.0e-12 for actual, expected in zip(cable.rest_lengths_m, expected_rest, strict=True)):
        raise ValueError("Production rest lengths do not match the 0.9525-m geometry.")
    if abs(cable.length_m - 0.9525) > 1.0e-12:
        raise ValueError("Production cable length must be 0.9525 m.")
    if settings.attachment_offset_body_m != (0.0, 0.0, -0.055):
        raise ValueError("Production UAV-reference-to-connector offset must be 55 mm down.")


def verify_active_parameters(
    settings: SimulatorSettings, manifest: dict[str, Any] | None = None
) -> None:
    """Reject implicit parameter substitution outside the pinned manifest."""

    selected = load_active_model_manifest() if manifest is None else manifest
    paths = active_model_paths(selected)
    uav = json.loads(
        (paths["uav_residual_freeze"] / "physical_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    cable = json.loads(
        (paths["development_cable_fit"] / "fitted_cable_parameters.json").read_text(
            encoding="utf-8"
        )
    )["fitted_parameters"]
    for name in ("K_p", "K_v", "k_a", "K_R", "K_omega"):
        if abs(float(getattr(settings.parameters.uav, name)) - float(uav[name])) > 1.0e-12:
            raise ValueError(f"Active UAV parameter {name} differs from the pinned freeze.")
    for name in ("EI", "Cb"):
        if abs(float(getattr(settings.parameters.cable, name)) - float(cable[name])) > 1.0e-15:
            raise ValueError(
                f"Active cable parameter {name} differs from the pinned development fit."
            )
    if selected.get("damping_backend") != VALIDATED_OPTIMIZED_DAMPING_BACKEND:
        raise ValueError("Active runtime must explicitly select validated PCG32.")


def load_production_residual(
    *, device: torch.device | str, dtype: torch.dtype
) -> CausalTranslationalResidual:
    manifest = load_active_model_manifest()
    freeze = active_model_paths(manifest)["uav_residual_freeze"]
    freeze_manifest = json.loads((freeze / "manifest.json").read_text(encoding="utf-8"))
    expected = str(freeze_manifest["artifact_hashes"]["residual_weights.pt"])
    weights = freeze / "residual_weights.pt"
    if sha256_file(weights) != expected:
        raise ValueError("Frozen UAV residual hash does not match its manifest.")
    normalization = json.loads(
        (freeze / "residual_normalization.json").read_text(encoding="utf-8")
    )
    model = CausalTranslationalResidual(
        torch.as_tensor(normalization["mean"], dtype=dtype, device=device),
        torch.as_tensor(
            normalization["standard_deviation"], dtype=dtype, device=device
        ),
        seed=42,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return model


def build_production_simulator(
    settings: SimulatorSettings,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> CoupledSimulator:
    """Build PR + rigid clamp + 12-node DDER from the explicit manifest."""

    manifest = load_active_model_manifest()
    verify_active_geometry(settings, manifest)
    verify_active_parameters(settings, manifest)
    selected_device = settings.torch_device() if device is None else torch.device(device)
    selected_dtype = (
        torch.float32
        if dtype is None and selected_device.type == "cuda"
        else torch.float64
        if dtype is None
        else dtype
    )
    residual = load_production_residual(device=selected_device, dtype=selected_dtype)
    return CoupledSimulator(
        settings.cable_configuration,
        settings.parameters,
        dt_s=settings.dt_s,
        device=selected_device,
        dtype=selected_dtype,
        uav_model=FullStateUAVModel(residual_model=residual),
        attachment_offset_body_m=settings.attachment_offset_body_m,
        attachment_tangent_body=settings.attachment_tangent_body,
    )
