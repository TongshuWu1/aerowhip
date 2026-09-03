"""Read-only portability and RTX workstation preflight.

Run this before opening the simulator or starting PPO on a new computer.  It
loads the immutable model and selected policy but performs no training, fitting,
hardware access, protected-test evaluation, or output-file writes.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
SELECTED_POLICY = (
    ROOT
    / "results"
    / "ppo"
    / "policies"
    / "PPO_WHIP_FORWARD_REVERSE_RELEASE_D50_V1"
)


def _cuda_version(value: str | None) -> tuple[int, int]:
    if not value:
        return (0, 0)
    components = value.split(".")
    return (int(components[0]), int(components[1]))


def main() -> int:
    checks: dict[str, Any] = {
        "schema": "portable_workstation_preflight_v1",
        "repository": str(ROOT),
        "python": sys.version.split()[0],
        "authorization": "READ_ONLY_SIMULATION_PREFLIGHT",
    }
    errors: list[str] = []
    try:
        import imageio_ffmpeg
        import matplotlib
        import numpy
        import PySide6
        import pyvista
        import pyvistaqt
        import torch
        import vtkmodules

        checks["packages"] = {
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
            "PySide6": PySide6.__version__,
            "pyvista": pyvista.__version__,
            "pyvistaqt": pyvistaqt.__version__,
            "vtkmodules": str(getattr(vtkmodules, "__version__", "installed")),
            "imageio_ffmpeg": imageio_ffmpeg.__version__,
        }
    except Exception as error:
        errors.append(f"Dependency import failed: {type(error).__name__}: {error}")
        checks["errors"] = errors
        checks["status"] = "FAIL"
        print(json.dumps(checks, indent=2))
        return 1

    checks["cuda"] = {
        "available": torch.cuda.is_available(),
        "pytorch_runtime": torch.version.cuda,
        "compiled_architectures": torch.cuda.get_arch_list(),
    }
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable. Select the CUDA-enabled PyTorch interpreter.")
    else:
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        checks["cuda"].update(
            {
                "device": device_name,
                "capability": list(capability),
                "required_architecture": required_arch,
                "rtx_5090_sm120_in_wheel": "sm_120" in torch.cuda.get_arch_list(),
            }
        )
        if _cuda_version(torch.version.cuda) < (12, 8):
            errors.append(
                "This PyTorch wheel is older than CUDA 12.8 and is unsuitable for "
                "an RTX 5090/Blackwell workstation."
            )
        if "sm_120" not in torch.cuda.get_arch_list():
            errors.append(
                "The installed PyTorch wheel does not contain sm_120 required by "
                "the destination RTX 5090; install the documented CUDA 12.8 build."
            )
        try:
            probe = torch.ones(4, device="cuda", dtype=torch.float32)
            checks["cuda"]["execution_probe"] = float((probe * probe).sum().item())
        except Exception as error:
            errors.append(f"CUDA execution probe failed: {type(error).__name__}: {error}")

    required_files = (
        ROOT / "config" / "active_model.json",
        ROOT / "config" / "default.json",
        ROOT / "results" / "common" / "context_normalizer.json",
        ROOT / "results" / "common" / "training_state_bank.npz",
        ROOT / "results" / "common" / "validation_state_bank.npz",
        SELECTED_POLICY / "config.json",
        SELECTED_POLICY / "checkpoints" / "terminal.pt",
        ROOT
        / "data"
        / "model_freezes"
        / "MODEL_FREEZE_DECOMPOSED_PRETEST"
        / "residual_weights.pt",
        ROOT
        / "data"
        / "model_freezes"
        / "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI"
        / "manifest.json",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    checks["required_files"] = {
        "count": len(required_files),
        "missing": missing,
    }
    if missing:
        errors.append("Portable production assets are missing from the clone.")

    if not errors:
        try:
            from planning.model_contract import verify_planning_model_integrity
            from planning.task import load_canonical_whip_task
            from run_simple_ppo import _build_agent, _load_config
            from run_simple_sac import _build_environment
            from simulator.parameters import SimulatorSettings

            settings = SimulatorSettings.load(ROOT / "config" / "default.json")
            task = load_canonical_whip_task(
                ROOT / "config" / "tasks" / "canonical_whip_v1.json"
            )
            model_audit = verify_planning_model_integrity(settings, task)
            checks["model_freeze"] = {
                "verified": bool(model_audit["verified"]),
                "name": task.model_freeze,
                "verified_artifacts": len(model_audit["verified_artifact_hashes"]),
            }

            policy_config = _load_config(SELECTED_POLICY / "config.json")
            environment, device = _build_environment(policy_config, batch_size=1)
            agent = _build_agent(policy_config, device)
            checkpoint = torch.load(
                SELECTED_POLICY / "checkpoints" / "terminal.pt",
                map_location=device,
                weights_only=False,
            )
            agent.policy.load_state_dict(checkpoint["policy"])
            agent.value.load_state_dict(checkpoint["value"])
            observation = environment.reset()
            with torch.no_grad():
                action = agent.deterministic_action(observation)
            finite = bool(torch.isfinite(observation).all() and torch.isfinite(action).all())
            checks["selected_policy"] = {
                "id": "PPO_WHIP_FORWARD_REVERSE_RELEASE_D50_V1",
                "checkpoint_episodes": int(checkpoint.get("episodes", -1)),
                "device": str(device),
                "observation_shape": list(observation.shape),
                "action_shape": list(action.shape),
                "finite_inference": finite,
            }
            if not finite:
                errors.append("Selected-policy inference produced a non-finite value.")
        except Exception as error:
            errors.append(
                f"Production model/policy load failed: {type(error).__name__}: {error}"
            )

    checks["status"] = "PASS" if not errors else "FAIL"
    checks["errors"] = errors
    print(json.dumps(checks, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
