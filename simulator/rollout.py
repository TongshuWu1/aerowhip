"""Deterministic rollouts for the force-controlled point-cable model."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .point_mass import ForceControlledPointCable


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def find_latest_policy_checkpoint(project_root: Path, algorithm: str = "ppo") -> Path | None:
    """Resolve the newest durable point-force policy checkpoint."""

    algorithm = algorithm.lower()
    if algorithm not in {"ppo", "sac"}:
        raise ValueError("Algorithm must be PPO or SAC.")
    runs_root = project_root / "runs" / algorithm
    active_pointer = runs_root / "ACTIVE_RUN.txt"
    if active_pointer.is_file():
        try:
            active = Path(active_pointer.read_text(encoding="utf-8-sig").strip())
            if not active.is_absolute():
                active = project_root / active
            for name in ("best_validation.pt", "latest.pt", "terminal.pt"):
                candidate = active / "checkpoints" / name
                if candidate.is_file():
                    return candidate.resolve()
        except OSError:
            pass
    candidates = [
        *runs_root.glob("*/checkpoints/best_validation.pt"),
        *runs_root.glob("*/checkpoints/latest.pt"),
        *runs_root.glob("*/checkpoints/terminal.pt"),
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).resolve()


def list_method_checkpoints(project_root: Path, algorithm: str) -> list[dict[str, Any]]:
    """List every replayable artifact for one comparison method."""

    algorithm = algorithm.lower()
    if algorithm not in {"ppo", "sac"}:
        raise ValueError("Algorithm must be PPO or SAC.")
    active = None
    pointer = project_root / "runs" / algorithm / "ACTIVE_RUN.txt"
    try:
        value = Path(pointer.read_text(encoding="utf-8-sig").strip())
        active = (value if value.is_absolute() else project_root / value).resolve()
    except OSError:
        pass
    kinds = {
        "best_validation.pt": "Best validation",
        "latest.pt": "Latest",
        "terminal.pt": "Terminal",
    }
    entries: list[dict[str, Any]] = []
    root = project_root / "runs" / algorithm
    if not root.is_dir():
        return entries
    for artifact in root.iterdir():
        if not artifact.is_dir():
            continue
        display_name = artifact.name
        status: dict[str, Any] = {}
        try:
            run = json.loads((artifact / "run.json").read_text(encoding="utf-8"))
            display_name = str(run.get("display_name", display_name))
        except (OSError, ValueError):
            pass
        try:
            status = json.loads((artifact / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        checkpoint_dir = artifact / "checkpoints"
        candidates = [
            (checkpoint_dir / filename, kind)
            for filename, kind in kinds.items()
        ]
        if algorithm == "ppo":
            candidates.extend(
                (path, "Manual snapshot")
                for path in checkpoint_dir.glob("manual_*.pt")
            )
        for path, kind in candidates:
            if not path.is_file():
                continue
            episodes = int(status.get("episodes", 0))
            if kind == "Manual snapshot":
                try:
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    episodes = int(checkpoint.get("episodes", episodes))
                    if checkpoint.get("schema") != "force_ppo_checkpoint_v1":
                        continue
                except (
                    AttributeError,
                    EOFError,
                    OSError,
                    pickle.UnpicklingError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    continue
            modified_ns = path.stat().st_mtime_ns
            if kind == "Manual snapshot":
                try:
                    modified_ns = int(path.stem.split("_")[2])
                except (IndexError, ValueError):
                    pass
            entries.append({
                "label": f"{display_name} · {kind}" + (f" · {episodes:,} ep" if episodes else "") + (" · CURRENT" if artifact.resolve() == active else ""),
                "path": path.resolve(), "artifact": artifact.resolve(),
                "kind": kind, "episodes": episodes,
                "modified_ns": modified_ns,
            })
    return sorted(entries, key=lambda item: int(item["modified_ns"]), reverse=True)


def _build_policy_agent(ppo_config: dict[str, Any], device: torch.device):
    from learning import POINT_FORCE_OBSERVATION_DIM, SimplePPOAgent

    source = ppo_config["ppo"]
    return SimplePPOAgent(
        POINT_FORCE_OBSERVATION_DIM,
        3,
        device=device,
        hidden_dim=int(source["hidden_dim"]),
        learning_rate=float(source["learning_rate"]),
        gamma=float(source["gamma"]),
        gae_lambda=float(source["gae_lambda"]),
        clip_ratio=float(source["clip_ratio"]),
        value_coefficient=float(source["value_coefficient"]),
        entropy_coefficient=float(source["entropy_coefficient"]),
        maximum_gradient_norm=float(source["maximum_gradient_norm"]),
        target_kl=float(source["target_kl"]),
        stochastic_action_indices=source.get("stochastic_action_indices"),
        initial_log_std=source.get("initial_log_std", -0.5),
        minimum_log_std=float(source.get("minimum_log_std", -10.0)),
    )


def _build_sac_agent(project_root: Path, device: torch.device, *, checkpoint=None):
    from learning import POINT_FORCE_OBSERVATION_DIM, SimpleSACAgent
    if checkpoint is not None and checkpoint.get("agent_config"):
        return SimpleSACAgent(POINT_FORCE_OBSERVATION_DIM, 3, device=device,
                              **checkpoint["agent_config"])
    source = load_json(project_root / "config" / "sac.json")["sac"]
    return SimpleSACAgent(
        POINT_FORCE_OBSERVATION_DIM, 3, device=device,
        **{key: source[key] for key in (
            "hidden_dim", "actor_learning_rate", "critic_learning_rate",
            "temperature_learning_rate", "gamma", "target_smoothing_tau",
            "initial_temperature", "stochastic_action_indices")}
    )


def simulate_policy(
    model_config: dict[str, Any],
    task_config: dict[str, Any],
    ppo_config: dict[str, Any],
    *,
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Replay one deterministic attempt with a durable PPO checkpoint."""

    from learning import POINT_FORCE_OBSERVATION_DIM, PointForceWhipEnvironment

    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {checkpoint_path}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
    environment = PointForceWhipEnvironment(
        model_config, task_config, ppo_config, batch_size=1, device=device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    schema = checkpoint.get("schema")
    if schema == "force_ppo_checkpoint_v1":
        agent = _build_policy_agent(ppo_config, device)
        agent.policy.set_action_prior(checkpoint.get("action_prior"))
        agent.policy.load_state_dict(checkpoint["policy"])
        agent.policy.eval()
        algorithm = "ppo"
    elif schema == "force_sac_checkpoint_v1":
        # New checkpoints carry their architecture; relocated files are valid.
        project_root = Path(__file__).resolve().parents[1]
        agent = _build_sac_agent(project_root, device, checkpoint=checkpoint)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.actor.eval()
        algorithm = "sac"
    else:
        raise ValueError("Checkpoint is not a point-force PPO or SAC artifact.")
    if int(checkpoint.get("observation_dim", POINT_FORCE_OBSERVATION_DIM)) != POINT_FORCE_OBSERVATION_DIM:
        raise ValueError("Checkpoint observation size is incompatible with this model.")
    if int(checkpoint.get("action_dim", 3)) != 3:
        raise ValueError("Checkpoint action size is incompatible with 3D force control.")

    observation = environment.reset()
    positions: list[torch.Tensor] = [
        environment.state.positions_m[0].detach().clone()
    ]
    velocities: list[torch.Tensor] = [
        environment.state.velocities_m_s[0].detach().clone()
    ]
    commands: list[torch.Tensor] = []
    reactions: list[torch.Tensor] = []
    normalized_actions: list[torch.Tensor] = []
    total_steps = (
        environment.control_step_count * environment.physics_steps_per_control
    )
    completed_steps = 0
    current_action = torch.zeros((1, 3), dtype=torch.float32, device=device)

    def record_physics_step(state, force, reaction) -> None:
        nonlocal completed_steps
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Policy simulation was cancelled.")
        positions.append(state.positions_m[0].detach().clone())
        velocities.append(state.velocities_m_s[0].detach().clone())
        commands.append(force[0].detach().clone())
        reactions.append(reaction[0].detach().clone())
        normalized_actions.append(current_action[0].detach().clone())
        completed_steps += 1
        if progress_callback is not None and (
            completed_steps == total_steps
            or completed_steps % max(1, total_steps // 100) == 0
        ):
            progress_callback(completed_steps, total_steps)

    with torch.no_grad():
        for control_index in range(environment.control_step_count):
            current_action = agent.deterministic_action(observation)
            result = environment.step(
                current_action, stop_when_all_done=True,
                physics_trace_callback=record_physics_step,
            )
            observation = result.next_observation
            if not bool(environment.active.any()):
                break

    if progress_callback is not None:
        progress_callback(completed_steps, completed_steps)

    def stack(values: list[torch.Tensor]) -> np.ndarray:
        return torch.stack(values).detach().cpu().numpy().astype(np.float64)

    position_array = stack(positions)
    velocity_array = stack(velocities)
    command_array = stack(commands)
    reaction_array = stack(reactions)
    action_array = stack(normalized_actions)
    dt_s = environment.physics_dt_s
    arrays = {
        "time_s": np.arange(completed_steps + 1, dtype=np.float64) * dt_s,
        "cable_node_position_world_m": position_array,
        "cable_node_velocity_world_m_s": velocity_array,
        "commanded_force_world_n": command_array,
        "effective_cable_reaction_on_point_world_n": reaction_array,
        "normalized_policy_action": action_array,
    }
    force_norm = np.linalg.norm(command_array, axis=1)
    summary = {
        "schema": "point_force_simulation_summary_v1",
        "source": f"latest_{algorithm}_policy",
        "algorithm": algorithm.upper(),
        "device": str(device),
        "dt_s": dt_s,
        "steps": completed_steps,
        "duration_s": completed_steps * dt_s,
        "maximum_duration_s": total_steps * dt_s,
        "hit_time_s": (
            float(environment.episode_hit_time_s[0])
            if bool(environment.episode_success[0]) else None
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_episodes": int(checkpoint.get("episodes", 0)),
        "checkpoint_gradient_updates": int(checkpoint.get("gradient_updates", 0)),
        "policy_action": "deterministic_mean",
        "success": bool(environment.episode_success[0].detach().cpu()),
        "non_tip_first": bool(
            environment.episode_non_tip_first[0].detach().cpu()
        ),
        "invalid_tip_entry": bool(
            environment.episode_invalid_tip_entry[0].detach().cpu()
        ),
        "timeout": bool(environment.episode_timed_out[0].detach().cpu()),
        "numerical_failure": bool(environment.failed[0].detach().cpu()),
        "minimum_tip_distance_m": float(
            environment.episode_minimum_tip_distance[0].detach().cpu()
        ),
        "point_displacement_integral_m_s": float(
            environment.episode_point_displacement_integral_m_s[0].detach().cpu()
        ),
        "point_displacement_cost_integral_s": float(
            environment.episode_point_displacement_cost_integral_s[0].detach().cpu()
        ),
        "maximum_force_norm_n": float(force_norm.max()),
        "final_root_position_world_m": position_array[-1, 0].tolist(),
        "final_tip_position_world_m": position_array[-1, -1].tolist(),
        "maximum_segment_error_m": float(
            environment.model.dder.maximum_segment_error_m(
                environment.state.positions_m
            )[0].detach().cpu()
        ),
        "finite": bool(
            np.isfinite(position_array).all()
            and np.isfinite(velocity_array).all()
            and np.isfinite(command_array).all()
            and np.isfinite(reaction_array).all()
        ),
    }
    return arrays, summary


def simulate_constant_force(
    model_config: dict[str, Any],
    task_config: dict[str, Any],
    *,
    force_world_n: tuple[float, float, float] | None = None,
    duration_s: float | None = None,
    device: torch.device = torch.device("cpu"),
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return one trajectory and a compact force/geometry summary."""

    model = ForceControlledPointCable.from_mapping(model_config)
    dt_s = float(model_config["simulation"]["dt_s"])
    duration = (
        float(task_config["episode_duration_s"])
        if duration_s is None
        else float(duration_s)
    )
    if duration <= 0.0:
        raise ValueError("duration_s must be positive.")
    steps = int(round(duration / dt_s))
    if steps < 1:
        raise ValueError("duration_s is shorter than one simulation step.")
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    root_position = torch.tensor(
        task_config["initial_root_position_m"], dtype=dtype, device=device
    )[None]
    root_velocity = torch.tensor(
        task_config["initial_root_velocity_m_s"], dtype=dtype, device=device
    )[None]
    state = model.hanging_state(root_position, root_velocity)
    command = (
        model.hover_force_world_n(dtype=dtype, device=device)
        if force_world_n is None
        else torch.tensor(force_world_n, dtype=dtype, device=device)
    )[None]

    positions = [state.positions_m[0].detach().cpu().numpy().copy()]
    velocities = [state.velocities_m_s[0].detach().cpu().numpy().copy()]
    cable_reactions: list[np.ndarray] = []
    for step_index in range(steps):
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Simulation was cancelled.")
        result = model.step_runtime(state, command, dt_s)
        state = result.state
        positions.append(state.positions_m[0].detach().cpu().numpy().copy())
        velocities.append(state.velocities_m_s[0].detach().cpu().numpy().copy())
        cable_reactions.append(
            result.forces.effective_cable_reaction_on_point_world_n[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        if progress_callback is not None and (
            step_index == steps - 1 or (step_index + 1) % max(1, steps // 100) == 0
        ):
            progress_callback(step_index + 1, steps)
    position_array = np.stack(positions).astype(np.float64)
    velocity_array = np.stack(velocities).astype(np.float64)
    reaction_array = np.stack(cable_reactions).astype(np.float64)
    command_array = np.broadcast_to(
        command[0].detach().cpu().numpy(), (steps, 3)
    ).copy().astype(np.float64)
    arrays = {
        "time_s": np.arange(steps + 1, dtype=np.float64) * dt_s,
        "cable_node_position_world_m": position_array,
        "cable_node_velocity_world_m_s": velocity_array,
        "commanded_force_world_n": command_array,
        "effective_cable_reaction_on_point_world_n": reaction_array,
    }
    summary = {
        "schema": "point_force_simulation_summary_v1",
        "source": "constant_force",
        "device": str(device),
        "dt_s": dt_s,
        "steps": steps,
        "duration_s": steps * dt_s,
        "point_mass_kg": model.point_mass_kg,
        "cable_mass_kg": model.cable_mass_kg,
        "system_mass_kg": model.system_mass_kg,
        "commanded_force_world_n": command_array[-1].tolist(),
        "hover_force_world_n": model.hover_force_world_n(
            dtype=dtype, device=device
        ).detach().cpu().tolist(),
        "initial_root_position_world_m": position_array[0, 0].tolist(),
        "final_root_position_world_m": position_array[-1, 0].tolist(),
        "final_root_velocity_world_m_s": velocity_array[-1, 0].tolist(),
        "final_tip_position_world_m": position_array[-1, -1].tolist(),
        "final_tip_velocity_world_m_s": velocity_array[-1, -1].tolist(),
        "final_cable_reaction_on_point_world_n": reaction_array[-1].tolist(),
        "maximum_segment_error_m": float(
            model.dder.maximum_segment_error_m(state.positions_m).max().detach().cpu()
        ),
        "finite": bool(
            np.isfinite(position_array).all()
            and np.isfinite(velocity_array).all()
            and np.isfinite(reaction_array).all()
        ),
    }
    return arrays, summary
