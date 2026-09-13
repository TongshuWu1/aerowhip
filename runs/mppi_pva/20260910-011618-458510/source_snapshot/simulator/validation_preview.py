"""Record five deterministic validation trials without touching a training job.

This module runs in a separate process. The actor plans from the initial state
estimate; the recorded positions come from the independent execution plant,
including PID recovery. Refused plans remain at their initial state.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from learning.deployment_rollout import execute_batch, plan_batch, sample_batch
from learning.point_force_env import PointForceWhipEnvironment
from simulator.rollout import _build_policy_agent, _build_sac_agent, load_json, resolve_device

TRIAL_COUNT = 5
SCHEMA = "five_trial_validation_preview_v1"


def checkpoint_signature(path: Path) -> list:
    stat = path.stat()
    return [str(path.resolve()), stat.st_mtime_ns, stat.st_size]


def latest_checkpoint(artifact: Path | None) -> Path | None:
    """The training preview follows latest, even when best validation is older."""
    if artifact is not None:
        for name in ("latest.pt", "terminal.pt"):
            path = artifact / "checkpoints" / name
            if path.is_file():
                return path
    return None


def load_snapshot(path: Path):
    # Training publishes with os.replace. Check the identity around the read so
    # the metadata describes precisely the immutable bytes evaluated below.
    for _ in range(3):
        signature = checkpoint_signature(path)
        data = path.read_bytes()
        if checkpoint_signature(path) == signature:
            return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False), signature
    raise RuntimeError("Checkpoint changed while loading; retry the preview.")


@torch.no_grad()
def record_trials(model, task, shared, agent, *, device, check_cancel=lambda: None):
    env = PointForceWhipEnvironment(model, task, shared, batch_size=TRIAL_COUNT, device=device)
    settings = shared["deployment"]
    generator = torch.Generator(device=device).manual_seed(int(settings["validation_seed"]))
    batch = sample_batch(env, settings, generator)
    print("Planning five validation strikes...", flush=True)
    class Planner:
        def deterministic_action(self, observation):
            check_cancel()
            return agent.deterministic_action(observation)

    forces, cutoffs = plan_batch(env, Planner(), batch)
    positions = [batch.truth.positions_m.clone()]
    commands = [env.hover_force_world_n.expand(TRIAL_COUNT, -1).clone()]
    hits = [torch.zeros(TRIAL_COUNT, device=device, dtype=torch.bool)]
    striking = [cutoffs > 0]

    def trace(_index, command, state, in_strike, success):
        if _index % 25 == 0:
            check_cancel()
        positions.append(state.positions_m.clone())
        commands.append(command)
        hits.append(success)
        striking.append(in_strike)

    print("Executing frozen force sequences and PID recovery...", flush=True)
    score = execute_batch(env, batch, forces, cutoffs, settings, trace=trace)
    arrays = {name: torch.stack(values).cpu().numpy() for name, values in (
        ("positions_m", positions), ("commanded_force_world_n", commands),
        ("hit", hits), ("striking", striking))}
    arrays["time_s"] = np.arange(len(positions), dtype=np.float64) * env.physics_dt_s
    result = dict(
        validation_seed=int(settings["validation_seed"]), trials=TRIAL_COUNT,
        target_position_m=env.target[0].cpu().tolist(),
        trial_target_positions_m=env.target.cpu().tolist(),
        desired_strike_direction_world=task["desired_strike_direction_world"],
        target_radius_m=task["success"]["tip_target_distance_m"],
        recovery_duration_s=float(settings["recovery_duration_s"]),
        planned=score.deployment["planned"].cpu().tolist(),
        success=score.episode_success.cpu().tolist(),
        recovered=score.deployment["recovered"].cpu().tolist(),
        failed=score.failed.cpu().tolist(),
        duration_s=score.deployment["duration_s"].cpu().tolist(),
        nominal=batch.nominal.cpu().tolist(),
        rewards=score.episode_reward.cpu().tolist(),
    )
    return arrays, result


def write_recording(path: Path, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata)))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_recording(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        if metadata["schema"] != SCHEMA or metadata["trials"] != TRIAL_COUNT:
            raise ValueError("Unsupported validation preview recording.")
        arrays = {name: payload[name].copy() for name in payload.files if name != "metadata"}
    return arrays, metadata


def generate(checkpoint_path: Path, output: Path, *, device_name="auto", cancel_file=None):
    def check_cancel():
        if cancel_file is not None and cancel_file.exists():
            raise InterruptedError("Validation preview cancelled.")

    check_cancel()
    root = Path(__file__).resolve().parents[1]
    artifact = checkpoint_path.parent.parent
    # Saved task/reward settings must win over edits made for the next run.
    configs = [load_json(artifact / name if (artifact / name).is_file() else root / "config" / name)
               for name in ("model.json", "task.json", "ppo.json")]
    model, task, shared = configs
    if not shared.get("deployment", {}).get("enabled"):
        raise ValueError("This run does not use initial-state-only validation.")
    if device_name == "auto":
        snapshot = artifact / "config_snapshot.json"
        effective = load_json(snapshot).get("effective_run", {}) if snapshot.is_file() else {}
        status_path = artifact / "status.json"
        try:
            status = load_json(status_path) if status_path.is_file() else {}
        except (OSError, ValueError):
            status = {}
        device_name = effective.get("device", status.get("device", shared["training"]["device"]))
    device = resolve_device(device_name)
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
    payload, signature = load_snapshot(checkpoint_path)
    schema = payload.get("schema")
    if schema == "force_ppo_checkpoint_v1":
        algorithm = "PPO"
        agent = _build_policy_agent(shared, device)
        agent.policy.set_action_prior(payload.get("action_prior"))
        agent.policy.load_state_dict(payload["policy"])
        agent.policy.eval()
    elif schema == "force_sac_checkpoint_v1":
        algorithm = "SAC"
        agent = _build_sac_agent(root, device, checkpoint=payload)
        agent.actor.load_state_dict(payload["actor"])
        agent.actor.eval()
    else:
        raise ValueError("The preview requires a PPO or SAC checkpoint.")
    arrays, metadata = record_trials(model, task, shared, agent, device=device, check_cancel=check_cancel)
    check_cancel()
    metadata.update(schema=SCHEMA, algorithm=algorithm, device=str(device),
                    checkpoint_signature=signature, episodes=int(payload.get("episodes", 0)),
                    execution_mode="initial_state_only_open_loop_once_with_pid_recovery")
    write_recording(output, arrays, metadata)
    print(f"Saved five {algorithm} validation trials at {metadata['episodes']:,} episodes.", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args()
    try:
        generate(args.checkpoint, args.output, device_name=args.device, cancel_file=args.cancel_file)
    except InterruptedError:
        print("Validation preview cancelled.", flush=True)
    finally:
        if args.cancel_file is not None:
            args.cancel_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
