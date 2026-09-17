"""Record five deterministic validation trials without touching a training job.

Historical recording helpers remain for independent execution-plant tests.
This module does not load agents or launch policy replay jobs.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from learning.deployment_rollout import execute_batch, plan_batch, sample_batch
from learning.point_force_env import PointForceWhipEnvironment

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
