"""Zero-training PPO feedback-dependence and trajectory-compiler audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from learning.normalization import FixedContextNormalizer
from learning.ppo_trajectory_compiler import (
    StateImpulse,
    compile_ppo_trajectory,
    metric_maximum_absolute_differences,
    metrics_from_feedback_environment,
    reindex_command_sequence,
    replay_compiled_trajectory,
    scaled_cable_parameters,
    summarize_execution_metrics,
)
from learning.ppo_validation import _state_distances
from learning.sequential_sac_env import SequentialWhipEnvironment, SimpleRewardWeights
from learning.state_bank import (
    InitialStateBank,
    initial_state_bank_from_state,
)
from planning.rollout import clone_state_batch, hover_preroll
from planning.task import load_canonical_whip_task
from run_simple_ppo import _build_agent, _load_config
from simulator.parameters import SimulatorParameters, SimulatorSettings
from simulator.production import build_production_simulator
from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "learning" / "ppo_open_loop_compiler_audit_v1.json"


def _utc_stamp() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H%M%S.") + f"{now.microsecond:06d}Z"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_safe) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reward_weights(config: dict[str, Any]) -> SimpleRewardWeights:
    reward = config["reward"]
    return SimpleRewardWeights(
        progress=float(reward["progress_weight"]),
        directed_speed_near_target=float(reward["directed_speed_near_target_weight"]),
        direction_near_target=float(reward["direction_near_target_weight"]),
        uav_displacement=float(reward["uav_displacement_weight"]),
        success_bonus=float(reward["success_bonus"]),
        numerical_failure=float(reward["numerical_failure_penalty"]),
        proximity_scale_m=float(reward["proximity_scale_m"]),
        non_tip_first=float(reward.get("non_tip_first_penalty", 0.0)),
        strike_quality_improvement=float(
            reward.get("strike_quality_improvement_weight", 0.0)
        ),
        maximum_displacement=float(reward.get("maximum_displacement_weight", 0.0)),
        displacement_integral=float(reward.get("displacement_integral_weight", 0.0)),
        uav_speed_integral=float(reward.get("uav_speed_integral_weight", 0.0)),
        acceleration_effort=float(reward.get("acceleration_effort_weight", 0.0)),
        body_rate_effort=float(reward.get("body_rate_effort_weight", 0.0)),
        action_smoothness=float(reward.get("action_smoothness_weight", 0.0)),
        time_to_success=float(reward.get("time_to_success_weight_per_s", 0.0)),
        directed_speed_reward_cap_m_s=float(
            reward.get("directed_speed_reward_cap_m_s", float("inf"))
        ),
        success_compactness_bonus=float(
            reward.get("success_compactness_bonus", 0.0)
        ),
        success_compactness_scale_m=float(
            reward.get("success_compactness_scale_m", 0.5)
        ),
        displacement_cost_scale_m=float(
            reward.get("displacement_cost_scale_m", 0.0)
        ),
    )


def _environment(
    simulator: CoupledSimulator,
    ppo_config: dict[str, Any],
    initial_state: SimulatorState,
    *,
    rollout_parameters: SimulatorParameters | None = None,
    record_commands: bool = False,
    record_states: bool = False,
    state_postprocessor: Any = None,
) -> SequentialWhipEnvironment:
    task = load_canonical_whip_task(ROOT / ppo_config["task_config"])
    action = ppo_config["action"]
    return SequentialWhipEnvironment(
        simulator,
        task,
        initial_state,
        FixedContextNormalizer.load(ROOT / ppo_config["context_normalizer"]),
        batch_size=initial_state.uav.batch_size,
        episode_duration_s=float(ppo_config["episode_duration_s"]),
        control_dt_s=float(ppo_config["control_dt_s"]),
        maximum_acceleration_m_s2=float(action["maximum_acceleration_norm_m_s2"]),
        maximum_body_rate_rad_s=float(action["maximum_body_rate_rad_s"]),
        observation_clip=float(ppo_config["observation"]["normalized_clip"]),
        reward_weights=_reward_weights(ppo_config),
        success_mode=str(ppo_config["success_mode"]),
        reward_mode=str(ppo_config["reward_mode"]),
        terminate_on_success=not bool(
            ppo_config.get("reported_success", {}).get(
                "episode_continues_after_success", True
            )
        ),
        rollout_parameters=rollout_parameters,
        record_fullstate_commands=record_commands,
        record_state_trajectory=record_states,
        state_postprocessor=state_postprocessor,
    )


@torch.no_grad()
def _feedback_rollout(environment: SequentialWhipEnvironment, agent: Any):
    was_training = bool(agent.policy.training)
    agent.policy.eval()
    observation = environment.reset()
    for _ in range(environment.control_step_count):
        observation = environment.step(
            agent.deterministic_action(observation)
        ).next_observation
    agent.policy.train(was_training)
    return metrics_from_feedback_environment(environment)


def _load_state_bank(audit_config: dict[str, Any]) -> InitialStateBank:
    root = ROOT / audit_config["state_bank_root"]
    return InitialStateBank.load(
        root / "validation_state_bank.npz",
        root / "validation_state_bank_manifest.json",
    )


def _matched_wrong_sources(distances: torch.Tensor, strata: int) -> torch.Tensor:
    order = torch.argsort(distances)
    groups = torch.tensor_split(order, int(strata))
    mapping = torch.empty_like(order)
    for group in groups:
        if group.numel() < 2:
            raise RuntimeError("Wrong-state pairing stratum contains fewer than two rows.")
        mapping[group] = torch.roll(group, shifts=-1)
    if bool(torch.any(mapping == torch.arange(mapping.numel()))):
        raise RuntimeError("Wrong-state pairing accidentally selected an identical state.")
    return mapping


def _metric_rows(metrics: Any, state_indices: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    fields = (
        "task_success",
        "endpoint_success",
        "legacy_scientific_success",
        "finite",
        "first_entry_marker",
        "first_entry_physics_step",
        "first_entry_time_s",
        "first_entry_tip_distance_m",
        "first_entry_tip_speed_m_s",
        "first_entry_directed_speed_m_s",
        "first_entry_direction_error_deg",
        "minimum_tip_distance_m",
        "maximum_uav_displacement_m",
        "maximum_uav_speed_m_s",
        "maximum_command_acceleration_m_s2",
    )
    cpu = {name: getattr(metrics, name).detach().cpu() for name in fields}
    for row_index, state_index in enumerate(state_indices.tolist()):
        row: dict[str, Any] = {"state_index": int(state_index)}
        for name in fields:
            value = cpu[name][row_index].item()
            row[name] = bool(value) if cpu[name].dtype == torch.bool else value
        rows.append(row)
    return rows


def _latency_summary(samples_s: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples_s, dtype=np.float64) * 1000.0
    return {
        "queries": int(values.size),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.quantile(values, 0.90)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "maximum_ms": float(np.max(values)),
    }


def _benchmark_compilation(
    ppo_config: dict[str, Any],
    bank: InitialStateBank,
    agent_checkpoint: dict[str, Any],
    *,
    fixed_evaluation_batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    settings = SimulatorSettings.load(ROOT / ppo_config["simulator_config"])
    simulator = build_production_simulator(settings)
    simulator.uav_model.set_fixed_evaluation_batch_size(fixed_evaluation_batch_size)
    selected = bank.select(torch.tensor([0]), device=simulator.device)
    environment = _environment(
        simulator,
        ppo_config,
        selected.state,
        record_commands=True,
    )
    agent = _build_agent(ppo_config, simulator.device)
    agent.policy.load_state_dict(agent_checkpoint["policy"])
    agent.value.load_state_dict(agent_checkpoint["value"])
    samples: list[float] = []
    for index in range(warmup + repeats):
        if simulator.device.type == "cuda":
            torch.cuda.synchronize(simulator.device)
        start = time.perf_counter()
        compiled = compile_ppo_trajectory(environment, agent)
        # Accessing the compiled tensors ensures command materialization is in
        # the timed path; no simulator or actor remains after this return.
        _ = compiled.commands.accelerations_m_s2
        if simulator.device.type == "cuda":
            torch.cuda.synchronize(simulator.device)
        elapsed = time.perf_counter() - start
        if index >= warmup:
            samples.append(elapsed)
    return {
        "logical_batch_size": 1,
        "fixed_uav_residual_evaluation_batch_size": fixed_evaluation_batch_size,
        "actor_queries": environment.control_step_count,
        "production_physics_steps": environment.physics_step_count,
        **_latency_summary(samples),
    }


def _save_commands(path: Path, commands: Any) -> None:
    np.savez_compressed(
        path,
        position_m=commands.positions_m.detach().cpu().numpy(),
        velocity_m_s=commands.velocities_m_s.detach().cpu().numpy(),
        acceleration_m_s2=commands.accelerations_m_s2.detach().cpu().numpy(),
        orientation_xyzw=commands.orientations_xyzw.detach().cpu().numpy(),
        angular_velocity_body_rad_s=(
            commands.angular_velocities_body_rad_s.detach().cpu().numpy()
        ),
    )


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _write_report(
    artifact: Path,
    *,
    audit_config: dict[str, Any],
    checkpoint: dict[str, Any],
    exact: dict[str, Any],
    nominal: dict[str, Any],
    mismatch: list[dict[str, Any]],
    disturbances: list[dict[str, Any]],
    latency: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    mismatch_lines = "\n".join(
        f"| {row['name']} | {row['EI_scale']:.2f} | {row['Cb_scale']:.2f} | "
        f"{_pct(row['feedback']['task_success_rate'])} | "
        f"{_pct(row['compiled_open_loop']['task_success_rate'])} | "
        f"{100.0 * row['feedback_minus_compiled_success_rate']:+.2f} pp |"
        for row in mismatch
    )
    disturbance_lines = "\n".join(
        f"| {row['name']} | {row['physics_step']} | "
        f"{_pct(row['feedback']['task_success_rate'])} | "
        f"{_pct(row['compiled_open_loop']['task_success_rate'])} | "
        f"{100.0 * row['feedback_minus_compiled_success_rate']:+.2f} pp |"
        for row in disturbances
    )
    optimized = latency["optimized_logical_batch_one"]
    fixed = latency["fixed_2048_residual_contract"]
    report = f"""# PPO Feedback Dependence and Open-Loop Trajectory-Compiler Report

## 1. Question and frozen scope

This zero-training milestone asks whether the terminal sequential PPO must remain a 10-Hz cable-state-feedback controller, or whether one initial measurement plus a fast PPO+sim rollout can compile a robust high-level open-loop FullState command sequence. The policy checkpoint, nominal model, canonical target, reward, 83-D context, 6-D control action, 10.0-s horizon, and success definition were frozen. No CEM, learning, target randomization, theta training, protected data, or hardware was used.

Checkpoint: `{audit_config['ppo_checkpoint']}` at {int(checkpoint['episodes']):,} episodes. This is the requested terminal checkpoint, not the best-validation checkpoint.

## 2. What was compiled

The existing PPO is queried 100 times at 0.1-s intervals. The production UAV/residual/12-node DDER simulator advances 1,000 times at 0.01 s. At every physics step, after the sequential controller has resolved its live-state re-anchoring, the compiler records:

`[p_cmd, v_cmd, a_cmd, q_cmd, omega_cmd]`.

The saved trajectory is therefore 1,000 physical FullState commands. Its replay API accepts no policy, context, or cable measurement. It is high-level open loop; the frozen low-level FullState UAV tracking dynamics remain part of the physical plant/controller contract.

## 3. Exact same-state replay gate

The normal feedback PPO rollout and compiled replay used all 512 bank states and identical deterministic float32 production physics. Maximum trajectory differences were:

```json
{json.dumps(exact['trajectory_maximum_absolute_difference'], indent=2)}
```

Hard classification/metric differences were:

```json
{json.dumps(exact['metric_differences'], indent=2)}
```

Gate: **{exact['classification']}**. The mismatch and disturbance experiments were run only because this gate passed.

## 4. Nominal 512-state architecture controls

| Mode | Task success | Endpoint success | Finite | Median max UAV displacement |
|---|---:|---:|---:|---:|
| Feedback PPO | {_pct(nominal['feedback']['task_success_rate'])} | {_pct(nominal['feedback']['endpoint_successes']/512)} | {_pct(nominal['feedback']['finite_rate'])} | {nominal['feedback']['maximum_uav_displacement_m']['median']:.4f} m |
| Matched compiled open loop | {_pct(nominal['compiled_open_loop']['task_success_rate'])} | {_pct(nominal['compiled_open_loop']['endpoint_successes']/512)} | {_pct(nominal['compiled_open_loop']['finite_rate'])} | {nominal['compiled_open_loop']['maximum_uav_displacement_m']['median']:.4f} m |
| Wrong-state compiled sequence | {_pct(nominal['wrong_initial_state_sequence']['task_success_rate'])} | {_pct(nominal['wrong_initial_state_sequence']['endpoint_successes']/512)} | {_pct(nominal['wrong_initial_state_sequence']['finite_rate'])} | {nominal['wrong_initial_state_sequence']['maximum_uav_displacement_m']['median']:.4f} m |

Matched compilation is expected to equal feedback under deterministic nominal physics. The wrong-state control isolates whether measuring/compiling for the actual initial state matters.

The wrong-state replay also reproduced the previously diagnosed FullState tracking-runaway failure mode: its p95 maximum displacement was {nominal['wrong_initial_state_sequence']['maximum_uav_displacement_m']['p95']:.3e} m and p95 UAV speed was {nominal['wrong_initial_state_sequence']['maximum_uav_speed_m_s']['p95']:.3e} m/s. Those absurd-but-finite post-failure values are not interpreted as physical exploration or task behavior. They strengthen the conclusion that a command sequence compiled for the wrong initial condition is unsafe to reuse.

## 5. EI/Cb model-mismatch sweep

PPO observations continued to contain nominal theta. Compilation was always performed under nominal theta; only execution physics changed.

| Case | EI scale | Cb scale | Feedback | Compiled open loop | Feedback - compiled |
|---|---:|---:|---:|---:|---:|
{mismatch_lines}

## 6. Post-planning disturbances

All impulses were applied after physics step 150 (1.50 s), after compilation had observed x0. Feedback PPO could react at subsequent 10-Hz queries; compiled replay could not.

| Disturbance | Step | Feedback | Compiled open loop | Feedback - compiled |
|---|---:|---:|---:|---:|
{disturbance_lines}

## 7. Compilation latency

Batch-one end-to-end timing includes 100 context builds, 100 PPO queries, 1,000 production UAV/residual/DDER steps, and command recording.

| Numerical path | Median | P95 | Max |
|---|---:|---:|---:|
| Logical B=1 residual evaluation | {optimized['median_ms']:.2f} ms | {optimized['p95_ms']:.2f} ms | {optimized['maximum_ms']:.2f} ms |
| Residual padded to fixed 2048 | {fixed['median_ms']:.2f} ms | {fixed['p95_ms']:.2f} ms | {fixed['maximum_ms']:.2f} ms |

The optimized B=1 timing is the relevant deployment compiler latency. The fixed-2048 result is retained to expose the numerical-padding overhead; neither path changes the model equations.

## 8. Architecture decision

Decision: **{decision['preferred_architecture']}**.

Reason: {decision['reason']}

The decision rule was fixed in the audit artifact: compilation must replay exactly, optimized B=1 p95 must be <=1.0 s, the mean feedback advantage across EI/Cb cases must be <=10 percentage points, the worst mismatch advantage <=20 points, and the mean disturbance advantage <=10 points. This is an architecture-screening rule, not a scientific task gate.

## 9. What did not happen

- PPO training: **NONE**
- CEM: **NOT USED**
- Target generalization: **NOT TESTED**
- Theta-conditioned policy learning: **NOT ENABLED**
- Displacement retraining: **NOT PERFORMED** (displacement was recorded)
- Protected `fig8vertical_002`: **NOT EVALUATED**
- Real hardware: **NOT EXECUTED**

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Policy:
        TERMINAL SEQUENTIAL PPO ({int(checkpoint['episodes']):,} episodes)

    New training:
        NONE

    State-bank contexts:
        512

    Actor queries during compilation:
        100

    Recorded FullState commands:
        1000

    Exact replay:
        {exact['classification']}

    Nominal feedback success:
        {_pct(nominal['feedback']['task_success_rate'])}

    Nominal compiled success:
        {_pct(nominal['compiled_open_loop']['task_success_rate'])}

    Wrong-state compiled success:
        {_pct(nominal['wrong_initial_state_sequence']['task_success_rate'])}

    Mean feedback advantage under EI/Cb mismatch:
        {100.0 * decision['mean_mismatch_feedback_advantage']:+.2f} percentage points

    Mean feedback advantage under post-t0 disturbance:
        {100.0 * decision['mean_disturbance_feedback_advantage']:+.2f} percentage points

    Optimized compilation median:
        {optimized['median_ms']:.2f} ms

    Optimized compilation p95:
        {optimized['p95_ms']:.2f} ms

    Preferred architecture:
        {decision['preferred_architecture']}

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
"""
    (artifact / "PPO_FEEDBACK_DEPENDENCE_AND_OPEN_LOOP_COMPILER_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    (ROOT / "PPO_FEEDBACK_DEPENDENCE_AND_OPEN_LOOP_COMPILER_REPORT.md").write_text(
        report, encoding="utf-8"
    )


def run(config_path: Path, artifact: Path) -> None:
    audit_config = json.loads(config_path.read_text(encoding="utf-8"))
    if audit_config.get("schema") != "ppo_open_loop_compiler_audit_v1":
        raise ValueError("Unsupported PPO compiler audit configuration.")
    if audit_config.get("training") != "NONE":
        raise ValueError("This milestone authorizes zero training.")
    ppo_config_path = ROOT / audit_config["ppo_config"]
    ppo_config = _load_config(ppo_config_path)
    checkpoint_path = ROOT / audit_config["ppo_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = int(audit_config["checkpoint_expected_episodes"])
    if int(checkpoint.get("episodes", -1)) != expected:
        raise ValueError("The checkpoint is not the requested terminal PPO checkpoint.")
    bank = _load_state_bank(audit_config)
    if len(bank) != int(audit_config["state_bank_count"]):
        raise ValueError("Unexpected state-bank size.")

    artifact.mkdir(parents=True, exist_ok=False)
    _write_json(artifact / "config.json", audit_config)
    print(f"artifact={artifact}", flush=True)
    print("stage=exact_replay_compile", flush=True)

    settings = SimulatorSettings.load(ROOT / ppo_config["simulator_config"])
    simulator = build_production_simulator(settings)
    if simulator.device.type != "cuda" or simulator.dtype != torch.float32:
        raise RuntimeError("The architecture audit requires production CUDA float32.")
    simulator.uav_model.set_fixed_evaluation_batch_size(len(bank))
    indices = torch.arange(len(bank), dtype=torch.int64)
    selected = bank.select(indices, device=simulator.device)
    initial_state = selected.state
    agent = _build_agent(ppo_config, simulator.device)
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.value.load_state_dict(checkpoint["value"])
    compile_environment = _environment(
        simulator,
        ppo_config,
        initial_state,
        record_commands=True,
        record_states=True,
    )
    compiled = compile_ppo_trajectory(compile_environment, agent)
    feedback_reference = compile_environment.recorded_state_trajectory(clone=False)
    replay = replay_compiled_trajectory(
        simulator,
        initial_state,
        compile_environment.task,
        compiled.commands,
        reference_state_trajectory=feedback_reference,
    )
    metric_differences = metric_maximum_absolute_differences(
        compiled.feedback_metrics, replay.metrics
    )
    trajectory_max = replay.reference_maximum_absolute_difference
    tolerance = float(
        audit_config["compiler"]["exact_replay_maximum_absolute_tolerance"]
    )
    exact_pass = max(trajectory_max.values(), default=0.0) <= tolerance and all(
        float(value) == 0.0 for value in metric_differences.values()
    )
    exact = {
        "classification": "PASS" if exact_pass else "FAIL",
        "tolerance": tolerance,
        "trajectory_maximum_absolute_difference": trajectory_max,
        "metric_differences": metric_differences,
        "contexts": len(bank),
    }
    _write_json(artifact / "exact_replay_verification.json", exact)
    if not exact_pass:
        raise RuntimeError("Exact replay failed; downstream experiments are prohibited.")
    print("stage=exact_replay_pass", flush=True)

    _save_commands(artifact / "compiled_fullstate_commands_512.npz", compiled.commands)
    # Free the large 1,001-frame reference trace before subsequent sweeps.
    del feedback_reference, compile_environment
    torch.cuda.empty_cache()

    canonical_simulator = build_production_simulator(settings)
    canonical_simulator.uav_model.set_fixed_evaluation_batch_size(1)
    base = hover_preroll(
        canonical_simulator,
        load_canonical_whip_task(ROOT / ppo_config["task_config"]),
    )
    canonical_bank = initial_state_bank_from_state(
        base,
        command_position_world_m=base.uav.position_m,
        command_velocity_world_m_s=base.uav.velocity_m_s,
        command_yaw_world_rad=0.0,
    )
    distances = _state_distances(bank, canonical_bank)
    wrong_sources = _matched_wrong_sources(
        distances, int(audit_config["wrong_initial_state_control"]["strata"])
    )
    wrong_commands = reindex_command_sequence(compiled.commands, wrong_sources)
    task = load_canonical_whip_task(ROOT / ppo_config["task_config"])
    wrong = replay_compiled_trajectory(
        simulator, initial_state, task, wrong_commands
    )
    nominal = {
        "feedback": summarize_execution_metrics(compiled.feedback_metrics),
        "compiled_open_loop": summarize_execution_metrics(replay.metrics),
        "wrong_initial_state_sequence": summarize_execution_metrics(wrong.metrics),
    }
    _write_json(
        artifact / "nominal_mode_comparison.json",
        {
            **nominal,
            "wrong_state_source_indices": wrong_sources.tolist(),
            "state_distances": distances.tolist(),
        },
    )
    _write_json(
        artifact / "nominal_state_rows.json",
        {
            "feedback": _metric_rows(compiled.feedback_metrics, indices),
            "compiled_open_loop": _metric_rows(replay.metrics, indices),
            "wrong_initial_state_sequence": _metric_rows(wrong.metrics, indices),
        },
    )
    print(
        "stage=nominal_complete "
        f"feedback={nominal['feedback']['task_success_rate']:.6f} "
        f"compiled={nominal['compiled_open_loop']['task_success_rate']:.6f} "
        f"wrong={nominal['wrong_initial_state_sequence']['task_success_rate']:.6f}",
        flush=True,
    )

    mismatch_results: list[dict[str, Any]] = []
    for case in audit_config["cable_parameter_sweeps"]:
        parameters = scaled_cable_parameters(
            simulator.parameters,
            ei_scale=float(case["EI_scale"]),
            cb_scale=float(case["Cb_scale"]),
        )
        feedback_environment = _environment(
            simulator,
            ppo_config,
            initial_state,
            rollout_parameters=parameters,
        )
        feedback = _feedback_rollout(feedback_environment, agent)
        open_loop = replay_compiled_trajectory(
            simulator,
            initial_state,
            task,
            compiled.commands,
            parameters=parameters,
        ).metrics
        feedback_summary = summarize_execution_metrics(feedback)
        open_summary = summarize_execution_metrics(open_loop)
        row = {
            **case,
            "feedback": feedback_summary,
            "compiled_open_loop": open_summary,
            "feedback_minus_compiled_success_rate": (
                feedback_summary["task_success_rate"]
                - open_summary["task_success_rate"]
            ),
        }
        mismatch_results.append(row)
        print(
            f"stage=mismatch case={case['name']} "
            f"feedback={feedback_summary['task_success_rate']:.6f} "
            f"compiled={open_summary['task_success_rate']:.6f}",
            flush=True,
        )
    _write_json(artifact / "model_mismatch_results.json", mismatch_results)

    disturbance_results: list[dict[str, Any]] = []
    for item in audit_config["post_planning_disturbances"]:
        impulse = StateImpulse(
            str(item["name"]),
            physics_step=int(item["physics_step"]),
            uav_velocity_delta_world_m_s=tuple(item["uav_velocity_delta_world_m_s"]),
            cable_velocity_delta_world_m_s=tuple(
                item["cable_velocity_delta_world_m_s"]
            ),
            cable_node_start=int(item["cable_node_start"]),
        )
        feedback_environment = _environment(
            simulator,
            ppo_config,
            initial_state,
            state_postprocessor=impulse,
        )
        feedback = _feedback_rollout(feedback_environment, agent)
        open_loop = replay_compiled_trajectory(
            simulator,
            initial_state,
            task,
            compiled.commands,
            state_postprocessor=impulse,
        ).metrics
        feedback_summary = summarize_execution_metrics(feedback)
        open_summary = summarize_execution_metrics(open_loop)
        row = {
            **item,
            "feedback": feedback_summary,
            "compiled_open_loop": open_summary,
            "feedback_minus_compiled_success_rate": (
                feedback_summary["task_success_rate"]
                - open_summary["task_success_rate"]
            ),
        }
        disturbance_results.append(row)
        print(
            f"stage=disturbance case={item['name']} "
            f"feedback={feedback_summary['task_success_rate']:.6f} "
            f"compiled={open_summary['task_success_rate']:.6f}",
            flush=True,
        )
    _write_json(artifact / "post_planning_disturbance_results.json", disturbance_results)

    print("stage=latency", flush=True)
    latency_config = audit_config["latency"]
    latency = {
        "optimized_logical_batch_one": _benchmark_compilation(
            ppo_config,
            bank,
            checkpoint,
            fixed_evaluation_batch_size=1,
            warmup=int(latency_config["optimized_batch_one_warmup"]),
            repeats=int(latency_config["optimized_batch_one_repeats"]),
        ),
        "fixed_2048_residual_contract": _benchmark_compilation(
            ppo_config,
            bank,
            checkpoint,
            fixed_evaluation_batch_size=2048,
            warmup=int(latency_config["fixed_2048_warmup"]),
            repeats=int(latency_config["fixed_2048_repeats"]),
        ),
    }
    _write_json(artifact / "compilation_latency.json", latency)

    mismatch_advantages = [
        float(row["feedback_minus_compiled_success_rate"])
        for row in mismatch_results
    ]
    disturbance_advantages = [
        float(row["feedback_minus_compiled_success_rate"])
        for row in disturbance_results
    ]
    mean_mismatch = float(np.mean(mismatch_advantages))
    worst_mismatch = float(np.max(mismatch_advantages))
    mean_disturbance = float(np.mean(disturbance_advantages))
    optimized_p95_s = latency["optimized_logical_batch_one"]["p95_ms"] / 1000.0
    compiler_preferred = (
        optimized_p95_s <= 1.0
        and mean_mismatch <= 0.10
        and worst_mismatch <= 0.20
        and mean_disturbance <= 0.10
    )
    preferred = (
        "PPO_SIM_TRAJECTORY_COMPILER"
        if compiler_preferred
        else "TEN_HZ_CLOSED_LOOP_PPO"
    )
    failed_reasons = []
    if optimized_p95_s > 1.0:
        failed_reasons.append(
            f"optimized compilation p95 {optimized_p95_s:.3f} s exceeds 1.0 s"
        )
    if mean_mismatch > 0.10:
        failed_reasons.append(
            f"mean mismatch feedback advantage {100*mean_mismatch:.2f} pp exceeds 10 pp"
        )
    if worst_mismatch > 0.20:
        failed_reasons.append(
            f"worst mismatch feedback advantage {100*worst_mismatch:.2f} pp exceeds 20 pp"
        )
    if mean_disturbance > 0.10:
        failed_reasons.append(
            f"mean disturbance feedback advantage {100*mean_disturbance:.2f} pp exceeds 10 pp"
        )
    decision = {
        "schema": "ppo_feedback_dependence_architecture_decision_v1",
        "preferred_architecture": preferred,
        "exact_replay_pass": True,
        "decision_rule": {
            "optimized_compilation_p95_max_s": 1.0,
            "mean_mismatch_feedback_advantage_max": 0.10,
            "worst_mismatch_feedback_advantage_max": 0.20,
            "mean_disturbance_feedback_advantage_max": 0.10,
        },
        "optimized_compilation_p95_s": optimized_p95_s,
        "mean_mismatch_feedback_advantage": mean_mismatch,
        "worst_mismatch_feedback_advantage": worst_mismatch,
        "mean_disturbance_feedback_advantage": mean_disturbance,
        "failed_criteria": failed_reasons,
        "reason": (
            "All predeclared compiler criteria passed."
            if compiler_preferred
            else "; ".join(failed_reasons)
        ),
        "training_performed": False,
        "protected_test_evaluated": False,
        "hardware_executed": False,
    }
    _write_json(artifact / "architecture_decision.json", decision)

    source_paths = (
        Path(__file__).resolve(),
        ROOT / "learning" / "ppo_trajectory_compiler.py",
        ROOT / "learning" / "sequential_sac_env.py",
        ROOT / "learning" / "simple_ppo.py",
        config_path,
        ppo_config_path,
        checkpoint_path,
        ROOT / ppo_config["simulator_config"],
        ROOT / ppo_config["task_config"],
        ROOT / "config" / "active_model.json",
    )
    _write_json(
        artifact / "source_hash_manifest.json",
        {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
    )
    _write_report(
        artifact,
        audit_config=audit_config,
        checkpoint=checkpoint,
        exact=exact,
        nominal=nominal,
        mismatch=mismatch_results,
        disturbances=disturbance_results,
        latency=latency,
        decision=decision,
    )
    _write_json(
        artifact / "final_summary.json",
        {
            "new_training_episodes": 0,
            "new_cem_solves": 0,
            "state_bank_contexts": len(bank),
            "exact_replay": exact["classification"],
            "nominal_feedback_success_rate": nominal["feedback"]["task_success_rate"],
            "nominal_compiled_success_rate": nominal["compiled_open_loop"][
                "task_success_rate"
            ],
            "wrong_state_success_rate": nominal["wrong_initial_state_sequence"][
                "task_success_rate"
            ],
            "compilation_latency": latency,
            "decision": decision,
        },
    )
    print(f"stage=complete preferred={preferred}", flush=True)
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path)
    arguments = parser.parse_args()
    artifact = (
        arguments.artifact.resolve()
        if arguments.artifact is not None
        else ROOT
        / "data"
        / "policy_training"
        / "ppo_open_loop_compiler_v1"
        / _utc_stamp()
    )
    run(arguments.config.resolve(), artifact)


if __name__ == "__main__":
    main()
