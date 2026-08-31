"""Milestone 5B: train one nominal-physics one-shot terminal SAC policy."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from learning.context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    sample_context_specification,
)
from learning.normalization import FixedContextNormalizer
from learning.policy_context import build_policy_context, policy_context_tensor_metadata
from learning.replay import TerminalReplayBuffer
from learning.sac import TerminalSacAgent
from learning.state_bank import (
    InitialStateBank,
    generate_initial_state_bank,
    initial_state_bank_from_state,
)
from learning.training import (
    TrainingOutcome,
    ValidationSuite,
    evaluate_deterministic_policy,
    estimate_fixed_context_normalizer,
    load_training_checkpoint,
    run_terminal_sac_training,
)
from planning.artifacts import (
    final_replay_metrics,
    save_command_csv,
    save_replay_npz,
)
from planning.cem_task import load_variable_duration_task
from planning.results import load_planning_result
from planning.rollout import hover_preroll
from planning.video import render_replay_video
from simulator.parameters import SimulatorSettings
from simulator.production import (
    active_model_paths,
    build_production_simulator,
    load_active_model_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "learning" / "oneshot_sac_nominal_v1.json"
REPORT_PATH = PROJECT_ROOT / "MILESTONE5B_ONESHOT_SAC_NOMINAL_TRAINING_REPORT.md"


def _write_json(path: Path, payload: Any) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return safe(value.item())
        return value

    path.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace(":", "").replace("+0000", "Z")


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "oneshot_terminal_sac_nominal_v1":
        raise ValueError("Unsupported Milestone 5B configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("Milestone 5B must use the PRE_MPPI production freeze.")
    if config.get("context_dimension") != 83 or config.get("action_dimension") != 49:
        raise ValueError("Milestone 5B context/action dimensions are frozen at 83/49.")
    if not config.get("nominal_physics_only"):
        raise ValueError("Milestone 5B forbids physics randomization.")
    policy = config["artifact_policy"]
    if policy["real_flight_authorized"] or policy["protected_test_evaluation_allowed"]:
        raise ValueError("Milestone 5B must remain simulated and protect the sealed test.")
    if policy["cem_training_data_allowed"]:
        raise ValueError("CEM data cannot enter Milestone 5B training.")


def _canonical_specification(simulator, canonical_bank, task) -> ContextSpecification:
    selected = canonical_bank.select(torch.tensor([0]), device=simulator.device)
    context = build_policy_context(
        simulator,
        selected.state,
        target_position_world_m=torch.tensor(task.target_position_m, device=simulator.device),
        desired_direction_world=torch.tensor(task.desired_direction, device=simulator.device),
        command_initial_position_world_m=selected.command_position_world_m,
        command_initial_velocity_world_m_s=selected.command_velocity_world_m_s,
        command_yaw_world_rad=selected.command_yaw_world_rad,
    )
    return ContextSpecification(
        torch.tensor([0]),
        context.target_position_local_m.detach().cpu(),
        context.target_direction_local.detach().cpu(),
        "canonical",
    )


def _record_policy_row(
    agent: TerminalSacAgent,
    normalizer: FixedContextNormalizer,
    simulator,
    task,
    bank,
    specification: ContextSpecification,
):
    context = build_context_from_specification(simulator, bank, specification)
    with torch.no_grad():
        action = agent.actor(normalizer.normalize(context.to_tensor()), deterministic=True).deterministic_mean_action
        result = __import__("learning.one_shot_env", fromlist=["evaluate_open_loop_batch"]).evaluate_open_loop_batch(
            simulator, context, action, task, record_trajectory=True
        )
    if result.trajectory is None:
        raise RuntimeError("Requested deterministic policy trajectory was not recorded.")
    return result, result.trajectory


def _source_hash_manifest(config_path: Path, extra: list[Path]) -> dict[str, Any]:
    paths = [
        config_path,
        PROJECT_ROOT / "learning" / "policy_context.py",
        PROJECT_ROOT / "learning" / "policy_action.py",
        PROJECT_ROOT / "learning" / "one_shot_env.py",
        PROJECT_ROOT / "learning" / "sac.py",
        PROJECT_ROOT / "learning" / "state_bank.py",
        PROJECT_ROOT / "learning" / "context_sampling.py",
        PROJECT_ROOT / "learning" / "normalization.py",
        PROJECT_ROOT / "learning" / "replay.py",
        PROJECT_ROOT / "learning" / "training.py",
        Path(__file__),
        *extra,
    ]
    return {
        "schema": "oneshot_sac_nominal_source_hashes_v1",
        "sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in paths
            if path.is_file()
        },
    }


def _write_training_csv(path: Path, history: tuple[dict[str, Any], ...]) -> None:
    if not history:
        path.write_text("episodes\n", encoding="utf-8")
        return
    keys = sorted({key for row in history for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _write_report(
    *,
    artifact_directory: Path,
    config: dict[str, Any],
    outcome,
    final_validation: dict[str, Any],
    canonical_metrics: dict[str, Any],
    runtime: dict[str, Any],
    freeze_path: Path | None,
    video_path: Path,
) -> None:
    canonical = final_validation["canonical"]
    iid = final_validation["held_out_iid"]
    edge = final_validation["edge_domain"]
    progression = "\n".join(
        f"| {row['episodes']} | {100*row['canonical']['hard_success_rate']:.1f}% | "
        f"{100*row['held_out_iid']['hard_success_rate']:.1f}% | "
        f"{100*row['held_out_iid']['feasible_rate']:.1f}% | "
        f"{100*row['edge_domain']['hard_success_rate']:.1f}% |"
        for row in outcome.evaluation_history
    ) or "| No validation checkpoint completed | — | — | — | — |"
    first = canonical_metrics
    report = f"""# Milestone 5B — One-Shot SAC Nominal-Physics Training Report

## 1. Scientific question

Milestone 5B tested whether a stochastic policy can map an initial UAV/cable state, target, desired strike direction, and nominal effective physics context to one complete variable-duration whip maneuver. The policy is queried exactly once. Its decoded FullState trajectory is then executed completely open loop with no callback, observation feedback, state correction, replanning, or receding horizon.

## 2. Frozen model and methodological boundary

The simulator remained `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`: frozen UAV physics, frozen causal UAV residual, rigid attachment, and frozen 12-node DDER using CUDA float32, PCG32, three DDER substeps, and four projections. EI, Cb, geometry, masses, gains, residual weights, and solver settings were not changed.

Theta remained present in every 83-D context but was nominal and identical for all training and validation episodes. This result therefore tests state/goal-conditioned control under nominal physics; it does not demonstrate learned physics conditioning. CEM actions, CEM elites, protected physical data, and real hardware were not used.

## 3. Context and action contracts

The context is the unchanged root-centered, yaw-aligned, gravity-preserving 83-D Milestone 5A tensor: UAV velocity/orientation/angular velocity; c1...c10 positions and velocities; local target and unit strike direction; and Kp, Kv, ka, KR, Komega, log(EI), log(Cb).

The action remains 49-D: sixteen three-axis acceleration knots plus one duration in [0.45, 1.20] s. FullState position and velocity are generated by exact integration of the linearly interpolated acceleration. Yaw remains the query-boundary yaw and omega command is zero.

## 4. Stochastic SAC distribution and Jacobian

Each raw Gaussian three-vector z is mapped into the open unit L2 ball with `y = tanh(||z||) z / ||z||`. Its analytic log-Jacobian is `2 log(tanh(r)/r) + log(sech(r)^2)` with a series expansion at small r. Duration uses scalar tanh with the stable standard Jacobian. Entropy is defined in normalized action coordinates; constant physical scales are excluded. The analytic radial determinant was checked against autograd 3x3 Jacobians, including zero and near-zero norms.

## 5. One-shot terminal SAC

One replay entry is `(context, complete normalized maneuver, terminal reward)`. Twin critic targets are exactly the scaled terminal reward, with no next state, discount bootstrap, or target critic. Actor loss is `alpha*log_pi - min(Q1,Q2)`. The actor and both critics use three 256-unit SiLU layers. Automatic entropy temperature uses target entropy -49.

## 6. Data generation and normalization

Training states: {config['state_banks']['training_count']} nominal-simulator causal states, seed {config['state_banks']['training_seed']}. Validation states: {config['state_banks']['validation_count']} separately generated states, seed {config['state_banks']['validation_seed']}. States were produced by smooth excitations bounded by 3 m/s² and include the full UAV state, c1...c10 state, residual FIFO, and causal FullState command boundary. Marker positions were never independently perturbed.

The fixed context normalizer was fitted before any gradient update from {config['context_normalization']['training_samples']} training contexts only. Near-zero-variance dimensions use scale one. Validation contexts never entered replay.

Targets used the frozen local Phase-1 domain x=[0.85,1.05] m, y=[-0.15,0.15] m, z=[-0.12,0.02] m, with horizontal root-to-target direction. Training used 20% near-canonical and 80% uniform Phase-1 contexts.

## 7. Reward, replay, and optimizer

The unchanged reward was `legacy_run_online_strike_margin_tuned_v4`, divided by 100 only for SAC numerical scale. Hard success gates were not changed. Replay capacity was 1,000,000 terminal episodes. No full trajectories were stored in replay. The run used seed 42, Adam at 3e-4, minibatches of 4096, 2048 simulator episodes per collection, four update cycles per collection after 50,000 initial actor-only exploration episodes, and gradient clipping at 10.

## 8. Validation progression

| Collected episodes | Canonical success | Held-out IID success | Held-out IID feasible | Edge success |
|---:|---:|---:|---:|---:|
{progression}

## 9. Final best-checkpoint evaluation

| Split | Success | Feasible | Finite | Tip-first | Median tip error | Median directed speed | Median direction error | Mean duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical | {100*canonical['hard_success_rate']:.2f}% | {100*canonical['feasible_rate']:.2f}% | {100*canonical['finite_rate']:.2f}% | {100*canonical['tip_first_rate']:.2f}% | {1000*canonical['median_tip_error_m']:.3f} mm | {canonical['median_directed_tip_speed_m_s']:.3f} m/s | {canonical['median_direction_error_deg']:.3f} deg | {canonical['mean_maneuver_duration_s']:.4f} s |
| Held-out IID | {100*iid['hard_success_rate']:.2f}% | {100*iid['feasible_rate']:.2f}% | {100*iid['finite_rate']:.2f}% | {100*iid['tip_first_rate']:.2f}% | {1000*iid['median_tip_error_m']:.3f} mm | {iid['median_directed_tip_speed_m_s']:.3f} m/s | {iid['median_direction_error_deg']:.3f} deg | {iid['mean_maneuver_duration_s']:.4f} s |
| Edge domain | {100*edge['hard_success_rate']:.2f}% | {100*edge['feasible_rate']:.2f}% | {100*edge['finite_rate']:.2f}% | {100*edge['tip_first_rate']:.2f}% | {1000*edge['median_tip_error_m']:.3f} mm | {edge['median_directed_tip_speed_m_s']:.3f} m/s | {edge['median_direction_error_deg']:.3f} deg | {edge['mean_maneuver_duration_s']:.4f} s |

Canonical deterministic result: success={first['success']}; tip error={1000*float(first['minimum_tip_target_distance_m']):.3f} mm; directed speed={float(first['reported_event_directed_tip_speed_m_s']):.3f} m/s; direction error={float(first['reported_event_direction_error_deg']):.3f} deg; duration={float(first['optimized_duration_s']):.4f} s.

Saved CEM reference only: tip error 1.756 mm, directed speed 4.599 m/s, direction error 19.615 deg, duration 1.1175 s. It was not used for initialization, replay, supervision, or reward shaping.

## 10. Runtime, diagnostics, and artifacts

Collected episodes: {outcome.episodes}. Gradient updates: {outcome.gradient_updates}. Training runtime: {outcome.runtime_s:.2f} s. Overall artifact runtime: {runtime['total_runtime_s']:.2f} s. Effective collection throughput: {runtime['mean_collection_episodes_per_s']:.1f} episodes/s. Peak CUDA allocation: {runtime['peak_cuda_memory_mb']:.1f} MB.

Training classification: **{outcome.classification}**. Policy freeze: `{str(freeze_path) if freeze_path else 'NOT CREATED'}`. Artifact directory: `{artifact_directory}`. Canonical video: `{video_path}`.

Protected test: **NOT EVALUATED**. Real hardware: **NOT EXECUTED**. Generated actions remain simulation-only and are not authorized for flight.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Learner:
        One-Shot Terminal SAC

    Policy query count per maneuver:
        1

    Execution:
        OPEN LOOP

    Physics during 5B:
        NOMINAL ONLY

    Context dimension:
        83

    Action dimension:
        49

    Reward:
        legacy_run_online_strike_margin_tuned_v4

    Training episodes:
        {outcome.episodes}

    Training runtime:
        {outcome.runtime_s:.2f} s

    Canonical:
        {'PASS' if canonical['hard_success_rate'] == 1.0 else 'FAIL'}
        tip error = {1000*float(first['minimum_tip_target_distance_m']):.3f} mm
        directed speed = {float(first['reported_event_directed_tip_speed_m_s']):.3f} m/s
        direction error = {float(first['reported_event_direction_error_deg']):.3f} deg
        duration = {float(first['optimized_duration_s']):.4f} s

    Held-out IID:
        success = {100*iid['hard_success_rate']:.2f} %
        feasible = {100*iid['feasible_rate']:.2f} %

    Edge-domain:
        success = {100*edge['hard_success_rate']:.2f} %

    SAC:
        {outcome.classification}

    CEM bootstrap:
        NOT USED

    Policy freeze:
        {freeze_path.name if freeze_path else 'NOT CREATED'}

    Physics-conditioning learned:
        NO — DEFERRED TO 5C

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    overall_start = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)

    artifact_directory = (
        PROJECT_ROOT / "data" / "policy_training" / config["run_id"] / _timestamp()
    )
    artifact_directory.mkdir(parents=True, exist_ok=False)
    _write_json(artifact_directory / "policy_config.json", config)
    _write_json(artifact_directory / "context_schema.json", policy_context_tensor_metadata())
    _write_json(
        artifact_directory / "stochastic_action_transform.json",
        {
            "schema": "radial_squashed_gaussian_action_v1",
            "acceleration_groups": 16,
            "group_dimension": 3,
            "transform": "y=tanh(norm(z))*z/norm(z)",
            "log_abs_det": "2*log(tanh(r)/r)+log(sech(r)^2)",
            "duration_transform": "tanh",
            "entropy_units": "normalized_action_space",
        },
    )

    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    task_path = (PROJECT_ROOT / config["task_config"]).resolve()
    task = load_variable_duration_task(task_path)
    task.validate_for_dt(simulator.dt_s)
    _write_json(artifact_directory / "reward_config_snapshot.json", task.snapshot())
    _write_json(artifact_directory / "target_domain.json", config["target_domain_local_m"])

    bank_config = config["state_banks"]
    training_bank = generate_initial_state_bank(
        simulator,
        task,
        count=int(bank_config["training_count"]),
        seed=int(bank_config["training_seed"]),
        logical_batch_size=int(bank_config["logical_generation_batch_size"]),
        excitation_max_acceleration_m_s2=float(bank_config["excitation_max_acceleration_m_s2"]),
        generation_horizon_s=float(bank_config["generation_horizon_s"]),
        excitation_duration_range_s=tuple(bank_config["excitation_duration_range_s"]),
        maximum_state_speed_m_s=float(bank_config["maximum_state_speed_m_s"]),
        maximum_state_displacement_m=float(bank_config["maximum_state_displacement_m"]),
    )
    training_bank.save(
        artifact_directory / "training_state_bank.npz",
        artifact_directory / "training_state_bank_manifest.json",
    )
    validation_bank = generate_initial_state_bank(
        simulator,
        task,
        count=int(bank_config["validation_count"]),
        seed=int(bank_config["validation_seed"]),
        logical_batch_size=int(bank_config["logical_generation_batch_size"]),
        excitation_max_acceleration_m_s2=float(bank_config["excitation_max_acceleration_m_s2"]),
        generation_horizon_s=float(bank_config["generation_horizon_s"]),
        excitation_duration_range_s=tuple(bank_config["excitation_duration_range_s"]),
        maximum_state_speed_m_s=float(bank_config["maximum_state_speed_m_s"]),
        maximum_state_displacement_m=float(bank_config["maximum_state_displacement_m"]),
    )
    validation_bank.save(
        artifact_directory / "validation_state_bank.npz",
        artifact_directory / "validation_state_bank_manifest.json",
    )
    base = hover_preroll(simulator, task)
    canonical_bank = initial_state_bank_from_state(
        base,
        command_position_world_m=torch.tensor(task.initial_uav_position_m, device=simulator.device),
        command_velocity_world_m_s=torch.tensor(task.initial_uav_velocity_m_s, device=simulator.device),
        command_yaw_world_rad=task.initial_yaw_rad,
        seed=int(config["seed"]),
    )
    canonical = _canonical_specification(simulator, canonical_bank, task)
    iid_generator = torch.Generator().manual_seed(int(config["validation"]["iid_seed"]))
    edge_generator = torch.Generator().manual_seed(int(config["validation"]["edge_seed"]))
    validation = ValidationSuite(
        canonical,
        sample_context_specification(
            validation_bank,
            count=int(config["validation"]["held_out_iid_count"]),
            generator=iid_generator,
            split="validation_iid",
        ),
        sample_context_specification(
            validation_bank,
            count=int(config["validation"]["edge_count"]),
            generator=edge_generator,
            split="edge",
        ),
    )
    validation.save(artifact_directory / "validation_contexts.npz")

    normalizer_config = config["context_normalization"]
    normalizer = estimate_fixed_context_normalizer(
        simulator,
        training_bank,
        sample_count=int(normalizer_config["training_samples"]),
        sample_batch_size=int(normalizer_config["sample_batch_size"]),
        seed=int(config["seed"]) + 500,
        standard_deviation_floor=float(normalizer_config["standard_deviation_floor"]),
        canonical_fraction=float(config["target_domain_local_m"]["canonical_training_fraction"]),
    )
    normalizer.save(artifact_directory / "context_normalizer.json")
    sac = config["sac"]
    agent = TerminalSacAgent.create(
        device=simulator.device,
        actor_lr=float(sac["actor_learning_rate"]),
        critic_lr=float(sac["critic_learning_rate"]),
        alpha_lr=float(sac["alpha_learning_rate"]),
        target_entropy=float(sac["target_entropy"]),
        gradient_clip_norm=float(sac["gradient_clip_norm"]),
    )
    replay = TerminalReplayBuffer(int(sac["replay_capacity"]))
    outcome = run_terminal_sac_training(
        agent=agent,
        normalizer=normalizer,
        replay=replay,
        simulator=simulator,
        task=task,
        training_bank=training_bank,
        canonical_bank=canonical_bank,
        validation_bank=validation_bank,
        validation=validation,
        config=config,
        artifact_directory=artifact_directory,
    )
    # The best validation checkpoint, not the final training step, is authoritative.
    restore_generator = torch.Generator().manual_seed(0)
    load_training_checkpoint(outcome.best_checkpoint, agent=agent, context_generator=restore_generator)
    final_validation = {
        "canonical": evaluate_deterministic_policy(agent, normalizer, simulator, task, canonical_bank, canonical),
        "held_out_iid": evaluate_deterministic_policy(agent, normalizer, simulator, task, validation_bank, validation.held_out_iid),
        "edge_domain": evaluate_deterministic_policy(agent, normalizer, simulator, task, validation_bank, validation.edge_domain),
    }
    _write_json(artifact_directory / "final_validation_metrics.json", final_validation)
    torch.save(agent.actor.state_dict(), artifact_directory / "best_actor.pt")
    torch.save(agent.critic1.state_dict(), artifact_directory / "best_critic1.pt")
    torch.save(agent.critic2.state_dict(), artifact_directory / "best_critic2.pt")
    _write_json(artifact_directory / "best_alpha.json", {"alpha": float(agent.alpha.detach())})
    _write_json(artifact_directory / "training_history.json", list(outcome.training_history))
    _write_training_csv(artifact_directory / "training_history.csv", outcome.training_history)
    _write_json(artifact_directory / "evaluation_history.json", list(outcome.evaluation_history))

    canonical_result, canonical_replay = _record_policy_row(
        agent, normalizer, simulator, task, canonical_bank, canonical
    )
    save_command_csv(artifact_directory / "canonical_fullstate_command.csv", canonical_replay)
    save_replay_npz(artifact_directory / "canonical_final_replay.npz", canonical_replay, task)
    canonical_metrics = final_replay_metrics(canonical_replay, task, settings)
    canonical_metrics.update(
        {
            "optimizer": "One-Shot SAC",
            "optimized_duration_s": float(canonical_result.decoded_action.duration_s[0]),
            "task_classification": "PASS" if bool(canonical_result.task_success[0]) else "FAIL",
            "policy_query_count": 1,
            "execution": "OPEN_LOOP",
        }
    )
    _write_json(artifact_directory / "canonical_final_metrics.json", canonical_metrics)

    # Make the saved policy result consumable by the established replay/video API.
    video_task = task.snapshot()
    video_task["task_id"] = "oneshot_sac_nominal_canonical"
    _write_json(artifact_directory / "task_config_snapshot.json", video_task)
    _write_json(artifact_directory / "final_metrics.json", canonical_metrics)
    _write_json(artifact_directory / "mppi_iteration_history.json", list(outcome.evaluation_history))
    shutil.copy2(artifact_directory / "canonical_final_replay.npz", artifact_directory / "final_replay.npz")
    video_path, video_metadata = render_replay_video(load_planning_result(artifact_directory))
    expected_video = artifact_directory / "oneshot_sac_nominal_canonical_final_replay.mp4"
    if video_path != expected_video:
        shutil.copy2(video_path, expected_video)
        video_path = expected_video
    _write_json(artifact_directory / "video_metadata.json", video_metadata)

    success_condition = (
        final_validation["canonical"]["hard_success_rate"] == 1.0
        and final_validation["held_out_iid"]["hard_success_rate"] >= 0.60
        and final_validation["held_out_iid"]["feasible_rate"] >= 0.95
        and outcome.classification == "TRAINED"
    )
    freeze_path: Path | None = None
    if success_condition:
        freeze_path = PROJECT_ROOT / "data" / "policy_freezes" / "POLICY_FREEZE_ONESHOT_SAC_NOMINAL_V1"
        freeze_path.mkdir(parents=True, exist_ok=False)
        for name in (
            "best_actor.pt",
            "context_normalizer.json",
            "context_schema.json",
            "stochastic_action_transform.json",
            "policy_config.json",
            "validation_contexts.npz",
            "final_validation_metrics.json",
        ):
            shutil.copy2(artifact_directory / name, freeze_path / name)
        _write_json(
            freeze_path / "manifest.json",
            {
                "schema": "oneshot_sac_nominal_policy_freeze_v1",
                "status": "SIMULATION_POLICY_NOMINAL_PHYSICS",
                "model_freeze": config["model_freeze"],
                "context_dimension": 83,
                "action_dimension": 49,
                "reward_profile": config["reward"]["profile"],
                "physics_conditioning_learned": False,
                "training_artifact": str(artifact_directory),
                "real_flight_authorized": False,
                "protected_test": "NOT EVALUATED",
                "hashes": {
                    name: _sha256(freeze_path / name)
                    for name in ("best_actor.pt", "context_normalizer.json", "validation_contexts.npz")
                },
            },
        )

    source_manifest = _source_hash_manifest(config_path, [task_path])
    _write_json(artifact_directory / "source_hash_manifest.json", source_manifest)
    collection_rates = [row["episodes_per_s"] for row in outcome.training_history]
    runtime = {
        "total_runtime_s": time.perf_counter() - overall_start,
        "training_runtime_s": outcome.runtime_s,
        "collected_episodes": outcome.episodes,
        "gradient_updates": outcome.gradient_updates,
        "mean_collection_episodes_per_s": sum(collection_rates) / max(len(collection_rates), 1),
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(simulator.device) / 1024**2,
        "device": torch.cuda.get_device_name(simulator.device),
    }
    _write_json(artifact_directory / "training_runtime.json", runtime)
    _write_report(
        artifact_directory=artifact_directory,
        config=config,
        outcome=outcome,
        final_validation=final_validation,
        canonical_metrics=canonical_metrics,
        runtime=runtime,
        freeze_path=freeze_path,
        video_path=video_path,
    )
    result = {
        "classification": outcome.classification,
        "artifact_directory": str(artifact_directory),
        "report": str(REPORT_PATH),
        "video": str(video_path),
        "policy_freeze": None if freeze_path is None else str(freeze_path),
        "episodes": outcome.episodes,
        "canonical_success": bool(canonical_result.task_success[0]),
        "iid_success_rate": final_validation["held_out_iid"]["hard_success_rate"],
        "protected_test": "NOT EVALUATED",
        "real_hardware": "NOT EXECUTED",
    }
    print("MILESTONE5B_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def _load_validation_suite(path: Path) -> ValidationSuite:
    with np.load(path, allow_pickle=False) as archive:
        def specification(prefix: str, split: str) -> ContextSpecification:
            return ContextSpecification(
                torch.from_numpy(np.asarray(archive[f"{prefix}_state_indices"]).copy()),
                torch.from_numpy(np.asarray(archive[f"{prefix}_target_position_local_m"]).copy()),
                torch.from_numpy(np.asarray(archive[f"{prefix}_desired_direction_local"]).copy()),
                split,
            )

        return ValidationSuite(
            specification("canonical", "canonical"),
            specification("iid", "validation_iid"),
            specification("edge", "edge"),
        )


def finalize_existing(artifact_directory: Path, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Resume only post-training evaluation/artifacts after an interrupted finalizer."""

    start = time.perf_counter()
    artifact_directory = artifact_directory.resolve()
    config = json.loads((artifact_directory / "policy_config.json").read_text(encoding="utf-8"))
    _validate_config(config)
    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    task_path = (PROJECT_ROOT / config["task_config"]).resolve()
    task = load_variable_duration_task(task_path)
    validation_bank = InitialStateBank.load(
        artifact_directory / "validation_state_bank.npz",
        artifact_directory / "validation_state_bank_manifest.json",
    )
    validation = _load_validation_suite(artifact_directory / "validation_contexts.npz")
    base = hover_preroll(simulator, task)
    canonical_bank = initial_state_bank_from_state(
        base,
        command_position_world_m=torch.tensor(task.initial_uav_position_m, device=simulator.device),
        command_velocity_world_m_s=torch.tensor(task.initial_uav_velocity_m_s, device=simulator.device),
        command_yaw_world_rad=task.initial_yaw_rad,
        seed=int(config["seed"]),
    )
    normalizer = FixedContextNormalizer.load(artifact_directory / "context_normalizer.json")
    sac = config["sac"]
    agent = TerminalSacAgent.create(
        device=simulator.device,
        actor_lr=float(sac["actor_learning_rate"]),
        critic_lr=float(sac["critic_learning_rate"]),
        alpha_lr=float(sac["alpha_learning_rate"]),
        target_entropy=float(sac["target_entropy"]),
        gradient_clip_norm=float(sac["gradient_clip_norm"]),
    )
    checkpoint_directory = artifact_directory / "checkpoints"
    latest_payload = torch.load(
        checkpoint_directory / "latest.pt", map_location="cpu", weights_only=False
    )
    restore_generator = torch.Generator().manual_seed(0)
    load_training_checkpoint(
        checkpoint_directory / "best_held_out_iid.pt",
        agent=agent,
        context_generator=restore_generator,
    )
    training_history = tuple(latest_payload["training_history"])
    evaluation_history = tuple(latest_payload["evaluation_history"])
    canonical_ever = max(
        (row["canonical"]["hard_success_rate"] for row in evaluation_history), default=0.0
    )
    iid_best = max(
        (row["held_out_iid"]["hard_success_rate"] for row in evaluation_history), default=0.0
    )
    classification = (
        "SAC_EXPLORATION_LIMITED"
        if canonical_ever < 1.0 or iid_best < 0.10
        else "SAC_TRAINING_FAILED"
    )
    training_runtime_s = float(training_history[-1]["elapsed_s"]) if training_history else 0.0
    outcome = TrainingOutcome(
        classification,
        int(latest_payload["episodes"]),
        int(latest_payload["gradient_updates"]),
        training_runtime_s,
        checkpoint_directory / "best_held_out_iid.pt",
        checkpoint_directory / "latest.pt",
        training_history,
        evaluation_history,
    )
    final_validation = {
        "canonical": evaluate_deterministic_policy(
            agent, normalizer, simulator, task, canonical_bank, validation.canonical
        ),
        "held_out_iid": evaluate_deterministic_policy(
            agent, normalizer, simulator, task, validation_bank, validation.held_out_iid
        ),
        "edge_domain": evaluate_deterministic_policy(
            agent, normalizer, simulator, task, validation_bank, validation.edge_domain
        ),
    }
    _write_json(artifact_directory / "final_validation_metrics.json", final_validation)
    torch.save(agent.actor.state_dict(), artifact_directory / "best_actor.pt")
    torch.save(agent.critic1.state_dict(), artifact_directory / "best_critic1.pt")
    torch.save(agent.critic2.state_dict(), artifact_directory / "best_critic2.pt")
    _write_json(artifact_directory / "best_alpha.json", {"alpha": float(agent.alpha.detach())})
    _write_json(artifact_directory / "training_history.json", list(training_history))
    _write_training_csv(artifact_directory / "training_history.csv", training_history)
    _write_json(artifact_directory / "evaluation_history.json", list(evaluation_history))

    canonical_result, canonical_replay = _record_policy_row(
        agent, normalizer, simulator, task, canonical_bank, validation.canonical
    )
    save_command_csv(artifact_directory / "canonical_fullstate_command.csv", canonical_replay)
    save_replay_npz(artifact_directory / "canonical_final_replay.npz", canonical_replay, task)
    canonical_metrics = final_replay_metrics(canonical_replay, task, settings)
    canonical_metrics.update(
        {
            "optimizer": "One-Shot SAC",
            "optimized_duration_s": float(canonical_result.decoded_action.duration_s[0]),
            "task_classification": "PASS" if bool(canonical_result.task_success[0]) else "FAIL",
            "policy_query_count": 1,
            "execution": "OPEN_LOOP",
        }
    )
    _write_json(artifact_directory / "canonical_final_metrics.json", canonical_metrics)

    # There was no held-out success; preserve one deterministic failure for diagnosis.
    failure_spec = ContextSpecification(
        validation.held_out_iid.state_indices[:1],
        validation.held_out_iid.target_position_local_m[:1],
        validation.held_out_iid.desired_direction_local[:1],
        "representative_failure",
    )
    _, failure_replay = _record_policy_row(
        agent, normalizer, simulator, task, validation_bank, failure_spec
    )
    save_replay_npz(
        artifact_directory / "representative_failure_replay.npz", failure_replay, task
    )

    video_task = task.snapshot()
    video_task["task_id"] = "oneshot_sac_nominal_canonical"
    _write_json(artifact_directory / "task_config_snapshot.json", video_task)
    _write_json(artifact_directory / "final_metrics.json", canonical_metrics)
    _write_json(artifact_directory / "mppi_iteration_history.json", list(evaluation_history))
    shutil.copy2(
        artifact_directory / "canonical_final_replay.npz",
        artifact_directory / "final_replay.npz",
    )
    video_path, video_metadata = render_replay_video(
        load_planning_result(artifact_directory), overwrite=True
    )
    _write_json(artifact_directory / "video_metadata.json", video_metadata)
    _write_json(
        artifact_directory / "source_hash_manifest.json",
        _source_hash_manifest(config_path, [task_path]),
    )
    collection_rates = [row["episodes_per_s"] for row in training_history]
    runtime = {
        "total_runtime_s": training_runtime_s + (time.perf_counter() - start),
        "training_runtime_s": training_runtime_s,
        "finalization_recovery_runtime_s": time.perf_counter() - start,
        "collected_episodes": outcome.episodes,
        "gradient_updates": outcome.gradient_updates,
        "mean_collection_episodes_per_s": sum(collection_rates) / max(len(collection_rates), 1),
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(simulator.device) / 1024**2,
        "device": torch.cuda.get_device_name(simulator.device),
        "training_was_rerun_during_recovery": False,
    }
    _write_json(artifact_directory / "training_runtime.json", runtime)
    _write_report(
        artifact_directory=artifact_directory,
        config=config,
        outcome=outcome,
        final_validation=final_validation,
        canonical_metrics=canonical_metrics,
        runtime=runtime,
        freeze_path=None,
        video_path=video_path,
    )
    result = {
        "classification": classification,
        "artifact_directory": str(artifact_directory),
        "report": str(REPORT_PATH),
        "video": str(video_path),
        "policy_freeze": None,
        "episodes": outcome.episodes,
        "canonical_success": bool(canonical_result.task_success[0]),
        "iid_success_rate": final_validation["held_out_iid"]["hard_success_rate"],
        "training_rerun": False,
        "protected_test": "NOT EVALUATED",
        "real_hardware": "NOT EXECUTED",
    }
    print("MILESTONE5B_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--finalize-existing",
        type=Path,
        help="Resume post-training artifacts from a completed checkpoint without collecting episodes.",
    )
    arguments = parser.parse_args()
    if arguments.finalize_existing is None:
        run(arguments.config.resolve())
    else:
        finalize_existing(arguments.finalize_existing, arguments.config.resolve())


if __name__ == "__main__":
    main()
