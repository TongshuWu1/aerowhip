from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
from threading import Event

from drone_mpc.model import load_cable_model
from drone_mpc.rl_env import TaskDistribution
from drone_mpc.sac import (
    SacSettings,
    SacTrainingUpdate,
    SacValidationPreview,
    train_sac,
)
from optitrack_offline.config import DEFAULT_MODEL_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the full-state 3-D cable-whip SAC policy."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/drone_mpc/sac_policy.pt"),
    )
    parser.add_argument("--transitions", type=int, default=500_000)
    parser.add_argument(
        "--endless",
        action="store_true",
        help="Train until Ctrl+C, the UI Stop button, or --stop-file requests a stop.",
    )
    parser.add_argument("--environments", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=100_000)
    parser.add_argument("--replay-capacity", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--nodes",
        type=int,
        default=15,
        help="DER nodes used by the SAC plant; 15 is the validated runtime/accuracy point.",
    )
    parser.add_argument("--evaluation-episodes", type=int, default=256)
    parser.add_argument(
        "--checkpoint-evaluation-episodes",
        type=int,
        default=None,
        help=(
            "Fixed validation targets used for checkpoint selection. The default "
            "is min(64, --evaluation-episodes)."
        ),
    )
    parser.add_argument("--actor-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--degradation-patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--her-goals-per-episode",
        type=int,
        default=2,
        help="Future achieved tip goals relabeled from every completed episode.",
    )
    parser.add_argument(
        "--her-minimum-achieved-speed",
        type=float,
        default=0.25,
        help="Ignore nearly static achieved states when selecting HER goals (m/s).",
    )
    parser.add_argument(
        "--checkpoint-validation-seed",
        type=int,
        default=None,
        help=(
            "Fixed target seed used to select checkpoints. By default this is "
            "the training seed plus 1000."
        ),
    )
    parser.add_argument(
        "--final-test-seed",
        type=int,
        default=None,
        help=(
            "Fixed target seed used for the final held-out evaluation. By "
            "default this is the training seed plus 10000."
        ),
    )
    parser.add_argument(
        "--target-distance-min",
        type=float,
        default=0.80,
        help="Minimum horizontal target radius from the initial drone (m).",
    )
    parser.add_argument(
        "--target-distance-max",
        type=float,
        default=0.80,
        help="Maximum radius; values may exceed the cable's initial static reach (m).",
    )
    parser.add_argument(
        "--target-height-min",
        type=float,
        default=-0.10,
        help="Minimum target-height offset relative to the initial drone (m).",
    )
    parser.add_argument(
        "--target-height-max",
        type=float,
        default=-0.10,
        help="Maximum target-height offset relative to the initial drone (m).",
    )
    parser.add_argument(
        "--target-azimuth-min",
        type=float,
        default=0.0,
        help="Minimum target azimuth around the initial drone (deg).",
    )
    parser.add_argument(
        "--target-azimuth-max",
        type=float,
        default=0.0,
        help="Maximum target azimuth around the initial drone (deg).",
    )
    parser.add_argument(
        "--impact-azimuth-offset-min",
        type=float,
        default=0.0,
        help="Minimum impact-vector yaw relative to the radial target direction (deg).",
    )
    parser.add_argument(
        "--impact-azimuth-offset-max",
        type=float,
        default=0.0,
        help="Maximum impact-vector yaw relative to the radial target direction (deg).",
    )
    parser.add_argument(
        "--impact-elevation",
        type=float,
        default=0.0,
        help="Desired free-tip impact elevation angle in the vertical target plane (deg).",
    )
    parser.add_argument(
        "--impact-angle",
        type=float,
        default=35.0,
        help="Allowed angular deviation from the desired impact direction (deg).",
    )
    parser.add_argument(
        "--minimum-extension-ratio",
        type=float,
        default=0.0,
        help=(
            "Minimum attachment-to-tip chord divided by cable length for a "
            "valid impact."
        ),
    )
    parser.add_argument(
        "--impact-speed",
        type=float,
        default=1.5,
        help="Fixed required directed cable-tip speed at impact (m/s).",
    )
    parser.add_argument("--target-keepout", type=float, default=0.25)
    parser.add_argument(
        "--hit-tolerance",
        type=float,
        default=0.05,
        help="Physical target radius used by swept tip-contact detection (m).",
    )
    parser.add_argument(
        "--progress-reward",
        type=float,
        default=5.0,
        help="Bounded tip-to-target potential shaping weight.",
    )
    parser.add_argument(
        "--strike-event-reward",
        type=float,
        default=0.0,
        help="Maximum bounded reward for the quality of a target-plane crossing.",
    )
    parser.add_argument(
        "--strike-accuracy-decay",
        type=float,
        default=2.5,
        help="Exponential target-plane miss-distance decay in inverse metres.",
    )
    parser.add_argument(
        "--success-reward",
        type=float,
        default=100.0,
        help="Dominant bonus for a geometrically and dynamically valid target hit.",
    )
    parser.add_argument(
        "--impact-drone-displacement-penalty",
        type=float,
        default=2.0,
        help="Weight on squared drone displacement from its start at cable-tip impact.",
    )
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit strict structured progress lines prefixed with SAC_EVENT_JSON=.",
    )
    parser.add_argument(
        "--validation-preview",
        action="store_true",
        help=(
            "Emit one deterministic validation trajectory at each checkpoint "
            "for UI playback. This does not add an evaluation rollout."
        ),
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="Stop cleanly when this file exists; the trainer does not delete it.",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    snapshot = load_cable_model(arguments.model)
    settings = SacSettings(
        total_transitions=arguments.transitions,
        environment_count=arguments.environments,
        replay_capacity=arguments.replay_capacity,
        warmup_transitions=arguments.warmup,
        batch_size=arguments.batch_size,
        actor_learning_rate=arguments.actor_learning_rate,
        critic_learning_rate=arguments.critic_learning_rate,
        entropy_learning_rate=arguments.entropy_learning_rate,
        controller_node_count=arguments.nodes,
        evaluation_episodes=arguments.evaluation_episodes,
        degradation_patience_evaluations=arguments.degradation_patience,
        hindsight_goals_per_episode=arguments.her_goals_per_episode,
        hindsight_minimum_achieved_speed_m_s=(
            arguments.her_minimum_achieved_speed
        ),
        seed=arguments.seed,
    )
    task = TaskDistribution(
        horizontal_distance_min_m=arguments.target_distance_min,
        horizontal_distance_max_m=arguments.target_distance_max,
        target_azimuth_min_deg=arguments.target_azimuth_min,
        target_azimuth_max_deg=arguments.target_azimuth_max,
        target_height_offset_min_m=arguments.target_height_min,
        target_height_offset_max_m=arguments.target_height_max,
        desired_impact_azimuth_offset_min_deg=(
            arguments.impact_azimuth_offset_min
        ),
        desired_impact_azimuth_offset_max_deg=(
            arguments.impact_azimuth_offset_max
        ),
        desired_impact_elevation_deg=arguments.impact_elevation,
        impact_angle_deg=arguments.impact_angle,
        minimum_extension_ratio=arguments.minimum_extension_ratio,
        minimum_impact_speed_min_m_s=arguments.impact_speed,
        minimum_impact_speed_max_m_s=arguments.impact_speed,
        hit_tolerance_m=arguments.hit_tolerance,
        drone_keepout_radius_m=arguments.target_keepout,
        strike_accuracy_decay_m_inv=arguments.strike_accuracy_decay,
        strike_event_reward=arguments.strike_event_reward,
        progress_reward=arguments.progress_reward,
        success_reward=arguments.success_reward,
        impact_drone_displacement_penalty=(
            arguments.impact_drone_displacement_penalty
        ),
    )
    transition_label = "endless" if arguments.endless else str(settings.total_transitions)
    print(
        f"Nominal cable-whip SAC | model={snapshot.source_path.name} "
        f"nodes={settings.controller_node_count} envs={settings.environment_count} "
        f"transitions={transition_label} | targets "
        f"r={arguments.target_distance_min:.2f}-{arguments.target_distance_max:.2f}m, "
        f"z={arguments.target_height_min:.2f}-{arguments.target_height_max:.2f}m, "
        f"az={arguments.target_azimuth_min:.0f}-{arguments.target_azimuth_max:.0f}deg, "
        f"impact yaw={arguments.impact_azimuth_offset_min:.0f}-"
        f"{arguments.impact_azimuth_offset_max:.0f}deg, "
        f"HER goals/episode={arguments.her_goals_per_episode}, "
        f"progress={arguments.progress_reward:g}, "
        f"valid hit bonus={arguments.success_reward:g}, "
        "full cable state, abrupt 3-D acceleration, future-goal HER"
    )
    if snapshot.provisional:
        print("WARNING: training uses the provisional two-holder EI/Cb transfer.")

    stop_file = arguments.stop_file.resolve() if arguments.stop_file else None
    stop_event = Event()

    def stop_requested() -> bool:
        return stop_event.is_set() or bool(
            stop_file is not None and stop_file.is_file()
        )

    def request_stop(_signal_number: int, _frame: object) -> None:
        if not stop_event.is_set():
            print("Stop requested; saving the latest SAC state cleanly.", flush=True)
            stop_event.set()

    def structured_progress(update: SacTrainingUpdate) -> None:
        if arguments.progress_json:
            print(
                "SAC_EVENT_JSON="
                + json.dumps(asdict(update), allow_nan=False, separators=(",", ":")),
                flush=True,
            )

    def structured_preview(preview: SacValidationPreview) -> None:
        if arguments.progress_json and arguments.validation_preview:
            print(
                "SAC_EVENT_JSON="
                + json.dumps(asdict(preview), allow_nan=False, separators=(",", ":")),
                flush=True,
            )

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        summary = train_sac(
            snapshot,
            arguments.output,
            settings,
            task,
            progress=print,
            cancelled=stop_requested,
            metrics_progress=structured_progress if arguments.progress_json else None,
            validation_preview_progress=(
                structured_preview
                if arguments.progress_json and arguments.validation_preview
                else None
            ),
            endless=arguments.endless,
            checkpoint_evaluation_episodes=(
                arguments.checkpoint_evaluation_episodes
            ),
            checkpoint_validation_seed=arguments.checkpoint_validation_seed,
            final_test_seed=arguments.final_test_seed,
        )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    if summary.held_out_evaluated:
        print(
            f"Completed: train success={100.0 * summary.training_success_rate:.1f}% "
            f"held-out simulation success={100.0 * summary.evaluation_success_rate:.1f}% "
            f"error={1000.0 * summary.evaluation_mean_minimum_error_m:.1f}mm "
            f"speed={summary.evaluation_mean_directional_speed_m_s:.3f}m/s "
            f"reward/episode={summary.evaluation_mean_episode_reward:.3f} "
            f"drone@hit={summary.evaluation_mean_hit_drone_displacement_m:.3f}m "
            f"unsafe={100.0 * summary.evaluation_unsafe_rate:.1f}% "
            f"peak cable energy={1000.0 * summary.evaluation_mean_peak_relative_cable_energy_j:.3f}mJ | "
            f"within-start-reach={100.0 * summary.evaluation_within_initial_reach_success_rate:.1f}% "
            f"beyond-start-reach={100.0 * summary.evaluation_beyond_initial_reach_success_rate:.1f}%"
        )
    else:
        print(
            f"Stopped cleanly at {summary.transitions} transitions; held-out test "
            "was not run."
        )
    print(f"Run status: {summary.run_status}")
    print(f"Policy: {summary.policy_path}")
    print(f"Best checkpoint transition: {summary.best_transitions}")
    print(f"Final diagnostic policy: {summary.final_policy_path}")
    print(f"Latest checkpoint: {summary.latest_policy_path}")
    print(f"Log: {summary.log_path}")


if __name__ == "__main__":
    main()
