"""Training loops for the permanent amortized-CEM diffusion policy and scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .action_diffusion import (
    ConditionalActionDiffusion,
    ExponentialMovingAverage,
    cosine_alpha_bar_schedule,
    diffusion_epsilon_loss,
)
from .amortized_cem_data import (
    ContextBalancedTeacherSampler,
    load_scorer_rows,
    load_teacher_rows,
)
from .normalization import FixedContextNormalizer
from .outcome_scorer import (
    ManeuverOutcomeScorer,
    OutcomeStratifiedSampler,
    OutcomeTargetNormalizer,
    binary_positive_weights,
    scorer_loss,
    scorer_validation_metrics,
    select_candidate_from_predictions,
)


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flat_teacher_validation(
    contexts: np.ndarray,
    actions_by_context: dict[int, np.ndarray],
    *,
    maximum_rows: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        (np.repeat(contexts[index][None], actions.shape[0], axis=0), actions)
        for index, actions in sorted(actions_by_context.items())
    ]
    context = np.concatenate([item[0] for item in rows], axis=0).astype(np.float32)
    action = np.concatenate([item[1] for item in rows], axis=0).astype(np.float32)
    if context.shape[0] > maximum_rows:
        rng = np.random.default_rng(12_345)
        selected = np.sort(rng.choice(context.shape[0], maximum_rows, replace=False))
        context, action = context[selected], action[selected]
    return torch.from_numpy(context), torch.from_numpy(action)


@torch.no_grad()
def _diffusion_validation_loss(
    model: torch.nn.Module,
    context: torch.Tensor,
    action: torch.Tensor,
    normalizer: FixedContextNormalizer,
    alpha_bar: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    losses, weights = [], []
    for start in range(0, context.shape[0], 1024):
        c = normalizer.normalize(context[start : start + 1024].to(device))
        a = action[start : start + 1024].to(device)
        result = diffusion_epsilon_loss(
            model, c, a, alpha_bar=alpha_bar, generator=generator
        )
        losses.append(float(result.loss) * a.shape[0])
        weights.append(a.shape[0])
    model.train()
    return float(sum(losses) / sum(weights))


def train_diffusion(
    artifact: str | Path,
    config: dict[str, Any],
    *,
    round_index: int = 0,
    device: torch.device | str = "cuda",
    model_factory: Callable[[], torch.nn.Module] = ConditionalActionDiffusion,
    architecture: dict[str, Any] | None = None,
    checkpoint_schema_prefix: str = "conditional_action_diffusion",
) -> dict[str, Any]:
    root = Path(artifact)
    selected = torch.device(device)
    torch.manual_seed(int(config["seed"]) + 1000 * round_index)
    torch.cuda.manual_seed_all(int(config["seed"]) + 1000 * round_index)
    source = config["diffusion"]
    manifest = json.loads((root / "diffusion_teacher_manifest.json").read_text(encoding="utf-8"))
    contexts, training_actions = load_teacher_rows(
        root / "context_table.npz",
        root / "diffusion_teacher_shards",
        manifest,
        split="TRAIN",
    )
    _, validation_actions = load_teacher_rows(
        root / "context_table.npz",
        root / "diffusion_teacher_shards",
        manifest,
        split="VALIDATION",
    )
    normalizer = FixedContextNormalizer.load(root / "context_normalizer.json")
    sampler = ContextBalancedTeacherSampler(
        contexts, training_actions, seed=int(config["seed"]) + 31 + round_index
    )
    validation_context, validation_action = _flat_teacher_validation(
        contexts, validation_actions
    )
    model = model_factory().to(selected)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(source["learning_rate"]),
        weight_decay=float(source["weight_decay"]),
    )
    ema = ExponentialMovingAverage(model, decay=float(source["ema_decay"]))
    alpha_bar = cosine_alpha_bar_schedule(int(source["training_steps"])).to(selected)
    batch_size = int(source["batch_size"])
    maximum = int(source["maximum_updates"])
    interval = int(source["validation_interval"])
    minimum = int(source["minimum_updates"])
    plateau = int(source["plateau_updates"])
    generator = torch.Generator(device=selected).manual_seed(
        int(config["seed"]) + 503 + round_index
    )
    history: list[dict[str, Any]] = []
    training_accumulator = 0.0
    best_validation = float("inf")
    best_update = 0
    evaluation_model = model_factory().to(selected)
    output_prefix = "" if round_index == 0 else f"aggregation_rounds/round_{round_index}/"
    output_root = root / output_prefix
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "diffusion_config.json",
        {
            **source,
            "architecture": architecture
            or {
                "context_encoder": "83-256-SiLU-256",
                "timestep_embedding": "sinusoidal64-256-SiLU-256",
                "action_projection": "49-256",
                "residual_blocks": 4,
                "residual_block": "LayerNorm-256x512-SiLU-512x256",
                "output": "LayerNorm-256x49",
            },
            "direct_normalized_action_space": True,
            "pca_bottleneck": False,
            "context_balanced_sampling": True,
        },
    )

    for update in range(1, maximum + 1):
        raw_context, action, _ = sampler.sample(batch_size)
        context = normalizer.normalize(raw_context.to(selected, non_blocking=True))
        action = action.to(selected, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        result = diffusion_epsilon_loss(
            model, context, action, alpha_bar=alpha_bar, generator=generator
        )
        result.loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(source["gradient_clip"]))
        )
        optimizer.step()
        ema.update(model)
        training_accumulator += float(result.loss.detach())

        if update % interval:
            continue
        ema.copy_to(evaluation_model)
        validation = _diffusion_validation_loss(
            evaluation_model,
            validation_context,
            validation_action,
            normalizer,
            alpha_bar,
            device=selected,
            seed=41_000 + round_index,
        )
        row = {
            "update": update,
            "training_epsilon_loss": training_accumulator / interval,
            "validation_ema_epsilon_loss": validation,
            "gradient_norm": gradient_norm,
        }
        history.append(row)
        training_accumulator = 0.0
        if validation < best_validation:
            best_validation = validation
            best_update = update
            torch.save(
                {
                    "schema": f"{checkpoint_schema_prefix}_v1",
                    "update": update,
                    "state_dict": _cpu_state(model),
                },
                output_root / "diffusion_best.pt",
            )
            torch.save(
                {
                    "schema": f"{checkpoint_schema_prefix}_ema_v1",
                    "update": update,
                    "state_dict": {
                        name: value.detach().cpu().clone() for name, value in ema.shadow.items()
                    },
                },
                output_root / "diffusion_ema_best.pt",
            )
        torch.save(
            {
                "schema": f"{checkpoint_schema_prefix}_latest_v1",
                "update": update,
                "state_dict": _cpu_state(model),
                "optimizer": optimizer.state_dict(),
                "ema": {
                    "decay": ema.decay,
                    "shadow": {
                        name: value.detach().cpu().clone() for name, value in ema.shadow.items()
                    },
                },
            },
            output_root / "diffusion_latest.pt",
        )
        _write_json(output_root / "diffusion_training_history.json", history)
        print(
            f"DIFFUSION round={round_index} update={update} "
            f"train={row['training_epsilon_loss']:.6f} val={validation:.6f}",
            flush=True,
        )
        if update >= minimum and update - best_update >= plateau:
            break
    summary = {
        "round": round_index,
        "updates": history[-1]["update"],
        "best_update": best_update,
        "best_validation_epsilon_loss": best_validation,
        "train_solved_contexts": len(training_actions),
        "validation_solved_contexts": len(validation_actions),
        "context_balanced_sampling": True,
        "early_stopped": history[-1]["update"] < maximum,
    }
    _write_json(output_root / "diffusion_training_summary.json", summary)
    return summary


def _scorer_context_indices(root: Path, manifest: dict[str, Any], split: str) -> np.ndarray:
    values: list[np.ndarray] = []
    for entry in manifest["shards"]:
        if entry["split"] != split:
            continue
        values.append(
            np.full(int(entry["row_count"]), int(entry["context_index"]), dtype=np.int64)
        )
    return np.concatenate(values)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def scorer_context_ranking_metrics(
    context_indices: np.ndarray,
    truth_continuous: np.ndarray,
    truth_binary: np.ndarray,
    predicted_continuous: np.ndarray,
    predicted_binary: np.ndarray,
) -> dict[str, Any]:
    top1, top4, correlations = [], [], []
    for context_index in np.unique(context_indices):
        mask = context_indices == context_index
        truth_success = truth_binary[mask, 2] > 0.5
        physical = torch.from_numpy(predicted_continuous[mask]).float()
        probability = torch.from_numpy(predicted_binary[mask]).float()
        from .outcome_scorer import PredictedPhysicalOutcomes

        outcomes = PredictedPhysicalOutcomes(physical, probability)
        selected = select_candidate_from_predictions(outcomes).selected_index
        top1.append(float(truth_success[selected]))
        gate = select_candidate_from_predictions(outcomes)
        # Deployment selector order: predicted gate, then minimum margin, then reward.
        key = (
            gate.predicted_gate_pass.float() * 1.0e6
            + gate.minimum_margin * 1.0e3
            + outcomes.reward
        ).numpy()
        order = np.argsort(-key, kind="stable")
        top4.append(float(truth_success[order[:4]].any()))
        true_reward = truth_continuous[mask, 0]
        predicted_reward = predicted_continuous[mask, 0]
        if true_reward.size > 1 and np.std(true_reward) > 0 and np.std(predicted_reward) > 0:
            correlations.append(
                float(np.corrcoef(_rankdata(true_reward), _rankdata(predicted_reward))[0, 1])
            )
    return {
        "context_count": int(np.unique(context_indices).size),
        "top1_true_success_rate": float(np.mean(top1)),
        "top4_contains_true_success_rate": float(np.mean(top4)),
        "mean_spearman_reward_correlation": float(np.mean(correlations)) if correlations else None,
    }


@torch.no_grad()
def _scorer_validation(
    model: ManeuverOutcomeScorer,
    context: np.ndarray,
    action: np.ndarray,
    continuous: np.ndarray,
    binary: np.ndarray,
    context_indices: np.ndarray,
    context_normalizer: FixedContextNormalizer,
    target_normalizer: OutcomeTargetNormalizer,
    positive_weights: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    predicted_continuous, predicted_binary, losses, weights = [], [], [], []
    for start in range(0, context.shape[0], 8192):
        raw_context = torch.from_numpy(context[start : start + 8192]).to(device)
        c = context_normalizer.normalize(raw_context)
        a = torch.from_numpy(action[start : start + 8192]).to(device)
        y_cont = torch.from_numpy(continuous[start : start + 8192]).to(device)
        y_bin = torch.from_numpy(binary[start : start + 8192]).to(device)
        standard, logits = model(c, a)
        loss = scorer_loss(
            standard,
            logits,
            y_cont,
            y_bin,
            target_normalizer=target_normalizer,
            positive_weights=positive_weights,
        )
        predicted_continuous.append(
            target_normalizer.denormalize(standard).cpu().numpy()
        )
        predicted_binary.append(torch.sigmoid(logits).cpu().numpy())
        losses.append(float(loss.total) * a.shape[0])
        weights.append(a.shape[0])
    predicted_continuous_array = np.concatenate(predicted_continuous)
    predicted_binary_array = np.concatenate(predicted_binary)
    metrics = scorer_validation_metrics(
        continuous, predicted_continuous_array, binary, predicted_binary_array
    )
    metrics["candidate_ranking"] = scorer_context_ranking_metrics(
        context_indices,
        continuous,
        binary,
        predicted_continuous_array,
        predicted_binary_array,
    )
    model.train()
    return float(sum(losses) / sum(weights)), metrics


def train_scorer(
    artifact: str | Path,
    config: dict[str, Any],
    *,
    round_index: int = 0,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    root = Path(artifact)
    selected = torch.device(device)
    torch.manual_seed(int(config["seed"]) + 2000 * round_index + 1)
    torch.cuda.manual_seed_all(int(config["seed"]) + 2000 * round_index + 1)
    source = config["scorer"]
    manifest = json.loads((root / "scorer_dataset_manifest.json").read_text(encoding="utf-8"))
    train = load_scorer_rows(
        root / "context_table.npz",
        root / "scorer_dataset_shards",
        manifest,
        split="TRAIN",
    )
    validation = load_scorer_rows(
        root / "context_table.npz",
        root / "scorer_dataset_shards",
        manifest,
        split="VALIDATION",
    )
    validation_indices = _scorer_context_indices(root, manifest, "VALIDATION")
    context_normalizer = FixedContextNormalizer.load(root / "context_normalizer.json")
    target_normalizer = OutcomeTargetNormalizer.fit(torch.from_numpy(train[2]))
    positive_weights = binary_positive_weights(torch.from_numpy(train[3]))
    model = ManeuverOutcomeScorer().to(selected)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(source["learning_rate"]),
        weight_decay=float(source["weight_decay"]),
    )
    sampler = OutcomeStratifiedSampler(train[2], train[3], seed=int(config["seed"]) + 73 + round_index)
    maximum = int(source["maximum_updates"])
    interval = int(source["validation_interval"])
    minimum = int(source["minimum_updates"])
    plateau = int(source["plateau_updates"])
    batch_size = int(source["batch_size"])
    history: list[dict[str, Any]] = []
    accumulator = 0.0
    best_validation = float("inf")
    best_update = 0
    output_prefix = "" if round_index == 0 else f"aggregation_rounds/round_{round_index}/"
    output_root = root / output_prefix
    output_root.mkdir(parents=True, exist_ok=True)
    target_normalizer.save(output_root / "scorer_target_normalization.json")
    _write_json(
        output_root / "scorer_config.json",
        {
            **source,
            "continuous_positive_weights_not_applicable": True,
            "binary_positive_weights": positive_weights.tolist(),
            "train_rows": int(train[0].shape[0]),
            "validation_rows": int(validation[0].shape[0]),
            "training_split_normalization_only": True,
        },
    )

    for update in range(1, maximum + 1):
        indices = sampler.sample(batch_size)
        context = context_normalizer.normalize(torch.from_numpy(train[0][indices]).to(selected))
        action = torch.from_numpy(train[1][indices]).to(selected)
        continuous = torch.from_numpy(train[2][indices]).to(selected)
        binary = torch.from_numpy(train[3][indices]).to(selected)
        optimizer.zero_grad(set_to_none=True)
        predicted_continuous, predicted_binary = model(context, action)
        losses = scorer_loss(
            predicted_continuous,
            predicted_binary,
            continuous,
            binary,
            target_normalizer=target_normalizer,
            positive_weights=positive_weights,
        )
        losses.total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        accumulator += float(losses.total.detach())
        if update % interval:
            continue
        validation_loss, metrics = _scorer_validation(
            model,
            *validation,
            validation_indices,
            context_normalizer,
            target_normalizer,
            positive_weights,
            device=selected,
        )
        row = {
            "update": update,
            "training_loss": accumulator / interval,
            "validation_loss": validation_loss,
            "gradient_norm": gradient_norm,
            "validation_metrics": metrics,
        }
        history.append(row)
        accumulator = 0.0
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_update = update
            torch.save(
                {
                    "schema": "maneuver_outcome_scorer_v1",
                    "update": update,
                    "state_dict": _cpu_state(model),
                },
                output_root / "scorer_best.pt",
            )
        torch.save(
            {
                "schema": "maneuver_outcome_scorer_latest_v1",
                "update": update,
                "state_dict": _cpu_state(model),
                "optimizer": optimizer.state_dict(),
            },
            output_root / "scorer_latest.pt",
        )
        _write_json(output_root / "scorer_training_history.json", history)
        print(
            f"SCORER round={round_index} update={update} "
            f"train={row['training_loss']:.6f} val={validation_loss:.6f}",
            flush=True,
        )
        if update >= minimum and update - best_update >= plateau:
            break
    best_payload = torch.load(
        output_root / "scorer_best.pt", map_location=selected, weights_only=True
    )
    model.load_state_dict(best_payload["state_dict"])
    best_loss, best_metrics = _scorer_validation(
        model,
        *validation,
        validation_indices,
        context_normalizer,
        target_normalizer,
        positive_weights,
        device=selected,
    )
    summary = {
        "round": round_index,
        "updates": history[-1]["update"],
        "best_update": best_update,
        "best_validation_loss": best_validation,
        "train_rows": int(train[0].shape[0]),
        "validation_rows": int(validation[0].shape[0]),
        "positive_weights": positive_weights.tolist(),
        "best_checkpoint_validation_loss_recomputed": best_loss,
        "best_checkpoint_validation_metrics": best_metrics,
        "early_stopped": history[-1]["update"] < maximum,
    }
    _write_json(output_root / "scorer_training_summary.json", summary)
    return summary
