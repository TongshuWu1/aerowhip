"""Supervised one-shot policy for amortized CEM trajectory optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .normalization import FixedContextNormalizer
from .policy_action import POLICY_ACTION_DIM
from .policy_context import POLICY_CONTEXT_DIM


class AmortizedTrajectoryActor(nn.Module):
    """One 83-D context query to one complete normalized 49-D maneuver."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(POLICY_CONTEXT_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, POLICY_ACTION_DIM),
            nn.Tanh(),
        )

    def forward(self, normalized_context: torch.Tensor) -> torch.Tensor:
        context = torch.as_tensor(normalized_context)
        if context.ndim != 2 or context.shape[1] != POLICY_CONTEXT_DIM:
            raise ValueError("Amortized actor input must have shape Bx83.")
        return self.network(context)


@dataclass(frozen=True, slots=True)
class SupervisedTrainingResult:
    best_epoch: int
    epochs_completed: int
    best_validation_loss: float
    training_history: tuple[dict[str, float], ...]
    training_metrics: dict[str, float]
    validation_metrics: dict[str, float]


def imitation_losses(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 49:
        raise ValueError("Supervised action tensors must have matching Bx49 shape.")
    acceleration = F.mse_loss(prediction[:, :48], target[:, :48])
    duration = F.mse_loss(prediction[:, 48], target[:, 48])
    total = acceleration + duration
    return total, {"acceleration_mse": acceleration, "duration_mse": duration}


def action_error_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction.detach() - target.detach()
    return {
        "normalized_action_rmse": float(torch.sqrt(error.square().mean())),
        "normalized_acceleration_rmse": float(torch.sqrt(error[:, :48].square().mean())),
        "normalized_duration_rmse": float(torch.sqrt(error[:, 48].square().mean())),
        "normalized_action_maximum_absolute_error": float(error.abs().max()),
    }


def train_amortized_actor(
    actor: AmortizedTrajectoryActor,
    normalizer: FixedContextNormalizer,
    train_context: torch.Tensor,
    train_action: torch.Tensor,
    validation_context: torch.Tensor,
    validation_action: torch.Tensor,
    *,
    seed: int = 42,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-5,
    batch_size: int = 32,
    maximum_epochs: int = 5000,
    patience_epochs: int = 500,
    checkpoint_path: str | Path | None = None,
) -> SupervisedTrainingResult:
    """Fit only teacher action labels; no critic, replay, reward, or simulator gradient."""

    device = next(actor.parameters()).device
    train_x = normalizer.normalize(train_context.to(device))
    train_y = train_action.to(device)
    validation_x = normalizer.normalize(validation_context.to(device))
    validation_y = validation_action.to(device)
    if train_x.shape[0] < 1 or validation_x.shape[0] < 1:
        raise ValueError("Training and validation teacher sets must be non-empty.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, maximum_epochs + 1):
        actor.train()
        order = torch.randperm(train_x.shape[0], generator=generator)
        training_total = 0.0
        batches = 0
        for start in range(0, train_x.shape[0], batch_size):
            indices = order[start : start + batch_size].to(device)
            prediction = actor(train_x[indices])
            loss, _ = imitation_losses(prediction, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            optimizer.step()
            training_total += float(loss.detach())
            batches += 1
        actor.eval()
        with torch.no_grad():
            validation_prediction = actor(validation_x)
            validation_loss, validation_parts = imitation_losses(
                validation_prediction, validation_y
            )
        row = {
            "epoch": float(epoch),
            "training_loss": training_total / max(batches, 1),
            "validation_loss": float(validation_loss),
            "validation_acceleration_mse": float(validation_parts["acceleration_mse"]),
            "validation_duration_mse": float(validation_parts["duration_mse"]),
        }
        history.append(row)
        if float(validation_loss) < best_loss - 1.0e-8:
            best_loss = float(validation_loss)
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in actor.state_dict().items()
            }
            if checkpoint_path is not None:
                torch.save(
                    {
                        "schema": "amortized_cem_actor_v1",
                        "epoch": epoch,
                        "validation_loss": best_loss,
                        "actor": best_state,
                    },
                    Path(checkpoint_path),
                )
        else:
            stale += 1
        if stale >= patience_epochs:
            break
    if best_state is None:
        raise RuntimeError("Supervised training did not produce a finite checkpoint.")
    actor.load_state_dict(best_state)
    actor.eval()
    with torch.no_grad():
        train_prediction = actor(train_x)
        validation_prediction = actor(validation_x)
    return SupervisedTrainingResult(
        best_epoch=best_epoch,
        epochs_completed=len(history),
        best_validation_loss=best_loss,
        training_history=tuple(history),
        training_metrics=action_error_metrics(train_prediction, train_y),
        validation_metrics=action_error_metrics(validation_prediction, validation_y),
    )


def save_teacher_dataset(
    path: str | Path,
    *,
    contexts: torch.Tensor,
    actions: torch.Tensor,
    split: np.ndarray,
    context_ids: np.ndarray,
    state_indices: np.ndarray,
    target_positions_local_m: np.ndarray,
    target_directions_local: np.ndarray,
    teacher_metrics_json: np.ndarray,
) -> None:
    np.savez_compressed(
        Path(path),
        contexts=np.asarray(contexts.detach().cpu(), dtype=np.float32),
        normalized_actions=np.asarray(actions.detach().cpu(), dtype=np.float32),
        split=np.asarray(split),
        context_ids=np.asarray(context_ids),
        state_indices=np.asarray(state_indices, dtype=np.int64),
        target_positions_local_m=np.asarray(target_positions_local_m, dtype=np.float32),
        target_directions_local=np.asarray(target_directions_local, dtype=np.float32),
        teacher_metrics_json=np.asarray(teacher_metrics_json),
        schema=np.asarray("amortized_cem_teacher_dataset_v1"),
        physics=np.asarray("NOMINAL_ONLY"),
    )
