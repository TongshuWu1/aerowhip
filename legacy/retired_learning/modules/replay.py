"""Compact fixed-size replay for one-decision terminal SAC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .policy_action import POLICY_ACTION_DIM
from .policy_context import POLICY_CONTEXT_DIM


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    context: torch.Tensor
    action: torch.Tensor
    scaled_reward: torch.Tensor


class TerminalReplayBuffer:
    """CPU float32 storage; complete trajectories are intentionally excluded."""

    def __init__(self, capacity: int = 1_000_000) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.context = torch.empty((capacity, POLICY_CONTEXT_DIM), dtype=torch.float32)
        self.action = torch.empty((capacity, POLICY_ACTION_DIM), dtype=torch.float32)
        self.scaled_reward = torch.empty((capacity, 1), dtype=torch.float32)
        self.raw_reward = torch.empty((capacity, 1), dtype=torch.float32)
        self.success = torch.empty((capacity, 1), dtype=torch.bool)
        self.feasible = torch.empty((capacity, 1), dtype=torch.bool)
        self.tip_distance_m = torch.empty((capacity, 1), dtype=torch.float32)
        self.directed_speed_m_s = torch.empty((capacity, 1), dtype=torch.float32)
        self.direction_error_deg = torch.empty((capacity, 1), dtype=torch.float32)
        self._size = 0
        self._position = 0
        self.total_inserted = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        *,
        context: torch.Tensor,
        action: torch.Tensor,
        scaled_reward: torch.Tensor,
        raw_reward: torch.Tensor,
        success: torch.Tensor,
        feasible: torch.Tensor,
        tip_distance_m: torch.Tensor,
        directed_speed_m_s: torch.Tensor,
        direction_error_deg: torch.Tensor,
    ) -> None:
        rows = int(context.shape[0])
        if rows < 1 or rows > self.capacity:
            raise ValueError("Invalid replay insertion batch size.")
        if context.shape != (rows, POLICY_CONTEXT_DIM) or action.shape != (rows, POLICY_ACTION_DIM):
            raise ValueError("Replay context/action shape mismatch.")
        indices = (torch.arange(rows) + self._position) % self.capacity

        def put(destination: torch.Tensor, source: torch.Tensor) -> None:
            value = torch.as_tensor(source).detach().to("cpu")
            if destination.dtype == torch.bool:
                value = value.to(torch.bool)
            else:
                value = value.to(torch.float32)
            value = value.reshape(rows, *destination.shape[1:])
            destination[indices] = value

        put(self.context, context)
        put(self.action, action)
        put(self.scaled_reward, scaled_reward)
        put(self.raw_reward, raw_reward)
        put(self.success, success)
        put(self.feasible, feasible)
        put(self.tip_distance_m, tip_distance_m)
        put(self.directed_speed_m_s, directed_speed_m_s)
        put(self.direction_error_deg, direction_error_deg)
        self._position = int((self._position + rows) % self.capacity)
        self._size = min(self.capacity, self._size + rows)
        self.total_inserted += rows

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> ReplayBatch:
        if self._size < batch_size:
            raise ValueError("Replay does not yet contain one requested minibatch.")
        indices = torch.randint(self._size, (batch_size,), generator=generator)
        selected = torch.device(device)
        return ReplayBatch(
            self.context[indices].to(selected, non_blocking=True),
            self.action[indices].to(selected, non_blocking=True),
            self.scaled_reward[indices].to(selected, non_blocking=True),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "terminal_sac_replay_metadata_v1",
            "capacity": self.capacity,
            "size": self._size,
            "position": self._position,
            "total_inserted": self.total_inserted,
            "context_dimension": POLICY_CONTEXT_DIM,
            "action_dimension": POLICY_ACTION_DIM,
            "trajectory_storage": False,
            "dtype": "float32",
        }

    def save(self, path: str | Path) -> None:
        size = self._size
        torch.save(
            {
                "schema": "terminal_sac_replay_v1",
                "capacity": self.capacity,
                "size": size,
                "position": self._position,
                "total_inserted": self.total_inserted,
                "context": self.context[:size].clone(),
                "action": self.action[:size].clone(),
                "scaled_reward": self.scaled_reward[:size].clone(),
                "raw_reward": self.raw_reward[:size].clone(),
                "success": self.success[:size].clone(),
                "feasible": self.feasible[:size].clone(),
                "tip_distance_m": self.tip_distance_m[:size].clone(),
                "directed_speed_m_s": self.directed_speed_m_s[:size].clone(),
                "direction_error_deg": self.direction_error_deg[:size].clone(),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TerminalReplayBuffer":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("schema") != "terminal_sac_replay_v1":
            raise ValueError("Unsupported terminal replay checkpoint.")
        result = cls(int(payload["capacity"]))
        size = int(payload["size"])
        for name in (
            "context",
            "action",
            "scaled_reward",
            "raw_reward",
            "success",
            "feasible",
            "tip_distance_m",
            "directed_speed_m_s",
            "direction_error_deg",
        ):
            getattr(result, name)[:size].copy_(payload[name])
        result._size = size
        result._position = int(payload["position"])
        result.total_inserted = int(payload["total_inserted"])
        return result
