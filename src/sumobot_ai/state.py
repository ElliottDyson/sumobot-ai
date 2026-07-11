from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import ACTION_DIM, DRIVEN_WHEEL_COUNT

ONGOING = -2
DRAW = -1
RED = 0
BLUE = 1


def _check_shape(name: str, tensor: torch.Tensor, expected: tuple[int | None, ...]) -> None:
    if tensor.ndim != len(expected):
        raise ValueError(f"{name} must have rank {len(expected)}, got {tuple(tensor.shape)}")
    for actual, wanted in zip(tensor.shape, expected, strict=True):
        if wanted is not None and actual != wanted:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")


@dataclass(frozen=True, slots=True)
class ArenaState:
    """Task-relevant state for a batch of two-robot arenas."""

    position: torch.Tensor  # (B, 2, 3), metres
    quaternion: torch.Tensor  # (B, 2, 4), xyzw
    linear_velocity: torch.Tensor  # (B, 2, 3), m/s
    angular_velocity: torch.Tensor  # (B, 2, 3), rad/s
    wheel_velocity: torch.Tensor  # (B, 2, 2), rad/s
    action_proposed: torch.Tensor  # (B, 2, 2), normalized policy command before latency/actuator effects
    action_exec: torch.Tensor  # (B, 2, 2), normalized command actually executed
    contact_force: torch.Tensor  # (B, 2, 3), N, net external contact summary
    edge_margin: torch.Tensor  # (B, 2), signed centre-to-edge margin, m
    support_margin: torch.Tensor  # (B, 2), max signed margin among the two wheels and skid, m
    stationary_time_s: torch.Tensor  # (B, 2), continuous time below movement threshold
    time_remaining_s: torch.Tensor  # (B,)

    def __post_init__(self) -> None:
        _check_shape("position", self.position, (None, 2, 3))
        batch = self.position.shape[0]
        shapes = {
            "quaternion": (batch, 2, 4),
            "linear_velocity": (batch, 2, 3),
            "angular_velocity": (batch, 2, 3),
            "wheel_velocity": (batch, 2, DRIVEN_WHEEL_COUNT),
            "action_proposed": (batch, 2, ACTION_DIM),
            "action_exec": (batch, 2, ACTION_DIM),
            "contact_force": (batch, 2, 3),
            "edge_margin": (batch, 2),
            "support_margin": (batch, 2),
            "stationary_time_s": (batch, 2),
            "time_remaining_s": (batch,),
        }
        for name, expected in shapes.items():
            _check_shape(name, getattr(self, name), expected)
        device = self.position.device
        if any(getattr(self, name).device != device for name in shapes):
            raise ValueError("all ArenaState tensors must use the same device")

    @property
    def batch_size(self) -> int:
        return self.position.shape[0]

    def swapped(self) -> ArenaState:
        index = torch.tensor([1, 0], device=self.position.device)
        return ArenaState(
            position=self.position.index_select(1, index),
            quaternion=self.quaternion.index_select(1, index),
            linear_velocity=self.linear_velocity.index_select(1, index),
            angular_velocity=self.angular_velocity.index_select(1, index),
            wheel_velocity=self.wheel_velocity.index_select(1, index),
            action_proposed=self.action_proposed.index_select(1, index),
            action_exec=self.action_exec.index_select(1, index),
            contact_force=self.contact_force.index_select(1, index),
            edge_margin=self.edge_margin.index_select(1, index),
            support_margin=self.support_margin.index_select(1, index),
            stationary_time_s=self.stationary_time_s.index_select(1, index),
            time_remaining_s=self.time_remaining_s,
        )


@dataclass(frozen=True, slots=True)
class ArenaTransition:
    previous: ArenaState
    current: ArenaState
    terminated: torch.Tensor  # (B,), physical terminal
    truncated: torch.Tensor  # (B,), time limit
    winner: torch.Tensor  # (B,), ONGOING=-2, DRAW=-1, RED=0, BLUE=1

    def __post_init__(self) -> None:
        if self.previous.batch_size != self.current.batch_size:
            raise ValueError("previous and current ArenaState batches must match")
        batch = self.current.batch_size
        _check_shape("terminated", self.terminated, (batch,))
        _check_shape("truncated", self.truncated, (batch,))
        _check_shape("winner", self.winner, (batch,))
        if self.terminated.dtype != torch.bool or self.truncated.dtype != torch.bool:
            raise ValueError("terminated and truncated must be bool tensors")
        valid = (self.winner >= ONGOING) & (self.winner <= BLUE)
        if not bool(valid.all()):
            raise ValueError("winner values must be ONGOING, DRAW, RED, or BLUE")
        finished = self.terminated | self.truncated
        if bool(((self.winner == ONGOING) & finished).any()):
            raise ValueError("finished transitions need a winner or DRAW outcome")
        if bool(((self.winner != ONGOING) & ~finished).any()):
            raise ValueError("unfinished transitions must use ONGOING")

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated

    @property
    def dt(self) -> torch.Tensor:
        return (self.previous.time_remaining_s - self.current.time_remaining_s).clamp_min(0.0)

    def swapped(self) -> ArenaTransition:
        winner = self.winner.clone()
        winner = torch.where(self.winner == RED, torch.full_like(winner, BLUE), winner)
        winner = torch.where(self.winner == BLUE, torch.full_like(winner, RED), winner)
        return ArenaTransition(
            previous=self.previous.swapped(),
            current=self.current.swapped(),
            terminated=self.terminated,
            truncated=self.truncated,
            winner=winner,
        )
