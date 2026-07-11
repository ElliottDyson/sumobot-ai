from __future__ import annotations

from dataclasses import dataclass

import torch

from .state import BLUE, DRAW, ONGOING, RED


@dataclass(frozen=True, slots=True)
class MatchResolution:
    terminated: torch.Tensor
    truncated: torch.Tensor
    winner: torch.Tensor
    ring_out: torch.Tensor
    inactivity: torch.Tensor
    numerical_failure: torch.Tensor


def resolve_match(
    out: torch.Tensor,
    inactive: torch.Tensor,
    timed_out: torch.Tensor,
    numerical_failure: torch.Tensor,
) -> MatchResolution:
    """Resolve an arena step with ring-out precedence over inactivity."""
    if out.ndim != 2 or out.shape[1] != 2 or inactive.shape != out.shape:
        raise ValueError("out and inactive must have shape (B, 2)")
    batch = out.shape[0]
    for name, value in (("timed_out", timed_out), ("numerical_failure", numerical_failure)):
        if value.shape != (batch,) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean tensor with shape (B,)")
    if out.dtype != torch.bool or inactive.dtype != torch.bool:
        raise ValueError("out and inactive must be boolean")

    ring_out = out.any(dim=-1) & ~numerical_failure
    inactivity = inactive.any(dim=-1) & ~ring_out & ~numerical_failure
    terminated = ring_out | inactivity | numerical_failure
    truncated = timed_out & ~terminated
    winner = torch.full((batch,), ONGOING, dtype=torch.int64, device=out.device)

    red_only_out = out[:, RED] & ~out[:, BLUE] & ring_out
    blue_only_out = out[:, BLUE] & ~out[:, RED] & ring_out
    winner = torch.where(red_only_out, torch.full_like(winner, BLUE), winner)
    winner = torch.where(blue_only_out, torch.full_like(winner, RED), winner)
    winner = torch.where(ring_out & ~(red_only_out | blue_only_out), torch.full_like(winner, DRAW), winner)

    red_only_inactive = inactive[:, RED] & ~inactive[:, BLUE] & inactivity
    blue_only_inactive = inactive[:, BLUE] & ~inactive[:, RED] & inactivity
    winner = torch.where(red_only_inactive, torch.full_like(winner, BLUE), winner)
    winner = torch.where(blue_only_inactive, torch.full_like(winner, RED), winner)
    winner = torch.where(inactivity & ~(red_only_inactive | blue_only_inactive), torch.full_like(winner, DRAW), winner)
    winner = torch.where(numerical_failure | truncated, torch.full_like(winner, DRAW), winner)
    return MatchResolution(terminated, truncated, winner, ring_out, inactivity, numerical_failure)
