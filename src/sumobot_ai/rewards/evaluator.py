from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ..state import ArenaTransition
from .spec import RewardSpec
from .terms import TERM_DEFINITIONS


@dataclass(frozen=True, slots=True)
class RewardResult:
    total: torch.Tensor  # (B, 2)
    components: Mapping[str, torch.Tensor]  # weighted, each (B, 2)


def evaluate_reward(spec: RewardSpec, transition: ArenaTransition) -> RewardResult:
    components: dict[str, torch.Tensor] = {}
    total = torch.zeros(
        (transition.current.batch_size, 2),
        dtype=transition.current.position.dtype,
        device=transition.current.position.device,
    )
    for term in spec.terms:
        raw = TERM_DEFINITIONS[term.name].function(transition, term.params)
        if tuple(raw.shape) != tuple(total.shape):
            raise RuntimeError(f"reward term {term.name!r} returned {tuple(raw.shape)}, expected {tuple(total.shape)}")
        weighted = raw * term.weight
        components[term.name] = weighted
        total = total + weighted
    if spec.clip is not None:
        unclipped = total
        total = total.clamp(*spec.clip)
        components["clip_delta"] = total - unclipped
    return RewardResult(total=total, components=components)


def evaluate_dual(red: RewardSpec, blue: RewardSpec, transition: ArenaTransition) -> RewardResult:
    """Evaluate independently authored rewards and select the owning side from each result."""
    red_result = evaluate_reward(red, transition)
    blue_result = evaluate_reward(blue, transition)
    total = torch.stack((red_result.total[:, 0], blue_result.total[:, 1]), dim=-1)
    components: dict[str, torch.Tensor] = {}
    for name, value in red_result.components.items():
        components[f"red.{name}"] = value[:, 0]
    for name, value in blue_result.components.items():
        components[f"blue.{name}"] = value[:, 1]
    return RewardResult(total=total, components=components)
