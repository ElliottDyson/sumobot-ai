from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from ..state import BLUE, DRAW, RED, ArenaTransition

TermFunction = Callable[[ArenaTransition, Mapping[str, float]], torch.Tensor]


@dataclass(frozen=True, slots=True)
class TermDefinition:
    function: TermFunction
    allowed_params: frozenset[str] = frozenset()
    defaults: Mapping[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.defaults is None:
            object.__setattr__(self, "defaults", {})

    def validate(self, params: Mapping[str, float]) -> None:
        if not all(math.isfinite(value) for value in params.values()):
            raise ValueError("reward parameters must be finite")
        if "scale_m" in params and params["scale_m"] <= 0:
            raise ValueError("scale_m must be positive")
        if "discount" in params and not 0.0 < params["discount"] <= 1.0:
            raise ValueError("discount must be in (0, 1]")


def _potential_delta(previous: torch.Tensor, current: torch.Tensor, discount: float) -> torch.Tensor:
    return discount * current - previous


def win_loss(transition: ArenaTransition, params: Mapping[str, float]) -> torch.Tensor:
    del params
    result = torch.zeros(
        (transition.current.batch_size, 2),
        dtype=transition.current.position.dtype,
        device=transition.current.position.device,
    )
    done = transition.done
    result[:, RED] = torch.where(
        done & (transition.winner == RED),
        torch.ones_like(result[:, RED]),
        torch.where(done & (transition.winner == BLUE), -torch.ones_like(result[:, RED]), result[:, RED]),
    )
    result[:, BLUE] = -result[:, RED]
    result = torch.where((done & (transition.winner == DRAW)).unsqueeze(-1), torch.zeros_like(result), result)
    return result


def approach_opponent(transition: ArenaTransition, params: Mapping[str, float]) -> torch.Tensor:
    scale = params["scale_m"]
    previous_distance = torch.linalg.vector_norm(
        transition.previous.position[:, 1, :2] - transition.previous.position[:, 0, :2], dim=-1
    )
    current_distance = torch.linalg.vector_norm(
        transition.current.position[:, 1, :2] - transition.current.position[:, 0, :2], dim=-1
    )
    previous_phi = -previous_distance / scale
    current_phi = -current_distance / scale
    value = _potential_delta(previous_phi, current_phi, params["discount"])
    return value.unsqueeze(-1).expand(-1, 2)


def push_opponent_to_edge(transition: ArenaTransition, params: Mapping[str, float]) -> torch.Tensor:
    scale = params["scale_m"]
    previous_phi = -transition.previous.edge_margin[:, [1, 0]] / scale
    current_phi = -transition.current.edge_margin[:, [1, 0]] / scale
    return _potential_delta(previous_phi, current_phi, params["discount"])


def protect_own_edge(transition: ArenaTransition, params: Mapping[str, float]) -> torch.Tensor:
    scale = params["scale_m"]
    previous_phi = transition.previous.edge_margin / scale
    current_phi = transition.current.edge_margin / scale
    return _potential_delta(previous_phi, current_phi, params["discount"])


def _forward_xy(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack((1.0 - 2.0 * (y.square() + z.square()), 2.0 * (x * y + z * w)), dim=-1)


def face_opponent(transition: ArenaTransition, params: Mapping[str, float]) -> torch.Tensor:
    del params
    relative_red = transition.current.position[:, 1, :2] - transition.current.position[:, 0, :2]
    norm = torch.linalg.vector_norm(relative_red, dim=-1, keepdim=True).clamp_min(1e-6)
    directions = torch.stack((relative_red / norm, -relative_red / norm), dim=1)
    forward = _forward_xy(transition.current.quaternion)
    cosine = (forward * directions).sum(dim=-1).clamp(-1.0, 1.0)
    return cosine * transition.dt.unsqueeze(-1)


def action_energy(transition: ArenaTransition, params: Mapping[str, float]) -> torch.Tensor:
    del params
    return -transition.current.action_exec.square().mean(dim=-1) * transition.dt.unsqueeze(-1)


_POTENTIAL_PARAMS = frozenset({"scale_m", "discount"})
_POTENTIAL_DEFAULTS = {"scale_m": 1.0, "discount": 1.0}

TERM_DEFINITIONS: dict[str, TermDefinition] = {
    "win_loss": TermDefinition(win_loss),
    "approach_opponent": TermDefinition(approach_opponent, _POTENTIAL_PARAMS, _POTENTIAL_DEFAULTS),
    "push_opponent_to_edge": TermDefinition(push_opponent_to_edge, _POTENTIAL_PARAMS, _POTENTIAL_DEFAULTS),
    "protect_own_edge": TermDefinition(protect_own_edge, _POTENTIAL_PARAMS, _POTENTIAL_DEFAULTS),
    "face_opponent": TermDefinition(face_opponent),
    "action_energy": TermDefinition(action_energy),
}


def register_reward_term(
    name: str,
    function: TermFunction,
    *,
    allowed_params: frozenset[str] = frozenset(),
    defaults: Mapping[str, float] | None = None,
) -> None:
    """Register a vectorized member term before loading its YAML specification."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("reward term names must use lowercase snake_case")
    if name in TERM_DEFINITIONS:
        raise ValueError(f"reward term {name!r} is already registered")
    defaults = dict(defaults or {})
    if not defaults.keys() <= allowed_params:
        raise ValueError("reward term defaults must be included in allowed_params")
    definition = TermDefinition(function, frozenset(allowed_params), defaults)
    definition.validate(defaults)
    TERM_DEFINITIONS[name] = definition
