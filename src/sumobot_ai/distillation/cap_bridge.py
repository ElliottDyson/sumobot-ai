from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..training.cpo import CapActorSuffix


def _cap_suffix_raw(cap_actor: Any, bottleneck: torch.Tensor, layers: int) -> torch.Tensor:
    value = bottleneck
    modules = cap_actor.mlp.layers
    for cap_index in range(1, layers + 1):
        value = modules._modules[f"actor_linear{cap_index}"](value)
        value = modules._modules[f"actor_norm{cap_index}"](value)
        value = modules._modules[f"actor_act{cap_index}"](value)
    return cap_actor.last(value)


def copy_actor_suffix_to_cap(teacher_suffix: CapActorSuffix, cap_actor: Any) -> dict[str, str]:
    """Copy only verified hidden blocks 1..N and the bounded-normal head into a CAP MLPHead."""
    modules = cap_actor.mlp.layers
    with torch.no_grad():
        for index, source in enumerate(teacher_suffix.blocks):
            cap_index = index + 1
            target_linear = modules._modules[f"actor_linear{cap_index}"]
            target_norm = modules._modules[f"actor_norm{cap_index}"]
            if target_linear.weight.shape != source.linear.weight.shape:
                raise ValueError(f"CAP actor linear {cap_index} has incompatible shape")
            if target_norm.weight.shape != source.norm.weight.shape:
                raise ValueError(f"CAP actor norm {cap_index} has incompatible shape")
            target_linear.weight.copy_(source.linear.weight)
            target_linear.bias.copy_(source.linear.bias)
            target_norm.weight.copy_(source.norm.weight)
        if cap_actor.last.weight.shape != teacher_suffix.last.weight.shape:
            raise ValueError("CAP actor output head has incompatible shape")
        cap_actor.last.weight.copy_(teacher_suffix.last.weight)
        cap_actor.last.bias.copy_(teacher_suffix.last.bias)
    return teacher_suffix.cap_state_dict_manifest()


@dataclass(frozen=True, slots=True)
class ActorParity:
    raw_max_abs_error: float
    mean_max_abs_error: float
    scale_max_abs_error: float
    passed: bool


@torch.no_grad()
def measure_actor_suffix_parity(
    teacher_suffix: CapActorSuffix,
    cap_actor: Any,
    bottleneck: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> ActorParity:
    teacher_raw = teacher_suffix.raw_parameters(bottleneck)
    cap_raw = _cap_suffix_raw(cap_actor, bottleneck, len(teacher_suffix.blocks))
    raw_error = (teacher_raw - cap_raw).abs().max().item()
    teacher_mean_raw, teacher_scale_raw = teacher_raw.chunk(2, dim=-1)
    cap_mean_raw, cap_scale_raw = cap_raw.chunk(2, dim=-1)
    teacher_mean = torch.tanh(teacher_mean_raw.float())
    cap_mean = torch.tanh(cap_mean_raw.float())
    teacher_scale = (teacher_suffix.max_std - teacher_suffix.min_std) * torch.sigmoid(
        teacher_scale_raw.float() + 2.0
    ) + teacher_suffix.min_std
    cap_scale = (teacher_suffix.max_std - teacher_suffix.min_std) * torch.sigmoid(
        cap_scale_raw.float() + 2.0
    ) + teacher_suffix.min_std
    mean_error = (teacher_mean - cap_mean).abs().max().item()
    scale_error = (teacher_scale - cap_scale).abs().max().item()
    return ActorParity(raw_error, mean_error, scale_error, max(raw_error, mean_error, scale_error) <= atol)
