from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Independent, Normal


def _weight_init(module: nn.Module, *, output_scale: float = 1.0) -> None:
    if isinstance(module, nn.RMSNorm):
        with torch.no_grad():
            module.weight.fill_(1.0)
        return
    weight = getattr(module, "weight", None)
    if weight is None or weight.numel() == 0:
        return
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    std = 1.1368 * math.sqrt(1.0 / fan_in)
    with torch.no_grad():
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        weight.mul_(output_scale)
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias.zero_()


class ActorBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=True)
        self.norm = nn.RMSNorm(width, eps=1e-4, dtype=torch.float32)
        self.activation = nn.SiLU()
        self.apply(_weight_init)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.linear(value)))


class CapActorSuffix(nn.Module):
    """CAP-12M-compatible hidden suffix and bounded-normal parameter head."""

    def __init__(
        self,
        action_dim: int,
        *,
        width: int = 256,
        layers: int = 2,
        min_std: float = 0.1,
        max_std: float = 1.0,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or width <= 0 or layers <= 0:
            raise ValueError("actor dimensions and layer count must be positive")
        if not 0 < min_std < max_std:
            raise ValueError("actor std bounds must satisfy 0 < min_std < max_std")
        self.width = width
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std
        self.blocks = nn.ModuleList(ActorBlock(width) for _ in range(layers))
        self.last = nn.Linear(width, action_dim * 2, bias=True)
        _weight_init(self.last, output_scale=0.01)

    def raw_parameters(self, bottleneck: torch.Tensor) -> torch.Tensor:
        value = bottleneck
        for block in self.blocks:
            value = block(value)
        return self.last(value)

    def forward(self, bottleneck: torch.Tensor) -> Independent:
        raw_mean, raw_std = self.raw_parameters(bottleneck).chunk(2, dim=-1)
        mean = torch.tanh(raw_mean.float())
        std = (self.max_std - self.min_std) * torch.sigmoid(raw_std.float() + 2.0) + self.min_std
        return Independent(Normal(mean, std), 1)

    def cap_state_dict_manifest(self, *, cap_actor_prefix: str = "actor") -> dict[str, str]:
        """Map local keys to the pinned CAP MLPHead's suffix keys."""
        manifest: dict[str, str] = {}
        for index in range(len(self.blocks)):
            cap_index = index + 1  # CAP block 0 maps the RSSM feature to the shared 256 bottleneck.
            manifest[f"blocks.{index}.linear.weight"] = f"{cap_actor_prefix}.mlp.layers.actor_linear{cap_index}.weight"
            manifest[f"blocks.{index}.linear.bias"] = f"{cap_actor_prefix}.mlp.layers.actor_linear{cap_index}.bias"
            manifest[f"blocks.{index}.norm.weight"] = f"{cap_actor_prefix}.mlp.layers.actor_norm{cap_index}.weight"
        manifest["last.weight"] = f"{cap_actor_prefix}.last.weight"
        manifest["last.bias"] = f"{cap_actor_prefix}.last.bias"
        return manifest


@dataclass(frozen=True, slots=True)
class CpoPolicyOutput:
    distribution: Independent
    value: torch.Tensor
    bottleneck: torch.Tensor


class TransplantableCPOActorCritic(nn.Module):
    """One conditioned CPO population with a privileged frontend and shared actor suffix."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        population_size: int = 6,
        *,
        policy_id_dim: int = 8,
        frontend_units: Sequence[int] = (256, 256),
        bottleneck_dim: int = 256,
        suffix_layers: int = 2,
        min_std: float = 0.1,
        max_std: float = 1.0,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or population_size <= 0 or policy_id_dim <= 0:
            raise ValueError("CPO observation and population dimensions must be positive")
        if bottleneck_dim != 256:
            raise ValueError("the first CAP-compatible teacher uses a 256-dimensional actor bottleneck")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.population_size = population_size
        self.policy_embedding = nn.Embedding(population_size, policy_id_dim)
        layers: list[nn.Module] = []
        input_dim = observation_dim + policy_id_dim
        for width in frontend_units:
            layers.extend((nn.Linear(input_dim, int(width)), nn.ELU()))
            input_dim = int(width)
        layers.extend(
            (
                nn.Linear(input_dim, bottleneck_dim),
                nn.RMSNorm(bottleneck_dim, eps=1e-4, dtype=torch.float32),
                nn.SiLU(),
            )
        )
        self.frontend = nn.Sequential(*layers)
        self.frontend.apply(_weight_init)
        self.actor_suffix = CapActorSuffix(
            action_dim,
            width=bottleneck_dim,
            layers=suffix_layers,
            min_std=min_std,
            max_std=max_std,
        )
        self.value = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ELU(),
            nn.Linear(bottleneck_dim, 1),
        )
        self.value.apply(_weight_init)

    def forward(self, observation: torch.Tensor, policy_id: torch.Tensor) -> CpoPolicyOutput:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(f"expected observation width {self.observation_dim}, got {observation.shape[-1]}")
        if policy_id.dtype != torch.long:
            raise ValueError("policy_id must have dtype torch.long")
        if tuple(policy_id.shape) != tuple(observation.shape[:-1]):
            raise ValueError("policy_id batch shape must match observation batch shape")
        if bool(((policy_id < 0) | (policy_id >= self.population_size)).any()):
            raise ValueError("policy_id is outside this CPO population")
        embedding = self.policy_embedding(policy_id)
        bottleneck = self.frontend(torch.cat((observation, embedding), dim=-1))
        return CpoPolicyOutput(self.actor_suffix(bottleneck), self.value(bottleneck).squeeze(-1), bottleneck)

    def leader(self, observation: torch.Tensor) -> CpoPolicyOutput:
        ids = torch.zeros(observation.shape[:-1], dtype=torch.long, device=observation.device)
        return self(observation, ids)


@dataclass(frozen=True, slots=True)
class CpoLossConfig:
    ppo_clip: float = 0.2
    follower_kl_coefficient: float = 0.1
    awac_temperature: float = 1.0
    awac_max_weight: float = 20.0
    ppo_scale: float = 1.0
    awac_scale: float = 1.0
    follower_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.ppo_clip < 1:
            raise ValueError("ppo_clip must be in (0, 1)")
        if self.follower_kl_coefficient < 0 or self.awac_temperature <= 0 or self.awac_max_weight <= 0:
            raise ValueError("CPO KL/AWAC parameters are invalid")


@dataclass(frozen=True, slots=True)
class CpoLossResult:
    total: torch.Tensor
    ppo: torch.Tensor
    follower_kl_ppo: torch.Tensor
    awac: torch.Tensor


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(value.dtype)
    return (value * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def cpo_actor_loss(
    *,
    old_log_prob: torch.Tensor,
    new_log_prob: torch.Tensor,
    leader_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    leader_online_mask: torch.Tensor,
    follower_online_mask: torch.Tensor,
    off_policy_mask: torch.Tensor,
    awac_mask: torch.Tensor,
    config: CpoLossConfig | None = None,
) -> CpoLossResult:
    """Leader PPO, follower KL-constrained PPO, and relabelled AWAC from the cited CPO implementation."""
    config = config or CpoLossConfig()
    tensors = (
        old_log_prob,
        new_log_prob,
        leader_log_prob,
        advantage,
        leader_online_mask,
        follower_online_mask,
        off_policy_mask,
        awac_mask,
    )
    if any(tensor.shape != old_log_prob.shape for tensor in tensors):
        raise ValueError("all CPO loss tensors must have the same shape")
    if any(mask.dtype != torch.bool for mask in (leader_online_mask, follower_online_mask, off_policy_mask, awac_mask)):
        raise ValueError("all CPO masks must be boolean")
    log_ratio = (new_log_prob - old_log_prob).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - config.ppo_clip, 1.0 + config.ppo_clip)

    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    ppo = _masked_mean(-surrogate, leader_online_mask | off_policy_mask)

    # The sampled-action log ratio estimates log(pi_follower / pi_leader). Subtracting it
    # from advantage implements the original follower PPO term's KL-to-leader pressure.
    follower_advantage = advantage - config.follower_kl_coefficient * (new_log_prob - leader_log_prob.detach())
    follower_surrogate = torch.minimum(ratio * follower_advantage, clipped_ratio * follower_advantage)
    follower = _masked_mean(-follower_surrogate, follower_online_mask)

    max_log_weight = math.log(config.awac_max_weight)
    awac_weight = torch.exp((advantage / config.awac_temperature).clamp(max=max_log_weight))
    awac = _masked_mean(-awac_weight * new_log_prob, awac_mask)
    total = config.ppo_scale * ppo + config.follower_scale * follower + config.awac_scale * awac
    return CpoLossResult(total=total, ppo=ppo, follower_kl_ppo=follower, awac=awac)


class DiversityDiscriminator(nn.Module):
    """Optional CPO-internal policy-ID classifier; unrelated to match win/loss reward."""

    def __init__(self, input_dim: int, population_size: int, hidden: Sequence[int] = (256, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for next_width in hidden:
            layers.extend((nn.Linear(width, int(next_width)), nn.ELU()))
            width = int(next_width)
        layers.append(nn.Linear(width, population_size))
        self.network = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)

    @staticmethod
    def follower_reward(logits: torch.Tensor, policy_id: torch.Tensor, coefficient: float) -> torch.Tensor:
        if coefficient < 0:
            raise ValueError("diversity coefficient cannot be negative")
        log_probability = torch.log_softmax(logits, dim=-1).gather(-1, policy_id.unsqueeze(-1)).squeeze(-1)
        return coefficient * log_probability * (policy_id != 0)
