from __future__ import annotations

from dataclasses import dataclass

import torch

from .cpo import CpoPolicyOutput, TransplantableCPOActorCritic


@dataclass(slots=True)
class CpoPopulationInstance:
    name: str
    model: TransplantableCPOActorCritic
    optimizer: torch.optim.Optimizer

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CPO population name cannot be empty")
        model_ids = {id(parameter) for parameter in self.model.parameters()}
        optimizer_ids = {id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]}
        if model_ids != optimizer_ids:
            raise ValueError("optimizer parameters must exactly match its CPO model")


class DualCpoBootstrapSession:
    """Update barrier for two literal competitors with independent models and optimizers."""

    def __init__(self, red: CpoPopulationInstance, blue: CpoPopulationInstance) -> None:
        if red.name == blue.name:
            raise ValueError("red and blue CPO populations need distinct names")
        red_parameters = {id(parameter) for parameter in red.model.parameters()}
        blue_parameters = {id(parameter) for parameter in blue.model.parameters()}
        if red_parameters & blue_parameters:
            raise ValueError("red and blue CPO populations cannot share Parameter objects")
        self.red = red
        self.blue = blue
        self._rollout_active = False
        self.rollout_index = 0

    @property
    def rollout_active(self) -> bool:
        return self._rollout_active

    def begin_rollout(self) -> int:
        if self._rollout_active:
            raise RuntimeError("a bootstrap rollout is already active")
        self._rollout_active = True
        self.red.model.train()
        self.blue.model.train()
        return self.rollout_index

    def act(
        self,
        red_observation: torch.Tensor,
        red_policy_id: torch.Tensor,
        blue_observation: torch.Tensor,
        blue_policy_id: torch.Tensor,
    ) -> tuple[CpoPolicyOutput, CpoPolicyOutput]:
        if not self._rollout_active:
            raise RuntimeError("begin_rollout() must be called before collecting actions")
        return (
            self.red.model(red_observation, red_policy_id),
            self.blue.model(blue_observation, blue_policy_id),
        )

    def finish_rollout(
        self,
        red_loss: torch.Tensor,
        blue_loss: torch.Tensor,
        *,
        max_grad_norm: float = 1.0,
    ) -> dict[str, float]:
        if not self._rollout_active:
            raise RuntimeError("there is no active rollout to update")
        if red_loss.ndim != 0 or blue_loss.ndim != 0:
            raise ValueError("population losses must be scalar")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self.red.optimizer.zero_grad(set_to_none=True)
        self.blue.optimizer.zero_grad(set_to_none=True)
        red_loss.backward()
        blue_loss.backward()
        red_norm = torch.nn.utils.clip_grad_norm_(self.red.model.parameters(), max_grad_norm)
        blue_norm = torch.nn.utils.clip_grad_norm_(self.blue.model.parameters(), max_grad_norm)
        self.red.optimizer.step()
        self.blue.optimizer.step()
        self._rollout_active = False
        self.rollout_index += 1
        return {
            "red_loss": float(red_loss.detach().item()),
            "blue_loss": float(blue_loss.detach().item()),
            "red_grad_norm": float(red_norm.detach().item()),
            "blue_grad_norm": float(blue_norm.detach().item()),
        }

    def cancel_rollout(self) -> None:
        self.red.optimizer.zero_grad(set_to_none=True)
        self.blue.optimizer.zero_grad(set_to_none=True)
        self._rollout_active = False
