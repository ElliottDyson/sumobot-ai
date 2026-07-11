from __future__ import annotations

from pathlib import Path

import torch

from sumobot_ai.rewards import RewardSpec, evaluate_dual, evaluate_reward, register_reward_term
from sumobot_ai.state import BLUE, DRAW, RED, ArenaState, ArenaTransition

ROOT = Path(__file__).resolve().parents[1]


def make_state(*, current: bool) -> ArenaState:
    batch = 3
    position = torch.zeros(batch, 2, 3)
    position[:, 0, 0] = -0.5 + (0.05 if current else 0.0)
    position[:, 1, 0] = 0.5 - (0.10 if current else 0.0)
    quaternion = torch.zeros(batch, 2, 4)
    quaternion[:, 0, 3] = 1.0
    quaternion[:, 1, 2] = 1.0
    edge = torch.tensor([[0.5, 0.3], [0.4, 0.2], [0.8, 0.7]])
    if current:
        edge = edge + torch.tensor([[0.01, -0.04], [0.02, -0.03], [0.0, 0.0]])
    return ArenaState(
        position=position,
        quaternion=quaternion,
        linear_velocity=torch.zeros(batch, 2, 3),
        angular_velocity=torch.zeros(batch, 2, 3),
        wheel_velocity=torch.zeros(batch, 2, 4),
        action_exec=torch.full((batch, 2, 4), 0.25),
        contact_force=torch.zeros(batch, 2, 3),
        edge_margin=edge,
        time_remaining_s=torch.full((batch,), 9.98 if current else 10.0),
    )


def make_transition() -> ArenaTransition:
    return ArenaTransition(
        previous=make_state(current=False),
        current=make_state(current=True),
        terminated=torch.tensor([True, True, False]),
        truncated=torch.tensor([False, False, True]),
        winner=torch.tensor([RED, BLUE, DRAW]),
    )


def test_sparse_outcome_reward_is_zero_sum_and_swap_symmetric() -> None:
    spec = RewardSpec.load(ROOT / "configs/rewards/sparse.yaml")
    transition = make_transition()
    reward = evaluate_reward(spec, transition).total
    assert torch.equal(reward, torch.tensor([[3.0, -3.0], [-3.0, 3.0], [0.0, 0.0]]))
    swapped = evaluate_reward(spec, transition.swapped()).total
    assert torch.equal(swapped, reward[:, [1, 0]])


def test_authored_dual_reward_selects_each_owner_side() -> None:
    baseline = RewardSpec.load(ROOT / "configs/rewards/baseline.yaml")
    sparse = RewardSpec.load(ROOT / "configs/rewards/sparse.yaml")
    transition = make_transition()
    result = evaluate_dual(baseline, sparse, transition)
    assert result.total.shape == (3, 2)
    assert torch.isfinite(result.total).all()
    assert result.total[:, 1].equal(evaluate_reward(sparse, transition).total[:, 1])
    assert any(name.startswith("red.") for name in result.components)
    assert any(name.startswith("blue.") for name in result.components)


def test_reward_hash_is_stable() -> None:
    first = RewardSpec.load(ROOT / "configs/rewards/bootstrap_competitive.yaml")
    second = RewardSpec.from_mapping(first.canonical_dict())
    assert first.digest == second.digest
    assert len(first.digest) == 64


def test_member_can_register_a_vectorized_custom_term() -> None:
    def custom_term(transition: ArenaTransition, params: dict[str, float]) -> torch.Tensor:
        return torch.ones_like(transition.current.edge_margin) * params["value"]

    register_reward_term(
        "test_custom_constant",
        custom_term,
        allowed_params=frozenset({"value"}),
        defaults={"value": 0.25},
    )
    spec = RewardSpec.from_mapping(
        {
            "version": 1,
            "name": "custom",
            "terms": [{"name": "test_custom_constant", "weight": 2.0}],
        }
    )
    result = evaluate_reward(spec, make_transition())
    assert torch.equal(result.total, torch.full((3, 2), 0.5))
