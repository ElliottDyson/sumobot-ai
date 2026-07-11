from __future__ import annotations

import torch

from sumobot_ai.training import CpoPopulationInstance, DualCpoBootstrapSession
from sumobot_ai.training.cpo import CpoLossConfig, DiversityDiscriminator, TransplantableCPOActorCritic, cpo_actor_loss


def test_transplantable_population_shapes_and_distribution() -> None:
    model = TransplantableCPOActorCritic(71, 4, population_size=6)
    observation = torch.randn(12, 71)
    policy_id = torch.arange(12) % 6
    output = model(observation, policy_id.long())
    action = output.distribution.rsample()
    assert action.shape == (12, 4)
    assert output.value.shape == (12,)
    assert output.bottleneck.shape == (12, 256)
    assert output.distribution.base_dist.loc.abs().max() <= 1.0
    assert output.distribution.base_dist.scale.min() >= 0.1
    assert output.distribution.base_dist.scale.max() <= 1.0
    manifest = model.actor_suffix.cap_state_dict_manifest()
    assert manifest["blocks.0.linear.weight"].endswith("actor_linear1.weight")
    assert manifest["last.weight"].endswith("actor.last.weight")


def test_cpo_loss_channels_are_finite() -> None:
    size = 8
    masks = {
        "leader_online_mask": torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool),
        "follower_online_mask": torch.tensor([0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.bool),
        "off_policy_mask": torch.tensor([0, 0, 0, 0, 1, 1, 0, 0], dtype=torch.bool),
        "awac_mask": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool),
    }
    result = cpo_actor_loss(
        old_log_prob=torch.zeros(size),
        new_log_prob=torch.linspace(-0.1, 0.1, size, requires_grad=True),
        leader_log_prob=torch.zeros(size),
        advantage=torch.ones(size),
        config=CpoLossConfig(),
        **masks,
    )
    assert torch.isfinite(result.total)
    result.total.backward()


def test_optional_diversity_reward_excludes_leader() -> None:
    discriminator = DiversityDiscriminator(10, 3)
    ids = torch.tensor([0, 1, 2])
    logits = discriminator(torch.randn(3, 10))
    reward = discriminator.follower_reward(logits, ids, 0.01)
    assert reward[0] == 0
    assert torch.isfinite(reward).all()


def test_dual_bootstrap_updates_two_independent_populations_after_barrier() -> None:
    red_model = TransplantableCPOActorCritic(12, 4, population_size=2)
    blue_model = TransplantableCPOActorCritic(12, 4, population_size=2)
    session = DualCpoBootstrapSession(
        CpoPopulationInstance("red", red_model, torch.optim.Adam(red_model.parameters(), lr=1e-3)),
        CpoPopulationInstance("blue", blue_model, torch.optim.Adam(blue_model.parameters(), lr=1e-3)),
    )
    red_before = red_model.actor_suffix.last.weight.detach().clone()
    blue_before = blue_model.actor_suffix.last.weight.detach().clone()
    session.begin_rollout()
    observation = torch.randn(6, 12)
    policy_id = torch.arange(6).remainder(2).long()
    red, blue = session.act(observation, policy_id, observation, policy_id)
    actions = torch.zeros(6, 4)
    metrics = session.finish_rollout(
        -red.distribution.log_prob(actions).mean(),
        -blue.distribution.log_prob(actions).mean(),
    )
    assert not session.rollout_active and session.rollout_index == 1
    assert not torch.equal(red_before, red_model.actor_suffix.last.weight)
    assert not torch.equal(blue_before, blue_model.actor_suffix.last.weight)
    assert set(metrics) == {"red_loss", "blue_loss", "red_grad_norm", "blue_grad_norm"}
