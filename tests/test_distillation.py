from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Independent, Normal

from sumobot_ai.distillation import (
    ReplayContract,
    copy_actor_suffix_to_cap,
    masked_teacher_kl,
    measure_actor_suffix_parity,
)
from sumobot_ai.training.cpo import CapActorSuffix


def replay_batch(batch: int = 5) -> dict[str, torch.Tensor]:
    return {
        "student_obs": torch.zeros(batch, 25),
        "privileged_obs": torch.zeros(batch, 71),
        "action_exec": torch.zeros(batch, 2),
        "action_student_raw": torch.zeros(batch, 2),
        "teacher_mean": torch.zeros(batch, 2),
        "teacher_scale": torch.ones(batch, 2) * 0.2,
        "teacher_action_mode": torch.zeros(batch, 2),
        "teacher_value_raw": torch.zeros(batch, 1),
        "teacher_feature": torch.zeros(batch, 256),
        "teacher_valid": torch.ones(batch, 1, dtype=torch.bool),
        "policy_id_executed": torch.zeros(batch, 1, dtype=torch.int64),
        "intervention": torch.zeros(batch, 1, dtype=torch.bool),
        "reward_env": torch.zeros(batch, 1),
        "reward_cpo_diversity": torch.zeros(batch, 1),
        "score_outcome": torch.full((batch, 1), -2, dtype=torch.int8),
        "is_first": torch.zeros(batch, 1, dtype=torch.bool),
        "is_last": torch.zeros(batch, 1, dtype=torch.bool),
        "is_terminal": torch.zeros(batch, 1, dtype=torch.bool),
        "domain_parameters": torch.zeros(batch, 24),
        "sensor_age_s": torch.zeros(batch, 2),
        "sensor_valid": torch.ones(batch, 5, dtype=torch.bool),
        "timestamp_s": torch.arange(batch, dtype=torch.float32).unsqueeze(-1),
        "episode": torch.zeros(batch, dtype=torch.int64),
    }


def fake_cap_actor(action_dim: int = 2) -> SimpleNamespace:
    modules = []
    for index in range(3):
        modules.extend(
            (
                (f"actor_linear{index}", nn.Linear(256, 256)),
                (f"actor_norm{index}", nn.RMSNorm(256, eps=1e-4)),
                (f"actor_act{index}", nn.SiLU()),
            )
        )
    return SimpleNamespace(
        mlp=SimpleNamespace(layers=nn.Sequential(OrderedDict(modules))),
        last=nn.Linear(256, 2 * action_dim),
    )


def test_replay_contract_and_masked_kl() -> None:
    batch = replay_batch()
    ReplayContract.validate(batch)
    student = Independent(Normal(torch.zeros(5, 2), torch.ones(5, 2) * 0.3), 1)
    loss = masked_teacher_kl(student, batch["teacher_mean"], batch["teacher_scale"], batch["teacher_valid"])
    assert torch.isfinite(loss) and loss >= 0


def test_replay_rejects_unexecuted_action_range() -> None:
    batch = replay_batch()
    batch["action_exec"][0, 0] = 1.2
    try:
        ReplayContract.validate(batch)
    except ValueError as error:
        assert "action_exec" in str(error)
    else:
        raise AssertionError("out-of-range executed action was accepted")


def test_actor_suffix_copy_has_numerical_parity() -> None:
    teacher = CapActorSuffix(2)
    cap = fake_cap_actor()
    manifest = copy_actor_suffix_to_cap(teacher, cap)
    parity = measure_actor_suffix_parity(teacher, cap, torch.randn(7, 256))
    assert parity.passed
    assert len(manifest) == 8
