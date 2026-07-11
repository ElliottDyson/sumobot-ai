from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sumobot_ai.config import ArenaConfig
from sumobot_ai.state import ArenaState
from sumobot_ai.training.bootstrap import BootstrapTrainingConfig, compute_gae
from sumobot_ai.training.validation import render_top_down

ROOT = Path(__file__).resolve().parents[1]


def _state() -> ArenaState:
    position = torch.tensor([[[-0.5, 0.1, 0.046], [0.5, -0.1, 0.046]]])
    quaternion = torch.zeros(1, 2, 4)
    quaternion[..., 3] = 1.0
    return ArenaState(
        position=position,
        quaternion=quaternion,
        linear_velocity=torch.zeros(1, 2, 3),
        angular_velocity=torch.zeros(1, 2, 3),
        wheel_velocity=torch.zeros(1, 2, 4),
        action_exec=torch.zeros(1, 2, 4),
        contact_force=torch.zeros(1, 2, 3),
        edge_margin=torch.ones(1, 2),
        time_remaining_s=torch.tensor([12.5]),
    )


def test_bootstrap_training_config_loads() -> None:
    config = BootstrapTrainingConfig.load(ROOT / "configs/training/bootstrap.yaml")
    assert config.num_envs == 256
    assert config.validation_every_updates == 10
    assert config.device == "cuda:0"


def test_gae_respects_terminal_boundary() -> None:
    reward = torch.tensor([[1.0], [2.0], [3.0]])
    value = torch.zeros_like(reward)
    done = torch.tensor([[False], [True], [False]])
    advantage, returns = compute_gae(reward, value, done, torch.tensor([4.0]), gamma=0.5, gae_lambda=1.0)
    assert torch.allclose(advantage[:, 0], torch.tensor([2.0, 2.0, 5.0]))
    assert torch.equal(returns, advantage)


def test_validation_renderer_produces_tensorboard_frames() -> None:
    config = ArenaConfig.load(ROOT / "configs/arena/flat_3x2.yaml")
    frame = render_top_down(_state(), config)
    assert frame.shape == (320, 480, 3)
    assert frame.dtype == np.uint8
    assert frame.max() > frame.min()
