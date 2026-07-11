from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sumobot_ai.config import ArenaConfig
from sumobot_ai.state import ArenaState
from sumobot_ai.training.bootstrap import BootstrapTrainingConfig, compute_gae
from sumobot_ai.training.validation import encode_validation_gif, render_top_down, validation_capture_steps

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
        wheel_velocity=torch.zeros(1, 2, 2),
        action_exec=torch.zeros(1, 2, 2),
        contact_force=torch.zeros(1, 2, 3),
        edge_margin=torch.ones(1, 2),
        stationary_time_s=torch.zeros(1, 2),
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


def test_validation_capture_schedule_reaches_match_timeout() -> None:
    steps = validation_capture_steps(1500, 50, 5, 151)
    assert len(steps) == 150
    assert min(steps) == 10
    assert max(steps) == 1500


def test_validation_capture_schedule_rejects_truncating_cap() -> None:
    try:
        validation_capture_steps(1500, 50, 5, 150)
    except ValueError as error:
        assert "truncates the match" in str(error)
    else:
        raise AssertionError("a video frame cap shorter than the match was accepted")


def test_validation_gif_preserves_frames_and_playback_timing() -> None:
    video = torch.zeros(1, 3, 3, 8, 8, dtype=torch.uint8)
    video[:, 1, 0] = 127
    video[:, 2, 1] = 255
    encoded = encode_validation_gif(video, fps=5)
    image = Image.open(BytesIO(encoded))
    durations = []
    for frame_index in range(image.n_frames):
        image.seek(frame_index)
        durations.append(image.info["duration"])
    assert image.n_frames == 3
    assert durations == [200, 200, 200]
