from __future__ import annotations

import torch

from sumobot_ai.match import resolve_match
from sumobot_ai.state import BLUE, DRAW, RED


def test_ring_out_inactivity_and_timeout_resolution() -> None:
    out = torch.tensor(
        [
            [True, False],
            [False, True],
            [True, True],
            [False, False],
            [False, False],
            [False, False],
        ]
    )
    inactive = torch.tensor(
        [
            [False, True],  # Ring-out takes precedence over blue's inactivity.
            [False, False],
            [False, False],
            [True, False],
            [True, True],
            [False, False],
        ]
    )
    timed_out = torch.tensor([False, False, False, False, False, True])
    numerical_failure = torch.zeros(6, dtype=torch.bool)
    result = resolve_match(out, inactive, timed_out, numerical_failure)
    assert torch.equal(result.winner, torch.tensor([BLUE, RED, DRAW, BLUE, DRAW, DRAW]))
    assert torch.equal(result.terminated, torch.tensor([True, True, True, True, True, False]))
    assert torch.equal(result.truncated, torch.tensor([False, False, False, False, False, True]))
    assert result.ring_out[:3].all()
    assert result.inactivity[3:5].all()


def test_numerical_failure_is_a_draw_not_a_learnable_win() -> None:
    result = resolve_match(
        torch.tensor([[True, False]]),
        torch.tensor([[False, False]]),
        torch.tensor([False]),
        torch.tensor([True]),
    )
    assert result.terminated.item()
    assert result.winner.item() == DRAW
    assert not result.ring_out.item()
