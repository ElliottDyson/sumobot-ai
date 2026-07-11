from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from sumobot_ai.sim.newton_backend import run_smoke

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.sim
@pytest.mark.gpu
def test_newton_mujoco_warp_smoke() -> None:
    result = run_smoke(ROOT / "configs/arena/flat_3x2.yaml", world_count=2, steps=20, device="cuda:0")
    assert result["finite"]
    assert result["worlds"] == 2
    assert result["mean_displacement_m"] > 0.02
