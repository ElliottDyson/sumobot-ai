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
    assert result["max_differential_yaw_rate_rad_s"] > 0.1
    assert 10.0 <= result["stationary_draw_after_s"] <= 15.0
    assert result["numerical_failures"] == 0
