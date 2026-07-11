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
    assert result["mean_normal_contact_force_n"] > 0.5
    assert result["maximum_static_load_relative_error"] < 0.10
    assert 0.0 <= result["first_step_executed_command_mean"] < 0.35
    assert result["robot_envelope_size_m"] == [0.04, 0.04, 0.08]
    assert result["student_observation_base_dim"] == 25
    assert result["support_rule_verified"]
    assert 10.0 <= result["stationary_draw_after_s"] <= 15.0
    assert result["numerical_failures"] == 0
