from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from sumobot_ai.config import ArenaConfig
from sumobot_ai.domain_randomization import DomainRandomizer
from sumobot_ai.observations import (
    StudentObservationHistory,
    StudentSensors,
    TeacherObservationHistory,
    build_student_observation,
    build_teacher_observation,
)
from sumobot_ai.state import ArenaState

ROOT = Path(__file__).resolve().parents[1]


def make_state(batch: int = 4) -> ArenaState:
    position = torch.zeros(batch, 2, 3)
    position[:, 0, 0] = -0.5
    position[:, 1, 0] = 0.5
    quaternion = torch.zeros(batch, 2, 4)
    quaternion[..., 3] = 1.0
    return ArenaState(
        position=position,
        quaternion=quaternion,
        linear_velocity=torch.zeros(batch, 2, 3),
        angular_velocity=torch.zeros(batch, 2, 3),
        wheel_velocity=torch.zeros(batch, 2, 2),
        action_proposed=torch.zeros(batch, 2, 2),
        action_exec=torch.zeros(batch, 2, 2),
        contact_force=torch.zeros(batch, 2, 3),
        edge_margin=torch.ones(batch, 2),
        support_margin=torch.ones(batch, 2),
        stationary_time_s=torch.zeros(batch, 2),
        time_remaining_s=torch.full((batch,), 30.0),
    )


def test_arena_config_contract() -> None:
    config = ArenaConfig.load(ROOT / "configs/arena/flat_3x2.yaml")
    assert config.board.size_m == (3.0, 2.0)
    assert config.version == 3
    assert config.robot.envelope_size_m == (0.04, 0.04, 0.08)
    assert config.robot.chassis_size_m == (0.04, 0.028, 0.074)
    assert config.robot.track_m + config.robot.wheel_width_m <= config.robot.envelope_size_m[1]
    assert config.robot.chassis_ground_clearance_m + config.robot.chassis_size_m[2] <= 0.08
    assert config.robot.action_order == ("left_wheel", "right_wheel")
    assert config.robot.skid_x_m < config.robot.wheel_axle_x_m
    assert config.match.inactivity_timeout_s == 10.0
    assert config.match.movement_confirmation_s == 0.2
    assert config.physics.backend == "mujoco_warp"
    assert config.physics.max_episode_steps == 1500


def test_privileged_and_student_observations_are_separate() -> None:
    config = ArenaConfig.load(ROOT / "configs/arena/flat_3x2.yaml")
    domain = DomainRandomizer(config.domain_randomization).sample(4, generator=torch.Generator().manual_seed(3))
    state = make_state()
    teacher_history = TeacherObservationHistory(state, config.sensors.proprio_history_steps)
    teacher = build_teacher_observation(state, domain, perspective=0, history=teacher_history)
    assert "self_position" in teacher.fields
    assert "opponent_position" in teacher.fields
    assert any(name.startswith("domain.") for name in teacher.fields)
    assert teacher.values.shape == (4, 295)
    assert teacher.field("proprioceptive_history").shape == (4, 170)
    assert all(field.stop > field.start for name, field in teacher.fields.items() if name.startswith("domain."))

    zeros = lambda width: torch.zeros(4, width)  # noqa: E731 - compact tensor fixture
    sensors = StudentSensors(
        wheel_velocity=zeros(2),
        imu_gyro=zeros(3),
        imu_acceleration=zeros(3),
        gravity_direction=zeros(3),
        edge_ranges=zeros(4),
        opponent_range_bearing_valid=zeros(4),
        previous_action_exec=zeros(2),
        edge_sensor_age_s=zeros(1),
        opponent_sensor_age_s=zeros(1),
        stationary_time_fraction=zeros(1),
        time_fraction=zeros(1),
    )
    student_history = StudentObservationHistory(sensors, config.sensors.proprio_history_steps)
    student = build_student_observation(sensors, student_history)
    assert student.values.shape == (4, 155)
    assert student.field("proprioceptive_history").shape == (4, 130)
    assert not any("position" in name or name.startswith("domain.") for name in student.fields)
    assert teacher.values.shape[-1] > student.values.shape[-1]


def test_domain_samples_respect_ranges() -> None:
    config = ArenaConfig.load(ROOT / "configs/arena/flat_3x2.yaml")
    batch = DomainRandomizer(config.domain_randomization).sample(128, generator=torch.Generator().manual_seed(11))
    for name, bounds in config.domain_randomization.items():
        value = getattr(batch, name)
        assert float(value.min()) >= bounds[0]
        assert float(value.max()) <= bounds[1]


def test_config_rejects_external_geometry_that_exceeds_legal_envelope() -> None:
    data = yaml.safe_load((ROOT / "configs/arena/flat_3x2.yaml").read_text(encoding="utf-8"))
    data["robot"]["track_m"] = 0.04
    with pytest.raises(ValueError, match="legal width envelope"):
        ArenaConfig.from_mapping(data)


def test_teacher_and_student_histories_shift_and_reset_without_leaking_opponent_actions() -> None:
    state = make_state(batch=2)
    teacher_history = TeacherObservationHistory(state, 10)
    proposed = state.action_proposed.clone()
    proposed[:, 0] = torch.tensor([0.4, -0.2])
    proposed[:, 1] = torch.tensor([-0.8, 0.7])
    teacher_history.update(replace(state, action_proposed=proposed))
    latest_red = teacher_history.perspective(0).reshape(2, 10, -1)[:, -1]
    assert torch.equal(latest_red[:, :2], proposed[:, 0])
    assert not torch.equal(latest_red[:, :2], proposed[:, 1])

    zeros = lambda width: torch.zeros(2, width)  # noqa: E731
    sensors = StudentSensors(
        wheel_velocity=zeros(2),
        imu_gyro=zeros(3),
        imu_acceleration=zeros(3),
        gravity_direction=zeros(3),
        edge_ranges=zeros(4),
        opponent_range_bearing_valid=zeros(4),
        previous_action_exec=zeros(2),
        edge_sensor_age_s=zeros(1),
        opponent_sensor_age_s=zeros(1),
        stationary_time_fraction=zeros(1),
        time_fraction=zeros(1),
    )
    history = StudentObservationHistory(sensors, 10)
    moved = replace(sensors, previous_action_exec=torch.ones(2, 2))
    history.update(moved, torch.tensor([False, True]))
    frames = history.values().reshape(2, 10, -1)
    assert torch.equal(frames[0, -1, -2:], torch.ones(2))
    assert torch.equal(frames[1, :, -2:], torch.ones(10, 2))
