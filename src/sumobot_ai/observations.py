from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .contracts import ACTION_DIM, DRIVEN_WHEEL_COUNT
from .domain_randomization import DomainBatch
from .state import ArenaState


@dataclass(frozen=True, slots=True)
class ObservationVector:
    values: torch.Tensor
    fields: Mapping[str, slice]

    def field(self, name: str) -> torch.Tensor:
        return self.values[..., self.fields[name]]


def _pack(named: list[tuple[str, torch.Tensor]]) -> ObservationVector:
    fields: dict[str, slice] = {}
    offset = 0
    for name, value in named:
        width = value.shape[-1]
        fields[name] = slice(offset, offset + width)
        offset += width
    return ObservationVector(torch.cat([value for _, value in named], dim=-1), fields)


def build_teacher_observation(state: ArenaState, domain: DomainBatch, perspective: int) -> ObservationVector:
    """Build the privileged vector for one side without a policy-ID embedding."""
    if perspective not in (0, 1):
        raise ValueError("perspective must be 0 (red) or 1 (blue)")
    if state.batch_size != domain.batch_size:
        raise ValueError("state and domain batches must match")
    opponent = 1 - perspective
    domain_values, domain_names = domain.teacher_tensor()
    named = [
        ("self_position", state.position[:, perspective]),
        ("self_quaternion_xyzw", state.quaternion[:, perspective]),
        ("self_linear_velocity", state.linear_velocity[:, perspective]),
        ("self_angular_velocity", state.angular_velocity[:, perspective]),
        ("opponent_position", state.position[:, opponent]),
        ("opponent_quaternion_xyzw", state.quaternion[:, opponent]),
        ("opponent_linear_velocity", state.linear_velocity[:, opponent]),
        ("opponent_angular_velocity", state.angular_velocity[:, opponent]),
        ("relative_position", state.position[:, opponent] - state.position[:, perspective]),
        ("relative_linear_velocity", state.linear_velocity[:, opponent] - state.linear_velocity[:, perspective]),
        ("self_wheel_velocity", state.wheel_velocity[:, perspective]),
        ("opponent_wheel_velocity", state.wheel_velocity[:, opponent]),
        ("self_contact_force", state.contact_force[:, perspective]),
        ("opponent_contact_force", state.contact_force[:, opponent]),
        ("edge_margins", state.edge_margin[:, [perspective, opponent]]),
        ("stationary_time_s", state.stationary_time_s[:, [perspective, opponent]]),
        ("time_remaining_s", state.time_remaining_s.unsqueeze(-1)),
    ]
    for index, name in enumerate(domain_names):
        named.append((f"domain.{name}", domain_values[:, index : index + 1]))
    return _pack(named)


@dataclass(frozen=True, slots=True)
class StudentSensors:
    wheel_velocity: torch.Tensor  # (B, 2)
    imu_gyro: torch.Tensor  # (B, 3)
    imu_acceleration: torch.Tensor  # (B, 3)
    gravity_direction: torch.Tensor  # (B, 3)
    edge_ranges: torch.Tensor  # (B, 4)
    opponent_range_bearing_valid: torch.Tensor  # (B, 4): range, sin(bearing), cos(bearing), valid
    previous_action_exec: torch.Tensor  # (B, 2)
    edge_sensor_age_s: torch.Tensor  # (B, 1)
    opponent_sensor_age_s: torch.Tensor  # (B, 1)
    stationary_time_fraction: torch.Tensor  # (B, 1), locally tracked from deployable odometry
    time_fraction: torch.Tensor  # (B, 1)

    def __post_init__(self) -> None:
        batch = self.wheel_velocity.shape[0]
        expected = {
            "wheel_velocity": (batch, DRIVEN_WHEEL_COUNT),
            "imu_gyro": (batch, 3),
            "imu_acceleration": (batch, 3),
            "gravity_direction": (batch, 3),
            "edge_ranges": (batch, 4),
            "opponent_range_bearing_valid": (batch, 4),
            "previous_action_exec": (batch, ACTION_DIM),
            "edge_sensor_age_s": (batch, 1),
            "opponent_sensor_age_s": (batch, 1),
            "stationary_time_fraction": (batch, 1),
            "time_fraction": (batch, 1),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            if value.device != self.wheel_velocity.device:
                raise ValueError("all StudentSensors tensors must use the same device")


def build_student_observation(sensors: StudentSensors) -> ObservationVector:
    """Build the deployment-only vector; privileged simulator fields are impossible inputs here."""
    return _pack(
        [
            ("wheel_velocity", sensors.wheel_velocity),
            ("imu_gyro", sensors.imu_gyro),
            ("imu_acceleration", sensors.imu_acceleration),
            ("gravity_direction", sensors.gravity_direction),
            ("edge_ranges", sensors.edge_ranges),
            ("opponent_range_bearing_valid", sensors.opponent_range_bearing_valid),
            ("previous_action_exec", sensors.previous_action_exec),
            ("edge_sensor_age_s", sensors.edge_sensor_age_s),
            ("opponent_sensor_age_s", sensors.opponent_sensor_age_s),
            ("stationary_time_fraction", sensors.stationary_time_fraction),
            ("time_fraction", sensors.time_fraction),
        ]
    )
