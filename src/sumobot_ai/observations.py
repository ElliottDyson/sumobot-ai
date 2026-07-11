from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .contracts import ACTION_DIM, DRIVEN_WHEEL_COUNT
from .domain_randomization import DomainBatch
from .geometry import quaternion_rotate_inverse_xyzw
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


TEACHER_HISTORY_FEATURE_DIM = 17


def _teacher_history_frame(state: ArenaState) -> torch.Tensor:
    frames: list[torch.Tensor] = []
    for perspective in (0, 1):
        opponent = 1 - perspective
        quaternion = state.quaternion[:, perspective]
        self_velocity_body = quaternion_rotate_inverse_xyzw(quaternion, state.linear_velocity[:, perspective])
        relative_position_body = quaternion_rotate_inverse_xyzw(
            quaternion, state.position[:, opponent] - state.position[:, perspective]
        )
        relative_velocity_body = quaternion_rotate_inverse_xyzw(
            quaternion, state.linear_velocity[:, opponent] - state.linear_velocity[:, perspective]
        )
        contact_body = quaternion_rotate_inverse_xyzw(quaternion, state.contact_force[:, perspective])
        frame = torch.cat(
            (
                state.action_proposed[:, perspective],
                state.action_exec[:, perspective],
                state.wheel_velocity[:, perspective],
                self_velocity_body[:, :2],
                state.angular_velocity[:, perspective, 2:3],
                relative_position_body[:, :2],
                relative_velocity_body[:, :2],
                contact_body,
                state.edge_margin[:, perspective : perspective + 1],
            ),
            dim=-1,
        )
        if frame.shape[-1] != TEACHER_HISTORY_FEATURE_DIM:
            raise RuntimeError("teacher history feature contract changed unexpectedly")
        frames.append(frame)
    return torch.stack(frames, dim=1)


class TeacherObservationHistory:
    """Finite proprioceptive/action history used by both CPO teacher populations."""

    def __init__(self, state: ArenaState, length: int) -> None:
        if length <= 0:
            raise ValueError("teacher history length must be positive")
        self.length = int(length)
        frame = _teacher_history_frame(state)
        self._buffer = frame.unsqueeze(2).expand(-1, -1, self.length, -1).clone()

    @property
    def batch_size(self) -> int:
        return self._buffer.shape[0]

    @property
    def width(self) -> int:
        return self.length * TEACHER_HISTORY_FEATURE_DIM

    def update(self, state: ArenaState, reset_mask: torch.Tensor | None = None) -> None:
        if state.batch_size != self.batch_size:
            raise ValueError("teacher history and state batches must match")
        frame = _teacher_history_frame(state)
        self._buffer = torch.roll(self._buffer, shifts=-1, dims=2)
        self._buffer[:, :, -1].copy_(frame)
        if reset_mask is not None:
            reset_mask = reset_mask.to(device=frame.device, dtype=torch.bool)
            if tuple(reset_mask.shape) != (self.batch_size,):
                raise ValueError("teacher history reset mask must have shape (B,)")
            repeated = frame[reset_mask].unsqueeze(2).expand(-1, -1, self.length, -1)
            self._buffer[reset_mask] = repeated

    def perspective(self, perspective: int) -> torch.Tensor:
        if perspective not in (0, 1):
            raise ValueError("perspective must be 0 (red) or 1 (blue)")
        return self._buffer[:, perspective].reshape(self.batch_size, self.width)


def build_teacher_observation(
    state: ArenaState,
    domain: DomainBatch,
    perspective: int,
    history: TeacherObservationHistory | None = None,
) -> ObservationVector:
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
        ("self_action_proposed", state.action_proposed[:, perspective]),
        ("self_action_exec", state.action_exec[:, perspective]),
        ("self_wheel_velocity", state.wheel_velocity[:, perspective]),
        ("opponent_wheel_velocity", state.wheel_velocity[:, opponent]),
        ("self_contact_force", state.contact_force[:, perspective]),
        ("opponent_contact_force", state.contact_force[:, opponent]),
        ("edge_margins", state.edge_margin[:, [perspective, opponent]]),
        ("support_margins", state.support_margin[:, [perspective, opponent]]),
        ("stationary_time_s", state.stationary_time_s[:, [perspective, opponent]]),
        ("time_remaining_s", state.time_remaining_s.unsqueeze(-1)),
    ]
    for index, name in enumerate(domain_names):
        named.append((f"domain.{name}", domain_values[:, index : index + 1]))
    if history is not None:
        if history.batch_size != state.batch_size:
            raise ValueError("teacher history and state batches must match")
        named.append(("proprioceptive_history", history.perspective(perspective)))
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


STUDENT_PROPRIO_HISTORY_FEATURE_DIM = 13


def _student_history_frame(sensors: StudentSensors) -> torch.Tensor:
    frame = torch.cat(
        (
            sensors.wheel_velocity,
            sensors.imu_gyro,
            sensors.imu_acceleration,
            sensors.gravity_direction,
            sensors.previous_action_exec,
        ),
        dim=-1,
    )
    if frame.shape[-1] != STUDENT_PROPRIO_HISTORY_FEATURE_DIM:
        raise RuntimeError("student proprioceptive history contract changed unexpectedly")
    return frame


class StudentObservationHistory:
    """Explicit deployable proprioceptive history, complementary to CAP's RSSM recurrence."""

    def __init__(self, sensors: StudentSensors, length: int) -> None:
        if length <= 0:
            raise ValueError("student history length must be positive")
        self.length = int(length)
        frame = _student_history_frame(sensors)
        self._buffer = frame.unsqueeze(1).expand(-1, self.length, -1).clone()

    @property
    def batch_size(self) -> int:
        return self._buffer.shape[0]

    @property
    def width(self) -> int:
        return self.length * STUDENT_PROPRIO_HISTORY_FEATURE_DIM

    def update(self, sensors: StudentSensors, reset_mask: torch.Tensor | None = None) -> None:
        frame = _student_history_frame(sensors)
        if frame.shape[0] != self.batch_size:
            raise ValueError("student history and sensor batches must match")
        self._buffer = torch.roll(self._buffer, shifts=-1, dims=1)
        self._buffer[:, -1].copy_(frame)
        if reset_mask is not None:
            reset_mask = reset_mask.to(device=frame.device, dtype=torch.bool)
            if tuple(reset_mask.shape) != (self.batch_size,):
                raise ValueError("student history reset mask must have shape (B,)")
            self._buffer[reset_mask] = frame[reset_mask].unsqueeze(1).expand(-1, self.length, -1)

    def values(self) -> torch.Tensor:
        return self._buffer.reshape(self.batch_size, self.width)


def build_student_observation(
    sensors: StudentSensors, history: StudentObservationHistory | None = None
) -> ObservationVector:
    """Build the deployment-only vector; privileged simulator fields are impossible inputs here."""
    named = [
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
    if history is not None:
        if history.batch_size != sensors.wheel_velocity.shape[0]:
            raise ValueError("student history and sensor batches must match")
        named.append(("proprioceptive_history", history.values()))
    return _pack(named)
