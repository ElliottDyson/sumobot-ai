from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import ClassVar

import torch


@dataclass(frozen=True, slots=True)
class DomainBatch:
    board_friction: torch.Tensor  # (B, 1)
    chassis_friction: torch.Tensor  # (B, 2)
    wheel_friction: torch.Tensor  # (B, 2)
    skid_friction: torch.Tensor  # (B, 2)
    wheel_torsional_friction: torch.Tensor  # (B, 2)
    wheel_rolling_friction: torch.Tensor  # (B, 2)
    skid_torsional_friction: torch.Tensor  # (B, 2)
    skid_rolling_friction: torch.Tensor  # (B, 2)
    restitution: torch.Tensor  # (B, 1)
    contact_stiffness_scale: torch.Tensor  # (B, 1)
    contact_damping_scale: torch.Tensor  # (B, 1)
    chassis_mass_scale: torch.Tensor  # (B, 2)
    wheel_mass_scale: torch.Tensor  # (B, 2)
    chassis_inertia_scale: torch.Tensor  # (B, 2)
    chassis_com_offset_x_m: torch.Tensor  # (B, 2)
    chassis_com_offset_y_m: torch.Tensor  # (B, 2)
    chassis_com_offset_z_m: torch.Tensor  # (B, 2)
    motor_strength_scale: torch.Tensor  # (B, 2)
    motor_time_constant_s: torch.Tensor  # (B, 2)
    motor_deadband: torch.Tensor  # (B, 2)
    motor_asymmetry: torch.Tensor  # (B, 2), signed left/right gain mismatch
    battery_voltage_scale: torch.Tensor  # (B, 2)
    action_latency_s: torch.Tensor  # (B, 2)
    encoder_noise_std_rad_s: torch.Tensor  # (B, 2)
    encoder_bias_std_rad_s: torch.Tensor  # (B, 2)
    encoder_latency_s: torch.Tensor  # (B, 2)
    imu_gyro_noise_std_rad_s: torch.Tensor  # (B, 2)
    imu_gyro_bias_std_rad_s: torch.Tensor  # (B, 2)
    imu_accel_noise_std_m_s2: torch.Tensor  # (B, 2)
    imu_accel_bias_std_m_s2: torch.Tensor  # (B, 2)
    imu_latency_s: torch.Tensor  # (B, 2)
    edge_range_noise_std_m: torch.Tensor  # (B, 2)
    edge_sensor_latency_s: torch.Tensor  # (B, 2)
    edge_sensor_dropout: torch.Tensor  # (B, 2)
    opponent_range_noise_std_m: torch.Tensor  # (B, 2)
    opponent_bearing_noise_std_rad: torch.Tensor  # (B, 2)
    opponent_sensor_latency_s: torch.Tensor  # (B, 2)
    opponent_sensor_dropout: torch.Tensor  # (B, 2)

    @property
    def batch_size(self) -> int:
        return self.board_friction.shape[0]

    def teacher_tensor(self) -> tuple[torch.Tensor, tuple[str, ...]]:
        items = tuple((field.name, getattr(self, field.name)) for field in fields(self))
        names: list[str] = []
        tensors: list[torch.Tensor] = []
        for name, value in items:
            if value.ndim != 2 or value.shape[0] != self.batch_size:
                raise ValueError(f"domain field {name} must have shape (B, W)")
            tensors.append(value)
            if value.shape[1] == 1:
                names.append(name)
            else:
                names.extend(f"{name}_{side}" for side in range(value.shape[1]))
        return torch.cat(tensors, dim=-1), tuple(names)


class DomainRandomizer:
    SHARED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"board_friction", "restitution", "contact_stiffness_scale", "contact_damping_scale"}
    )
    ROBOT_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "chassis_friction",
            "wheel_friction",
            "skid_friction",
            "wheel_torsional_friction",
            "wheel_rolling_friction",
            "skid_torsional_friction",
            "skid_rolling_friction",
            "chassis_mass_scale",
            "wheel_mass_scale",
            "chassis_inertia_scale",
            "chassis_com_offset_x_m",
            "chassis_com_offset_y_m",
            "chassis_com_offset_z_m",
            "motor_strength_scale",
            "motor_time_constant_s",
            "motor_deadband",
            "motor_asymmetry",
            "battery_voltage_scale",
            "action_latency_s",
            "encoder_noise_std_rad_s",
            "encoder_bias_std_rad_s",
            "encoder_latency_s",
            "imu_gyro_noise_std_rad_s",
            "imu_gyro_bias_std_rad_s",
            "imu_accel_noise_std_m_s2",
            "imu_accel_bias_std_m_s2",
            "imu_latency_s",
            "edge_range_noise_std_m",
            "edge_sensor_latency_s",
            "edge_sensor_dropout",
            "opponent_range_noise_std_m",
            "opponent_bearing_noise_std_rad",
            "opponent_sensor_latency_s",
            "opponent_sensor_dropout",
        }
    )
    REQUIRED_FIELDS: ClassVar[frozenset[str]] = SHARED_FIELDS | ROBOT_FIELDS
    SIGNED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"chassis_com_offset_x_m", "chassis_com_offset_y_m", "chassis_com_offset_z_m", "motor_asymmetry"}
    )
    PROBABILITY_FIELDS: ClassVar[frozenset[str]] = frozenset({"edge_sensor_dropout", "opponent_sensor_dropout"})
    STRICTLY_POSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "contact_stiffness_scale",
            "contact_damping_scale",
            "chassis_mass_scale",
            "wheel_mass_scale",
            "chassis_inertia_scale",
            "motor_strength_scale",
            "motor_time_constant_s",
            "battery_voltage_scale",
        }
    )

    def __init__(self, bounds: Mapping[str, tuple[float, float]]) -> None:
        missing = self.REQUIRED_FIELDS - bounds.keys()
        unknown = bounds.keys() - self.REQUIRED_FIELDS
        if missing:
            raise ValueError(f"missing domain-randomization fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"unknown domain-randomization fields: {sorted(unknown)}")
        self.bounds = {name: (float(low), float(high)) for name, (low, high) in bounds.items()}
        for name, (low, high) in self.bounds.items():
            if low > high:
                raise ValueError(f"{name} has low > high")
            if name not in self.SIGNED_FIELDS and low < 0.0:
                raise ValueError(f"{name} cannot be negative")
            if name in self.STRICTLY_POSITIVE_FIELDS and low <= 0.0:
                raise ValueError(f"{name} must be strictly positive")
            if name in self.PROBABILITY_FIELDS and high > 1.0:
                raise ValueError(f"{name} must remain in [0, 1]")
        if max(abs(value) for value in self.bounds["motor_asymmetry"]) >= 0.9:
            raise ValueError("motor_asymmetry must remain strictly within (-0.9, 0.9)")
        if self.bounds["motor_deadband"][1] >= 1.0:
            raise ValueError("motor_deadband must remain below one")

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> DomainBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        device = torch.device(device)

        def uniform(name: str, width: int) -> torch.Tensor:
            low, high = self.bounds[name]
            random = torch.rand((batch_size, width), device=device, generator=generator)
            return random.mul(high - low).add(low)

        values = {name: uniform(name, 1 if name in self.SHARED_FIELDS else 2) for name in self.REQUIRED_FIELDS}
        return DomainBatch(**values)
