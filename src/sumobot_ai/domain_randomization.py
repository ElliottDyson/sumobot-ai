from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass(frozen=True, slots=True)
class DomainBatch:
    board_friction: torch.Tensor  # (B, 1)
    chassis_friction: torch.Tensor  # (B, 2)
    wheel_friction: torch.Tensor  # (B, 2)
    restitution: torch.Tensor  # (B, 1)
    chassis_mass_scale: torch.Tensor  # (B, 2)
    wheel_mass_scale: torch.Tensor  # (B, 2)
    motor_strength_scale: torch.Tensor  # (B, 2)
    action_latency_s: torch.Tensor  # (B, 2)
    encoder_noise_std_rad_s: torch.Tensor  # (B, 2)
    imu_gyro_noise_std_rad_s: torch.Tensor  # (B, 2)
    imu_accel_noise_std_m_s2: torch.Tensor  # (B, 2)
    opponent_sensor_dropout: torch.Tensor  # (B, 2)

    @property
    def batch_size(self) -> int:
        return self.board_friction.shape[0]

    def teacher_tensor(self) -> tuple[torch.Tensor, tuple[str, ...]]:
        items = (
            ("board_friction", self.board_friction),
            ("chassis_friction", self.chassis_friction),
            ("wheel_friction", self.wheel_friction),
            ("restitution", self.restitution),
            ("chassis_mass_scale", self.chassis_mass_scale),
            ("wheel_mass_scale", self.wheel_mass_scale),
            ("motor_strength_scale", self.motor_strength_scale),
            ("action_latency_s", self.action_latency_s),
            ("encoder_noise_std_rad_s", self.encoder_noise_std_rad_s),
            ("imu_gyro_noise_std_rad_s", self.imu_gyro_noise_std_rad_s),
            ("imu_accel_noise_std_m_s2", self.imu_accel_noise_std_m_s2),
            ("opponent_sensor_dropout", self.opponent_sensor_dropout),
        )
        names: list[str] = []
        tensors: list[torch.Tensor] = []
        for name, value in items:
            tensors.append(value)
            if value.shape[1] == 1:
                names.append(name)
            else:
                names.extend(f"{name}_{side}" for side in range(value.shape[1]))
        return torch.cat(tensors, dim=-1), tuple(names)


class DomainRandomizer:
    SHARED_FIELDS: ClassVar[frozenset[str]] = frozenset({"board_friction", "restitution"})
    ROBOT_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "chassis_friction",
            "wheel_friction",
            "chassis_mass_scale",
            "wheel_mass_scale",
            "motor_strength_scale",
            "action_latency_s",
            "encoder_noise_std_rad_s",
            "imu_gyro_noise_std_rad_s",
            "imu_accel_noise_std_m_s2",
            "opponent_sensor_dropout",
        }
    )
    REQUIRED_FIELDS: ClassVar[frozenset[str]] = SHARED_FIELDS | ROBOT_FIELDS

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
