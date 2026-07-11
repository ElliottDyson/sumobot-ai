from __future__ import annotations

import math
from typing import ClassVar

import torch

from ..config import ArenaConfig
from ..domain_randomization import DomainBatch
from ..geometry import quaternion_rotate_inverse_xyzw, quaternion_rotate_xyzw
from ..observations import StudentSensors
from ..state import ArenaState


class StudentSensorSuite:
    """Stateful, delayed deployable sensor simulation for both robots in every arena."""

    _WIDTHS: ClassVar[dict[str, int]] = {
        "wheel": 2,
        "gyro": 3,
        "accel": 3,
        "gravity": 3,
        "edge": 4,
        "opponent": 4,
    }

    def __init__(
        self,
        config: ArenaConfig,
        world_count: int,
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> None:
        self.config = config
        self.world_count = world_count
        self.device = device
        self.generator = generator
        self.dt = config.physics.control_dt
        maximum_latency = max(
            config.domain_randomization[name][1]
            for name in (
                "encoder_latency_s",
                "imu_latency_s",
                "edge_sensor_latency_s",
                "opponent_sensor_latency_s",
            )
        )
        slowest_period = 1.0 / min(
            config.sensors.encoder_update_hz,
            config.sensors.imu_update_hz,
            config.sensors.edge_update_hz,
            config.sensors.opponent_update_hz,
        )
        self.history_length = math.ceil((maximum_latency + slowest_period) / self.dt) + 2
        self._history = {
            name: torch.zeros((self.history_length, world_count, 2, width), dtype=torch.float32, device=device)
            for name, width in self._WIDTHS.items()
        }
        self._cache = {
            name: torch.zeros((world_count, 2, width), dtype=torch.float32, device=device)
            for name, width in self._WIDTHS.items()
        }
        self._sample_id = {
            name: torch.full((world_count, 2), -1, dtype=torch.int64, device=device) for name in self._WIDTHS
        }
        self._previous_linear_velocity = torch.zeros((world_count, 2, 3), device=device)
        self._encoder_bias = torch.zeros((world_count, 2, 2), device=device)
        self._gyro_bias = torch.zeros((world_count, 2, 3), device=device)
        self._accel_bias = torch.zeros((world_count, 2, 3), device=device)
        self._initialized = False

    def _normal(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, device=self.device, generator=self.generator)

    def _edge_truth(self, state: ArenaState) -> torch.Tensor:
        local = torch.tensor(
            self.config.sensors.edge_sensor_positions_m, dtype=state.position.dtype, device=self.device
        ).view(1, 1, 4, 3)
        local = local.expand(self.world_count, 2, -1, -1)
        quaternion = state.quaternion.unsqueeze(2)
        origins = state.position.unsqueeze(2) + quaternion_rotate_xyzw(quaternion, local)
        down = torch.zeros_like(local)
        down[..., 2] = -1.0
        directions = quaternion_rotate_xyzw(quaternion, down)
        denominator = directions[..., 2]
        distance = (self.config.board.top_z_m - origins[..., 2]) / denominator.clamp(max=-1e-6)
        hit_xy = origins[..., :2] + distance.unsqueeze(-1) * directions[..., :2]
        half_x, half_y = (value / 2.0 for value in self.config.board.size_m)
        valid = (
            (denominator < -1e-6)
            & (distance >= 0.0)
            & (distance <= self.config.sensors.edge_max_range_m)
            & (hit_xy[..., 0].abs() <= half_x)
            & (hit_xy[..., 1].abs() <= half_y)
        )
        maximum = torch.full_like(distance, self.config.sensors.edge_max_range_m)
        return torch.where(valid, distance, maximum)

    def _opponent_truth(self, state: ArenaState) -> torch.Tensor:
        opponent_position = state.position[:, [1, 0]]
        relative_body = quaternion_rotate_inverse_xyzw(state.quaternion, opponent_position - state.position)
        distance = torch.linalg.vector_norm(relative_body[..., :2], dim=-1)
        bearing = torch.atan2(relative_body[..., 1], relative_body[..., 0])
        valid = (distance <= self.config.sensors.opponent_max_range_m) & (
            bearing.abs() <= self.config.sensors.opponent_horizontal_fov_rad / 2.0
        )
        maximum = torch.full_like(distance, self.config.sensors.opponent_max_range_m)
        return torch.stack(
            (
                torch.where(valid, distance, maximum),
                torch.where(valid, torch.sin(bearing), torch.zeros_like(bearing)),
                torch.where(valid, torch.cos(bearing), torch.zeros_like(bearing)),
                valid.to(distance.dtype),
            ),
            dim=-1,
        )

    def _truth(self, state: ArenaState, *, reset_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        quaternion = state.quaternion
        world_acceleration = (state.linear_velocity - self._previous_linear_velocity) / self.dt
        if reset_mask is not None:
            world_acceleration = torch.where(
                reset_mask[:, None, None], torch.zeros_like(world_acceleration), world_acceleration
            )
        gravity_world = torch.zeros_like(world_acceleration)
        gravity_world[..., 2] = self.config.physics.gravity_m_s2
        specific_force = quaternion_rotate_inverse_xyzw(quaternion, world_acceleration - gravity_world)
        gravity_direction_world = torch.zeros_like(world_acceleration)
        gravity_direction_world[..., 2] = -1.0
        return {
            "wheel": state.wheel_velocity,
            "gyro": quaternion_rotate_inverse_xyzw(quaternion, state.angular_velocity),
            "accel": specific_force,
            "gravity": quaternion_rotate_inverse_xyzw(quaternion, gravity_direction_world),
            "edge": self._edge_truth(state),
            "opponent": self._opponent_truth(state),
        }

    def _resample_biases(self, domain: DomainBatch, mask: torch.Tensor) -> None:
        count = int(mask.sum().item())
        if count == 0:
            return
        self._encoder_bias[mask] = self._normal((count, 2, 2)) * domain.encoder_bias_std_rad_s[mask].unsqueeze(-1)
        self._gyro_bias[mask] = self._normal((count, 2, 3)) * domain.imu_gyro_bias_std_rad_s[mask].unsqueeze(-1)
        self._accel_bias[mask] = self._normal((count, 2, 3)) * domain.imu_accel_bias_std_m_s2[mask].unsqueeze(-1)

    def reset(self, state: ArenaState, domain: DomainBatch, mask: torch.Tensor) -> None:
        mask = mask.to(device=self.device, dtype=torch.bool)
        if tuple(mask.shape) != (self.world_count,):
            raise ValueError("sensor reset mask must have shape (B,)")
        truth = self._truth(state, reset_mask=mask)
        for name, value in truth.items():
            repeated = value[mask].unsqueeze(0).expand(self.history_length, -1, -1, -1)
            self._history[name][:, mask] = repeated
            self._cache[name][mask] = value[mask]
            self._sample_id[name][mask] = -1
        self._previous_linear_velocity[mask] = state.linear_velocity[mask]
        self._resample_biases(domain, mask)
        self._initialized = True

    def advance(self, state: ArenaState) -> None:
        if not self._initialized:
            raise RuntimeError("sensor suite must be reset before it advances")
        truth = self._truth(state)
        for name, value in truth.items():
            self._history[name] = torch.roll(self._history[name], shifts=1, dims=0)
            self._history[name][0].copy_(value)
        self._previous_linear_velocity.copy_(state.linear_velocity)

    def _delayed(
        self,
        name: str,
        latency: torch.Tensor,
        elapsed_steps: torch.Tensor,
        perspective: int,
        update_hz: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        period_steps = self.config.physics.control_hz // update_hz
        delay_steps = torch.ceil(latency[:, perspective] / self.dt).to(torch.int64)
        available_step = (elapsed_steps - delay_steps).clamp_min(0)
        capture_step = torch.div(available_step, period_steps, rounding_mode="floor") * period_steps
        age_steps = (elapsed_steps - capture_step).clamp(0, self.history_length - 1)
        world = torch.arange(self.world_count, device=self.device)
        sample = self._history[name][age_steps, world, perspective]
        return sample, capture_step, age_steps.to(torch.float32) * self.dt

    def _update_cache(
        self,
        name: str,
        perspective: int,
        sample: torch.Tensor,
        capture_step: torch.Tensor,
    ) -> torch.Tensor:
        changed = capture_step != self._sample_id[name][:, perspective]
        self._cache[name][:, perspective] = torch.where(
            changed.unsqueeze(-1), sample, self._cache[name][:, perspective]
        )
        self._sample_id[name][:, perspective] = torch.where(
            changed, capture_step, self._sample_id[name][:, perspective]
        )
        return self._cache[name][:, perspective]

    def observe(
        self,
        state: ArenaState,
        domain: DomainBatch,
        elapsed_steps: torch.Tensor,
        perspective: int,
    ) -> StudentSensors:
        if perspective not in (0, 1):
            raise ValueError("perspective must be 0 (red) or 1 (blue)")
        if not self._initialized:
            raise RuntimeError("sensor suite has not been initialized")
        sensor_cfg = self.config.sensors

        wheel, wheel_id, _ = self._delayed(
            "wheel", domain.encoder_latency_s, elapsed_steps, perspective, sensor_cfg.encoder_update_hz
        )
        quantum = 2.0 * math.pi * sensor_cfg.encoder_update_hz / sensor_cfg.encoder_counts_per_revolution
        wheel = torch.round(wheel / quantum) * quantum
        wheel = wheel + self._encoder_bias[:, perspective]
        wheel = (
            wheel + self._normal(tuple(wheel.shape)) * domain.encoder_noise_std_rad_s[:, perspective : perspective + 1]
        )
        wheel = self._update_cache("wheel", perspective, wheel, wheel_id)

        gyro, imu_id, _ = self._delayed(
            "gyro", domain.imu_latency_s, elapsed_steps, perspective, sensor_cfg.imu_update_hz
        )
        accel, _, _ = self._delayed("accel", domain.imu_latency_s, elapsed_steps, perspective, sensor_cfg.imu_update_hz)
        gravity, _, _ = self._delayed(
            "gravity", domain.imu_latency_s, elapsed_steps, perspective, sensor_cfg.imu_update_hz
        )
        gyro = gyro + self._gyro_bias[:, perspective]
        gyro = (
            gyro + self._normal(tuple(gyro.shape)) * domain.imu_gyro_noise_std_rad_s[:, perspective : perspective + 1]
        )
        accel = accel + self._accel_bias[:, perspective]
        accel = (
            accel + self._normal(tuple(accel.shape)) * domain.imu_accel_noise_std_m_s2[:, perspective : perspective + 1]
        )
        gyro = self._update_cache("gyro", perspective, gyro, imu_id)
        accel = self._update_cache("accel", perspective, accel, imu_id)
        gravity = self._update_cache("gravity", perspective, gravity, imu_id)

        edge, edge_id, edge_age = self._delayed(
            "edge", domain.edge_sensor_latency_s, elapsed_steps, perspective, sensor_cfg.edge_update_hz
        )
        edge = edge + self._normal(tuple(edge.shape)) * domain.edge_range_noise_std_m[:, perspective : perspective + 1]
        edge_dropout = (
            torch.rand((self.world_count,), device=self.device, generator=self.generator)
            < domain.edge_sensor_dropout[:, perspective]
        )
        edge = torch.where(edge_dropout.unsqueeze(-1), torch.full_like(edge, sensor_cfg.edge_max_range_m), edge).clamp(
            0.0, sensor_cfg.edge_max_range_m
        )
        edge = self._update_cache("edge", perspective, edge, edge_id)

        opponent, opponent_id, opponent_age = self._delayed(
            "opponent",
            domain.opponent_sensor_latency_s,
            elapsed_steps,
            perspective,
            sensor_cfg.opponent_update_hz,
        )
        valid = opponent[:, 3] > 0.5
        noisy_range = (
            opponent[:, 0] + self._normal((self.world_count,)) * domain.opponent_range_noise_std_m[:, perspective]
        )
        bearing = torch.atan2(opponent[:, 1], opponent[:, 2])
        bearing = bearing + self._normal((self.world_count,)) * domain.opponent_bearing_noise_std_rad[:, perspective]
        dropout = (
            torch.rand((self.world_count,), device=self.device, generator=self.generator)
            < domain.opponent_sensor_dropout[:, perspective]
        )
        valid &= ~dropout
        opponent = torch.stack(
            (
                torch.where(
                    valid,
                    noisy_range.clamp(0.0, sensor_cfg.opponent_max_range_m),
                    torch.full_like(noisy_range, sensor_cfg.opponent_max_range_m),
                ),
                torch.where(valid, torch.sin(bearing), torch.zeros_like(bearing)),
                torch.where(valid, torch.cos(bearing), torch.zeros_like(bearing)),
                valid.to(opponent.dtype),
            ),
            dim=-1,
        )
        opponent = self._update_cache("opponent", perspective, opponent, opponent_id)

        return StudentSensors(
            wheel_velocity=wheel,
            imu_gyro=gyro,
            imu_acceleration=accel,
            gravity_direction=gravity,
            edge_ranges=edge,
            opponent_range_bearing_valid=opponent,
            previous_action_exec=state.action_exec[:, perspective],
            edge_sensor_age_s=edge_age.unsqueeze(-1),
            opponent_sensor_age_s=opponent_age.unsqueeze(-1),
            stationary_time_fraction=(
                state.stationary_time_s[:, perspective : perspective + 1] / self.config.match.inactivity_timeout_s
            ).clamp(0.0, 1.0),
            time_fraction=(state.time_remaining_s / self.config.physics.episode_seconds).unsqueeze(-1),
        )
