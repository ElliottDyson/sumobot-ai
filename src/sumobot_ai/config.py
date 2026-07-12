from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import ACTION_ORDER


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _float_tuple(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    result = tuple(float(item) for item in value)
    if not all(item == item and abs(item) != float("inf") for item in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


@dataclass(frozen=True, slots=True)
class BoardConfig:
    size_m: tuple[float, float]
    thickness_m: float
    top_z_m: float
    out_rule: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BoardConfig:
        result = cls(
            size_m=_float_tuple(data["size_m"], 2, "board.size_m"),
            thickness_m=float(data["thickness_m"]),
            top_z_m=float(data.get("top_z_m", 0.0)),
            out_rule=str(data.get("out_rule", "support_lost")),
        )
        if min(result.size_m) <= 0 or result.thickness_m <= 0:
            raise ValueError("board dimensions must be positive")
        if result.out_rule != "support_lost":
            raise ValueError(f"unsupported board.out_rule: {result.out_rule!r}")
        return result


@dataclass(frozen=True, slots=True)
class RobotConfig:
    envelope_size_m: tuple[float, float, float]
    chassis_size_m: tuple[float, float, float]
    chassis_mass_kg: float
    chassis_ground_clearance_m: float
    chassis_com_offset_m: tuple[float, float, float]
    chassis_inertia_diagonal_kg_m2: tuple[float, float, float]
    wheel_radius_m: float
    wheel_half_width_m: float
    wheel_mass_kg: float
    wheel_axle_x_m: float
    track_m: float
    skid_x_m: float
    skid_size_m: tuple[float, float, float]
    max_wheel_speed_rad_s: float
    max_wheel_torque_nm: float
    velocity_servo_gain: float
    motor_command_slew_per_s: float
    motor_command_quantization_levels: int
    motor_min_torque_fraction: float
    action_order: tuple[str, str]

    @property
    def wheel_width_m(self) -> float:
        return 2.0 * self.wheel_half_width_m

    @property
    def chassis_center_height_m(self) -> float:
        return self.chassis_ground_clearance_m + self.chassis_size_m[2] / 2.0

    @property
    def total_mass_kg(self) -> float:
        return self.chassis_mass_kg + 2.0 * self.wheel_mass_kg

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RobotConfig:
        result = cls(
            envelope_size_m=_float_tuple(data["envelope_size_m"], 3, "robot.envelope_size_m"),
            chassis_size_m=_float_tuple(data["chassis_size_m"], 3, "robot.chassis_size_m"),
            chassis_mass_kg=float(data["chassis_mass_kg"]),
            chassis_ground_clearance_m=float(data["chassis_ground_clearance_m"]),
            chassis_com_offset_m=_float_tuple(data["chassis_com_offset_m"], 3, "robot.chassis_com_offset_m"),
            chassis_inertia_diagonal_kg_m2=_float_tuple(
                data["chassis_inertia_diagonal_kg_m2"], 3, "robot.chassis_inertia_diagonal_kg_m2"
            ),
            wheel_radius_m=float(data["wheel_radius_m"]),
            wheel_half_width_m=float(data["wheel_half_width_m"]),
            wheel_mass_kg=float(data["wheel_mass_kg"]),
            wheel_axle_x_m=float(data["wheel_axle_x_m"]),
            track_m=float(data["track_m"]),
            skid_x_m=float(data["skid_x_m"]),
            skid_size_m=_float_tuple(data["skid_size_m"], 3, "robot.skid_size_m"),
            max_wheel_speed_rad_s=float(data["max_wheel_speed_rad_s"]),
            max_wheel_torque_nm=float(data["max_wheel_torque_nm"]),
            velocity_servo_gain=float(data["velocity_servo_gain"]),
            motor_command_slew_per_s=float(data["motor_command_slew_per_s"]),
            motor_command_quantization_levels=int(data["motor_command_quantization_levels"]),
            motor_min_torque_fraction=float(data["motor_min_torque_fraction"]),
            action_order=tuple(str(item) for item in data["action_order"]),
        )
        numeric = (
            *result.envelope_size_m,
            *result.chassis_size_m,
            result.chassis_mass_kg,
            *result.chassis_inertia_diagonal_kg_m2,
            result.wheel_radius_m,
            result.wheel_half_width_m,
            result.wheel_mass_kg,
            result.track_m,
            *result.skid_size_m,
            result.max_wheel_speed_rad_s,
            result.max_wheel_torque_nm,
            result.velocity_servo_gain,
            result.motor_command_slew_per_s,
        )
        if min(numeric) <= 0:
            raise ValueError("robot dimensions, masses, limits, and speeds must be positive")
        finite = (
            *numeric,
            result.chassis_ground_clearance_m,
            *result.chassis_com_offset_m,
            result.wheel_axle_x_m,
            result.skid_x_m,
            result.motor_min_torque_fraction,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("robot geometry, masses, limits, and speeds must be finite")
        if result.chassis_ground_clearance_m < 0:
            raise ValueError("robot.chassis_ground_clearance_m cannot be negative")
        if result.motor_command_quantization_levels < 2:
            raise ValueError("robot.motor_command_quantization_levels must be at least two")
        if not 0.0 <= result.motor_min_torque_fraction <= 1.0:
            raise ValueError("robot.motor_min_torque_fraction must be in [0, 1]")
        if result.action_order != ACTION_ORDER:
            raise ValueError(f"robot.action_order must be {list(ACTION_ORDER)}")
        if any(
            chassis > envelope + 1e-9
            for chassis, envelope in zip(result.chassis_size_m, result.envelope_size_m, strict=True)
        ):
            raise ValueError("robot chassis must fit inside robot.envelope_size_m")
        envelope_x, envelope_y, envelope_z = result.envelope_size_m
        if result.chassis_ground_clearance_m + result.chassis_size_m[2] > envelope_z + 1e-9:
            raise ValueError("robot chassis and ground clearance exceed the legal height envelope")
        if result.track_m + result.wheel_width_m > envelope_y + 1e-9:
            raise ValueError("wheel track and full wheel widths exceed the legal width envelope")
        if abs(result.wheel_axle_x_m) + result.wheel_radius_m > envelope_x / 2.0 + 1e-9:
            raise ValueError("wheel footprint exceeds the legal length envelope")
        if 2.0 * result.wheel_radius_m > envelope_z + 1e-9:
            raise ValueError("wheel diameter exceeds the legal height envelope")
        if abs(result.skid_x_m) + result.skid_size_m[0] / 2.0 > envelope_x / 2.0 + 1e-9:
            raise ValueError("robot skid footprint exceeds the legal length envelope")
        if result.skid_size_m[1] > envelope_y + 1e-9 or result.skid_size_m[2] > envelope_z + 1e-9:
            raise ValueError("robot skid exceeds the legal envelope")
        half_chassis = tuple(value / 2.0 for value in result.chassis_size_m)
        if any(abs(offset) >= half for offset, half in zip(result.chassis_com_offset_m, half_chassis, strict=True)):
            raise ValueError("robot.chassis_com_offset_m must remain inside the chassis")
        return result


@dataclass(frozen=True, slots=True)
class SensorConfig:
    proprio_history_steps: int
    encoder_counts_per_revolution: int
    encoder_update_hz: int
    imu_update_hz: int
    edge_update_hz: int
    opponent_update_hz: int
    edge_max_range_m: float
    edge_sensor_positions_m: tuple[tuple[float, float, float], ...]
    opponent_max_range_m: float
    opponent_horizontal_fov_rad: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, control_hz: int) -> SensorConfig:
        raw_positions = data["edge_sensor_positions_m"]
        if (
            not isinstance(raw_positions, Sequence)
            or isinstance(raw_positions, (str, bytes))
            or len(raw_positions) != 4
        ):
            raise ValueError("sensors.edge_sensor_positions_m must contain four xyz positions")
        result = cls(
            proprio_history_steps=int(data["proprio_history_steps"]),
            encoder_counts_per_revolution=int(data["encoder_counts_per_revolution"]),
            encoder_update_hz=int(data["encoder_update_hz"]),
            imu_update_hz=int(data["imu_update_hz"]),
            edge_update_hz=int(data["edge_update_hz"]),
            opponent_update_hz=int(data["opponent_update_hz"]),
            edge_max_range_m=float(data["edge_max_range_m"]),
            edge_sensor_positions_m=tuple(
                _float_tuple(position, 3, f"sensors.edge_sensor_positions_m[{index}]")
                for index, position in enumerate(raw_positions)
            ),
            opponent_max_range_m=float(data["opponent_max_range_m"]),
            opponent_horizontal_fov_rad=float(data["opponent_horizontal_fov_rad"]),
        )
        rates = (
            result.encoder_update_hz,
            result.imu_update_hz,
            result.edge_update_hz,
            result.opponent_update_hz,
        )
        if min(result.proprio_history_steps, result.encoder_counts_per_revolution, *rates) <= 0:
            raise ValueError("sensor resolutions and update rates must be positive")
        if any(rate > control_hz or control_hz % rate for rate in rates):
            raise ValueError("every sensor update rate must divide physics.control_hz")
        if result.edge_max_range_m <= 0 or result.opponent_max_range_m <= 0:
            raise ValueError("sensor maximum ranges must be positive")
        if not 0 < result.opponent_horizontal_fov_rad <= 2.0 * math.pi:
            raise ValueError("sensors.opponent_horizontal_fov_rad must be in (0, 2*pi]")
        return result


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    backend: str
    control_hz: int
    substeps: int
    solver_iterations: int
    solver_ls_iterations: int
    use_mujoco_contacts: bool
    gravity_m_s2: float
    episode_seconds: float

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def simulation_dt(self) -> float:
        return self.control_dt / self.substeps

    @property
    def max_episode_steps(self) -> int:
        return round(self.episode_seconds * self.control_hz)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PhysicsConfig:
        result = cls(
            backend=str(data["backend"]),
            control_hz=int(data["control_hz"]),
            substeps=int(data["substeps"]),
            solver_iterations=int(data["solver_iterations"]),
            solver_ls_iterations=int(data["solver_ls_iterations"]),
            use_mujoco_contacts=bool(data["use_mujoco_contacts"]),
            gravity_m_s2=float(data.get("gravity_m_s2", -9.81)),
            episode_seconds=float(data["episode_seconds"]),
        )
        if result.backend != "mujoco_warp":
            raise ValueError("the initial challenge requires physics.backend=mujoco_warp")
        if min(result.control_hz, result.substeps, result.solver_iterations, result.solver_ls_iterations) <= 0:
            raise ValueError("physics rates and solver iteration counts must be positive")
        if result.episode_seconds <= 0:
            raise ValueError("physics.episode_seconds must be positive")
        if not result.use_mujoco_contacts:
            raise ValueError("the initial challenge requires physics.use_mujoco_contacts=true")
        return result


@dataclass(frozen=True, slots=True)
class MatchConfig:
    inactivity_timeout_s: float
    movement_speed_threshold_m_s: float
    movement_confirmation_s: float
    ring_out_confirmation_s: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, episode_seconds: float) -> MatchConfig:
        result = cls(
            inactivity_timeout_s=float(data["inactivity_timeout_s"]),
            movement_speed_threshold_m_s=float(data["movement_speed_threshold_m_s"]),
            movement_confirmation_s=float(data["movement_confirmation_s"]),
            ring_out_confirmation_s=float(data["ring_out_confirmation_s"]),
        )
        if not 0 < result.inactivity_timeout_s < episode_seconds:
            raise ValueError("match.inactivity_timeout_s must be between zero and the episode duration")
        if result.movement_speed_threshold_m_s <= 0:
            raise ValueError("match.movement_speed_threshold_m_s must be positive")
        if not 0 < result.movement_confirmation_s < result.inactivity_timeout_s:
            raise ValueError("match.movement_confirmation_s must be positive and shorter than inactivity timeout")
        if not 0 < result.ring_out_confirmation_s < result.inactivity_timeout_s:
            raise ValueError("match.ring_out_confirmation_s must be positive and shorter than inactivity timeout")
        return result


@dataclass(frozen=True, slots=True)
class InitializationConfig:
    red_position_m: tuple[float, float, float]
    red_yaw_rad: float
    blue_position_m: tuple[float, float, float]
    blue_yaw_rad: float
    position_jitter_m: tuple[float, float]
    yaw_jitter_rad: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> InitializationConfig:
        result = cls(
            red_position_m=_float_tuple(data["red_position_m"], 3, "initialization.red_position_m"),
            red_yaw_rad=float(data["red_yaw_rad"]),
            blue_position_m=_float_tuple(data["blue_position_m"], 3, "initialization.blue_position_m"),
            blue_yaw_rad=float(data["blue_yaw_rad"]),
            position_jitter_m=_float_tuple(data["position_jitter_m"], 2, "initialization.position_jitter_m"),
            yaw_jitter_rad=float(data["yaw_jitter_rad"]),
        )
        if min(result.position_jitter_m) < 0 or result.yaw_jitter_rad < 0:
            raise ValueError("initialization jitter cannot be negative")
        return result


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    version: int
    name: str
    board: BoardConfig
    robot: RobotConfig
    physics: PhysicsConfig
    sensors: SensorConfig
    match: MatchConfig
    initialization: InitializationConfig
    domain_randomization: Mapping[str, tuple[float, float]]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ArenaConfig:
        dr_data = _mapping(data["domain_randomization"], "domain_randomization")
        dr = {name: _float_tuple(bounds, 2, f"domain_randomization.{name}") for name, bounds in dr_data.items()}
        for name, (low, high) in dr.items():
            if low > high:
                raise ValueError(f"domain_randomization.{name} has low > high")
        physics = PhysicsConfig.from_mapping(_mapping(data["physics"], "physics"))
        result = cls(
            version=int(data["version"]),
            name=str(data["name"]),
            board=BoardConfig.from_mapping(_mapping(data["board"], "board")),
            robot=RobotConfig.from_mapping(_mapping(data["robot"], "robot")),
            physics=physics,
            sensors=SensorConfig.from_mapping(_mapping(data["sensors"], "sensors"), control_hz=physics.control_hz),
            match=MatchConfig.from_mapping(_mapping(data["match"], "match"), episode_seconds=physics.episode_seconds),
            initialization=InitializationConfig.from_mapping(_mapping(data["initialization"], "initialization")),
            domain_randomization=dr,
        )
        if result.version != 3:
            raise ValueError(f"unsupported arena config version: {result.version}")
        half_x, half_y = (value / 2 for value in result.board.size_m)
        initial_positions = (
            ("red", result.initialization.red_position_m),
            ("blue", result.initialization.blue_position_m),
        )
        for side, pos in initial_positions:
            if abs(pos[0]) >= half_x or abs(pos[1]) >= half_y:
                raise ValueError(f"initialization.{side}_position_m must begin on the board")
            expected_z = result.board.top_z_m + result.robot.chassis_center_height_m
            if abs(pos[2] - expected_z) > 1e-9:
                raise ValueError(
                    f"initialization.{side}_position_m z must be {expected_z} so the wheels and skid touch the board"
                )
        return result

    @classmethod
    def load(cls, path: str | Path) -> ArenaConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        return cls.from_mapping(_mapping(data, str(path)))
