from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
            out_rule=str(data.get("out_rule", "center_crosses_edge")),
        )
        if min(result.size_m) <= 0 or result.thickness_m <= 0:
            raise ValueError("board dimensions must be positive")
        if result.out_rule != "center_crosses_edge":
            raise ValueError(f"unsupported board.out_rule: {result.out_rule!r}")
        return result


@dataclass(frozen=True, slots=True)
class RobotConfig:
    chassis_size_m: tuple[float, float, float]
    chassis_mass_kg: float
    wheel_radius_m: float
    wheel_width_m: float
    wheel_mass_kg: float
    wheelbase_m: float
    track_m: float
    max_wheel_speed_rad_s: float
    max_wheel_torque_nm: float
    action_order: tuple[str, str, str, str]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RobotConfig:
        result = cls(
            chassis_size_m=_float_tuple(data["chassis_size_m"], 3, "robot.chassis_size_m"),
            chassis_mass_kg=float(data["chassis_mass_kg"]),
            wheel_radius_m=float(data["wheel_radius_m"]),
            wheel_width_m=float(data["wheel_width_m"]),
            wheel_mass_kg=float(data["wheel_mass_kg"]),
            wheelbase_m=float(data["wheelbase_m"]),
            track_m=float(data["track_m"]),
            max_wheel_speed_rad_s=float(data["max_wheel_speed_rad_s"]),
            max_wheel_torque_nm=float(data["max_wheel_torque_nm"]),
            action_order=tuple(str(item) for item in data["action_order"]),
        )
        numeric = (
            *result.chassis_size_m,
            result.chassis_mass_kg,
            result.wheel_radius_m,
            result.wheel_width_m,
            result.wheel_mass_kg,
            result.wheelbase_m,
            result.track_m,
            result.max_wheel_speed_rad_s,
            result.max_wheel_torque_nm,
        )
        if min(numeric) <= 0:
            raise ValueError("robot dimensions, masses, limits, and speeds must be positive")
        expected = ("front_left", "front_right", "rear_left", "rear_right")
        if result.action_order != expected:
            raise ValueError(f"robot.action_order must be {list(expected)}")
        if result.wheelbase_m > result.chassis_size_m[0]:
            raise ValueError("robot.wheelbase_m cannot exceed chassis x size")
        if result.track_m < result.chassis_size_m[1]:
            raise ValueError("robot.track_m must span at least the chassis y size")
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
    initialization: InitializationConfig
    domain_randomization: Mapping[str, tuple[float, float]]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ArenaConfig:
        dr_data = _mapping(data["domain_randomization"], "domain_randomization")
        dr = {name: _float_tuple(bounds, 2, f"domain_randomization.{name}") for name, bounds in dr_data.items()}
        for name, (low, high) in dr.items():
            if low > high:
                raise ValueError(f"domain_randomization.{name} has low > high")
        result = cls(
            version=int(data["version"]),
            name=str(data["name"]),
            board=BoardConfig.from_mapping(_mapping(data["board"], "board")),
            robot=RobotConfig.from_mapping(_mapping(data["robot"], "robot")),
            physics=PhysicsConfig.from_mapping(_mapping(data["physics"], "physics")),
            initialization=InitializationConfig.from_mapping(_mapping(data["initialization"], "initialization")),
            domain_randomization=dr,
        )
        if result.version != 1:
            raise ValueError(f"unsupported arena config version: {result.version}")
        half_x, half_y = (value / 2 for value in result.board.size_m)
        initial_positions = (
            ("red", result.initialization.red_position_m),
            ("blue", result.initialization.blue_position_m),
        )
        for side, pos in initial_positions:
            if abs(pos[0]) >= half_x or abs(pos[1]) >= half_y:
                raise ValueError(f"initialization.{side}_position_m must begin on the board")
        return result

    @classmethod
    def load(cls, path: str | Path) -> ArenaConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        return cls.from_mapping(_mapping(data, str(path)))
