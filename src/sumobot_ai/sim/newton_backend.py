from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import ArenaConfig
from ..contracts import ACTION_DIM, DRIVEN_WHEEL_COUNT
from ..domain_randomization import DomainBatch, DomainRandomizer
from ..match import resolve_match
from ..state import DRAW, ArenaState

try:  # Keep lightweight reward/curriculum tooling usable without the simulation extra.
    import newton
    import warp as wp
except ImportError:  # pragma: no cover - exercised by the doctor command
    newton = None
    wp = None


if wp is not None:

    @wp.kernel
    def _scatter_wheel_targets(
        actions: wp.array(dtype=wp.float32),
        targets: wp.array(dtype=wp.float32),
        max_wheel_speed: float,
    ):
        action_index = wp.tid()
        world = action_index // 4
        within_world = action_index - world * 4
        robot = within_world // 2
        wheel = within_world - robot * 2
        # Each robot has a six-DOF free joint followed by two driven-wheel hinges.
        target_index = world * 16 + robot * 8 + 6 + wheel
        targets[target_index] = actions[action_index] * max_wheel_speed


@dataclass(frozen=True, slots=True)
class PhysicsStep:
    state: ArenaState
    action_proposed: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    winner: torch.Tensor
    ring_out: torch.Tensor
    inactivity: torch.Tensor
    numerical_failure: torch.Tensor

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated


def _require_sim() -> None:
    if newton is None or wp is None:
        raise RuntimeError("Newton simulation dependencies are missing; install the project with the 'sim' extra")


class NewtonSumoArena:
    """GPU-vectorized two-robot arena using Newton's MuJoCo-Warp contact solver."""

    BODIES_PER_WORLD = 6
    SHAPES_PER_WORLD = 9
    DOFS_PER_WORLD = 16
    COORDS_PER_WORLD = 18
    CHASSIS_BODY_INDICES = (0, 3)

    def __init__(
        self,
        config: ArenaConfig,
        world_count: int,
        *,
        device: str = "cuda:0",
        seed: int = 0,
    ) -> None:
        _require_sim()
        if world_count <= 0:
            raise ValueError("world_count must be positive")
        self.config = config
        self.world_count = world_count
        self.torch_device = torch.device(device)
        self.generator = torch.Generator(device=self.torch_device).manual_seed(seed)
        self.randomizer = DomainRandomizer(config.domain_randomization)

        template = self._build_world_template()
        if template.body_count != self.BODIES_PER_WORLD or template.shape_count != self.SHAPES_PER_WORLD:
            raise RuntimeError(
                f"unexpected template layout: {template.body_count} bodies, {template.shape_count} shapes"
            )
        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        scene.replicate(template, world_count, spacing=(0.0, 0.0, 0.0))
        self.model = scene.finalize(device=device)
        if self.model.joint_dof_count != world_count * self.DOFS_PER_WORLD:
            raise RuntimeError("joint DOF layout no longer matches the wheel-target scatter contract")
        if self.model.joint_coord_count != world_count * self.COORDS_PER_WORLD:
            raise RuntimeError("joint coordinate layout no longer matches the reset contract")

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            iterations=config.physics.solver_iterations,
            ls_iterations=config.physics.solver_ls_iterations,
            use_mujoco_contacts=True,
            nconmax=128,
            njmax=512,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if self.control.joint_target_qd is None:
            raise RuntimeError("wheel velocity actuators did not create joint_target_qd")
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self._base_body_mass = self.model.body_mass.numpy().copy()
        self._base_body_inertia = self.model.body_inertia.numpy().copy()
        self._base_joint_effort = self.model.joint_effort_limit.numpy().copy()
        self._base_joint_target_kd = self.model.joint_target_kd.numpy().copy()
        max_delay = max(config.domain_randomization["action_latency_s"])
        self._history_length = math.ceil(max_delay / config.physics.control_dt) + 2
        self._action_history = torch.zeros(
            (self._history_length, world_count, 2, ACTION_DIM), dtype=torch.float32, device=self.torch_device
        )
        self._last_action_exec = torch.zeros(
            (world_count, 2, ACTION_DIM), dtype=torch.float32, device=self.torch_device
        )
        self._elapsed_steps = torch.zeros(world_count, dtype=torch.int64, device=self.torch_device)
        self._stationary_steps = torch.zeros((world_count, 2), dtype=torch.int64, device=self.torch_device)
        self._movement_steps = torch.zeros((world_count, 2), dtype=torch.int64, device=self.torch_device)
        self._movement_confirmation_steps = math.ceil(config.match.movement_confirmation_s / config.physics.control_dt)
        self.domain = self.randomizer.sample(world_count, device=self.torch_device, generator=self.generator)
        self._apply_domain_parameters()
        self.reset()

    def _build_world_template(self) -> Any:
        config = self.config
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.gravity = config.physics.gravity_m_s2
        board_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8, restitution=0.0)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, 0.0, config.board.top_z_m - config.board.thickness_m / 2.0), wp.quat_identity()),
            hx=config.board.size_m[0] / 2.0,
            hy=config.board.size_m[1] / 2.0,
            hz=config.board.thickness_m / 2.0,
            cfg=board_cfg,
            label="board",
            color=(0.18, 0.18, 0.20),
        )
        self._add_robot(
            builder,
            "red",
            config.initialization.red_position_m,
            config.initialization.red_yaw_rad,
            color=(0.75, 0.12, 0.10),
        )
        self._add_robot(
            builder,
            "blue",
            config.initialization.blue_position_m,
            config.initialization.blue_yaw_rad,
            color=(0.10, 0.25, 0.78),
        )
        return builder

    def _add_robot(
        self,
        builder: Any,
        label: str,
        position: tuple[float, float, float],
        yaw: float,
        *,
        color: tuple[float, float, float],
    ) -> None:
        robot = self.config.robot
        orientation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
        chassis = builder.add_link(xform=wp.transform(position, orientation), label=f"{label}/chassis")
        chassis_volume = math.prod(robot.chassis_size_m)
        chassis_cfg = newton.ModelBuilder.ShapeConfig(
            density=robot.chassis_mass_kg / chassis_volume,
            mu=0.55,
            restitution=0.02,
        )
        builder.add_shape_box(
            chassis,
            hx=robot.chassis_size_m[0] / 2.0,
            hy=robot.chassis_size_m[1] / 2.0,
            hz=robot.chassis_size_m[2] / 2.0,
            cfg=chassis_cfg,
            label=f"{label}/chassis_collision",
            color=color,
        )
        skid_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.22, restitution=0.02)
        builder.add_shape_sphere(
            chassis,
            xform=wp.transform(
                (robot.skid_x_m, 0.0, -robot.chassis_size_m[2] / 2.0),
                wp.quat_identity(),
            ),
            radius=robot.skid_radius_m,
            cfg=skid_cfg,
            label=f"{label}/skid_collision",
            color=(0.15, 0.15, 0.15),
        )
        joints = [builder.add_joint_free(chassis, label=f"{label}/free")]
        wheel_density = robot.wheel_mass_kg / (math.pi * robot.wheel_radius_m**2 * (2.0 * robot.wheel_width_m))
        wheel_cfg = newton.ModelBuilder.ShapeConfig(density=wheel_density, mu=1.0, restitution=0.02)
        wheel_shape_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        for wheel_name, local_y in (
            ("left_wheel", robot.track_m / 2.0),
            ("right_wheel", -robot.track_m / 2.0),
        ):
            local_x = robot.wheel_axle_x_m
            world_x = position[0] + cos_yaw * local_x - sin_yaw * local_y
            world_y = position[1] + sin_yaw * local_x + cos_yaw * local_y
            wheel_z = self.config.board.top_z_m + robot.wheel_radius_m
            wheel = builder.add_link(
                xform=wp.transform((world_x, world_y, wheel_z), orientation),
                label=f"{label}/{wheel_name}",
            )
            builder.add_shape_cylinder(
                wheel,
                xform=wp.transform((0.0, 0.0, 0.0), wheel_shape_rotation),
                radius=robot.wheel_radius_m,
                half_height=robot.wheel_width_m,
                cfg=wheel_cfg,
                label=f"{label}/{wheel_name}_collision",
                color=(0.04, 0.04, 0.04),
            )
            axis = newton.ModelBuilder.JointDofConfig(
                axis=(0.0, 1.0, 0.0),
                target_vel=0.0,
                target_ke=0.0,
                target_kd=0.02,
                damping=1e-5,
                armature=1e-6,
                effort_limit=robot.max_wheel_torque_nm,
                velocity_limit=robot.max_wheel_speed_rad_s,
                actuator_mode=newton.JointTargetMode.VELOCITY,
            )
            joints.append(
                builder.add_joint_revolute(
                    parent=chassis,
                    child=wheel,
                    axis=axis,
                    parent_xform=wp.transform((local_x, local_y, wheel_z - position[2]), wp.quat_identity()),
                    child_xform=wp.transform_identity(),
                    label=f"{label}/{wheel_name}_joint",
                    collision_filter_parent=True,
                )
            )
        builder.add_articulation(joints, label=f"{label}/robot")

    def _replace_domain_rows(self, fresh: DomainBatch, mask: torch.Tensor) -> None:
        values: dict[str, torch.Tensor] = {}
        selector = mask.unsqueeze(-1)
        for field in fields(DomainBatch):
            old_value = getattr(self.domain, field.name)
            new_value = getattr(fresh, field.name)
            values[field.name] = torch.where(selector, new_value, old_value)
        self.domain = DomainBatch(**values)

    def _apply_domain_parameters(self) -> None:
        domain = self.domain
        board_mu = domain.board_friction.detach().cpu().numpy()
        chassis_mu = domain.chassis_friction.detach().cpu().numpy()
        wheel_mu = domain.wheel_friction.detach().cpu().numpy()
        skid_mu = domain.skid_friction.detach().cpu().numpy()
        restitution_values = domain.restitution.detach().cpu().numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        restitution = self.model.shape_material_restitution.numpy()
        for world in range(self.world_count):
            offset = world * self.SHAPES_PER_WORLD
            shape_mu[offset] = board_mu[world, 0]
            restitution[offset] = restitution_values[world, 0]
            for robot in range(2):
                shape_start = offset + 1 + robot * 4
                shape_mu[shape_start] = chassis_mu[world, robot]
                shape_mu[shape_start + 1] = skid_mu[world, robot]
                shape_mu[shape_start + 2 : shape_start + 4] = wheel_mu[world, robot]
                restitution[shape_start : shape_start + 4] = restitution_values[world, 0]
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_restitution.assign(restitution)

        body_mass = self._base_body_mass.copy()
        body_inertia = self._base_body_inertia.copy()
        chassis_scale = domain.chassis_mass_scale.detach().cpu().numpy()
        wheel_scale = domain.wheel_mass_scale.detach().cpu().numpy()
        for world in range(self.world_count):
            body_offset = world * self.BODIES_PER_WORLD
            for robot in range(2):
                chassis_index = body_offset + robot * 3
                body_mass[chassis_index] *= chassis_scale[world, robot]
                body_inertia[chassis_index] *= chassis_scale[world, robot]
                wheel_slice = slice(chassis_index + 1, chassis_index + 3)
                body_mass[wheel_slice] *= wheel_scale[world, robot]
                body_inertia[wheel_slice] *= wheel_scale[world, robot]
        self.model.body_mass.assign(body_mass)
        self.model.body_inertia.assign(body_inertia)
        self.model.body_inv_mass.assign(np.reciprocal(body_mass))
        self.model.body_inv_inertia.assign(np.linalg.inv(body_inertia))

        joint_effort = self._base_joint_effort.copy()
        joint_target_kd = self._base_joint_target_kd.copy()
        motor_scale = domain.motor_strength_scale.detach().cpu().numpy()
        for world in range(self.world_count):
            dof_offset = world * self.DOFS_PER_WORLD
            for robot in range(2):
                wheels = slice(dof_offset + robot * 8 + 6, dof_offset + robot * 8 + 8)
                joint_effort[wheels] *= motor_scale[world, robot]
                joint_target_kd[wheels] *= motor_scale[world, robot]
        self.model.joint_effort_limit.assign(joint_effort)
        self.model.joint_target_kd.assign(joint_target_kd)
        self.solver.notify_model_changed(
            newton.ModelFlags.SHAPE_PROPERTIES
            | newton.ModelFlags.BODY_INERTIAL_PROPERTIES
            | newton.ModelFlags.JOINT_DOF_PROPERTIES
        )

    def _set_random_initial_coordinates(self, mask: torch.Tensor) -> None:
        coordinates = self.state_0.joint_q.numpy()
        mask_cpu = mask.detach().cpu().numpy()
        jitter_xy = self.config.initialization.position_jitter_m
        yaw_jitter = self.config.initialization.yaw_jitter_rad
        random_xy = (
            (torch.rand((self.world_count, 2, 2), device=self.torch_device, generator=self.generator) * 2.0 - 1.0)
            .cpu()
            .numpy()
        )
        random_yaw = (
            (torch.rand((self.world_count, 2), device=self.torch_device, generator=self.generator) * 2.0 - 1.0)
            .cpu()
            .numpy()
        )
        base = (
            (self.config.initialization.red_position_m, self.config.initialization.red_yaw_rad),
            (self.config.initialization.blue_position_m, self.config.initialization.blue_yaw_rad),
        )
        for world in range(self.world_count):
            if not mask_cpu[world]:
                continue
            coordinate_offset = world * self.COORDS_PER_WORLD
            for robot, (position, yaw) in enumerate(base):
                free_start = coordinate_offset + robot * 9
                coordinates[free_start : free_start + 3] = (
                    position[0] + random_xy[world, robot, 0] * jitter_xy[0],
                    position[1] + random_xy[world, robot, 1] * jitter_xy[1],
                    position[2],
                )
                sampled_yaw = yaw + random_yaw[world, robot] * yaw_jitter
                coordinates[free_start + 3 : free_start + 7] = (
                    0.0,
                    0.0,
                    math.sin(sampled_yaw / 2.0),
                    math.cos(sampled_yaw / 2.0),
                )
                coordinates[free_start + 7 : free_start + 9] = 0.0
        self.state_0.joint_q.assign(coordinates)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def reset(self, mask: torch.Tensor | None = None) -> ArenaState:
        if mask is None:
            mask = torch.ones(self.world_count, dtype=torch.bool, device=self.torch_device)
        else:
            mask = mask.to(device=self.torch_device, dtype=torch.bool)
            if tuple(mask.shape) != (self.world_count,):
                raise ValueError(f"reset mask must have shape ({self.world_count},)")
        fresh = self.randomizer.sample(self.world_count, device=self.torch_device, generator=self.generator)
        self._replace_domain_rows(fresh, mask)
        self._apply_domain_parameters()
        warp_mask = wp.from_torch(mask.contiguous(), dtype=wp.bool)
        self.solver.reset(self.state_0, world_mask=warp_mask)
        self._set_random_initial_coordinates(mask)
        self._elapsed_steps[mask] = 0
        self._stationary_steps[mask] = 0
        self._movement_steps[mask] = 0
        self._action_history[:, mask] = 0.0
        self._last_action_exec[mask] = 0.0
        return self.snapshot()

    def _delay_actions(self, proposed: torch.Tensor) -> torch.Tensor:
        self._action_history = torch.roll(self._action_history, shifts=1, dims=0)
        self._action_history[0].copy_(proposed)
        delay = self.domain.action_latency_s / self.config.physics.control_dt
        lower = torch.floor(delay).to(torch.long).clamp(max=self._history_length - 2)
        fraction = (delay - lower).unsqueeze(-1)
        world_index = torch.arange(self.world_count, device=self.torch_device).unsqueeze(-1).expand(-1, 2)
        robot_index = torch.arange(2, device=self.torch_device).unsqueeze(0).expand(self.world_count, -1)
        newer = self._action_history[lower, world_index, robot_index]
        older = self._action_history[lower + 1, world_index, robot_index]
        return newer * (1.0 - fraction) + older * fraction

    def step(self, actions: torch.Tensor) -> PhysicsStep:
        expected = (self.world_count, 2, ACTION_DIM)
        if tuple(actions.shape) != expected:
            raise ValueError(f"actions must have shape {expected}")
        proposed = actions.to(device=self.torch_device, dtype=torch.float32).clamp(-1.0, 1.0)
        executed = self._delay_actions(proposed)
        self._last_action_exec.copy_(executed)
        action_warp = wp.from_torch(executed.contiguous().view(-1), dtype=wp.float32)
        wp.launch(
            _scatter_wheel_targets,
            dim=self.world_count * 2 * DRIVEN_WHEEL_COUNT,
            inputs=(action_warp, self.control.joint_target_qd, self.config.robot.max_wheel_speed_rad_s),
            device=self.model.device,
        )
        for _ in range(self.config.physics.substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                None,
                self.config.physics.simulation_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        self._elapsed_steps += 1
        state = self.snapshot()
        planar_speed = torch.linalg.vector_norm(state.linear_velocity[..., :2], dim=-1)
        movement_candidate = planar_speed >= self.config.match.movement_speed_threshold_m_s
        self._movement_steps = torch.where(
            movement_candidate, self._movement_steps + 1, torch.zeros_like(self._movement_steps)
        )
        moving = self._movement_steps >= self._movement_confirmation_steps
        self._stationary_steps = torch.where(
            moving, torch.zeros_like(self._stationary_steps), self._stationary_steps + 1
        )
        stationary_time = self._stationary_steps.to(torch.float32) * self.config.physics.control_dt
        state = replace(state, stationary_time_s=stationary_time)
        out = state.edge_margin < 0.0
        fallen = state.position[:, :, 2] < self.config.board.top_z_m - self.config.robot.wheel_radius_m
        out = out | fallen
        finite = torch.ones((self.world_count, 2), dtype=torch.bool, device=self.torch_device)
        for value in (
            state.position,
            state.quaternion,
            state.linear_velocity,
            state.angular_velocity,
            state.wheel_velocity,
        ):
            finite &= torch.isfinite(value).flatten(start_dim=2).all(dim=-1)
        numerical_failure = ~finite.all(dim=-1)
        inactive = stationary_time >= self.config.match.inactivity_timeout_s
        timed_out = self._elapsed_steps >= self.config.physics.max_episode_steps
        resolution = resolve_match(out, inactive, timed_out, numerical_failure)
        return PhysicsStep(
            state=state,
            action_proposed=proposed,
            terminated=resolution.terminated,
            truncated=resolution.truncated,
            winner=resolution.winner,
            ring_out=resolution.ring_out,
            inactivity=resolution.inactivity,
            numerical_failure=resolution.numerical_failure,
        )

    def snapshot(self) -> ArenaState:
        body_q = wp.to_torch(self.state_0.body_q).reshape(self.world_count, self.BODIES_PER_WORLD, 7)
        body_qd = wp.to_torch(self.state_0.body_qd).reshape(self.world_count, self.BODIES_PER_WORLD, 6)
        chassis = torch.tensor(self.CHASSIS_BODY_INDICES, dtype=torch.long, device=self.torch_device)
        pose = body_q.index_select(1, chassis)
        twist = body_qd.index_select(1, chassis)
        joint_qd = wp.to_torch(self.state_0.joint_qd).reshape(self.world_count, self.DOFS_PER_WORLD)
        wheel_velocity = torch.stack((joint_qd[:, 6:8], joint_qd[:, 14:16]), dim=1)
        position = pose[..., :3].clone()
        half_x, half_y = (value / 2.0 for value in self.config.board.size_m)
        edge_margin = torch.minimum(half_x - position[..., 0].abs(), half_y - position[..., 1].abs())
        time_remaining = (
            self.config.physics.episode_seconds - self._elapsed_steps.to(torch.float32) * self.config.physics.control_dt
        ).clamp_min(0.0)
        return ArenaState(
            position=position,
            quaternion=pose[..., 3:7].clone(),
            linear_velocity=twist[..., :3].clone(),
            angular_velocity=twist[..., 3:6].clone(),
            wheel_velocity=wheel_velocity.clone(),
            action_exec=self._last_action_exec.clone(),
            contact_force=torch.zeros((self.world_count, 2, 3), device=self.torch_device),
            edge_margin=edge_margin,
            stationary_time_s=self._stationary_steps.to(torch.float32) * self.config.physics.control_dt,
            time_remaining_s=time_remaining,
        )


def run_smoke(config_path: Path, *, world_count: int, steps: int, device: str) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    config = ArenaConfig.load(config_path)
    arena = NewtonSumoArena(config, world_count, device=device, seed=7)
    initial = arena.snapshot()
    actions = torch.ones((world_count, 2, ACTION_DIM), dtype=torch.float32, device=device) * 0.35
    result = None
    for _ in range(steps):
        result = arena.step(actions)
        if bool(result.done.all()):
            break
    assert result is not None
    final = result.state
    tensors = (final.position, final.quaternion, final.linear_velocity, final.wheel_velocity)
    finite = all(bool(torch.isfinite(value).all()) for value in tensors)
    displacement = torch.linalg.vector_norm(final.position[..., :2] - initial.position[..., :2], dim=-1)
    mean_displacement = float(displacement.mean().item())
    minimum_height = float(final.position[..., 2].min().item())
    maximum_height = float(final.position[..., 2].max().item())
    if not finite:
        raise RuntimeError("Newton/MuJoCo-Warp smoke rollout produced non-finite state")
    if steps >= 10 and mean_displacement < 0.02:
        raise RuntimeError("wheel commands did not produce meaningful planar motion")
    if minimum_height < config.board.top_z_m - 0.01 or maximum_height > config.board.top_z_m + 0.10:
        raise RuntimeError("chassis height left the calibrated board-contact envelope")

    arena.reset()
    turn_actions = torch.zeros((world_count, 2, ACTION_DIM), dtype=torch.float32, device=device)
    turn_actions[..., 0] = -0.35
    turn_actions[..., 1] = 0.35
    turn_result = None
    for _ in range(steps):
        turn_result = arena.step(turn_actions)
        if bool(turn_result.done.all()):
            break
    assert turn_result is not None
    max_yaw_rate = float(turn_result.state.angular_velocity[..., 2].abs().max().item())
    if steps >= 10 and max_yaw_rate < 0.1:
        raise RuntimeError("independent left/right wheel commands did not produce differential turning")

    arena.reset()
    idle_actions = torch.zeros((world_count, 2, ACTION_DIM), dtype=torch.float32, device=device)
    idle_result = None
    idle_steps = 0
    idle_limit = round((config.match.inactivity_timeout_s + 5.0) * config.physics.control_hz)
    for _ in range(idle_limit):
        idle_steps += 1
        idle_result = arena.step(idle_actions)
        if bool(idle_result.done.all()):
            break
    assert idle_result is not None
    if not bool(idle_result.inactivity.all()) or not bool((idle_result.winner == DRAW).all()):
        raise RuntimeError("two inactive robots did not end in a draw within the inactivity window")
    return {
        "backend": "Newton SolverMuJoCo / MuJoCo-Warp contacts",
        "device": str(arena.model.device),
        "worlds": world_count,
        "steps": steps,
        "finite": finite,
        "mean_displacement_m": mean_displacement,
        "max_speed_m_s": float(torch.linalg.vector_norm(final.linear_velocity, dim=-1).max().item()),
        "max_differential_yaw_rate_rad_s": max_yaw_rate,
        "stationary_draw_after_s": idle_steps * config.physics.control_dt,
        "chassis_height_range_m": [minimum_height, maximum_height],
        "terminated_worlds": int(result.terminated.sum().item()),
        "numerical_failures": int(
            result.numerical_failure.sum().item()
            + turn_result.numerical_failure.sum().item()
            + idle_result.numerical_failure.sum().item()
        ),
        "board_size_m": list(config.board.size_m),
        "robot_chassis_size_m": list(config.robot.chassis_size_m),
    }
