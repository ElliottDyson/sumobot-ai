from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch

from ..config import ArenaConfig
from ..contracts import ACTION_DIM, DRIVEN_WHEEL_COUNT
from ..domain_randomization import DomainBatch, DomainRandomizer
from ..geometry import quaternion_rotate_xyzw
from ..match import resolve_match
from ..observations import StudentSensors
from ..state import DRAW, ArenaState
from .sensors import StudentSensorSuite

try:  # Keep lightweight reward/curriculum tooling usable without the simulation extra.
    import newton
    import warp as wp
except ImportError:  # pragma: no cover - exercised by the doctor command
    newton = None
    wp = None


if wp is not None:

    @wp.kernel
    def _scatter_wheel_actuation(
        actions: wp.array(dtype=wp.float32),
        targets: wp.array(dtype=wp.float32),
        effort_limits: wp.array(dtype=wp.float32),
        joint_velocity: wp.array(dtype=wp.float32),
        motor_strength: wp.array(dtype=wp.float32),
        battery_scale: wp.array(dtype=wp.float32),
        motor_asymmetry: wp.array(dtype=wp.float32),
        max_wheel_speed: float,
        stall_torque: float,
        minimum_torque_fraction: float,
    ):
        action_index = wp.tid()
        world = action_index // 4
        within_world = action_index - world * 4
        robot = within_world // 2
        wheel = within_world - robot * 2
        # Each robot has a six-DOF free joint followed by two driven-wheel hinges.
        target_index = world * 16 + robot * 8 + 6 + wheel
        robot_index = world * 2 + robot
        side_gain = 1.0 + motor_asymmetry[robot_index]
        if wheel == 1:
            side_gain = 1.0 - motor_asymmetry[robot_index]
        side_gain = wp.max(side_gain, 0.1)
        voltage = wp.max(battery_scale[robot_index], 0.1)
        no_load_speed = max_wheel_speed * voltage * side_gain
        targets[target_index] = actions[action_index] * no_load_speed
        speed_fraction = wp.abs(joint_velocity[target_index]) / wp.max(no_load_speed, 1.0e-6)
        torque_fraction = wp.max(minimum_torque_fraction, 1.0 - speed_fraction)
        effort_limits[target_index] = stall_torque * motor_strength[robot_index] * voltage * side_gain * torque_fraction

    @wp.kernel
    def _accumulate_contact_impulses(
        contact_count: wp.array(dtype=wp.int32),
        shape0: wp.array(dtype=wp.int32),
        shape1: wp.array(dtype=wp.int32),
        contact_force: wp.array(dtype=wp.spatial_vector),
        impulse: wp.array(dtype=wp.vec3),
        dt: float,
    ):
        contact_index = wp.tid()
        if contact_index >= contact_count[0]:
            return
        first = shape0[contact_index]
        second = shape1[contact_index]
        force = wp.spatial_top(contact_force[contact_index]) * dt
        first_local = first % 9
        second_local = second % 9
        if first_local >= 1 and first_local <= 8:
            first_robot = first // 9 * 2
            if first_local >= 5:
                first_robot += 1
            wp.atomic_add(impulse, first_robot, force)
        if second_local >= 1 and second_local <= 8:
            second_robot = second // 9 * 2
            if second_local >= 5:
                second_robot += 1
            wp.atomic_add(impulse, second_robot, -force)


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

        self.model.request_contact_attributes("force")
        self.contacts = self.model.contacts()
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

        # Keep the immutable template values on the simulation device.  Episode
        # resets are frequent in a large vectorized run; staging these arrays
        # through NumPy made every partial reset copy and loop over every world.
        self._base_body_mass = wp.to_torch(self.model.body_mass).reshape(world_count, self.BODIES_PER_WORLD).clone()
        self._base_body_inertia = (
            wp.to_torch(self.model.body_inertia).reshape(world_count, self.BODIES_PER_WORLD, 3, 3).clone()
        )
        self._base_body_com = wp.to_torch(self.model.body_com).reshape(world_count, self.BODIES_PER_WORLD, 3).clone()
        self._base_joint_effort = (
            wp.to_torch(self.model.joint_effort_limit).reshape(world_count, self.DOFS_PER_WORLD).clone()
        )
        self._base_joint_target_kd = (
            wp.to_torch(self.model.joint_target_kd).reshape(world_count, self.DOFS_PER_WORLD).clone()
        )
        self._base_shape_ke = (
            wp.to_torch(self.model.shape_material_ke).reshape(world_count, self.SHAPES_PER_WORLD).clone()
        )
        self._base_shape_kd = (
            wp.to_torch(self.model.shape_material_kd).reshape(world_count, self.SHAPES_PER_WORLD).clone()
        )
        max_delay = max(config.domain_randomization["action_latency_s"])
        self._history_length = math.ceil(max_delay / config.physics.control_dt) + 2
        self._action_history = torch.zeros(
            (self._history_length, world_count, 2, ACTION_DIM), dtype=torch.float32, device=self.torch_device
        )
        self._last_action_exec = torch.zeros(
            (world_count, 2, ACTION_DIM), dtype=torch.float32, device=self.torch_device
        )
        self._last_action_proposed = torch.zeros_like(self._last_action_exec)
        self._motor_command = torch.zeros_like(self._last_action_exec)
        self._contact_impulse = torch.zeros((world_count, 2, 3), dtype=torch.float32, device=self.torch_device)
        self._elapsed_steps = torch.zeros(world_count, dtype=torch.int64, device=self.torch_device)
        self._stationary_steps = torch.zeros((world_count, 2), dtype=torch.int64, device=self.torch_device)
        self._movement_steps = torch.zeros((world_count, 2), dtype=torch.int64, device=self.torch_device)
        self._out_steps = torch.zeros((world_count, 2), dtype=torch.int64, device=self.torch_device)
        self._movement_confirmation_steps = math.ceil(config.match.movement_confirmation_s / config.physics.control_dt)
        self._ring_out_confirmation_steps = math.ceil(config.match.ring_out_confirmation_s / config.physics.control_dt)
        self.domain = self.randomizer.sample(world_count, device=self.torch_device, generator=self.generator)
        self._apply_domain_parameters()
        self.sensor_suite = StudentSensorSuite(config, world_count, device=self.torch_device, generator=self.generator)
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
        builder.body_mass[chassis] = robot.chassis_mass_kg
        builder.body_com[chassis] = wp.vec3(*robot.chassis_com_offset_m)
        ixx, iyy, izz = robot.chassis_inertia_diagonal_kg_m2
        builder.body_inertia[chassis] = wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz)

        skid_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=0.22,
            restitution=0.02,
            mu_torsional=0.008,
            mu_rolling=0.0005,
        )
        skid_z = self.config.board.top_z_m + robot.skid_size_m[2] / 2.0 - position[2]
        builder.add_shape_box(
            chassis,
            xform=wp.transform(
                (robot.skid_x_m, 0.0, skid_z),
                wp.quat_identity(),
            ),
            hx=robot.skid_size_m[0] / 2.0,
            hy=robot.skid_size_m[1] / 2.0,
            hz=robot.skid_size_m[2] / 2.0,
            cfg=skid_cfg,
            label=f"{label}/skid_collision",
            color=(0.15, 0.15, 0.15),
        )
        joints = [builder.add_joint_free(chassis, label=f"{label}/free")]
        wheel_density = robot.wheel_mass_kg / (math.pi * robot.wheel_radius_m**2 * robot.wheel_width_m)
        wheel_cfg = newton.ModelBuilder.ShapeConfig(
            density=wheel_density,
            mu=1.0,
            restitution=0.02,
            mu_torsional=0.003,
            mu_rolling=0.0001,
        )
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
                half_height=robot.wheel_half_width_m,
                cfg=wheel_cfg,
                label=f"{label}/{wheel_name}_collision",
                color=(0.04, 0.04, 0.04),
            )
            axis = newton.ModelBuilder.JointDofConfig(
                axis=(0.0, 1.0, 0.0),
                target_vel=0.0,
                target_ke=0.0,
                target_kd=robot.velocity_servo_gain,
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

    def _replace_domain_rows(self, fresh: DomainBatch, indices: torch.Tensor) -> None:
        if fresh.batch_size != indices.numel():
            raise ValueError("fresh domain batch must contain exactly one row per reset world")
        for field in fields(DomainBatch):
            getattr(self.domain, field.name).index_copy_(0, indices, getattr(fresh, field.name))

    def _apply_domain_parameters(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            indices = torch.arange(self.world_count, device=self.torch_device)
        else:
            indices = mask.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return

        domain = self.domain
        shape_mu = wp.to_torch(self.model.shape_material_mu).reshape(self.world_count, self.SHAPES_PER_WORLD)
        selected_mu = shape_mu.index_select(0, indices)
        selected_mu[:, 0] = domain.board_friction[indices, 0]
        selected_mu[:, (1, 5)] = domain.chassis_friction[indices]
        selected_mu[:, (2, 6)] = domain.skid_friction[indices]
        selected_mu[:, 3:5] = domain.wheel_friction[indices, 0:1]
        selected_mu[:, 7:9] = domain.wheel_friction[indices, 1:2]
        shape_mu.index_copy_(0, indices, selected_mu)

        shape_torsional = wp.to_torch(self.model.shape_material_mu_torsional).reshape(
            self.world_count, self.SHAPES_PER_WORLD
        )
        selected_torsional = shape_torsional.index_select(0, indices)
        selected_torsional[:, (2, 6)] = domain.skid_torsional_friction[indices]
        selected_torsional[:, 3:5] = domain.wheel_torsional_friction[indices, 0:1]
        selected_torsional[:, 7:9] = domain.wheel_torsional_friction[indices, 1:2]
        shape_torsional.index_copy_(0, indices, selected_torsional)

        shape_rolling = wp.to_torch(self.model.shape_material_mu_rolling).reshape(
            self.world_count, self.SHAPES_PER_WORLD
        )
        selected_rolling = shape_rolling.index_select(0, indices)
        selected_rolling[:, (2, 6)] = domain.skid_rolling_friction[indices]
        selected_rolling[:, 3:5] = domain.wheel_rolling_friction[indices, 0:1]
        selected_rolling[:, 7:9] = domain.wheel_rolling_friction[indices, 1:2]
        shape_rolling.index_copy_(0, indices, selected_rolling)

        restitution = wp.to_torch(self.model.shape_material_restitution).reshape(
            self.world_count, self.SHAPES_PER_WORLD
        )
        restitution.index_copy_(0, indices, domain.restitution[indices].expand(-1, self.SHAPES_PER_WORLD))
        stiffness = wp.to_torch(self.model.shape_material_ke).reshape(self.world_count, self.SHAPES_PER_WORLD)
        damping = wp.to_torch(self.model.shape_material_kd).reshape(self.world_count, self.SHAPES_PER_WORLD)
        stiffness.index_copy_(0, indices, self._base_shape_ke[indices] * domain.contact_stiffness_scale[indices])
        damping.index_copy_(0, indices, self._base_shape_kd[indices] * domain.contact_damping_scale[indices])

        chassis_columns = (0, 3)
        wheel_columns = (1, 2, 4, 5)
        chassis_scale = domain.chassis_mass_scale[indices]
        wheel_scale = domain.wheel_mass_scale[indices].repeat_interleave(2, dim=-1)
        body_mass = self._base_body_mass[indices].clone()
        body_mass[:, chassis_columns] *= chassis_scale
        body_mass[:, wheel_columns] *= wheel_scale
        model_body_mass = wp.to_torch(self.model.body_mass).reshape(self.world_count, self.BODIES_PER_WORLD)
        model_body_mass.index_copy_(0, indices, body_mass)
        wp.to_torch(self.model.body_inv_mass).reshape(self.world_count, self.BODIES_PER_WORLD).index_copy_(
            0, indices, body_mass.reciprocal()
        )

        body_inertia = self._base_body_inertia[indices].clone()
        body_inertia[:, chassis_columns] *= (
            (chassis_scale * domain.chassis_inertia_scale[indices]).unsqueeze(-1).unsqueeze(-1)
        )
        body_inertia[:, wheel_columns] *= wheel_scale.unsqueeze(-1).unsqueeze(-1)
        wp.to_torch(self.model.body_inertia).reshape(self.world_count, self.BODIES_PER_WORLD, 3, 3).index_copy_(
            0, indices, body_inertia
        )
        wp.to_torch(self.model.body_inv_inertia).reshape(self.world_count, self.BODIES_PER_WORLD, 3, 3).index_copy_(
            0, indices, torch.linalg.inv(body_inertia)
        )

        body_com = self._base_body_com[indices].clone()
        body_com[:, chassis_columns] += torch.stack(
            (
                domain.chassis_com_offset_x_m[indices],
                domain.chassis_com_offset_y_m[indices],
                domain.chassis_com_offset_z_m[indices],
            ),
            dim=-1,
        )
        wp.to_torch(self.model.body_com).reshape(self.world_count, self.BODIES_PER_WORLD, 3).index_copy_(
            0, indices, body_com
        )

        wheel_dofs = (6, 7, 14, 15)
        motor_scale = domain.motor_strength_scale[indices].repeat_interleave(2, dim=-1)
        joint_effort = self._base_joint_effort[indices].clone()
        joint_target_kd = self._base_joint_target_kd[indices].clone()
        joint_effort[:, wheel_dofs] *= motor_scale
        joint_target_kd[:, wheel_dofs] *= motor_scale
        wp.to_torch(self.model.joint_effort_limit).reshape(self.world_count, self.DOFS_PER_WORLD).index_copy_(
            0, indices, joint_effort
        )
        wp.to_torch(self.model.joint_target_kd).reshape(self.world_count, self.DOFS_PER_WORLD).index_copy_(
            0, indices, joint_target_kd
        )
        self.solver.notify_model_changed(
            newton.ModelFlags.SHAPE_PROPERTIES
            | newton.ModelFlags.BODY_INERTIAL_PROPERTIES
            | newton.ModelFlags.JOINT_DOF_PROPERTIES
        )

    def _set_random_initial_coordinates(self, mask: torch.Tensor) -> None:
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return
        coordinates = wp.to_torch(self.state_0.joint_q).reshape(self.world_count, self.COORDS_PER_WORLD)
        selected = torch.zeros(
            (indices.numel(), self.COORDS_PER_WORLD), dtype=coordinates.dtype, device=self.torch_device
        )
        jitter_xy = self.config.initialization.position_jitter_m
        yaw_jitter = self.config.initialization.yaw_jitter_rad
        random_xy = (
            torch.rand((indices.numel(), 2, 2), device=self.torch_device, generator=self.generator).mul_(2.0).sub_(1.0)
        )
        random_yaw = (
            torch.rand((indices.numel(), 2), device=self.torch_device, generator=self.generator).mul_(2.0).sub_(1.0)
        )
        base = (
            (self.config.initialization.red_position_m, self.config.initialization.red_yaw_rad),
            (self.config.initialization.blue_position_m, self.config.initialization.blue_yaw_rad),
        )
        for robot, (position, yaw) in enumerate(base):
            free_start = robot * 9
            selected[:, free_start] = position[0] + random_xy[:, robot, 0] * jitter_xy[0]
            selected[:, free_start + 1] = position[1] + random_xy[:, robot, 1] * jitter_xy[1]
            selected[:, free_start + 2] = position[2]
            sampled_yaw = yaw + random_yaw[:, robot] * yaw_jitter
            selected[:, free_start + 5] = torch.sin(sampled_yaw / 2.0)
            selected[:, free_start + 6] = torch.cos(sampled_yaw / 2.0)
        coordinates.index_copy_(0, indices, selected)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def reset(self, mask: torch.Tensor | None = None) -> ArenaState:
        if mask is None:
            mask = torch.ones(self.world_count, dtype=torch.bool, device=self.torch_device)
        else:
            mask = mask.to(device=self.torch_device, dtype=torch.bool)
            if tuple(mask.shape) != (self.world_count,):
                raise ValueError(f"reset mask must have shape ({self.world_count},)")
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return self.snapshot()
        fresh = self.randomizer.sample(indices.numel(), device=self.torch_device, generator=self.generator)
        self._replace_domain_rows(fresh, indices)
        self._apply_domain_parameters(mask)
        warp_mask = wp.from_torch(mask.contiguous(), dtype=wp.bool)
        self.solver.reset(self.state_0, world_mask=warp_mask)
        self._set_random_initial_coordinates(mask)
        self._elapsed_steps[mask] = 0
        self._stationary_steps[mask] = 0
        self._movement_steps[mask] = 0
        self._out_steps[mask] = 0
        self._action_history[:, mask] = 0.0
        self._last_action_proposed[mask] = 0.0
        self._last_action_exec[mask] = 0.0
        self._motor_command[mask] = 0.0
        self._contact_impulse[mask] = 0.0
        state = self.snapshot()
        self.sensor_suite.reset(state, self.domain, mask)
        return state

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

    def _motor_response(self, delayed: torch.Tensor) -> torch.Tensor:
        deadband = self.domain.motor_deadband.unsqueeze(-1)
        magnitude = ((delayed.abs() - deadband) / (1.0 - deadband).clamp_min(1e-6)).clamp(0.0, 1.0)
        desired = delayed.sign() * magnitude
        levels = self.config.robot.motor_command_quantization_levels
        desired = torch.round(desired * levels) / levels
        alpha = 1.0 - torch.exp(-self.config.physics.control_dt / self.domain.motor_time_constant_s)
        first_order_delta = alpha.unsqueeze(-1) * (desired - self._motor_command)
        maximum_delta = self.config.robot.motor_command_slew_per_s * self.config.physics.control_dt
        self._motor_command.add_(first_order_delta.clamp(-maximum_delta, maximum_delta)).clamp_(-1.0, 1.0)
        return self._motor_command

    def _support_margin(self, position: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
        robot = self.config.robot
        contact_z = -robot.chassis_center_height_m
        local = torch.tensor(
            (
                (robot.wheel_axle_x_m, robot.track_m / 2.0, contact_z),
                (robot.wheel_axle_x_m, -robot.track_m / 2.0, contact_z),
                (robot.skid_x_m, 0.0, contact_z),
            ),
            dtype=position.dtype,
            device=self.torch_device,
        ).view(1, 1, 3, 3)
        support_world = position.unsqueeze(2) + quaternion_rotate_xyzw(quaternion.unsqueeze(2), local)
        half_x, half_y = (value / 2.0 for value in self.config.board.size_m)
        point_margin = torch.minimum(half_x - support_world[..., 0].abs(), half_y - support_world[..., 1].abs())
        return point_margin.max(dim=-1).values

    def step(self, actions: torch.Tensor) -> PhysicsStep:
        expected = (self.world_count, 2, ACTION_DIM)
        if tuple(actions.shape) != expected:
            raise ValueError(f"actions must have shape {expected}")
        proposed = actions.to(device=self.torch_device, dtype=torch.float32).clamp(-1.0, 1.0)
        self._last_action_proposed.copy_(proposed)
        executed = self._motor_response(self._delay_actions(proposed))
        self._last_action_exec.copy_(executed)
        action_warp = wp.from_torch(executed.contiguous().view(-1), dtype=wp.float32)
        strength_warp = wp.from_torch(self.domain.motor_strength_scale.contiguous().view(-1), dtype=wp.float32)
        battery_warp = wp.from_torch(self.domain.battery_voltage_scale.contiguous().view(-1), dtype=wp.float32)
        asymmetry_warp = wp.from_torch(self.domain.motor_asymmetry.contiguous().view(-1), dtype=wp.float32)
        impulse_warp = wp.from_torch(self._contact_impulse.view(-1, 3), dtype=wp.vec3)
        self._contact_impulse.zero_()
        for _ in range(self.config.physics.substeps):
            wp.launch(
                _scatter_wheel_actuation,
                dim=self.world_count * 2 * DRIVEN_WHEEL_COUNT,
                inputs=(
                    action_warp,
                    self.control.joint_target_qd,
                    self.model.joint_effort_limit,
                    self.state_0.joint_qd,
                    strength_warp,
                    battery_warp,
                    asymmetry_warp,
                    self.config.robot.max_wheel_speed_rad_s,
                    self.config.robot.max_wheel_torque_nm,
                    self.config.robot.motor_min_torque_fraction,
                ),
                device=self.model.device,
            )
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                None,
                self.config.physics.simulation_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.solver.update_contacts(self.contacts, self.state_0)
            wp.launch(
                _accumulate_contact_impulses,
                dim=self.contacts.rigid_contact_max,
                inputs=(
                    self.contacts.rigid_contact_count,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self.contacts.force,
                    impulse_warp,
                    self.config.physics.simulation_dt,
                ),
                device=self.model.device,
            )
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
        fallen = state.position[:, :, 2] < self.config.board.top_z_m - self.config.robot.wheel_radius_m
        out_candidate = (state.support_margin < 0.0) | fallen
        self._out_steps = torch.where(out_candidate, self._out_steps + 1, torch.zeros_like(self._out_steps))
        out = self._out_steps >= self._ring_out_confirmation_steps
        finite = torch.ones((self.world_count, 2), dtype=torch.bool, device=self.torch_device)
        for value in (
            state.position,
            state.quaternion,
            state.linear_velocity,
            state.angular_velocity,
            state.wheel_velocity,
            state.action_exec,
            state.contact_force,
        ):
            finite &= torch.isfinite(value).flatten(start_dim=2).all(dim=-1)
        finite &= torch.isfinite(state.edge_margin) & torch.isfinite(state.support_margin)
        numerical_failure = ~finite.all(dim=-1)
        inactive = stationary_time >= self.config.match.inactivity_timeout_s
        timed_out = self._elapsed_steps >= self.config.physics.max_episode_steps
        resolution = resolve_match(out, inactive, timed_out, numerical_failure)
        self.sensor_suite.advance(state)
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
        quaternion = pose[..., 3:7].clone()
        support_margin = self._support_margin(position, quaternion)
        time_remaining = (
            self.config.physics.episode_seconds - self._elapsed_steps.to(torch.float32) * self.config.physics.control_dt
        ).clamp_min(0.0)
        return ArenaState(
            position=position,
            quaternion=quaternion,
            linear_velocity=twist[..., :3].clone(),
            angular_velocity=twist[..., 3:6].clone(),
            wheel_velocity=wheel_velocity.clone(),
            action_proposed=self._last_action_proposed.clone(),
            action_exec=self._last_action_exec.clone(),
            contact_force=self._contact_impulse / self.config.physics.control_dt,
            edge_margin=edge_margin,
            support_margin=support_margin,
            stationary_time_s=self._stationary_steps.to(torch.float32) * self.config.physics.control_dt,
            time_remaining_s=time_remaining,
        )

    def student_sensors(self, perspective: int) -> StudentSensors:
        """Return the delayed/noisy deployable sensor packet for one side."""
        return self.sensor_suite.observe(self.snapshot(), self.domain, self._elapsed_steps, perspective)


def run_smoke(config_path: Path, *, world_count: int, steps: int, device: str) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    config = ArenaConfig.load(config_path)
    arena = NewtonSumoArena(config, world_count, device=device, seed=7)
    initial = arena.snapshot()
    first_step_exec_mean = 0.0
    actions = torch.ones((world_count, 2, ACTION_DIM), dtype=torch.float32, device=device) * 0.35
    result = None
    for step_index in range(steps):
        result = arena.step(actions)
        if step_index == 0:
            first_step_exec_mean = float(result.state.action_exec.abs().mean().item())
        if bool(result.done.all()):
            break
    assert result is not None
    final = result.state
    tensors = (final.position, final.quaternion, final.linear_velocity, final.wheel_velocity, final.contact_force)
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
    mean_normal_contact = float(final.contact_force[..., 2].mean().item())
    if mean_normal_contact <= 0.0:
        raise RuntimeError("MuJoCo-Warp contact forces were not propagated into ArenaState")
    red_sensors = arena.student_sensors(0)
    if not bool(torch.isfinite(red_sensors.wheel_velocity).all()):
        raise RuntimeError("deployable sensor simulation produced non-finite values")
    if not bool(
        ((red_sensors.edge_ranges >= 0.0) & (red_sensors.edge_ranges <= config.sensors.edge_max_range_m)).all()
    ):
        raise RuntimeError("edge sensors left their physical range")

    probe_position = torch.zeros((world_count, 2, 3), dtype=torch.float32, device=device)
    probe_quaternion = torch.zeros((world_count, 2, 4), dtype=torch.float32, device=device)
    probe_quaternion[..., 3] = 1.0
    probe_position[..., 0] = config.board.size_m[0] / 2.0 + 0.005
    partially_supported = arena._support_margin(probe_position, probe_quaternion)
    probe_position[..., 0] = config.board.size_m[0] / 2.0 + 0.020
    unsupported = arena._support_margin(probe_position, probe_quaternion)
    support_rule_verified = bool((partially_supported > 0.0).all() and (unsupported < 0.0).all())
    if not support_rule_verified:
        raise RuntimeError("support-point ring-out did not distinguish partial support from complete support loss")

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
    body_mass = wp.to_torch(arena.model.body_mass).reshape(world_count, arena.BODIES_PER_WORLD)
    robot_mass = torch.stack((body_mass[:, 0:3].sum(dim=-1), body_mass[:, 3:6].sum(dim=-1)), dim=-1)
    static_load_ratio = idle_result.state.contact_force[..., 2] / (
        robot_mass * abs(config.physics.gravity_m_s2)
    ).clamp_min(1e-6)
    maximum_static_load_error = float((static_load_ratio - 1.0).abs().max().item())
    if maximum_static_load_error > 0.10:
        raise RuntimeError("settled contact-force summary does not reproduce robot weight")
    return {
        "backend": "Newton SolverMuJoCo / MuJoCo-Warp contacts",
        "device": str(arena.model.device),
        "worlds": world_count,
        "steps": steps,
        "finite": finite,
        "mean_displacement_m": mean_displacement,
        "first_step_executed_command_mean": first_step_exec_mean,
        "max_speed_m_s": float(torch.linalg.vector_norm(final.linear_velocity, dim=-1).max().item()),
        "max_differential_yaw_rate_rad_s": max_yaw_rate,
        "mean_normal_contact_force_n": mean_normal_contact,
        "maximum_static_load_relative_error": maximum_static_load_error,
        "stationary_draw_after_s": idle_steps * config.physics.control_dt,
        "chassis_height_range_m": [minimum_height, maximum_height],
        "terminated_worlds": int(result.terminated.sum().item()),
        "numerical_failures": int(
            result.numerical_failure.sum().item()
            + turn_result.numerical_failure.sum().item()
            + idle_result.numerical_failure.sum().item()
        ),
        "board_size_m": list(config.board.size_m),
        "robot_envelope_size_m": list(config.robot.envelope_size_m),
        "robot_chassis_size_m": list(config.robot.chassis_size_m),
        "student_observation_base_dim": 25,
        "support_rule_verified": support_rule_verified,
    }
