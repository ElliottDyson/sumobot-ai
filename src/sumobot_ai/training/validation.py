from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..config import ArenaConfig
from ..observations import build_teacher_observation
from ..rewards import RewardSpec, evaluate_reward
from ..state import BLUE, DRAW, RED, ArenaState, ArenaTransition

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from ..sim.newton_backend import NewtonSumoArena
    from .cpo import TransplantableCPOActorCritic


def validation_capture_steps(
    max_episode_steps: int,
    control_hz: int,
    video_fps: int,
    video_max_frames: int,
) -> frozenset[int]:
    """Return post-step frame indices, including the match-time terminal frame."""
    if min(max_episode_steps, control_hz, video_fps, video_max_frames) <= 0:
        raise ValueError("validation timing and video settings must be positive")
    if video_fps > control_hz or control_hz % video_fps:
        raise ValueError("validation_video_fps must divide the control frequency")
    interval = control_hz // video_fps
    steps = list(range(interval, max_episode_steps + 1, interval))
    if not steps or steps[-1] != max_episode_steps:
        steps.append(max_episode_steps)
    required_frames = 1 + len(steps)  # Include the reset frame at t=0.
    if video_max_frames < required_frames:
        raise ValueError(
            f"validation_video_max_frames={video_max_frames} truncates the match; "
            f"at least {required_frames} frames are required"
        )
    return frozenset(steps)


def encode_validation_gif(video: torch.Tensor, fps: int) -> bytes:
    """Encode a TensorBoard-compatible GIF with explicit, non-zero frame timing."""
    if video.dtype != torch.uint8 or video.ndim != 5 or video.shape[0] != 1 or video.shape[2] not in (1, 3, 4):
        raise ValueError("video must be uint8 with shape (1, T, C, H, W) and 1, 3, or 4 channels")
    if fps <= 0:
        raise ValueError("video fps must be positive")
    frame_duration_ms = max(10, round(1000 / fps))
    array = video[0].permute(0, 2, 3, 1).contiguous().cpu().numpy()
    frames = [Image.fromarray(frame.squeeze(-1) if frame.shape[-1] == 1 else frame) for frame in array]
    if not frames:
        raise ValueError("video must contain at least one frame")
    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return output.getvalue()


def add_tensorboard_video(
    writer: SummaryWriter,
    tag: str,
    video: torch.Tensor,
    global_step: int,
    *,
    fps: int,
) -> None:
    """Write an animated GIF directly, avoiding MoviePy/ImageIO timing incompatibilities."""
    from tensorboard.compat.proto.summary_pb2 import Summary

    encoded = encode_validation_gif(video, fps)
    _, _, channels, height, width = video.shape
    image = Summary.Image(
        height=height,
        width=width,
        colorspace=channels,
        encoded_image_string=encoded,
    )
    summary = Summary(value=[Summary.Value(tag=tag, image=image)])
    writer._get_file_writer().add_summary(summary, global_step)


def _yaw_from_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def render_top_down(
    state: ArenaState,
    config: ArenaConfig,
    *,
    arena_index: int = 0,
    width: int = 480,
    height: int = 320,
) -> np.ndarray:
    """Render a lightweight validation view without adding a deployment camera."""
    if not 0 <= arena_index < state.batch_size:
        raise IndexError("arena_index is outside the state batch")
    image = Image.new("RGB", (width, height), (25, 27, 31))
    draw = ImageDraw.Draw(image)
    margin = 8
    draw.rectangle(
        (margin, margin, width - margin - 1, height - margin - 1),
        fill=(225, 222, 210),
        outline=(8, 8, 8),
        width=3,
    )
    board_x, board_y = config.board.size_m
    usable_x, usable_y = width - 2 * margin, height - 2 * margin

    def to_pixel(x: float, y: float) -> tuple[float, float]:
        return margin + (x / board_x + 0.5) * usable_x, margin + (0.5 - y / board_y) * usable_y

    poses = state.position[arena_index].detach().cpu().numpy()
    quaternions = state.quaternion[arena_index].detach().cpu().numpy()
    colors = ((205, 50, 45), (40, 85, 205))
    # Preserve true centers/headings but make the tiny 40 mm footprint visible in TensorBoard.
    display_length = max(config.robot.chassis_size_m[0], 0.08)
    display_width = max(config.robot.chassis_size_m[1], 0.08)
    for side in (RED, BLUE):
        x, y = float(poses[side, 0]), float(poses[side, 1])
        yaw = _yaw_from_xyzw(quaternions[side])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        corners: list[tuple[float, float]] = []
        for local_x, local_y in (
            (display_length / 2, display_width / 2),
            (display_length / 2, -display_width / 2),
            (-display_length / 2, -display_width / 2),
            (-display_length / 2, display_width / 2),
        ):
            world_x = x + cosine * local_x - sine * local_y
            world_y = y + sine * local_x + cosine * local_y
            corners.append(to_pixel(world_x, world_y))
        draw.polygon(corners, fill=colors[side], outline=(255, 255, 255))
        center = to_pixel(x, y)
        nose = to_pixel(x + cosine * display_length, y + sine * display_length)
        draw.line((center, nose), fill=(255, 245, 100), width=2)
    draw.text((14, 13), f"t={float(state.time_remaining_s[arena_index]):5.2f}s", fill=(15, 15, 15))
    red_idle, blue_idle = (float(value) for value in state.stationary_time_s[arena_index])
    draw.text((14, 29), f"idle R={red_idle:4.1f}s B={blue_idle:4.1f}s", fill=(15, 15, 15))
    return np.asarray(image).copy()


@dataclass(frozen=True, slots=True)
class ValidationResult:
    metrics: dict[str, float]
    video: torch.Tensor | None  # (1, T, C, H, W), uint8 CPU


class LeaderValidation:
    def __init__(
        self,
        arena: NewtonSumoArena,
        arena_config: ArenaConfig,
        reward_spec: RewardSpec,
        *,
        seed: int,
        video_fps: int,
        video_max_frames: int,
    ) -> None:
        if video_fps <= 0 or video_max_frames <= 0:
            raise ValueError("validation video settings must be positive")
        self.arena = arena
        self.arena_config = arena_config
        self.reward_spec = reward_spec
        self.seed = seed
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames
        self.capture_steps = validation_capture_steps(
            arena_config.physics.max_episode_steps,
            arena_config.physics.control_hz,
            video_fps,
            video_max_frames,
        )

    @torch.no_grad()
    def run(
        self,
        red_model: TransplantableCPOActorCritic,
        blue_model: TransplantableCPOActorCritic,
    ) -> ValidationResult:
        red_was_training, blue_was_training = red_model.training, blue_model.training
        red_model.eval()
        blue_model.eval()
        self.arena.generator.manual_seed(self.seed)
        state = self.arena.reset()
        count = self.arena.world_count
        device = state.position.device
        active = torch.ones(count, dtype=torch.bool, device=device)
        returns = torch.zeros(count, 2, dtype=torch.float32, device=device)
        component_returns: dict[str, torch.Tensor] = {}
        lengths = torch.zeros(count, dtype=torch.int32, device=device)
        outcomes = torch.full((count,), DRAW, dtype=torch.int64, device=device)
        ending_counts = torch.zeros(3, dtype=torch.int64, device=device)
        frames = [render_top_down(state, self.arena_config)]

        for step in range(self.arena_config.physics.max_episode_steps):
            red_observation = build_teacher_observation(state, self.arena.domain, RED).values
            blue_observation = build_teacher_observation(state, self.arena.domain, BLUE).values
            red_action = red_model.leader(red_observation).distribution.mean
            blue_action = blue_model.leader(blue_observation).distribution.mean
            actions = torch.stack((red_action, blue_action), dim=1)
            actions = torch.where(active[:, None, None], actions, torch.zeros_like(actions))
            physics = self.arena.step(actions)
            transition = ArenaTransition(
                previous=state,
                current=physics.state,
                terminated=physics.terminated,
                truncated=physics.truncated,
                winner=physics.winner,
            )
            reward_result = evaluate_reward(self.reward_spec, transition)
            reward = reward_result.total
            returns += reward * active.unsqueeze(-1)
            for name, value in reward_result.components.items():
                component_returns[name] = component_returns.get(
                    name, torch.zeros_like(returns)
                ) + value * active.unsqueeze(-1)
            lengths += active
            newly_done = active & physics.done
            outcomes = torch.where(newly_done, physics.winner, outcomes)
            ending_counts[0] += (newly_done & physics.ring_out).sum()
            ending_counts[1] += (newly_done & physics.inactivity).sum()
            ending_counts[2] += (newly_done & physics.numerical_failure).sum()
            control_step = step + 1
            scheduled = control_step in self.capture_steps
            terminal_frame = bool(newly_done[0])
            if (scheduled or terminal_frame) and len(frames) < self.video_max_frames and bool(active[0]):
                frames.append(render_top_down(physics.state, self.arena_config))
            active &= ~newly_done
            if not bool(active.any()):
                state = physics.state
                break
            state = self.arena.reset(newly_done) if bool(newly_done.any()) else physics.state

        if red_was_training:
            red_model.train()
        if blue_was_training:
            blue_model.train()
        video = None
        if frames:
            video_array = np.stack(frames)
            video = torch.from_numpy(video_array).permute(0, 3, 1, 2).unsqueeze(0)
        metrics = {
            "red_win_rate": float((outcomes == RED).float().mean().item()),
            "blue_win_rate": float((outcomes == BLUE).float().mean().item()),
            "draw_rate": float((outcomes == DRAW).float().mean().item()),
            "red_return": float(returns[:, RED].mean().item()),
            "blue_return": float(returns[:, BLUE].mean().item()),
            "episode_length": float(lengths.float().mean().item()),
            "video_frames": float(len(frames)),
            "ring_outs": float(ending_counts[0].item()),
            "inactivity_endings": float(ending_counts[1].item()),
            "numerical_failures": float(ending_counts[2].item()),
        }
        for name, value in component_returns.items():
            metrics[f"red_reward/{name}"] = float(value[:, RED].mean().item())
            metrics[f"blue_reward/{name}"] = float(value[:, BLUE].mean().item())
        return ValidationResult(metrics=metrics, video=video)
