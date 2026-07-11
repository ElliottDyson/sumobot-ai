from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from ..config import ArenaConfig
from ..contracts import ACTION_DIM
from ..observations import build_teacher_observation
from ..rewards import RewardSpec, evaluate_reward
from ..sim.newton_backend import NewtonSumoArena
from ..state import BLUE, DRAW, RED, ArenaState, ArenaTransition
from .cpo import CpoLossConfig, TransplantableCPOActorCritic, cpo_actor_loss
from .matchmaking import BootstrapMatchmaker
from .validation import LeaderValidation, add_tensorboard_video


@dataclass(frozen=True, slots=True)
class BootstrapTrainingConfig:
    version: int
    seed: int
    device: str
    num_envs: int
    total_environment_steps: int
    rollout_steps: int
    learning_rate: float
    update_epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    value_coefficient: float
    entropy_coefficient: float
    leader_awac_coefficient: float
    leader_awac_temperature: float
    leader_awac_max_weight: float
    max_grad_norm: float
    checkpoint_every_updates: int
    validation_every_updates: int
    validation_envs: int
    validation_seed: int
    validation_video_fps: int
    validation_video_max_frames: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BootstrapTrainingConfig:
        result = cls(**{field: data[field] for field in cls.__dataclass_fields__})
        if result.version != 1:
            raise ValueError(f"unsupported bootstrap training version: {result.version}")
        integer_positive = (
            result.num_envs,
            result.total_environment_steps,
            result.rollout_steps,
            result.update_epochs,
            result.minibatch_size,
            result.validation_envs,
            result.validation_video_fps,
            result.validation_video_max_frames,
        )
        if min(integer_positive) <= 0:
            raise ValueError("bootstrap sizes, steps, epochs, and video settings must be positive")
        if result.minibatch_size > result.num_envs * result.rollout_steps:
            raise ValueError("minibatch_size cannot exceed one flattened rollout")
        if not 0 < result.gamma <= 1 or not 0 < result.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")
        if result.learning_rate <= 0 or result.max_grad_norm <= 0:
            raise ValueError("learning rate and max gradient norm must be positive")
        if min(result.value_coefficient, result.entropy_coefficient, result.leader_awac_coefficient) < 0:
            raise ValueError("loss coefficients cannot be negative")
        if result.leader_awac_temperature <= 0 or result.leader_awac_max_weight <= 0:
            raise ValueError("leader AWAC parameters must be positive")
        if result.checkpoint_every_updates < 0 or result.validation_every_updates < 0:
            raise ValueError("checkpoint and validation intervals cannot be negative")
        return result

    @classmethod
    def load(cls, path: str | Path) -> BootstrapTrainingConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, Mapping):
            raise ValueError("bootstrap training config root must be a mapping")
        return cls.from_mapping(data)


@dataclass(frozen=True, slots=True)
class PopulationBatch:
    observation: torch.Tensor  # (T, N, O)
    policy_id: torch.Tensor  # (T, N)
    action_raw: torch.Tensor  # (T, N, A)
    old_log_prob: torch.Tensor  # (T, N)
    value: torch.Tensor  # (T, N)
    reward: torch.Tensor  # (T, N)
    done: torch.Tensor  # (T, N)
    advantage: torch.Tensor  # (T, N)
    returns: torch.Tensor  # (T, N)

    def flatten(self) -> dict[str, torch.Tensor]:
        time, environments = self.policy_id.shape
        batch = time * environments
        return {
            "observation": self.observation.reshape(batch, self.observation.shape[-1]),
            "policy_id": self.policy_id.reshape(batch),
            "action_raw": self.action_raw.reshape(batch, self.action_raw.shape[-1]),
            "old_log_prob": self.old_log_prob.reshape(batch),
            "value": self.value.reshape(batch),
            "reward": self.reward.reshape(batch),
            "done": self.done.reshape(batch),
            "advantage": self.advantage.reshape(batch),
            "returns": self.returns.reshape(batch),
        }


def compute_gae(
    reward: torch.Tensor,
    value: torch.Tensor,
    done: torch.Tensor,
    last_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reward.shape != value.shape or reward.shape != done.shape:
        raise ValueError("reward, value, and done must have identical (T, N) shapes")
    if last_value.shape != reward.shape[1:]:
        raise ValueError("last_value must have shape (N,)")
    advantage = torch.zeros_like(reward)
    accumulator = torch.zeros_like(last_value)
    next_value = last_value
    for step in reversed(range(reward.shape[0])):
        nonterminal = 1.0 - done[step].to(reward.dtype)
        delta = reward[step] + gamma * next_value * nonterminal - value[step]
        accumulator = delta + gamma * gae_lambda * nonterminal * accumulator
        advantage[step] = accumulator
        next_value = value[step]
    return advantage, advantage + value


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class BootstrapTrainer:
    def __init__(
        self,
        arena_config: ArenaConfig,
        reward_spec: RewardSpec,
        training_config: BootstrapTrainingConfig,
        cpo_config: Mapping[str, Any],
        logdir: str | Path,
        *,
        resume: bool = True,
    ) -> None:
        self.arena_config = arena_config
        self.reward_spec = reward_spec
        reward_spec.validate_for_episode(arena_config.physics.episode_seconds)
        self.config = training_config
        self.cpo_config = dict(cpo_config)
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.logdir / "checkpoint.pt"
        self.writer = SummaryWriter(log_dir=str(self.logdir), flush_secs=10)
        self.device = torch.device(training_config.device)
        self._set_seeds(training_config.seed)
        if float(cpo_config.get("diversity_reward_coefficient", 0.0)) != 0.0:
            raise ValueError("the bootstrap runner currently requires diversity_reward_coefficient=0.0")

        self.arena = NewtonSumoArena(
            arena_config,
            training_config.num_envs,
            device=training_config.device,
            seed=training_config.seed,
        )
        initial_state = self.arena.snapshot()
        observation_dim = build_teacher_observation(initial_state, self.arena.domain, RED).values.shape[-1]
        self.population_size = int(cpo_config["population_size"])
        self.red_model = self._make_model(observation_dim)
        self.blue_model = self._make_model(observation_dim)
        if {id(parameter) for parameter in self.red_model.parameters()} & {
            id(parameter) for parameter in self.blue_model.parameters()
        }:
            raise RuntimeError("bootstrap CPO populations unexpectedly share parameters")
        self.red_optimizer = torch.optim.Adam(self.red_model.parameters(), lr=training_config.learning_rate, eps=1e-5)
        self.blue_optimizer = torch.optim.Adam(self.blue_model.parameters(), lr=training_config.learning_rate, eps=1e-5)
        self.matchmaker = BootstrapMatchmaker(self.population_size)
        self.cpo_loss_config = CpoLossConfig(
            ppo_clip=float(cpo_config["ppo_clip"]),
            follower_kl_coefficient=float(cpo_config["lambda_follower_kl"]),
            awac_temperature=float(cpo_config["lambda_awac"]),
            awac_max_weight=float(cpo_config["awac_max_weight"]),
            awac_scale=0.0,
        )
        self.validation: LeaderValidation | None = None
        if training_config.validation_every_updates > 0:
            validation_arena = NewtonSumoArena(
                arena_config,
                training_config.validation_envs,
                device=training_config.device,
                seed=training_config.validation_seed,
            )
            self.validation = LeaderValidation(
                validation_arena,
                arena_config,
                reward_spec,
                seed=training_config.validation_seed,
                video_fps=training_config.validation_video_fps,
                video_max_frames=training_config.validation_video_max_frames,
            )

        self.global_step = 0
        self.update_index = 0
        self._stop_requested = False
        self._write_run_metadata(observation_dim)
        if resume and self.checkpoint_path.exists():
            self._load_checkpoint()
        self._install_signal_handlers()

    @staticmethod
    def load_cpo_config(path: str | Path) -> Mapping[str, Any]:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, Mapping) or int(data.get("version", -1)) != 1:
            raise ValueError("CPO config must be a version-1 mapping")
        return data

    def _set_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_model(self, observation_dim: int) -> TransplantableCPOActorCritic:
        return TransplantableCPOActorCritic(
            observation_dim=observation_dim,
            action_dim=ACTION_DIM,
            population_size=self.population_size,
            policy_id_dim=int(self.cpo_config["policy_id_dim"]),
            frontend_units=tuple(int(value) for value in self.cpo_config["teacher_frontend_units"]),
            bottleneck_dim=int(self.cpo_config["actor_bottleneck"]),
            suffix_layers=int(self.cpo_config["actor_suffix_layers"]),
            min_std=float(self.cpo_config["actor_min_std"]),
            max_std=float(self.cpo_config["actor_max_std"]),
        ).to(self.device)

    def _write_run_metadata(self, observation_dim: int) -> None:
        metadata = {
            "arena": self.arena_config.name,
            "reward": self.reward_spec.canonical_dict(),
            "reward_spec_sha256": self.reward_spec.digest,
            "training": asdict(self.config),
            "cpo": self.cpo_config,
            "teacher_observation_dim": observation_dim,
            "action_dim": ACTION_DIM,
        }
        path = self.logdir / "run_config.json"
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        self.writer.add_text("run/config", f"```json\n{json.dumps(metadata, indent=2, sort_keys=True)}\n```", 0)

    def _install_signal_handlers(self) -> None:
        def request_stop(signum: int, _frame: Any) -> None:
            print(f"received {signal.Signals(signum).name}; checkpointing after the current update", flush=True)
            self._stop_requested = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, request_stop)

    def _save_checkpoint(self) -> None:
        checkpoint = {
            "version": 1,
            "global_step": self.global_step,
            "update_index": self.update_index,
            "red_model": self.red_model.state_dict(),
            "blue_model": self.blue_model.state_dict(),
            "red_optimizer": self.red_optimizer.state_dict(),
            "blue_optimizer": self.blue_optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        temporary = self.checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, self.checkpoint_path)
        (self.logdir / "checkpoint.meta.json").write_text(
            json.dumps({"global_step": self.global_step, "update_index": self.update_index}), encoding="utf-8"
        )

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if int(checkpoint.get("version", -1)) != 1:
            raise ValueError("unsupported bootstrap checkpoint version")
        self.red_model.load_state_dict(checkpoint["red_model"])
        self.blue_model.load_state_dict(checkpoint["blue_model"])
        self.red_optimizer.load_state_dict(checkpoint["red_optimizer"])
        self.blue_optimizer.load_state_dict(checkpoint["blue_optimizer"])
        self.global_step = int(checkpoint["global_step"])
        self.update_index = int(checkpoint["update_index"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        print(f"resumed bootstrap training at step {self.global_step}, update {self.update_index}", flush=True)

    def _policy_ids(self) -> tuple[torch.Tensor, torch.Tensor]:
        assignments = self.matchmaker.assign(self.config.num_envs, rollout_index=self.update_index)
        red = torch.tensor(
            [assignment.red.policy_id for assignment in assignments], device=self.device, dtype=torch.long
        )
        blue = torch.tensor(
            [assignment.blue.policy_id for assignment in assignments], device=self.device, dtype=torch.long
        )
        return red, blue

    def _collect_rollout(
        self, state: ArenaState
    ) -> tuple[PopulationBatch, PopulationBatch, ArenaState, dict[str, float]]:
        red_policy_id, blue_policy_id = self._policy_ids()
        red_storage: dict[str, list[torch.Tensor]] = {
            key: [] for key in ("observation", "policy_id", "action", "log_prob", "value", "reward", "done")
        }
        blue_storage = {key: [] for key in red_storage}
        episode_return = torch.zeros(self.config.num_envs, 2, device=self.device)
        episode_length = torch.zeros(self.config.num_envs, dtype=torch.int32, device=self.device)
        completed_returns: list[torch.Tensor] = []
        completed_lengths: list[torch.Tensor] = []
        wins = torch.zeros(3, dtype=torch.int64, device=self.device)  # red, blue, draw
        endings = torch.zeros(3, dtype=torch.int64, device=self.device)  # ring-out, inactivity, numerical failure
        component_sums: dict[str, torch.Tensor] = {}

        for _ in range(self.config.rollout_steps):
            with torch.no_grad():
                red_observation = build_teacher_observation(state, self.arena.domain, RED).values
                blue_observation = build_teacher_observation(state, self.arena.domain, BLUE).values
                red_output = self.red_model(red_observation, red_policy_id)
                blue_output = self.blue_model(blue_observation, blue_policy_id)
                red_action = red_output.distribution.rsample()
                blue_action = blue_output.distribution.rsample()
                red_log_prob = red_output.distribution.log_prob(red_action)
                blue_log_prob = blue_output.distribution.log_prob(blue_action)
            physics = self.arena.step(torch.stack((red_action, blue_action), dim=1))
            transition = ArenaTransition(
                previous=state,
                current=physics.state,
                terminated=physics.terminated,
                truncated=physics.truncated,
                winner=physics.winner,
            )
            reward_result = evaluate_reward(self.reward_spec, transition)
            reward = reward_result.total
            for name, value in reward_result.components.items():
                component_sums[name] = component_sums.get(name, torch.zeros(2, device=self.device)) + value.sum(dim=0)
            for storage, observation, policy_id, action, log_prob, value, side in (
                (red_storage, red_observation, red_policy_id, red_action, red_log_prob, red_output.value, RED),
                (blue_storage, blue_observation, blue_policy_id, blue_action, blue_log_prob, blue_output.value, BLUE),
            ):
                storage["observation"].append(observation)
                storage["policy_id"].append(policy_id)
                storage["action"].append(action)
                storage["log_prob"].append(log_prob)
                storage["value"].append(value)
                storage["reward"].append(reward[:, side])
                storage["done"].append(physics.done)
            episode_return += reward
            episode_length += 1
            endings[0] += physics.ring_out.sum()
            endings[1] += physics.inactivity.sum()
            endings[2] += physics.numerical_failure.sum()
            if bool(physics.done.any()):
                indices = physics.done.nonzero(as_tuple=False).squeeze(-1)
                completed_returns.append(episode_return[indices].clone())
                completed_lengths.append(episode_length[indices].clone())
                wins[RED] += (physics.winner[indices] == RED).sum()
                wins[BLUE] += (physics.winner[indices] == BLUE).sum()
                wins[2] += (physics.winner[indices] == DRAW).sum()
                episode_return[indices] = 0.0
                episode_length[indices] = 0
                state = self.arena.reset(physics.done)
            else:
                state = physics.state

        with torch.no_grad():
            red_last = self.red_model(
                build_teacher_observation(state, self.arena.domain, RED).values, red_policy_id
            ).value
            blue_last = self.blue_model(
                build_teacher_observation(state, self.arena.domain, BLUE).values, blue_policy_id
            ).value

        def finalize(storage: dict[str, list[torch.Tensor]], last_value: torch.Tensor) -> PopulationBatch:
            observation = torch.stack(storage["observation"])
            policy_id = torch.stack(storage["policy_id"])
            action = torch.stack(storage["action"])
            log_prob = torch.stack(storage["log_prob"])
            value = torch.stack(storage["value"])
            reward = torch.stack(storage["reward"])
            done = torch.stack(storage["done"])
            advantage, returns = compute_gae(
                reward,
                value,
                done,
                last_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            return PopulationBatch(observation, policy_id, action, log_prob, value, reward, done, advantage, returns)

        episodes = int(wins.sum().item())
        summary = {
            "red_reward_mean": float(torch.stack(red_storage["reward"]).mean().item()),
            "blue_reward_mean": float(torch.stack(blue_storage["reward"]).mean().item()),
            "episodes": float(episodes),
            "red_win_rate": float(wins[RED].item() / max(episodes, 1)),
            "blue_win_rate": float(wins[BLUE].item() / max(episodes, 1)),
            "draw_rate": float(wins[2].item() / max(episodes, 1)),
            "ring_outs": float(endings[0].item()),
            "inactivity_endings": float(endings[1].item()),
            "numerical_failures": float(endings[2].item()),
        }
        if completed_returns:
            all_returns = torch.cat(completed_returns)
            all_lengths = torch.cat(completed_lengths)
            summary["completed_red_return"] = float(all_returns[:, RED].mean().item())
            summary["completed_blue_return"] = float(all_returns[:, BLUE].mean().item())
            summary["completed_length"] = float(all_lengths.float().mean().item())
        transition_count = self.config.num_envs * self.config.rollout_steps
        for name, value in component_sums.items():
            summary[f"reward/red/{name}"] = float((value[RED] / transition_count).item())
            summary[f"reward/blue/{name}"] = float((value[BLUE] / transition_count).item())
        return finalize(red_storage, red_last), finalize(blue_storage, blue_last), state, summary

    def _update_population(
        self,
        model: TransplantableCPOActorCritic,
        optimizer: torch.optim.Optimizer,
        rollout: PopulationBatch,
    ) -> dict[str, float]:
        batch = rollout.flatten()
        advantage = batch["advantage"]
        batch["advantage"] = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
        total = batch["policy_id"].shape[0]
        sums = {
            "loss": 0.0,
            "actor": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "leader_awac": 0.0,
            "approx_kl": 0.0,
            "follower_kl": 0.0,
            "grad_norm": 0.0,
        }
        updates = 0
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(total, device=self.device)
            for start in range(0, total, self.config.minibatch_size):
                index = permutation[start : start + self.config.minibatch_size]
                observation = batch["observation"][index]
                policy_id = batch["policy_id"][index]
                action = batch["action_raw"][index]
                old_log_prob = batch["old_log_prob"][index]
                minibatch_advantage = batch["advantage"][index]
                output = model(observation, policy_id)
                leader = model.leader(observation)
                new_log_prob = output.distribution.log_prob(action)
                leader_log_prob = leader.distribution.log_prob(action)
                leader_mask = policy_id == 0
                follower_mask = ~leader_mask
                empty = torch.zeros_like(leader_mask)
                actor_result = cpo_actor_loss(
                    old_log_prob=old_log_prob,
                    new_log_prob=new_log_prob,
                    leader_log_prob=leader_log_prob,
                    advantage=minibatch_advantage,
                    leader_online_mask=leader_mask,
                    follower_online_mask=follower_mask,
                    off_policy_mask=empty,
                    awac_mask=empty,
                    config=self.cpo_loss_config,
                )
                awac_weight = torch.exp(
                    (minibatch_advantage / self.config.leader_awac_temperature).clamp(
                        max=math.log(self.config.leader_awac_max_weight)
                    )
                ).detach()
                leader_awac = _masked_mean(-awac_weight * leader_log_prob, follower_mask)
                value_loss = 0.5 * (output.value - batch["returns"][index]).square().mean()
                entropy = output.distribution.entropy().mean()
                loss = (
                    actor_result.total
                    + self.config.leader_awac_coefficient * leader_awac
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    approximate_kl = (old_log_prob - new_log_prob).mean()
                    follower_kl_values = torch.distributions.kl_divergence(output.distribution, leader.distribution)
                    follower_kl = _masked_mean(follower_kl_values, follower_mask)
                values = {
                    "loss": loss,
                    "actor": actor_result.total,
                    "value": value_loss,
                    "entropy": entropy,
                    "leader_awac": leader_awac,
                    "approx_kl": approximate_kl,
                    "follower_kl": follower_kl,
                    "grad_norm": grad_norm,
                }
                for name, value in values.items():
                    sums[name] += float(value.detach().item())
                updates += 1
        return {name: value / max(updates, 1) for name, value in sums.items()}

    def _log_training(
        self,
        rollout_metrics: Mapping[str, float],
        red_metrics: Mapping[str, float],
        blue_metrics: Mapping[str, float],
        *,
        fps: float,
    ) -> None:
        for name, value in rollout_metrics.items():
            self.writer.add_scalar(f"rollout/{name}", value, self.global_step)
        for population, metrics in (("red", red_metrics), ("blue", blue_metrics)):
            for name, value in metrics.items():
                self.writer.add_scalar(f"train/{population}/{name}", value, self.global_step)
        self.writer.add_scalar("performance/environment_steps_per_second", fps, self.global_step)
        self.writer.add_scalar("performance/update", self.update_index, self.global_step)

    def _run_validation(self) -> None:
        if self.validation is None:
            return
        started = time.perf_counter()
        result = self.validation.run(self.red_model, self.blue_model)
        for name, value in result.metrics.items():
            self.writer.add_scalar(f"validation/{name}", value, self.global_step)
        if result.video is not None:
            add_tensorboard_video(
                self.writer,
                "validation/leader_match",
                result.video,
                self.global_step,
                fps=self.config.validation_video_fps,
            )
        self.writer.add_scalar("validation/wall_seconds", time.perf_counter() - started, self.global_step)
        self.writer.flush()

    def run(self) -> None:
        state = self.arena.reset()
        if self.validation is not None and self.global_step == 0:
            self._run_validation()
        try:
            while self.global_step < self.config.total_environment_steps and not self._stop_requested:
                started = time.perf_counter()
                red_rollout, blue_rollout, state, rollout_metrics = self._collect_rollout(state)
                self.global_step += self.config.num_envs * self.config.rollout_steps
                red_metrics = self._update_population(self.red_model, self.red_optimizer, red_rollout)
                blue_metrics = self._update_population(self.blue_model, self.blue_optimizer, blue_rollout)
                self.update_index += 1
                elapsed = time.perf_counter() - started
                fps = self.config.num_envs * self.config.rollout_steps / max(elapsed, 1e-6)
                self._log_training(rollout_metrics, red_metrics, blue_metrics, fps=fps)
                print(
                    json.dumps(
                        {
                            "step": self.global_step,
                            "update": self.update_index,
                            "fps": round(fps, 1),
                            "red_reward": round(rollout_metrics["red_reward_mean"], 5),
                            "blue_reward": round(rollout_metrics["blue_reward_mean"], 5),
                            "red_loss": round(red_metrics["loss"], 5),
                            "blue_loss": round(blue_metrics["loss"], 5),
                        }
                    ),
                    flush=True,
                )
                if (
                    self.config.checkpoint_every_updates
                    and self.update_index % self.config.checkpoint_every_updates == 0
                ):
                    self._save_checkpoint()
                if (
                    self.validation is not None
                    and self.config.validation_every_updates
                    and self.update_index % self.config.validation_every_updates == 0
                ):
                    self._run_validation()
            self._save_checkpoint()
            if self.global_step >= self.config.total_environment_steps:
                (self.logdir / "COMPLETED").write_text(str(self.global_step), encoding="utf-8")
        finally:
            self.writer.flush()
            self.writer.close()


def run_bootstrap_training(
    *,
    arena_config_path: str | Path,
    reward_spec_path: str | Path,
    training_config_path: str | Path,
    cpo_config_path: str | Path,
    logdir: str | Path,
    resume: bool = True,
    overrides: Mapping[str, Any] | None = None,
) -> None:
    training_data = asdict(BootstrapTrainingConfig.load(training_config_path))
    training_data.update(dict(overrides or {}))
    training_config = BootstrapTrainingConfig.from_mapping(training_data)
    trainer = BootstrapTrainer(
        ArenaConfig.load(arena_config_path),
        RewardSpec.load(reward_spec_path),
        training_config,
        BootstrapTrainer.load_cpo_config(cpo_config_path),
        logdir,
        resume=resume,
    )
    trainer.run()
