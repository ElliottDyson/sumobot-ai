from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch.distributions import Independent, Normal

from ..contracts import ACTION_DIM


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    schema_version: int
    arena_name: str
    arena_config_sha256: str
    reward_spec_sha256: str
    reward_code_sha256: str
    teacher_checkpoint_sha256: str
    teacher_population: str
    teacher_leader_id: int
    student_observation_version: int
    privileged_observation_version: int
    newton_commit: str
    cap_dreamer_commit: str

    def __post_init__(self) -> None:
        if self.schema_version != ReplayContract.VERSION:
            raise ValueError(f"dataset schema version must be {ReplayContract.VERSION}")
        for name in (
            "arena_config_sha256",
            "reward_spec_sha256",
            "reward_code_sha256",
            "teacher_checkpoint_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.teacher_leader_id != 0:
            raise ValueError("CPO leader ID is fixed to 0 in schema version 2")

    @property
    def digest(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class ReplayContract:
    """Validator for current-observation labels and executed-action world-model transitions."""

    VERSION = 2
    REQUIRED_FIELDS = frozenset(
        {
            "student_obs",
            "privileged_obs",
            "action_exec",
            "action_student_raw",
            "teacher_mean",
            "teacher_scale",
            "teacher_action_mode",
            "teacher_value_raw",
            "teacher_feature",
            "teacher_valid",
            "policy_id_executed",
            "intervention",
            "reward_env",
            "reward_cpo_diversity",
            "score_outcome",
            "is_first",
            "is_last",
            "is_terminal",
            "domain_parameters",
            "sensor_age_s",
            "sensor_valid",
            "timestamp_s",
            "episode",
        }
    )
    BOOL_FIELDS = frozenset({"teacher_valid", "intervention", "is_first", "is_last", "is_terminal", "sensor_valid"})
    INTEGER_FIELDS = frozenset({"policy_id_executed", "score_outcome", "episode"})
    ACTION_FIELDS = frozenset(
        {"action_exec", "action_student_raw", "teacher_mean", "teacher_scale", "teacher_action_mode"}
    )

    @classmethod
    def validate(cls, batch: Mapping[str, torch.Tensor], *, action_dim: int = ACTION_DIM) -> None:
        missing = cls.REQUIRED_FIELDS - batch.keys()
        if missing:
            raise ValueError(f"replay batch is missing fields: {sorted(missing)}")
        non_tensors = {name for name, value in batch.items() if not isinstance(value, torch.Tensor)}
        if non_tensors:
            raise TypeError(f"replay values must be tensors: {sorted(non_tensors)}")
        batch_sizes = {value.shape[0] for value in batch.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"replay fields have inconsistent leading dimensions: {sorted(batch_sizes)}")
        for name in cls.BOOL_FIELDS:
            if batch[name].dtype != torch.bool:
                raise ValueError(f"{name} must be boolean")
        for name in cls.INTEGER_FIELDS:
            if batch[name].dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError(f"{name} must be an integer tensor")
        for name in cls.ACTION_FIELDS:
            if batch[name].shape[-1] != action_dim:
                raise ValueError(f"{name} must have final dimension {action_dim}")
        for name, value in batch.items():
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN or infinity")
        if bool((batch["teacher_scale"] <= 0).any()):
            raise ValueError("teacher_scale must be strictly positive")
        if bool((batch["action_exec"].abs() > 1.0 + 1e-6).any()):
            raise ValueError("action_exec must contain the final normalized action in [-1, 1]")
        if bool((batch["teacher_action_mode"].abs() > 1.0 + 1e-6).any()):
            raise ValueError("teacher_action_mode must use the environment action range [-1, 1]")
        if bool((batch["is_terminal"] & ~batch["is_last"]).any()):
            raise ValueError("every terminal transition must also be last")
        if bool((batch["teacher_valid"] & (batch["teacher_scale"].amin(dim=-1, keepdim=True) <= 0)).any()):
            raise ValueError("valid teacher labels require positive scale")
        valid_outcomes = (batch["score_outcome"] >= -2) & (batch["score_outcome"] <= 1)
        if not bool(valid_outcomes.all()):
            raise ValueError("score_outcome must use -2 ongoing, -1 draw, 0 red, or 1 blue")


def masked_teacher_kl(
    student: Independent,
    teacher_mean: torch.Tensor,
    teacher_scale: torch.Tensor,
    teacher_valid: torch.Tensor,
) -> torch.Tensor:
    """Mean KL(teacher || student), matching the CAP bounded-normal parameter space."""
    teacher = Independent(Normal(teacher_mean.float(), teacher_scale.float()), 1)
    kl = torch.distributions.kl_divergence(teacher, student)
    mask = teacher_valid.squeeze(-1) if teacher_valid.ndim == kl.ndim + 1 else teacher_valid
    if mask.shape != kl.shape:
        raise ValueError(f"teacher_valid shape {tuple(mask.shape)} does not match KL shape {tuple(kl.shape)}")
    weights = mask.to(kl.dtype)
    return (kl * weights).sum() / weights.sum().clamp_min(1.0)
