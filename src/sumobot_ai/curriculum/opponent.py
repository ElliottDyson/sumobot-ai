from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    version: int = 1
    window_size: int = 24
    min_results: int = 12
    promotion_win_rate: float = 0.70
    demotion_win_rate: float = 0.30
    easier_probability: float = 0.15
    current_probability: float = 0.70
    harder_probability: float = 0.15
    max_level: int = 12
    draw_score: float = 0.5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CurriculumConfig:
        probabilities = data.get("neighbor_probabilities", {})
        if not isinstance(probabilities, Mapping):
            raise ValueError("neighbor_probabilities must be a mapping")
        result = cls(
            version=int(data.get("version", 1)),
            window_size=int(data.get("window_size", 24)),
            min_results=int(data.get("min_results", 12)),
            promotion_win_rate=float(data.get("promotion_win_rate", 0.70)),
            demotion_win_rate=float(data.get("demotion_win_rate", 0.30)),
            easier_probability=float(probabilities.get("easier", 0.15)),
            current_probability=float(probabilities.get("current", 0.70)),
            harder_probability=float(probabilities.get("harder", 0.15)),
            max_level=int(data.get("max_level", 12)),
            draw_score=float(data.get("draw_score", 0.5)),
        )
        if result.version != 1:
            raise ValueError(f"unsupported curriculum version: {result.version}")
        if result.window_size <= 0 or not 0 < result.min_results <= result.window_size:
            raise ValueError("curriculum requires 0 < min_results <= window_size")
        if not 0 <= result.demotion_win_rate < result.promotion_win_rate <= 1:
            raise ValueError("curriculum thresholds must satisfy 0 <= demotion < promotion <= 1")
        probabilities_tuple = (
            result.easier_probability,
            result.current_probability,
            result.harder_probability,
        )
        if min(probabilities_tuple) < 0 or abs(sum(probabilities_tuple) - 1.0) > 1e-6:
            raise ValueError("neighbor probabilities must be non-negative and sum to 1")
        if result.max_level < 0 or not 0 <= result.draw_score <= 1:
            raise ValueError("max_level and draw_score are invalid")
        return result

    @classmethod
    def load(cls, path: str | Path) -> CurriculumConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, Mapping):
            raise ValueError("curriculum config root must be a mapping")
        return cls.from_mapping(data)


@dataclass(frozen=True, slots=True)
class OpponentSnapshot:
    snapshot_id: str
    population: str
    checkpoint_uri: str
    checkpoint_sha256: str
    rating: float
    level: int
    validation_score: float
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.population or not self.checkpoint_uri:
            raise ValueError("snapshot identity, population, and checkpoint URI are required")
        if len(self.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        if self.level < 0:
            raise ValueError("snapshot level cannot be negative")
        if not self.frozen:
            raise ValueError("curriculum snapshots must be immutable")


class OpponentLeague:
    """Immutable checkpoint catalog grouped by validated difficulty level."""

    def __init__(self) -> None:
        self._by_id: dict[str, OpponentSnapshot] = {}
        self._by_level: dict[int, list[OpponentSnapshot]] = {}

    def add(self, snapshot: OpponentSnapshot) -> None:
        if snapshot.snapshot_id in self._by_id:
            raise ValueError(f"duplicate snapshot_id: {snapshot.snapshot_id}")
        self._by_id[snapshot.snapshot_id] = snapshot
        level = self._by_level.setdefault(snapshot.level, [])
        level.append(snapshot)
        level.sort(key=lambda item: (item.rating, item.validation_score, item.snapshot_id))

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_level))

    def at_level(self, level: int) -> tuple[OpponentSnapshot, ...]:
        return tuple(self._by_level.get(level, ()))

    def nearest_populated_level(self, target: int) -> int:
        if not self._by_level:
            raise RuntimeError("opponent league is empty")
        return min(self._by_level, key=lambda level: (abs(level - target), level))

    def sample(self, target_level: int, rng: random.Random) -> OpponentSnapshot:
        level = self.nearest_populated_level(target_level)
        return rng.choice(self._by_level[level])

    def __len__(self) -> int:
        return len(self._by_id)


@dataclass(frozen=True, slots=True)
class CurriculumUpdate:
    arena_id: int
    old_level: int
    new_level: int
    window_score: float
    promoted: bool
    demoted: bool


class ArenaCurriculum:
    """Extreme-Parkour-style per-arena progression where levels select models."""

    def __init__(self, num_arenas: int, config: CurriculumConfig, *, initial_level: int = 0) -> None:
        if num_arenas <= 0:
            raise ValueError("num_arenas must be positive")
        if not 0 <= initial_level <= config.max_level:
            raise ValueError("initial_level is outside curriculum bounds")
        self.config = config
        self.levels = [initial_level] * num_arenas
        self._history = [deque(maxlen=config.window_size) for _ in range(num_arenas)]

    @property
    def num_arenas(self) -> int:
        return len(self.levels)

    def record_score(self, arena_id: int, score: float) -> CurriculumUpdate:
        if not 0 <= arena_id < self.num_arenas:
            raise IndexError(f"arena_id {arena_id} is out of range")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be in [0, 1]")
        history = self._history[arena_id]
        history.append(float(score))
        mean = sum(history) / len(history)
        old = self.levels[arena_id]
        new = old
        if len(history) >= self.config.min_results:
            if mean >= self.config.promotion_win_rate and old < self.config.max_level:
                new = old + 1
            elif mean <= self.config.demotion_win_rate and old > 0:
                new = old - 1
            if new != old:
                history.clear()
                self.levels[arena_id] = new
        return CurriculumUpdate(
            arena_id=arena_id,
            old_level=old,
            new_level=new,
            window_score=mean,
            promoted=new > old,
            demoted=new < old,
        )

    def record_outcome(self, arena_id: int, outcome: str) -> CurriculumUpdate:
        scores = {"win": 1.0, "draw": self.config.draw_score, "loss": 0.0}
        try:
            return self.record_score(arena_id, scores[outcome])
        except KeyError as error:
            raise ValueError("outcome must be 'win', 'draw', or 'loss'") from error

    def sample_target_level(self, arena_id: int, rng: random.Random) -> int:
        if not 0 <= arena_id < self.num_arenas:
            raise IndexError(f"arena_id {arena_id} is out of range")
        current = self.levels[arena_id]
        candidates = (max(0, current - 1), current, min(self.config.max_level, current + 1))
        weights = (
            self.config.easier_probability,
            self.config.current_probability,
            self.config.harder_probability,
        )
        return rng.choices(candidates, weights=weights, k=1)[0]

    def select_opponent(self, arena_id: int, league: OpponentLeague, rng: random.Random) -> OpponentSnapshot:
        return league.sample(self.sample_target_level(arena_id, rng), rng)
