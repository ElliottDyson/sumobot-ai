from __future__ import annotations

import random
from dataclasses import dataclass

from ..curriculum import ArenaCurriculum, OpponentLeague


@dataclass(frozen=True, slots=True)
class PolicyRef:
    kind: str
    population: str
    policy_id: int | None = None
    snapshot_id: str | None = None
    checkpoint_uri: str | None = None
    collect_on_policy: bool = False

    def __post_init__(self) -> None:
        if self.kind == "live_cpo":
            if self.policy_id is None or self.snapshot_id is not None:
                raise ValueError("live_cpo requires policy_id and forbids snapshot_id")
        elif self.kind == "frozen_snapshot":
            if self.snapshot_id is None or self.checkpoint_uri is None or self.policy_id is not None:
                raise ValueError("frozen_snapshot requires snapshot/checkpoint and forbids policy_id")
            if self.collect_on_policy:
                raise ValueError("a frozen opponent cannot collect on-policy learner data")
        else:
            raise ValueError(f"unknown policy kind: {self.kind!r}")


@dataclass(frozen=True, slots=True)
class MatchAssignment:
    arena_id: int
    red: PolicyRef
    blue: PolicyRef
    phase: str
    learner_side: int | None
    curriculum_level: int | None


class BootstrapMatchmaker:
    """Pair two simultaneously learning, completely separate CPO populations."""

    def __init__(self, population_size: int, *, red_name: str = "bootstrap_red", blue_name: str = "bootstrap_blue"):
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        self.population_size = population_size
        self.red_name = red_name
        self.blue_name = blue_name

    def assign(self, num_arenas: int, *, rollout_index: int = 0) -> list[MatchAssignment]:
        if num_arenas <= 0 or rollout_index < 0:
            raise ValueError("num_arenas must be positive and rollout_index non-negative")
        assignments: list[MatchAssignment] = []
        n = self.population_size
        start = rollout_index * num_arenas
        for arena_id in range(num_arenas):
            sequence_index = start + arena_id
            red_id = sequence_index % n
            latin_row = (sequence_index // n) % n
            blue_id = (red_id + latin_row) % n
            assignments.append(
                MatchAssignment(
                    arena_id=arena_id,
                    red=PolicyRef("live_cpo", self.red_name, policy_id=red_id, collect_on_policy=True),
                    blue=PolicyRef("live_cpo", self.blue_name, policy_id=blue_id, collect_on_policy=True),
                    phase="bootstrap_dual_cpo",
                    learner_side=None,
                    curriculum_level=None,
                )
            )
        return assignments


class MemberCurriculumMatchmaker:
    """Pair one member CPO population against frozen curriculum inference models."""

    def __init__(
        self,
        population_size: int,
        learner_population: str,
        curriculum: ArenaCurriculum,
        league: OpponentLeague,
        *,
        seed: int = 0,
        alternate_sides: bool = True,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if len(league) == 0:
            raise ValueError("member training needs a non-empty frozen opponent league")
        self.population_size = population_size
        self.learner_population = learner_population
        self.curriculum = curriculum
        self.league = league
        self.rng = random.Random(seed)
        self.alternate_sides = alternate_sides

    def assign(self, *, rollout_index: int = 0) -> list[MatchAssignment]:
        assignments: list[MatchAssignment] = []
        for arena_id in range(self.curriculum.num_arenas):
            policy_id = (rollout_index * self.curriculum.num_arenas + arena_id) % self.population_size
            learner = PolicyRef("live_cpo", self.learner_population, policy_id=policy_id, collect_on_policy=True)
            snapshot = self.curriculum.select_opponent(arena_id, self.league, self.rng)
            opponent = PolicyRef(
                "frozen_snapshot",
                snapshot.population,
                snapshot_id=snapshot.snapshot_id,
                checkpoint_uri=snapshot.checkpoint_uri,
            )
            learner_side = (rollout_index + arena_id) % 2 if self.alternate_sides else 0
            red, blue = (learner, opponent) if learner_side == 0 else (opponent, learner)
            assignments.append(
                MatchAssignment(
                    arena_id=arena_id,
                    red=red,
                    blue=blue,
                    phase="member_single_cpo_vs_curriculum",
                    learner_side=learner_side,
                    curriculum_level=snapshot.level,
                )
            )
        return assignments
