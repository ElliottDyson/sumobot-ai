from __future__ import annotations

from sumobot_ai.curriculum import ArenaCurriculum, CurriculumConfig, OpponentLeague, OpponentSnapshot
from sumobot_ai.training.matchmaking import BootstrapMatchmaker, MemberCurriculumMatchmaker


def curriculum_config() -> CurriculumConfig:
    return CurriculumConfig(window_size=4, min_results=2, promotion_win_rate=0.75, demotion_win_rate=0.25, max_level=3)


def snapshot(level: int) -> OpponentSnapshot:
    return OpponentSnapshot(
        snapshot_id=f"opponent-{level}",
        population="bootstrap_blue",
        checkpoint_uri=f"checkpoints/opponent-{level}.pt",
        checkpoint_sha256=hex(level + 1)[2:] * 64,
        rating=1000.0 + level * 100,
        level=level,
        validation_score=0.5 + level * 0.1,
    )


def test_per_arena_promotion_and_demotion() -> None:
    curriculum = ArenaCurriculum(2, curriculum_config(), initial_level=1)
    curriculum.record_outcome(0, "win")
    promoted = curriculum.record_outcome(0, "win")
    curriculum.record_outcome(1, "loss")
    demoted = curriculum.record_outcome(1, "loss")
    assert promoted.promoted and curriculum.levels[0] == 2
    assert demoted.demoted and curriculum.levels[1] == 0


def test_bootstrap_pairs_two_live_populations_with_full_pair_coverage() -> None:
    assignments = BootstrapMatchmaker(6).assign(36)
    pairs = {(item.red.policy_id, item.blue.policy_id) for item in assignments}
    assert len(pairs) == 36
    assert all(item.red.collect_on_policy and item.blue.collect_on_policy for item in assignments)
    assert all(item.phase == "bootstrap_dual_cpo" and item.learner_side is None for item in assignments)
    assert {item.red.population for item in assignments} == {"bootstrap_red"}
    assert {item.blue.population for item in assignments} == {"bootstrap_blue"}


def test_member_session_has_one_live_population_and_one_frozen_opponent() -> None:
    curriculum = ArenaCurriculum(4, curriculum_config())
    league = OpponentLeague()
    for level in range(3):
        league.add(snapshot(level))
    assignments = MemberCurriculumMatchmaker(6, "member_alice", curriculum, league, seed=4).assign()
    for assignment in assignments:
        refs = (assignment.red, assignment.blue)
        live = [ref for ref in refs if ref.kind == "live_cpo"]
        frozen = [ref for ref in refs if ref.kind == "frozen_snapshot"]
        assert len(live) == len(frozen) == 1
        assert live[0].population == "member_alice" and live[0].collect_on_policy
        assert not frozen[0].collect_on_policy
        assert assignment.phase == "member_single_cpo_vs_curriculum"
    assert {assignment.learner_side for assignment in assignments} == {0, 1}
