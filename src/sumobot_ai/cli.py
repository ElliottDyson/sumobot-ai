from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .curriculum import ArenaCurriculum, CurriculumConfig
from .doctor import run_checks
from .rewards import RewardSpec
from .rewards import terms as reward_terms


def _doctor(_args: argparse.Namespace) -> int:
    checks = run_checks()
    width = max(len(check.name) for check in checks)
    for check in checks:
        marker = "PASS" if check.ok else "FAIL"
        print(f"{marker:4}  {check.name:<{width}}  {check.detail}")
    return 0 if all(check.ok for check in checks) else 1


def _reward_lint(args: argparse.Namespace) -> int:
    source_paths = [Path(reward_terms.__file__).resolve()]
    for module in args.plugin:
        imported = importlib.import_module(module)
        if getattr(imported, "__file__", None):
            source_paths.append(Path(imported.__file__).resolve())
    spec = RewardSpec.load(args.path)
    from .config import ArenaConfig

    arena = ArenaConfig.load(args.arena)
    spec.validate_for_episode(arena.physics.episode_seconds)
    code_hash = hashlib.sha256()
    for path in sorted(set(source_paths)):
        code_hash.update(str(path.name).encode())
        code_hash.update(path.read_bytes())
    print(f"valid reward: {spec.name}")
    print(f"spec_sha256: {spec.digest}")
    print(f"code_sha256: {code_hash.hexdigest()}")
    print(f"clip: {spec.clip}")
    guidance_budget = spec.guidance_max_abs_per_second * arena.physics.episode_seconds
    print(f"win_reward: {spec.win_reward:g}")
    print(f"guidance_budget: ±{guidance_budget:g} per {arena.physics.episode_seconds:g}s match")
    for term in spec.terms:
        print(f"  {term.name}: weight={term.weight:g} params={dict(term.params)}")
    return 0


def _curriculum_demo(args: argparse.Namespace) -> int:
    config = CurriculumConfig.load(args.config)
    curriculum = ArenaCurriculum(3, config)
    updates = []
    for arena_id, outcome in ((0, "win"), (1, "draw"), (2, "loss")):
        for _ in range(config.min_results):
            update = curriculum.record_outcome(arena_id, outcome)
        updates.append(update)
    print(json.dumps({"levels": curriculum.levels, "updates": [asdict(update) for update in updates]}, indent=2))
    return 0


def _sim_smoke(args: argparse.Namespace) -> int:
    from .sim.newton_backend import run_smoke

    result = run_smoke(Path(args.config), world_count=args.worlds, steps=args.steps, device=args.device)
    print(json.dumps(result, indent=2))
    return 0


def _train_bootstrap(args: argparse.Namespace) -> int:
    from .training.bootstrap import run_bootstrap_training

    override_names = ("num_envs", "total_environment_steps", "rollout_steps", "validation_every_updates")
    overrides = {name: getattr(args, name) for name in override_names if getattr(args, name) is not None}
    run_bootstrap_training(
        arena_config_path=args.arena,
        reward_spec_path=args.reward,
        training_config_path=args.config,
        cpo_config_path=args.cpo_config,
        logdir=args.logdir,
        resume=not args.no_resume,
        overrides=overrides,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sumobot-ai")
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser("doctor", help="validate CUDA and optional simulation dependencies")
    doctor.set_defaults(func=_doctor)

    reward = subcommands.add_parser("reward", help="reward authoring tools")
    reward_subcommands = reward.add_subparsers(dest="reward_command", required=True)
    lint = reward_subcommands.add_parser("lint", help="validate and hash a reward YAML")
    lint.add_argument("path", type=Path)
    lint.add_argument("--arena", default="configs/arena/flat_3x2.yaml")
    lint.add_argument(
        "--plugin", action="append", default=[], help="import a Python module that registers custom terms"
    )
    lint.set_defaults(func=_reward_lint)

    curriculum = subcommands.add_parser("curriculum", help="opponent curriculum tools")
    curriculum_subcommands = curriculum.add_subparsers(dest="curriculum_command", required=True)
    demo = curriculum_subcommands.add_parser("demo", help="exercise promotion/demotion logic")
    demo.add_argument("--config", default="configs/training/curriculum.yaml")
    demo.set_defaults(func=_curriculum_demo)

    sim = subcommands.add_parser("sim", help="Newton/MuJoCo-Warp tools")
    sim_subcommands = sim.add_subparsers(dest="sim_command", required=True)
    smoke = sim_subcommands.add_parser("smoke", help="run a short vectorized physics rollout")
    smoke.add_argument("--config", default="configs/arena/flat_3x2.yaml")
    smoke.add_argument("--worlds", type=int, default=8)
    smoke.add_argument("--steps", type=int, default=20)
    smoke.add_argument("--device", default="cuda:0")
    smoke.set_defaults(func=_sim_smoke)

    train = subcommands.add_parser("train", help="run privileged teacher training")
    train_subcommands = train.add_subparsers(dest="train_command", required=True)
    bootstrap = train_subcommands.add_parser("bootstrap", help="train two independent CPO populations competitively")
    bootstrap.add_argument("--arena", default="configs/arena/flat_3x2.yaml")
    bootstrap.add_argument("--reward", default="configs/rewards/bootstrap_competitive.yaml")
    bootstrap.add_argument("--config", default="configs/training/bootstrap.yaml")
    bootstrap.add_argument("--cpo-config", default="configs/training/cpo.yaml")
    bootstrap.add_argument("--logdir", default="runs/bootstrap")
    bootstrap.add_argument("--num-envs", type=int)
    bootstrap.add_argument("--total-environment-steps", type=int)
    bootstrap.add_argument("--rollout-steps", type=int)
    bootstrap.add_argument("--validation-every-updates", type=int)
    bootstrap.add_argument("--no-resume", action="store_true")
    bootstrap.set_defaults(func=_train_bootstrap)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
