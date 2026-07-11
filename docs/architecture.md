# Architecture and research plan

## Scope of the first release

The first release is a deliberately small, measurable environment: two four-wheel robots on a finite flat table.
It establishes interfaces that later Sumobot tasks can extend without changing the training algorithms' basic data
contracts. It does not yet claim a trained policy or sim-to-real transfer.

The challenge has three distinct objectives which must not be conflated:

1. **Tournament objective:** fixed, zero-sum win/loss/draw scoring. This is the only ranking signal.
2. **Training task reward:** bootstrap uses a shared win/loss-dominant competitive reward; a later member session uses
   that member's reproducible YAML reward.
3. **CPO diversity reward:** an optional follower-only signal that can make policies within one CPO population
   distinguishable. It is disabled for the first bootstrap run and never stored as CAP-Dreamer's environment reward.

## Arena and control contract

- Coordinates are metres, seconds, kilograms, radians, and Newton uses `+z` as up.
- The table is a finite `3 x 2 x 0.05 m` box whose top is at `z=0`.
- The robot axis convention is `x=forward`, `y=left`, `z=up`, with a `40 x 40 mm` footprint and `80 mm` height.
- Actions are four normalized wheel-velocity commands in `[front_left, front_right, rear_left, rear_right]` order.
- The environment clips, latency-delays, and motor-scales an action before it reaches physics. The resulting
  `action_exec`, rather than a policy proposal, is fed to recurrent state and replay.
- Control runs at 50 Hz with four MuJoCo-Warp substeps initially. Both are configuration values.
- The provisional out rule is that a chassis centre crosses a board edge. Before a public challenge, replace or
  ratify this with the society's physical rule and add golden tests.

## Observation boundary

The teacher receives a fixed-order privileged vector containing both 3D poses, both 3D twists, wheel speeds,
contact summaries, signed edge margins, time remaining, relative motion, and sampled domain parameters. It also
receives the CPO policy identifier through the policy network, not through the physical observation.

The student receives only a deployment-feasible sensor vector: wheel encoders, IMU angular velocity and
acceleration, gravity direction, four edge sensors, an opponent range/bearing/valid tuple, previous executed action,
sensor ages, and time remaining. This sensor suite is provisional because physical hardware was not specified. It
is intentionally centralized in one schema so a later camera, lidar, or ToF decision changes one contract rather
than leaking privileged state into training.

Every exported student model is checked for privileged keys. Domain parameters may be auxiliary prediction targets
but may not be actor inputs at deployment.

## Phase A: two-population CPO bootstrap

There are two independent CPO populations, `red` and `blue`. A policy from red always controls robot 0 and a policy
from blue always controls robot 1. During bootstrap, a balanced cyclic pairing covers all leader/follower policy-ID
pairs across vectorized arenas. Rollouts are collected with frozen old-policy snapshots for both populations, then
both populations update after the rollout barrier. This avoids changing either behaviour policy halfway through an
on-policy batch.

Within each population, policy ID 0 is the leader and IDs 1..N are followers. The implementation follows the cited
CPO structure:

- the leader uses PPO and may learn from selected follower experience;
- followers use PPO with a KL-to-leader term on their own trajectories;
- relabelled transitions support an AWAC term;
- a classifier over state/action identifies policy IDs and supplies a small follower-only diversity reward.

The red-vs-blue task is adversarial in the literal game-theoretic sense: winning is positive, losing is negative, and
the two separately optimized models are each other's changing opponent. This win/loss-dominant reward is the primary
bootstrap signal. That is separate from CPO's optional classifier reward. Keeping the two channels separate makes
reward ablations and student training interpretable.

Training writes one TensorBoard event stream containing outcome rates, rewards, red/blue losses, KL diagnostics,
gradient norms, and simulator throughput. At a fixed update interval, a seeded leader-vs-leader validation batch runs
without exploration and records aggregate metrics plus a lightweight top-down video. Checkpoints contain both model
and optimizer states and all random-number-generator states needed to resume the paired populations together.

## Phase B: one member CPO population against the curriculum

Once bootstrap has produced a validated opponent league, a member training session has only one learning CPO
population of `x` conditioned policies. Its agents collect experience against frozen curriculum models; the opponent
side performs inference only and is never updated by that session. The member's authored reward trains their CPO
leader/followers. Arena sides alternate to prevent red/blue geometry or reset bias.

## Model curriculum

Extreme Parkour tracks a terrain level per environment and moves an environment to a neighboring level according to
performance. Here, `level` indexes frozen opponent snapshots rather than terrain. At an episode boundary:

- sustained success above the promotion threshold raises that arena one level;
- sustained failure below the demotion threshold lowers it one level;
- matchmaking samples mostly at the current level with smaller probabilities for one easier and one harder level;
- levels contain immutable, hashed checkpoints, not mutable live model objects.

Snapshots are ordered by held-out league rating and validation gates, not merely training step. A mixture of recent,
historical, and exploitability-check opponents is retained to reduce forgetting and self-play cycles. During initial
development, when no snapshot league exists, both live CPO populations play in every arena.

## Teacher design and CAP-Dreamer bridge

The first implementation uses the practical, decoupled route:

1. Train privileged CPO teachers and collect broad coverage from leaders and followers.
2. Label every stored state with the frozen leader distribution, even when a follower or student executed the action.
3. Pretrain CAP-Dreamer's encoder/RSSM and native heads on deployable observations, executed actions, fixed task
   rewards, and correct continuation flags.
4. Copy only the actor suffix whose tensor shapes and distribution semantics are intentionally identical.
5. Distill the leader distribution on real posterior states.
6. Run mixed-execution and then student-executed DAgger.
7. Enable CAP-Dreamer's imagined actor learning only after multi-step prediction and closed-loop gates pass.

The shared actor suffix uses the CAP-Dreamer 12M convention: 256-wide linear blocks with FP32 RMSNorm
(`eps=1e-4`), SiLU, and a bounded-normal parameter head. The teacher frontend maps privileged input plus policy ID
to a 256-dimensional bottleneck. CAP's first actor block maps the RSSM feature to that same bottleneck; subsequent
blocks and the output head can be copied exactly. A manifest and numerical parity test are required at export.

CAP's current bounded-normal head applies `tanh` to its Gaussian mean and bounds standard deviation with
`(max-min)*sigmoid(raw+2)+min`; it does not apply a tanh-transform Jacobian. CPO must calculate PPO likelihoods with
that same distribution. Final action clipping remains an environment operation and must be logged in `action_exec`.

### Replay alignment

At observation time `t`, store the policy proposal and leader label for that observation. At physics time, store the
post-latency/post-clipping action actually executed. The world-model transition is always:

```text
(student_obs_t, action_exec_t, reward_env_{t+1}, student_obs_{t+1})
```

Required replay fields are versioned in `sumobot_ai.distillation.schema`. `reward_cpo_diversity` is present for audit
only. `reward_env` is the team's authored task reward during teacher/student training; `score_outcome` is the fixed
tournament result used for evaluation. Dataset metadata includes reward-spec and reward-code hashes, teacher hash, arena version,
observation schema version, and simulator/domain-randomization version.

## Domain randomization

Randomization is sampled per arena and per episode. The first ranges cover board, chassis and wheel friction;
restitution; chassis and wheel mass; motor strength; action latency; encoder/IMU noise; and opponent-sensor dropout.
Newton material changes are batched and followed by `SolverMuJoCo.notify_model_changed(...)`. Parameters are visible
to the privileged teacher and dataset diagnostics, never directly to the deployed actor.

Later ranges should include centre-of-mass offsets, inertial tensors, voltage/battery effects, wheel eccentricity,
motor deadband, controller jitter, sensor extrinsics, timestamp error, and external impulses. Randomization must be
validated against plausible physical measurements rather than made arbitrarily wide.

## Validation gates

1. **Contract gate:** dimensions, units, action order, reset masks, out rule, reward symmetry, and observation leakage.
2. **Physics gate:** finite rollouts, wheel direction, static stability, contact/friction response, and deterministic
   reset on both CPU reference and GPU MuJoCo-Warp.
3. **CPO gate:** old/new log-prob parity, mask coverage, leader/follower returns, KL, discriminator accuracy, and
   balanced red/blue matchup coverage.
4. **Curriculum gate:** no mutable checkpoints, bounded levels, promotion/demotion hysteresis, and held-out rating
   monotonicity.
5. **Distillation gate:** actor-suffix numerical parity, teacher/student KL by state source, no privileged export keys,
   and executed-action alignment.
6. **World-model gate:** one- and multi-step task-variable error, continuation calibration, CAP confidence unlocks,
   and predicted-versus-real return.
7. **Closed-loop gate:** win rate, draw rate, interventions, edge failures, recovery, held-out randomization, inference
   latency, and student-only execution.

## Milestones

- **M0 — contracts and physics:** settle physical dimensions/sensors/out rule; validate thousands of parallel Newton
  arenas and randomization.
- **M1 — reward workshop:** publish term documentation, linting, reward hacking examples, baseline/sparse rewards, and
  fixed evaluation harness.
- **M2 — dual CPO bootstrap:** train both privileged populations from scratch, freeze validated snapshots, and build
  league ratings.
- **M3 — opponent curriculum:** activate per-arena snapshot levels and test against uniform/latest-only baselines.
- **M4 — CAP pretraining/distillation:** add replay fields and losses to the pinned CAP branch; run actor parity,
  offline cloning, and DAgger.
- **M5 — sim-to-real:** calibrate dynamics/sensors from standardized hardware, export student-only inference, and use
  staged safety tests before physical competition.

## Pinned upstream references

Pins record the APIs inspected for this scaffold and should be changed through a compatibility PR:

- Newton: `newton-physics/newton@82526c0aa7322569de4faf461b0db87b294a8117`
- CPO: `Naoki04/paper-cpo-code@d1597a4fc870124cef98922f737998bfabf8e6d4`
- CAP-Dreamer hybrid: `ElliottDyson/CAP-Dreamer@b39108d75a3bc01a972776f3f45a03914949470e`
- Extreme Parkour code (conceptual reference only):
  `chengxuxin/extreme-parkour@d2ffe27ba59a3229fad22a9fc94c38010bb1f519`

Extreme Parkour's code is CC BY-NC 4.0, so this repository uses the curriculum idea and paper attribution without
copying its implementation.
