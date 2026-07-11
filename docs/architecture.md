# Architecture and research plan

## Scope of the first release

The first release is a deliberately small, measurable environment: two differential-drive robots on a finite flat
table, each using two driven wheels and a passive skid.
It establishes interfaces that later Sumobot tasks can extend without changing the training algorithms' basic data
contracts. It does not yet claim a trained policy or sim-to-real transfer.

The challenge has three distinct objectives which must not be conflated:

1. **Tournament objective:** fixed, zero-sum win/loss/draw scoring. This is the only ranking signal.
2. **Training task reward:** bootstrap uses a shared outcome-dominant competitive reward; a later member session uses
   that member's reproducible YAML reward. The aggregate non-outcome channel is rate-capped.
3. **CPO diversity reward:** an optional follower-only signal that can make policies within one CPO population
   distinguishable. It is disabled for the first bootstrap run and never stored as CAP-Dreamer's environment reward.

## Arena and control contract

- Coordinates are metres, seconds, kilograms, radians, and Newton uses `+z` as up.
- The table is a finite `3 x 2 x 0.05 m` box whose top is at `z=0`.
- The robot axis convention is `x=forward`, `y=left`, `z=up`. The complete external envelope, including wheels and
  skid, is `40 x 40 x 80 mm`; the collision chassis is smaller so those appendages remain inside it.
- Each robot has one driven wheel on each side of a common axle and a finite, low-friction passive rear skid pad.
- Actions are two normalized wheel-velocity commands in `[left_wheel, right_wheel]` order.
- The environment clips, latency-delays, deadband-corrects, quantizes, slew-limits, and first-order filters a command.
  A torque-speed curve, battery scale, and left/right gain mismatch then govern the velocity actuators. The resulting
  normalized `action_exec`, rather than the raw policy proposal, is fed to recurrent state and replay.
- Control runs at 50 Hz with four MuJoCo-Warp substeps initially. Both are configuration values.
- A robot rings out after all three wheel/skid support projections remain outside the tabletop for `0.1 s`. Partial
  support does not count as out; a simultaneous confirmed support loss is a draw.
- A robot below `0.01 m/s` planar chassis speed for ten continuous seconds loses. Movement must remain above the
  threshold for `0.2 s` before it resets the timer, preventing a single solver-jitter or one-tick command spike from
  evading the rule. Physical movement, including being pushed, counts. If both timers expire on the same control
  step, the result is a draw. Ring-out takes precedence if ring-out and inactivity occur together.

## Observation boundary

The teacher receives a fixed-order privileged vector containing both 3D poses, both 3D twists, wheel speeds, real
MuJoCo-Warp contact-force summaries, centre and support edge margins, time remaining, relative motion, actuator state,
and sampled domain parameters. It also receives ten 50 Hz samples (200 ms) of compact self action/proprioception and
opponent-relative motion. Opponent private commands are deliberately excluded. The CPO policy identifier enters
through the policy network, not through the physical observation.

The student receives only a deployment-feasible sensor vector: quantized wheel encoders, IMU angular velocity and
specific force, gravity direction, four downward ray-based edge sensors, an FOV-limited opponent
range/bearing/valid tuple, previous executed action, sensor ages, local inactivity state, and time remaining. Sensor
rates, latency, bias, noise, and dropout are simulated. A ten-sample compact proprioceptive/action window accompanies
the current packet, while CAP's RSSM supplies longer recurrent memory. The suite remains centralized so later physical
sensor selection changes one versioned contract rather than leaking privileged state into training.

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

The red-vs-blue task is adversarial in the literal game-theoretic sense: winning is `+10`, losing is `-10`, and the
two separately optimized models are each other's changing opponent. All authored non-outcome terms are summed into a
guidance channel and clipped to `±0.05 × transition_seconds`. Its absolute full-match budget is therefore at most
`1.5`, or 15% of a win, regardless of the number or scale of custom terms. Load-time validation rejects a guidance
budget above 25% of the win magnitude. This is separate from CPO's optional classifier reward.

Training writes one TensorBoard event stream containing outcome rates, termination causes, rewards, red/blue losses,
KL diagnostics, gradient norms, and simulator throughput. At a fixed update interval, a seeded leader-vs-leader
validation batch runs without exploration and records aggregate metrics plus a lightweight top-down video. Its frame
schedule is validated to include the reset and terminal match-time frames rather than truncating at a frame cap.
Checkpoints contain both model
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

Randomization is sampled per arena and per episode. Current physical ranges cover board/chassis/wheel/skid sliding,
torsional, and rolling friction; restitution and contact stiffness/damping; chassis/wheel mass; COM offsets and
inertia; motor strength, response time, deadband, left/right mismatch, battery voltage, and action latency. Sensor
ranges cover encoder/IMU bias, noise and latency, edge-sensor noise/dropout/latency, and opponent-sensor
range/bearing noise, dropout, and latency. Newton material and inertial changes are followed by
`SolverMuJoCo.notify_model_changed(...)`. Sampled parameters are visible to the privileged teacher and diagnostics,
never directly to the deployed actor.

These are bounded starting distributions, not substitutes for system identification. Hardware acceptance must measure
mass/COM/inertia, wheel speed and torque response, turn-in-place skid scrub, straight-line drift, encoder/IMU error,
and sensor latency, then update and narrow the ranges. Wheel-radius/eccentricity and external-impulse experiments can
be added once their physical distributions are known.

## Validation gates

1. **Contract gate:** external-envelope dimensions, units, action order, history/reset masks, support out rule, reward
   symmetry, and observation leakage.
2. **Physics gate:** finite rollouts, wheel direction, motor step response, straight/yaw kinematics, static stability,
   measured contact load, skid scrub, and deterministic reset on GPU MuJoCo-Warp.
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
