# UBRobotics Sumobot AI

This repository is the initial training platform for a two-robot, simulated sumo challenge in which teams design
their own training rewards. Two privileged CPO populations first learn against each other and produce an opponent
curriculum. A member then trains one CPO population against that curriculum with their reward, and distils its leader
into a deployable CAP-Dreamer student.

The first arena is a flat, finite `3 m x 2 m` table. Each arena contains two standardized differential-drive robots,
each with two independently driven wheels and a passive rear skid, and is replicated on the GPU. Newton supplies the
simulation model and `SolverMuJoCo` runs MuJoCo-Warp contact physics.

## What exists now

- A versioned YAML reward format with linting and a vectorized reward evaluator.
- Strict privileged-teacher and deployable-student observation contracts.
- Domain-randomization sampling with teacher-only parameter export.
- A per-arena opponent curriculum based on promotion and demotion against frozen model snapshots.
- A bootstrap matchmaker where two independent live CPO populations compete, plus a member-training matchmaker where
  one live CPO population trains against frozen curriculum inference models.
- CPO-compatible actor, loss, and diversity-discriminator primitives with a CAP-compatible actor suffix.
- A replay schema that keeps executed actions, teacher labels, environment rewards, and CPO diversity rewards separate.
- A Newton/MuJoCo-Warp arena backend and CUDA/system diagnostics.
- A legal-envelope two-wheel/skid rigid-body model with filtered torque-speed-limited motors, real contact summaries,
  support-based ring-out, and delayed/noisy deployable sensors.
- Ten-step proprioceptive/action histories for both the privileged CPO teacher interface and the deployable student,
  with CAP's RSSM retaining its longer recurrent belief state.
- An executable dual-population bootstrap trainer with checkpoints, TensorBoard metrics, and validation-match videos.

The repository deliberately treats the robot dimensions and deployable sensor suite as versioned challenge
contracts. The complete robot—including wheels and skid—must fit the `40 mm` fore-aft, `40 mm` wide, and `80 mm`
high envelope. Policy actions are the normalized target velocities of the left and right driven wheels; the finite
rear skid pad is passive. Current inertial, motor, skid-material, and sensor values are measured-style starting
estimates and must be replaced or narrowed using the standardized physical robot before sim-to-real deployment.

## Quick start

The development host exposes the NVIDIA driver outside the default loader path. Source the bootstrap before any
CUDA process:

```bash
source scripts/activate_gpu.sh
python -m sumobot_ai doctor
```

Install the lightweight development package and run the CPU contract tests:

```bash
python -m pip install -e '.[dev]'
pytest -q -m 'not sim and not gpu'
python -m sumobot_ai reward lint configs/rewards/baseline.yaml
python -m sumobot_ai curriculum demo
```

Install and exercise Newton/MuJoCo-Warp:

```bash
python -m pip install -e '.[sim]'
python -m sumobot_ai sim smoke --config configs/arena/flat_3x2.yaml --worlds 8 --steps 20
```

Fetch the exact research implementations inspected by this repository and verify the real CAP actor suffix:

```bash
scripts/checkout_upstreams.sh
python scripts/verify_cap_actor_bridge.py
```

The upstream checkouts live under the ignored `.upstreams/` directory. They are pinned inputs, not vendored copies.

Start or resume competitive bootstrap training:

```bash
source scripts/activate_gpu.sh
python -m sumobot_ai train bootstrap --logdir runs/bootstrap
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

The red and blue CPO populations use separate parameters and optimizers while sharing the same 24,576 physical
arenas. Each population has six policy-conditioning coefficient blocks, so every red policy and every blue policy
controls exactly 4,096 robots per rollout (49,152 simultaneous agent instances in total). The checked-in production
configuration uses the CPO reference geometry of a 16-step horizon and 32,768-sample minibatches; startup rejects a
configuration whose arena count or horizon disagrees with that contract. TensorBoard receives rollout outcomes,
per-population PPO/CPO losses, throughput, held-out leader-vs-leader metrics, and periodic top-down validation video.
`SIGINT`, `SIGTERM`, or `SIGUSR1` requests a checkpoint after the current update; rerunning the command resumes from
`checkpoint.pt` unless `--no-resume` is supplied.

The full design, phase gates, and non-negotiable data invariants are in [the architecture document](docs/architecture.md).
Member-facing reward semantics are in [the reward authoring guide](docs/reward_authoring.md).

## Training shape

```text
bootstrap red CPO  ─┐
                    ├─ competitive Newton arenas ── frozen leader snapshots
bootstrap blue CPO ─┘                                      │
                                                  opponent curriculum
                                                          │
member CPO + custom reward ───────────────────────────────┘
              │
              └─ privileged leader labels + deployable observations + executed actions
                                                          │
                                              CAP-Dreamer + online DAgger
```

Bootstrap's primary adversarial signal is the literal zero-sum match outcome. In later member sessions, authored
shaping affects their training but never tournament ranking. Ring-out takes precedence; otherwise, a robot that stays
below the configured planar movement threshold for ten continuous seconds loses, with simultaneous inactivity scored
as a draw. Ring-out means all three wheel/skid support projections have left the tabletop for the confirmation
interval—not merely that the chassis centre crossed an edge. All non-outcome reward terms share a strict per-second
cap, so their full-match contribution cannot overpower win/loss.
