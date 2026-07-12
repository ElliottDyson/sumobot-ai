# ADR 0004: two-wheel drive, inactivity, and bounded guidance

Status: accepted; physical geometry and ring-out details refined by ADR 0005

The standardized robot has two independently driven side wheels and one passive rear skid. Its policy action is
therefore exactly `[left_wheel_velocity, right_wheel_velocity]`; there is no steering or skid action. The skid is a
massless collision shape fixed to the chassis, with its own randomized low-friction material. Its current configured
position is provisional hardware geometry and can be calibrated without changing the two-action policy contract.

Match resolution uses this precedence:

1. A ring-out is resolved first. Exactly one out robot loses; simultaneous ring-out is a draw.
2. Otherwise, ten continuous seconds of physical planar inactivity loses. Both timers expiring together is a draw.
3. Otherwise, the 30-second time limit is a draw.
4. A simulator numerical failure is always a draw and is logged; it is never a learnable win.

Movement means chassis planar speed of at least `0.01 m/s`, sustained for `0.2 s` before resetting the inactivity
timer. This filters solver jitter and single-tick pulses. Being physically pushed counts as movement because the rule
is defined from observable robot motion, not hidden policy intent.

The terminal training reward is `+10/-10/0`. Every non-outcome term is combined into one bounded guidance channel:

```text
abs(guidance_t) <= 0.05 * dt
abs(sum_t guidance_t) <= 1.5 over a 30-second match
```

Thus even a winner receiving the worst possible guidance has at least `+8.5`, while a loser receiving the best
possible guidance has at most `-8.5`. Arbitrary member terms can change what guidance encourages but cannot change
that ordering. Validation rejects any configured full-match guidance budget above 25% of the win magnitude.

This changes the arena and replay contracts to version 2. Four-action teacher checkpoints and datasets are
intentionally incompatible and must not be resumed. Validation GIFs also carry an explicit frame delay and a schedule
that includes the terminal frame, so TensorBoard playback covers the entire realized match.
