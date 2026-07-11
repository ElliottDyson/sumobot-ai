# Reward authoring

Reward design is the member-facing experiment in Sumobot AI. A reward file controls learning but cannot change the
fixed match outcome or tournament score.

## Two training contexts

- **Bootstrap:** two independent live CPO populations compete. Both use `bootstrap_competitive.yaml`, whose dominant
  term is literal win/loss. These policies become the frozen opponent curriculum.
- **Member session:** one live CPO population of leader/followers competes against curriculum snapshots. The member's
  YAML is evaluated only for their learner side. Opponent inference models never update in this session.

## Format

```yaml
version: 1
name: my_reward_v1
description: What behavior this reward is intended to produce.
guidance_max_abs_per_second: 0.05
clip: [-12.0, 12.0]
terms:
  - name: win_loss
    weight: 10.0
  - name: approach_opponent
    weight: 0.25
    params:
      scale_m: 1.0
      discount: 1.0
```

Validate and content-hash it with:

```bash
python -m sumobot_ai reward lint path/to/reward.yaml
```

The command reports separate specification and reward-code hashes; both are stored in every dataset/checkpoint
metadata record. Unknown terms, duplicate terms, invalid parameters, and non-finite weights fail before a training job
starts.

Every valid challenge reward must contain exactly one positive `win_loss` term. All other terms—including custom
Python terms—are summed and symmetrically rate-limited:

```text
guidance_t = clamp(sum(non_outcome_terms_t), ±guidance_max_abs_per_second × dt)
reward_t   = weighted_win_loss_t + guidance_t
```

The shipped rate is `0.05/s`, so guidance can total at most `±1.5` over a 30-second match versus `±10` for win/loss.
The validator rejects configurations whose maximum full-match guidance exceeds 25% of the win magnitude, and also
rejects a final clip that could erase that dominance. Members retain freedom over what states and behaviours guidance
values; they cannot use its scale to redefine winning.

## Shipped bootstrap guidance

| Term | Raw weight |
| --- | ---: |
| `win_loss` | `10.0` (not guidance-capped) |
| `approach_opponent` | `0.25` |
| `push_opponent_to_edge` | `0.50` |
| `protect_own_edge` | `0.15` |
| `face_opponent` | `0.01` |
| `action_energy` | `0.002` |

The raw weights determine how the limited guidance budget is shared; their combined realized contribution still
cannot exceed the aggregate cap. The sparse ablation contains only `win_loss`.

## Built-in terms

| Term | Meaning | Form |
| --- | --- | --- |
| `win_loss` | `+1` to the winner, `-1` to the loser, `0` for a draw | terminal, zero-sum |
| `approach_opponent` | Potential difference for reducing planar opponent distance | dense, symmetric |
| `push_opponent_to_edge` | Potential difference for reducing the opponent's signed edge margin | dense, per side |
| `protect_own_edge` | Potential difference for increasing one's own signed edge margin | dense, per side |
| `face_opponent` | Heading alignment with the opponent, integrated over the control interval | dense |
| `action_energy` | Negative mean squared executed action, integrated over the control interval | dense cost |

The three progress terms are potential differences rather than raw state rewards. This reduces incentives to collect
the same shaping reward indefinitely without making progress. It does not eliminate reward hacking; all submissions
still need closed-loop evaluation against the fixed score.

## Custom Python terms

Members who need more than weights and built-ins can register a batched Torch function in an importable module:

```python
from sumobot_ai.rewards import register_reward_term


def lateral_speed(transition, params):
    return transition.current.linear_velocity[..., 1].abs() * transition.dt.unsqueeze(-1)


register_reward_term("lateral_speed", lateral_speed)
```

Then lint with `--plugin my_reward_terms` and use `name: lateral_speed` in YAML. A term must return one value per arena
and side with shape `(B, 2)`. Plugin import executes member code, so public training workers must run submissions in
an isolated job/container with resource limits. The registered source revision joins the reward YAML hash in final
submission metadata; a YAML hash alone is insufficient for custom code.

## Audit rules

- Components are logged separately; never log only their sum.
- `guidance_clip_delta` records exactly how much raw shaping the aggregate cap removed.
- `action_energy` uses the delayed/clipped action actually executed.
- CPO's optional policy-ID diversity bonus is `reward_cpo_diversity`, not a YAML term.
- Curriculum promotion uses fixed win/draw/loss outcomes, not the member's reward return.
- Final evaluation uses no custom shaping in the ranking calculation.
- Compare sparse, baseline, and authored rewards against the same seeds, opponent snapshots, and domain draws.

The match engine—not member reward code—also assigns a loss after ten continuous seconds below the physical movement
threshold. Movement must be sustained for 0.2 seconds to reset that timer; simultaneous inactivity is a draw. Plot raw
components, bounded guidance, terminal cause, and policies near the edge: that is where apparently sensible rewards
most often discover undesirable shortcuts.
