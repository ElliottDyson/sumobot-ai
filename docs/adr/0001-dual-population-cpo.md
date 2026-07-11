# ADR 0001: two independent CPO populations

Status: accepted for bootstrap

Each arena contains one red-controlled robot and one blue-controlled robot. Red and blue use independent CPO model,
optimizer, normalization, discriminator, and checkpoint state. A rollout barrier freezes both old policies until all
arena transitions for that update have been collected.

This matches the requested literal adversarial development setup while retaining valid on-policy ratios: each live
population improves against the other and a win is the primary positive outcome. Weight sharing between sides is an
ablation, not the default. CPO's optional intra-population classifier reward is logged separately from the competitive
win/loss task reward.

After bootstrap, member training uses one live CPO population against frozen curriculum policies. Those inference
opponents do not update during the member's session.
