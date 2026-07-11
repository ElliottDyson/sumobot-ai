# ADR 0002: authored training rewards do not define tournament score

Status: accepted

Teams may select and weight registered reward terms through a versioned YAML file. The simulator records every
component and a content hash. Tournament ranking always uses the fixed out/draw rules. Training reward, CPO diversity
reward, and tournament outcome are separate tensors in replay and metrics.

This makes creative reward design possible without letting a team redefine what winning means, and it lets judges
replay or audit a submission deterministically.
