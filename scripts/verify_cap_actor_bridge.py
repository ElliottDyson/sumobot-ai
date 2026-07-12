#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from sumobot_ai.distillation import copy_actor_suffix_to_cap, measure_actor_suffix_parity
from sumobot_ai.training.cpo import CapActorSuffix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cap_source", nargs="?", default=".upstreams/CAP-Dreamer")
    args = parser.parse_args()
    source = Path(args.cap_source).resolve()
    if not (source / "networks.py").is_file():
        raise SystemExit(f"CAP-Dreamer source not found at {source}; run scripts/checkout_upstreams.sh")
    sys.path.insert(0, str(source))
    import networks

    config = SimpleNamespace(
        shape=(2,),
        layers=3,
        units=256,
        act="SiLU",
        symlog_inputs=False,
        device="cpu",
        name="actor",
        outscale=0.01,
        dist=SimpleNamespace(name="bounded_normal", min_std=0.1, max_std=1.0),
    )
    cap_actor = networks.MLPHead(config, inp_dim=256, validate_args=True)
    teacher_suffix = CapActorSuffix(action_dim=2)
    manifest = copy_actor_suffix_to_cap(teacher_suffix, cap_actor)
    generator = torch.Generator().manual_seed(17)
    bottleneck = torch.randn(32, 256, generator=generator)
    parity = measure_actor_suffix_parity(teacher_suffix, cap_actor, bottleneck)
    print(json.dumps({"parity": asdict(parity), "manifest": manifest}, indent=2))
    return 0 if parity.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
