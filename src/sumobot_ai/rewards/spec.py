from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .terms import TERM_DEFINITIONS


@dataclass(frozen=True, slots=True)
class RewardTerm:
    name: str
    weight: float
    params: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class RewardSpec:
    version: int
    name: str
    description: str
    terms: tuple[RewardTerm, ...]
    clip: tuple[float, float] | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RewardSpec:
        if int(data.get("version", -1)) != 1:
            raise ValueError("reward spec version must be 1")
        raw_terms = data.get("terms")
        if not isinstance(raw_terms, list) or not raw_terms:
            raise ValueError("reward spec must contain a non-empty terms list")
        terms: list[RewardTerm] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_terms):
            if not isinstance(raw, Mapping):
                raise ValueError(f"terms[{index}] must be a mapping")
            name = str(raw.get("name", ""))
            if name not in TERM_DEFINITIONS:
                raise ValueError(f"unknown reward term {name!r}; available: {sorted(TERM_DEFINITIONS)}")
            if name in seen:
                raise ValueError(f"reward term {name!r} appears more than once")
            seen.add(name)
            weight = float(raw.get("weight", 0.0))
            if not math.isfinite(weight):
                raise ValueError(f"reward term {name!r} has a non-finite weight")
            params_raw = raw.get("params", {})
            if not isinstance(params_raw, Mapping):
                raise ValueError(f"reward term {name!r} params must be a mapping")
            definition = TERM_DEFINITIONS[name]
            unknown_params = set(params_raw) - definition.allowed_params
            if unknown_params:
                raise ValueError(f"reward term {name!r} has unknown params: {sorted(unknown_params)}")
            params = dict(definition.defaults)
            params.update({str(key): float(value) for key, value in params_raw.items()})
            definition.validate(params)
            terms.append(RewardTerm(name=name, weight=weight, params=params))
        clip_raw = data.get("clip")
        clip = None
        if clip_raw is not None:
            if not isinstance(clip_raw, list) or len(clip_raw) != 2:
                raise ValueError("clip must be [minimum, maximum]")
            clip = (float(clip_raw[0]), float(clip_raw[1]))
            if not all(math.isfinite(value) for value in clip) or clip[0] >= clip[1]:
                raise ValueError("clip bounds must be finite and increasing")
        return cls(
            version=1,
            name=str(data.get("name", "unnamed")),
            description=str(data.get("description", "")),
            terms=tuple(terms),
            clip=clip,
        )

    @classmethod
    def load(cls, path: str | Path) -> RewardSpec:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, Mapping):
            raise ValueError("reward spec root must be a mapping")
        return cls.from_mapping(data)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "clip": list(self.clip) if self.clip is not None else None,
            "terms": [
                {"name": term.name, "weight": term.weight, "params": dict(sorted(term.params.items()))}
                for term in self.terms
            ],
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
