"""Frozen, hashable config base shared by every benchmark's PolicyConfig."""

from __future__ import annotations

import dataclasses
import hashlib
import json


@dataclasses.dataclass(frozen=True)
class PolicyConfig:
    """Base class for benchmark-specific policy configs.

    Frozen so a config can be safely reused across every coalition run in a
    Shapley game without risk of one run's code mutating another's inputs,
    and hashable (via config_hash) so results stay reproducible/comparable
    across runs with the same parameters.
    """


def config_hash(config: PolicyConfig) -> str:
    """Stable short hash of a frozen dataclass config, for grouping and
    reproducibility keys (stored alongside every result, per the brief's
    provenance requirement).

    `default=str` lets this handle non-JSON-native field types (e.g. an
    enum or a frozenset) without needing every config subclass to define
    its own serialization.
    """
    as_dict = dataclasses.asdict(config)
    canonical = json.dumps(as_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
