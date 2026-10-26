from __future__ import annotations

import dataclasses

from castvalue.core import PolicyConfig, config_hash


@dataclasses.dataclass(frozen=True)
class SampleConfig(PolicyConfig):
    x: float = 1.0
    y: str = "a"


def test_same_config_same_hash():
    assert config_hash(SampleConfig()) == config_hash(SampleConfig())


def test_different_config_different_hash():
    assert config_hash(SampleConfig(x=1.0)) != config_hash(SampleConfig(x=2.0))


def test_hash_is_a_short_hex_string():
    h = config_hash(SampleConfig())
    assert isinstance(h, str)
    assert len(h) == 16
    int(h, 16)  # must be valid hex
