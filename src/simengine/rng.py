from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np


def _name_words(name: str) -> list[int]:
    """Stable 128-bit fingerprint of a name as four 32-bit integers."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]


class SeededRng:
    """Seeded random source with named, order-independent child streams."""

    def __init__(self, seed: int, path: tuple[str, ...] = ()) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self._seed = seed
        self._path = path
        entropy: list[int] = [seed]
        for part in path:
            entropy.extend(_name_words(part))
        sequence = np.random.SeedSequence(entropy)
        self._generator = np.random.Generator(np.random.PCG64(sequence))

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def path(self) -> tuple[str, ...]:
        return self._path

    @property
    def generator(self) -> np.random.Generator:
        return self._generator

    def child(self, name: str) -> SeededRng:
        """New stream fixed by (seed, path, name). Ignores parent draws."""
        return SeededRng(self._seed, self._path + (name,))

    def get_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._generator.bit_generator.state)

    def set_state(self, state: dict[str, Any]) -> None:
        self._generator.bit_generator.state = copy.deepcopy(state)