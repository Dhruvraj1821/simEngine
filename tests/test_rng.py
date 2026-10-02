import json
import os
import subprocess
import sys

import numpy as np
import pytest

from simengine.rng import SeededRng


def draw(rng: SeededRng, n: int = 5) -> np.ndarray:
    return rng.generator.random(n)


def test_same_seed_same_sequence() -> None:
    assert np.array_equal(draw(SeededRng(42)), draw(SeededRng(42)))


def test_different_seeds_differ() -> None:
    assert not np.array_equal(draw(SeededRng(1)), draw(SeededRng(2)))


def test_different_child_names_differ() -> None:
    root = SeededRng(42)
    assert not np.array_equal(draw(root.child("ecology")), draw(root.child("events")))


def test_child_independent_of_parent_draws() -> None:
    used = SeededRng(42)
    draw(used, 100)
    fresh = SeededRng(42)
    assert np.array_equal(draw(used.child("x")), draw(fresh.child("x")))


def test_nested_child_differs_from_flat() -> None:
    root = SeededRng(42)
    assert not np.array_equal(
        draw(root.child("a").child("b")), draw(root.child("b"))
    )


def test_state_restore_resumes_exactly() -> None:
    rng = SeededRng(7)
    draw(rng, 10)
    saved = rng.get_state()
    expected = draw(rng, 5)
    rng.set_state(saved)
    assert np.array_equal(draw(rng, 5), expected)


def test_state_survives_json_roundtrip() -> None:
    rng = SeededRng(7)
    draw(rng, 10)
    saved = json.loads(json.dumps(rng.get_state()))
    expected = draw(rng, 5)
    other = SeededRng(7)
    other.set_state(saved)
    assert np.array_equal(draw(other, 5), expected)


def test_child_stable_across_hash_seeds() -> None:
    code = (
        "from simengine.rng import SeededRng;"
        "print(repr(SeededRng(42).child('ecology').generator.random()))"
    )
    outputs = []
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError):
        SeededRng(-1)