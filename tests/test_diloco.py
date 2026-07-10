"""Tests for the DiLoCo building blocks: serialization, averaging, outer step."""

import numpy as np
import pytest

from onepool.train.outer import NesterovOuter
from onepool.train.tensors import pack_state, state_delta, unpack_state, weighted_average


def test_pack_unpack_roundtrip():
    state = {
        "layer.weight": np.random.randn(4, 8).astype(np.float32),
        "layer.bias": np.random.randn(8).astype(np.float32),
    }
    restored = unpack_state(pack_state(state))
    assert set(restored) == set(state)
    for name in state:
        np.testing.assert_array_equal(restored[name], state[name])
        assert restored[name].flags.writeable  # workers mutate weights in place


def test_pack_casts_to_float32():
    state = {"w": np.array([1.0, 2.0], dtype=np.float64)}
    restored = unpack_state(pack_state(state))
    assert restored["w"].dtype == np.float32


def test_state_delta_is_start_minus_end():
    start = {"w": np.array([3.0], dtype=np.float32)}
    end = {"w": np.array([1.0], dtype=np.float32)}
    np.testing.assert_allclose(state_delta(start, end)["w"], [2.0])


def test_weighted_average():
    a = {"w": np.array([1.0], dtype=np.float32)}
    b = {"w": np.array([4.0], dtype=np.float32)}
    avg = weighted_average([a, b], [1.0, 2.0])
    np.testing.assert_allclose(avg["w"], [3.0])  # (1*1 + 2*4) / 3


def test_weighted_average_rejects_zero_weights():
    a = {"w": np.zeros(1, dtype=np.float32)}
    with pytest.raises(ValueError):
        weighted_average([a], [0.0])


def test_nesterov_outer_first_step():
    outer = NesterovOuter({"w": np.array([1.0], dtype=np.float32)}, lr=0.7, momentum=0.9)
    new = outer.step({"w": np.array([0.1], dtype=np.float32)})
    # v = 0.1; update = g + mu*v = 0.1 + 0.09 = 0.19; w = 1 - 0.7*0.19
    np.testing.assert_allclose(new["w"], [1 - 0.7 * 0.19], rtol=1e-6)


def test_nesterov_momentum_accumulates():
    outer = NesterovOuter({"w": np.array([0.0], dtype=np.float32)}, lr=1.0, momentum=0.5)
    g = {"w": np.array([1.0], dtype=np.float32)}
    outer.step(g)  # v=1.0, step = 1 + 0.5 = 1.5
    new = outer.step(g)  # v=1.5, step = 1 + 0.75 = 1.75 -> total 3.25
    np.testing.assert_allclose(new["w"], [-3.25], rtol=1e-6)


def test_outer_converges_toward_worker_consensus():
    """If every worker keeps pulling toward w=5, the outer weights get there."""
    outer = NesterovOuter({"w": np.array([0.0], dtype=np.float32)}, lr=0.7, momentum=0.9)
    target = 5.0
    for _ in range(200):
        current = outer.weights["w"].copy()
        # each round, workers move 10% of the way toward the target
        local = current + 0.1 * (target - current)
        outer.step({"w": current - local})
    np.testing.assert_allclose(outer.weights["w"], [target], atol=0.05)
