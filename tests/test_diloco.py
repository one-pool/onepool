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


def test_int8_roundtrip_close():
    state = {"w": (np.random.randn(64, 32) * 0.01).astype(np.float32)}
    restored = unpack_state(pack_state(state, compression="int8"))
    # error bounded by half a quantization bucket
    max_err = np.abs(state["w"]).max() / 127
    np.testing.assert_allclose(restored["w"], state["w"], atol=max_err)


def test_int8_is_4x_smaller():
    state = {"w": np.random.randn(1000).astype(np.float32)}
    raw = pack_state(state)["w"]["data"]
    packed = pack_state(state, compression="int8")["w"]["data"]
    assert len(raw) == 4 * len(packed)


def test_int8_all_zero_tensor():
    state = {"w": np.zeros(16, dtype=np.float32)}
    restored = unpack_state(pack_state(state, compression="int8"))
    np.testing.assert_array_equal(restored["w"], state["w"])


def test_dataset_ships_and_materializes(tmp_path):
    from onepool.jobs import TrainJob
    from onepool.train.distributed import dataset_payload, materialize_dataset

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello pool\n" * 100)
    job = TrainJob(model="m", dataset=str(corpus))

    payload = dataset_payload(job)
    assert payload["name"] == "corpus.txt"

    restored = materialize_dataset(payload)
    assert restored != str(corpus)  # lands in a temp dir, not the original path
    with open(restored, encoding="utf-8") as f:
        assert f.read() == "hello pool\n" * 100


def test_hub_dataset_id_not_shipped():
    from onepool.jobs import TrainJob
    from onepool.train.distributed import dataset_payload

    job = TrainJob(model="m", dataset="roneneldan/TinyStories")
    assert dataset_payload(job) is None


def test_round_plan_prepends_calibration():
    from onepool.jobs import TrainJob
    from onepool.train.distributed import round_plan

    job = TrainJob(model="m", dataset="d", steps=200, inner_steps=100)
    assert round_plan(job) == [5, 100, 100]
    tiny = TrainJob(model="m", dataset="d", steps=6, inner_steps=3)
    assert round_plan(tiny) == [3, 3, 3]  # calibration never exceeds inner_steps


def test_scale_steps_proportional():
    from onepool.train.distributed import scale_steps

    assert scale_steps(100, 5.0, 10.0) == 50  # half speed -> half steps
    assert scale_steps(100, 10.0, 10.0) == 100
    assert scale_steps(100, 0.1, 10.0) == 10  # 10% floor
    assert scale_steps(100, None, 10.0) == 100  # unknown speed -> full round
    assert scale_steps(100, 5.0, 0.0) == 100


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
