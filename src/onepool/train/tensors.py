"""Serialize named float32 tensors for the wire (msgpack-friendly).

Everything that crosses the network — adapter states, pseudo-gradients — is a
``dict[str, np.ndarray]`` in float32, packed as raw little-endian bytes with
shape metadata. No pickle: peers exchange data, never executable objects.
"""

from __future__ import annotations

import numpy as np

WireState = dict[str, dict]  # name -> {"shape": [...], "data": bytes}


def pack_state(state: dict[str, np.ndarray]) -> WireState:
    out: WireState = {}
    for name, arr in state.items():
        arr32 = np.ascontiguousarray(arr, dtype="<f4")
        out[name] = {"shape": list(arr32.shape), "data": arr32.tobytes()}
    return out


def unpack_state(wire: WireState) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, entry in wire.items():
        arr = np.frombuffer(entry["data"], dtype="<f4").reshape(entry["shape"])
        out[name] = arr.copy()  # own the memory; frombuffer views are read-only
    return out


def state_delta(start: dict[str, np.ndarray], end: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """DiLoCo pseudo-gradient: where local training moved away from the start."""
    return {name: start[name] - end[name] for name in start}


def weighted_average(
    deltas: list[dict[str, np.ndarray]], weights: list[float]
) -> dict[str, np.ndarray]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    names = deltas[0].keys()
    return {
        name: sum(w * d[name] for w, d in zip(weights, deltas, strict=True)) / total
        for name in names
    }
