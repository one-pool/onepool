"""Serialize named float32 tensors for the wire (msgpack-friendly).

Everything that crosses the network — adapter states, pseudo-gradients — is a
``dict[str, np.ndarray]`` in float32, packed as raw little-endian bytes with
shape metadata. No pickle: peers exchange data, never executable objects.

Pseudo-gradients support int8 quantization (symmetric, per-tensor scale):
4x smaller sync payloads for negligible quality cost — DiLoCo deltas are
famously tolerant of it. Canonical weights always travel in float32.
"""

from __future__ import annotations

import numpy as np

WireState = dict[str, dict]  # name -> {"shape": [...], "data": bytes, "enc": ..., "scale": ...}


def pack_state(state: dict[str, np.ndarray], compression: str = "none") -> WireState:
    out: WireState = {}
    for name, arr in state.items():
        arr32 = np.ascontiguousarray(arr, dtype="<f4")
        if compression == "int8":
            scale = float(np.abs(arr32).max()) / 127.0
            if scale == 0.0:
                scale = 1.0  # all-zero tensor; quantizes to zeros either way
            q = np.clip(np.round(arr32 / scale), -127, 127).astype(np.int8)
            out[name] = {
                "shape": list(arr32.shape),
                "data": q.tobytes(),
                "enc": "i8",
                "scale": scale,
            }
        else:
            out[name] = {"shape": list(arr32.shape), "data": arr32.tobytes(), "enc": "f4"}
    return out


def unpack_state(wire: WireState) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, entry in wire.items():
        if entry.get("enc", "f4") == "i8":
            q = np.frombuffer(entry["data"], dtype=np.int8).reshape(entry["shape"])
            out[name] = q.astype(np.float32) * entry["scale"]
        else:
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
