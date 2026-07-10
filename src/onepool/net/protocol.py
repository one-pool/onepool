"""Wire protocol: length-prefixed msgpack frames over TLS/TCP.

Every message is a msgpack map with a ``t`` (type) key. Frames are capped so a
misbehaving peer cannot make the other side allocate unbounded memory.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any

import msgpack

# Large enough for LoRA pseudo-gradients / adapter states with headroom;
# still a hard stop against a peer streaming garbage lengths.
MAX_FRAME = 256 * 1024 * 1024

# Handshake
HELLO = "hello"  # client -> host: {t, code_id, nonce}
CHALLENGE = "challenge"  # host -> client: {t, nonce}
AUTH = "auth"  # client -> host: {t, mac, node}
WELCOME = "welcome"  # host -> client: {t, member_id, members}
REJECT = "reject"  # host -> client: {t, reason}

# Steady state
MEMBERS = "members"  # host -> all: {t, members}
PING = "ping"  # client -> host: {t}
PONG = "pong"  # host -> client: {t}
LEAVE = "leave"  # client -> host: {t}

# Training (DiLoCo rounds)
TRAIN_START = "train_start"  # host -> worker: {t, job, weights, shard, shards, steps, round}
ROUND_RESULT = "round_result"  # worker -> host: {t, round, steps, samples, loss, tok_s, delta}
ROUND_UPDATE = "round_update"  # host -> worker: {t, round, weights, shard, shards, steps}
TRAIN_END = "train_end"  # host -> worker: {t, reason}
TRAIN_ERR = "train_err"  # worker -> host: {t, error}

TRAIN_TYPES = {TRAIN_START, ROUND_RESULT, ROUND_UPDATE, TRAIN_END, TRAIN_ERR}


class ProtocolError(Exception):
    pass


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    header = await reader.readexactly(4)
    (length,) = struct.unpack(">I", header)
    if length > MAX_FRAME:
        raise ProtocolError(f"frame of {length} bytes exceeds cap {MAX_FRAME}")
    payload = await reader.readexactly(length)
    msg = msgpack.unpackb(payload)
    if not isinstance(msg, dict) or "t" not in msg:
        raise ProtocolError("malformed message: expected map with 't' key")
    return msg


async def write_frame(writer: asyncio.StreamWriter, msg: dict[str, Any]) -> None:
    payload = msgpack.packb(msg)
    writer.write(struct.pack(">I", len(payload)) + payload)
    await writer.drain()


async def expect(reader: asyncio.StreamReader, msg_type: str) -> dict[str, Any]:
    msg = await read_frame(reader)
    if msg["t"] == REJECT:
        raise ProtocolError(f"rejected by peer: {msg.get('reason', 'unspecified')}")
    if msg["t"] != msg_type:
        raise ProtocolError(f"expected {msg_type!r}, got {msg['t']!r}")
    return msg
