"""The pool host: accepts joins, tracks liveness, broadcasts membership.

The machine that runs ``onepool up`` becomes the host. Hub-and-spoke on
purpose: it matches the star all-reduce planned for training, and a session
tool doesn't need leader election — if the host goes away, the session is over.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field

from onepool.hw.probe import NodeSpec
from onepool.net import protocol, tlsutil
from onepool.pool import Member, PoolState, new_member_id
from onepool.session import SessionCode, new_nonce

log = logging.getLogger(__name__)

HEARTBEAT_TIMEOUT = 12.0  # seconds without a ping before a member is dropped
SWEEP_INTERVAL = 3.0
AUTH_FAILURE_DELAY = 1.0  # throttles online guessing of session codes


@dataclass
class PoolHost:
    session: SessionCode
    spec: NodeSpec
    state: PoolState = field(init=False)
    port: int = field(init=False, default=0)
    fingerprint: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.state = PoolState(self.session.code)
        # training messages land here for the coordinator: (member_id, msg)
        self.inbox: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self._writers: dict[str, asyncio.StreamWriter] = {}
        self._server: asyncio.Server | None = None
        self._sweeper: asyncio.Task | None = None

    async def start(self) -> None:
        cert_pem, key_pem, fp = tlsutil.make_session_identity()
        self.fingerprint = fp
        ctx = tlsutil.host_ssl_context(cert_pem, key_pem)
        self._server = await asyncio.start_server(self._handle, host="0.0.0.0", port=0, ssl=ctx)
        self.port = self._server.sockets[0].getsockname()[1]
        self.state.add(Member.from_spec(self.spec, is_host=True))
        self._sweeper = asyncio.create_task(self._sweep_stale())

    async def stop(self) -> None:
        if self._sweeper:
            self._sweeper.cancel()
        for writer in list(self._writers.values()):
            writer.close()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        member_id: str | None = None
        try:
            member_id = await self._handshake(reader, writer)
            if member_id is None:
                return
            await self._serve_member(member_id, reader)
        except (asyncio.IncompleteReadError, ConnectionError, protocol.ProtocolError) as e:
            log.debug("connection ended: %s", e)
        finally:
            if member_id and self.state.remove(member_id):
                self._writers.pop(member_id, None)
                await self._broadcast_members()
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _handshake(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> str | None:
        hello = await protocol.expect(reader, protocol.HELLO)
        if hello.get("code_id") != self.session.code_id:
            await self._reject(writer, "unknown session")
            return None

        host_nonce = new_nonce()
        await protocol.write_frame(writer, {"t": protocol.CHALLENGE, "nonce": host_nonce})

        auth = await protocol.expect(reader, protocol.AUTH)
        expected = self.session.auth_mac(host_nonce, hello["nonce"], self.fingerprint)
        if not _constant_time_eq(auth.get("mac", b""), expected):
            await asyncio.sleep(AUTH_FAILURE_DELAY)
            await self._reject(writer, "authentication failed")
            return None

        machine_id = (auth.get("node") or {}).get("machine_id")
        if machine_id and not _same_machine_allowed():
            already = any(
                m.spec.get("machine_id") == machine_id for m in self.state.members.values()
            )
            if already:
                await self._reject(
                    writer,
                    "this machine is already in the pool — one node per machine "
                    "(set ONEPOOL_ALLOW_SAME_MACHINE=1 on the host to override for testing)",
                )
                return None

        member = Member(member_id=new_member_id(), spec=auth["node"])
        self.state.add(member)
        self._writers[member.member_id] = writer
        await protocol.write_frame(
            writer,
            {
                "t": protocol.WELCOME,
                "member_id": member.member_id,
                "members": self.state.snapshot()["members"],
            },
        )
        await self._broadcast_members(exclude=member.member_id)
        log.info("member joined: %s (%s)", member.member_id, member.spec.get("hostname"))
        return member.member_id

    async def _serve_member(self, member_id: str, reader: asyncio.StreamReader) -> None:
        writer = self._writers[member_id]
        while True:
            msg = await protocol.read_frame(reader)
            if msg["t"] == protocol.PING:
                self.state.touch(member_id)
                await protocol.write_frame(writer, {"t": protocol.PONG})
            elif msg["t"] == protocol.LEAVE:
                return
            elif msg["t"] in protocol.TRAIN_TYPES:
                self.state.touch(member_id)  # a training frame proves liveness too
                await self.inbox.put((member_id, msg))

    async def send_to(self, member_id: str, msg: dict) -> bool:
        writer = self._writers.get(member_id)
        if writer is None:
            return False
        try:
            await protocol.write_frame(writer, msg)
            return True
        except (ConnectionError, OSError):
            return False

    async def _broadcast_members(self, exclude: str | None = None) -> None:
        members = self.state.snapshot()["members"]
        for member_id, writer in list(self._writers.items()):
            if member_id == exclude:
                continue
            # on failure the sweeper or the member's read loop cleans it up
            with contextlib.suppress(ConnectionError, OSError):
                await protocol.write_frame(writer, {"t": protocol.MEMBERS, "members": members})

    async def _sweep_stale(self) -> None:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL)
            dropped = False
            for member in self.state.stale(HEARTBEAT_TIMEOUT):
                log.info("member timed out: %s", member.member_id)
                self.state.remove(member.member_id)
                writer = self._writers.pop(member.member_id, None)
                if writer:
                    writer.close()
                dropped = True
            if dropped:
                await self._broadcast_members()

    async def _reject(self, writer: asyncio.StreamWriter, reason: str) -> None:
        with contextlib.suppress(ConnectionError, OSError):
            await protocol.write_frame(writer, {"t": protocol.REJECT, "reason": reason})


def _constant_time_eq(a: bytes, b: bytes) -> bool:
    import hmac

    return isinstance(a, bytes) and hmac.compare_digest(a, b)


def _same_machine_allowed() -> bool:
    import os

    return os.environ.get("ONEPOOL_ALLOW_SAME_MACHINE") == "1"
