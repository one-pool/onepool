"""The joining side of a pool: authenticate, report hardware, stay alive."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from onepool.hw.probe import NodeSpec
from onepool.net import protocol, tlsutil
from onepool.session import SessionCode, new_nonce

log = logging.getLogger(__name__)

PING_INTERVAL = 3.0
CONNECT_TIMEOUT = 10.0


class JoinRejected(Exception):
    pass


@dataclass
class PoolClient:
    session: SessionCode
    spec: NodeSpec
    member_id: str = field(init=False, default="")
    members: list[dict[str, Any]] = field(init=False, default_factory=list)
    on_members_changed: Callable[[list[dict[str, Any]]], None] | None = None

    def __post_init__(self) -> None:
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self, host: str, port: int, expected_fingerprint: str | None) -> None:
        """Join the pool at host:port. Raises JoinRejected on auth failure."""
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=tlsutil.client_ssl_context()),
            timeout=CONNECT_TIMEOUT,
        )
        seen_fp = tlsutil.peer_fingerprint(self._writer)
        if expected_fingerprint and seen_fp != expected_fingerprint:
            raise JoinRejected(
                "TLS certificate does not match the pool advertisement — "
                "possible interception, refusing to join"
            )

        client_nonce = new_nonce()
        await protocol.write_frame(
            self._writer,
            {"t": protocol.HELLO, "code_id": self.session.code_id, "nonce": client_nonce},
        )
        try:
            challenge = await protocol.expect(self._reader, protocol.CHALLENGE)
            # The MAC binds the certificate we actually saw; the host verifies it
            # against its own certificate, closing the machine-in-the-middle window.
            mac = self.session.auth_mac(challenge["nonce"], client_nonce, seen_fp)
            await protocol.write_frame(
                self._writer, {"t": protocol.AUTH, "mac": mac, "node": asdict(self.spec)}
            )
            welcome = await protocol.expect(self._reader, protocol.WELCOME)
        except protocol.ProtocolError as e:
            raise JoinRejected(str(e)) from e
        self.member_id = welcome["member_id"]
        self.members = welcome["members"]

    async def run(self) -> None:
        """Heartbeat + membership updates until cancelled or the host goes away."""
        assert self._reader and self._writer
        ping_task = asyncio.create_task(self._ping_loop())
        try:
            while True:
                msg = await protocol.read_frame(self._reader)
                if msg["t"] == protocol.MEMBERS:
                    self.members = msg["members"]
                    if self.on_members_changed:
                        self.on_members_changed(self.members)
        except (asyncio.IncompleteReadError, ConnectionError):
            log.info("lost connection to pool host")
        finally:
            ping_task.cancel()

    async def leave(self) -> None:
        if self._writer:
            with contextlib.suppress(ConnectionError, OSError):
                await protocol.write_frame(self._writer, {"t": protocol.LEAVE})
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()

    async def _ping_loop(self) -> None:
        assert self._writer
        while True:
            await asyncio.sleep(PING_INTERVAL)
            with contextlib.suppress(ConnectionError, OSError):
                await protocol.write_frame(self._writer, {"t": protocol.PING})
