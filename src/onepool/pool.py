"""In-memory pool state shared by the host, dashboard, and (later) the trainer."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from onepool.hw.probe import NodeSpec


def new_member_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Member:
    member_id: str
    spec: dict[str, Any]  # serialized NodeSpec (crosses the wire as msgpack)
    is_host: bool = False
    joined_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @classmethod
    def from_spec(cls, spec: NodeSpec, *, is_host: bool = False) -> Member:
        return cls(member_id=new_member_id(), spec=asdict(spec), is_host=is_host)

    def summary(self) -> dict[str, Any]:
        accels = self.spec.get("accelerators") or []
        gpus = [
            {
                "name": a["name"],
                "backend": a["backend"],
                "vram_gb": a["vram_gb"],
                "usable": a["via"] == "torch",
            }
            for a in accels
        ]
        return {
            "member_id": self.member_id,
            "hostname": self.spec.get("hostname", "?"),
            "os": f"{self.spec.get('os_name', '?')} {self.spec.get('os_version', '')}".strip(),
            "cpu": self.spec.get("cpu_name", "?"),
            "cores": self.spec.get("cpu_cores_logical", 0),
            "ram_gb": self.spec.get("ram_gb", 0),
            "gpus": gpus,
            "is_host": self.is_host,
            "joined_at": self.joined_at,
        }


class PoolState:
    """Membership registry with change notifications (asyncio-side, not thread-safe)."""

    def __init__(self, session_code: str) -> None:
        self.session_code = session_code
        self.started_at = time.time()
        self.members: dict[str, Member] = {}
        self._listeners: list[Callable[[], None]] = []

    def add(self, member: Member) -> None:
        self.members[member.member_id] = member
        self._notify()

    def remove(self, member_id: str) -> Member | None:
        member = self.members.pop(member_id, None)
        if member:
            self._notify()
        return member

    def touch(self, member_id: str) -> None:
        member = self.members.get(member_id)
        if member:
            member.last_seen = time.time()

    def stale(self, timeout: float) -> list[Member]:
        cutoff = time.time() - timeout
        return [m for m in self.members.values() if not m.is_host and m.last_seen < cutoff]

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_code": self.session_code,
            "started_at": self.started_at,
            "members": [m.summary() for m in self.members.values()],
        }

    def on_change(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()


def wake(event: asyncio.Event) -> Callable[[], None]:
    """Adapter: PoolState change listener that sets an asyncio.Event."""

    def _listener() -> None:
        event.set()

    return _listener
