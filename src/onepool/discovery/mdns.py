"""mDNS advertisement and lookup for pools.

The host advertises ``<code_id>._onepool._tcp.local.`` where ``code_id`` is a
salted hash of the session code — someone sniffing mDNS learns that a pool
exists, but not its code. Joiners know the code, derive the same ``code_id``,
and resolve the service by exact name (no browsing needed). The TXT record
carries the host's TLS certificate fingerprint, which the join handshake then
binds cryptographically.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

from zeroconf import IPVersion
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

SERVICE_TYPE = "_onepool._tcp.local."
PROTOCOL_VERSION = "1"


def _service_name(code_id: str) -> str:
    return f"{code_id}.{SERVICE_TYPE}"


@dataclass
class PoolLocation:
    host: str
    port: int
    fingerprint: str | None


class PoolAdvertisement:
    """Keeps a pool visible on the LAN for the lifetime of the session."""

    def __init__(self, code_id: str, port: int, fingerprint: str) -> None:
        self._azc = AsyncZeroconf(ip_version=IPVersion.V4Only)
        self._info = AsyncServiceInfo(
            SERVICE_TYPE,
            _service_name(code_id),
            port=port,
            properties={"v": PROTOCOL_VERSION, "fp": fingerprint},
            server=f"{socket.gethostname()}.local.",
            addresses=[socket.inet_aton(_primary_ipv4())],
        )

    async def start(self) -> None:
        await self._azc.async_register_service(self._info)

    async def stop(self) -> None:
        await self._azc.async_unregister_service(self._info)
        await self._azc.async_close()


async def find_pool(code_id: str, timeout: float = 5.0) -> PoolLocation | None:
    """Resolve the pool advertised for this code, or None if nothing answers."""
    azc = AsyncZeroconf(ip_version=IPVersion.V4Only)
    try:
        info = AsyncServiceInfo(SERVICE_TYPE, _service_name(code_id))
        if not await info.async_request(azc.zeroconf, timeout * 1000):
            return None
        addresses = info.parsed_addresses(IPVersion.V4Only)
        if not addresses:
            return None
        props = {
            k.decode(): v.decode() if isinstance(v, bytes) else v
            for k, v in (info.properties or {}).items()
            if v is not None
        }
        return PoolLocation(host=addresses[0], port=info.port or 0, fingerprint=props.get("fp"))
    finally:
        await azc.async_close()


def _primary_ipv4() -> str:
    """The LAN address this machine would use to reach other machines.

    The UDP 'connection' never sends a packet; it just asks the OS which
    interface routes outward — the standard trick for multi-homed machines.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 80))  # TEST-NET address, never actually reached
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()
