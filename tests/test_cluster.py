"""Integration tests: host + clients forming a pool over loopback TLS.

mDNS is deliberately not exercised here (flaky on CI runners); clients connect
directly, which is also a supported real-world path (--host).
"""

import asyncio

import pytest

from onepool.hw.probe import NodeSpec
from onepool.net.client import JoinRejected, PoolClient
from onepool.net.server import PoolHost
from onepool.session import SessionCode


def _spec(name: str, machine_id: str | None = None) -> NodeSpec:
    return NodeSpec(
        hostname=name,
        os_name="TestOS",
        os_version="1",
        arch="x86_64",
        python_version="3.12.0",
        cpu_name="Test CPU",
        cpu_cores_physical=2,
        cpu_cores_logical=4,
        ram_gb=8.0,
        machine_id=machine_id if machine_id is not None else f"machine-{name}",
    )


@pytest.fixture
async def host():
    session = SessionCode.parse("amber-fox-73")
    pool_host = PoolHost(session=session, spec=_spec("host-node"))
    await pool_host.start()
    yield pool_host
    await pool_host.stop()


async def test_join_and_membership(host: PoolHost):
    client = PoolClient(session=host.session, spec=_spec("worker-1"))
    await client.connect("127.0.0.1", host.port, host.fingerprint)

    assert client.member_id
    assert len(client.members) == 2  # host + this worker
    assert len(host.state.members) == 2
    await client.leave()


async def test_wrong_code_rejected(host: PoolHost):
    imposter = PoolClient(session=SessionCode.parse("bold-owl-42"), spec=_spec("imposter"))
    with pytest.raises(JoinRejected):
        await imposter.connect("127.0.0.1", host.port, host.fingerprint)
    assert len(host.state.members) == 1  # only the host


async def test_wrong_fingerprint_refused(host: PoolHost):
    client = PoolClient(session=host.session, spec=_spec("cautious"))
    with pytest.raises(JoinRejected, match="certificate"):
        await client.connect("127.0.0.1", host.port, "not-the-real-fingerprint")


async def test_same_machine_join_rejected(host: PoolHost, monkeypatch):
    monkeypatch.delenv("ONEPOOL_ALLOW_SAME_MACHINE", raising=False)
    # host's spec is _spec("host-node") -> machine_id "machine-host-node"
    twin = PoolClient(session=host.session, spec=_spec("twin", machine_id="machine-host-node"))
    with pytest.raises(JoinRejected, match="already in the pool"):
        await twin.connect("127.0.0.1", host.port, host.fingerprint)
    assert len(host.state.members) == 1


async def test_same_machine_join_allowed_with_override(host: PoolHost, monkeypatch):
    monkeypatch.setenv("ONEPOOL_ALLOW_SAME_MACHINE", "1")
    twin = PoolClient(session=host.session, spec=_spec("twin", machine_id="machine-host-node"))
    await twin.connect("127.0.0.1", host.port, host.fingerprint)
    assert len(host.state.members) == 2
    await twin.leave()


async def test_blank_machine_id_not_treated_as_duplicate(host: PoolHost, monkeypatch):
    monkeypatch.delenv("ONEPOOL_ALLOW_SAME_MACHINE", raising=False)
    # old clients (pre-fix) send no machine_id; two of them must still both join
    a = PoolClient(session=host.session, spec=_spec("old-a", machine_id=""))
    b = PoolClient(session=host.session, spec=_spec("old-b", machine_id=""))
    await a.connect("127.0.0.1", host.port, host.fingerprint)
    await b.connect("127.0.0.1", host.port, host.fingerprint)
    assert len(host.state.members) == 3
    await a.leave()
    await b.leave()


async def test_disconnect_removes_member(host: PoolHost):
    client = PoolClient(session=host.session, spec=_spec("flaky"))
    await client.connect("127.0.0.1", host.port, host.fingerprint)
    assert len(host.state.members) == 2

    await client.leave()
    for _ in range(50):  # host notices the closed connection promptly
        if len(host.state.members) == 1:
            break
        await asyncio.sleep(0.1)
    assert len(host.state.members) == 1


async def test_second_client_sees_first(host: PoolHost):
    first = PoolClient(session=host.session, spec=_spec("worker-1"))
    await first.connect("127.0.0.1", host.port, host.fingerprint)

    updates: list[int] = []
    first.on_members_changed = lambda members: updates.append(len(members))
    run_task = asyncio.create_task(first.run())

    second = PoolClient(session=host.session, spec=_spec("worker-2"))
    await second.connect("127.0.0.1", host.port, host.fingerprint)
    assert len(second.members) == 3

    for _ in range(50):  # first client receives the broadcast
        if updates and updates[-1] == 3:
            break
        await asyncio.sleep(0.1)
    assert updates and updates[-1] == 3

    run_task.cancel()
    await second.leave()
    await first.leave()
