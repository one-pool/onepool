"""Dashboard server lifecycle tests."""

import asyncio
import json

import pytest
import websockets

from onepool.dash.app import serve, shutdown
from onepool.pool import PoolState


@pytest.fixture
async def dash():
    state = PoolState("test-pool")
    server, port = await serve(state, port=17070)
    yield state, server, port
    await shutdown(server)


async def test_pool_api(dash):
    state, _, port = dash
    import urllib.request

    def fetch() -> dict:  # blocking client must not run on the server's own loop
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/pool", timeout=5) as resp:
            return json.load(resp)

    data = await asyncio.to_thread(fetch)
    assert data["session_code"] == "test-pool"


async def test_websocket_receives_updates(dash):
    state, _, port = dash
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
        first = json.loads(await ws.recv())
        assert first["session_code"] == "test-pool"
        state.update_job(model="m", round=1)
        update = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert update["job"]["round"] == 1


async def test_shutdown_with_open_websocket_is_clean():
    """The exact field-reported failure: browser tab still connected at teardown."""
    state = PoolState("test-pool")
    server, port = await serve(state, port=17090)
    ws = await websockets.connect(f"ws://127.0.0.1:{port}/ws")
    await ws.recv()  # initial snapshot — connection is live

    # must complete without raising and without leaving the serve task dangling
    await asyncio.wait_for(shutdown(server), timeout=10)

    task = server._onepool_task
    assert task.done()
    assert task.exception() is None  # no CancelledError escaped uvicorn
