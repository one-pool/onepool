"""Dashboard server: pool state as JSON, live updates over WebSocket."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.resources
import logging
import socket

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from onepool.pool import PoolState, wake

log = logging.getLogger(__name__)

DEFAULT_PORT = 7070


def build_app(state: PoolState) -> FastAPI:
    app = FastAPI(title="onepool", docs_url=None, redoc_url=None)
    page = (
        importlib.resources.files("onepool.dash").joinpath("index.html").read_text("utf-8")
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return page

    @app.get("/api/pool")
    async def pool() -> dict:
        return state.snapshot()

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        changed = asyncio.Event()
        state.on_change(wake(changed))
        try:
            await socket.send_json(state.snapshot())
            while True:
                # push on change, and every few seconds as a keepalive/refresh
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(changed.wait(), timeout=5.0)
                changed.clear()
                await socket.send_json(state.snapshot())
        except WebSocketDisconnect:
            pass

    return app


async def serve(state: PoolState, port: int = DEFAULT_PORT) -> tuple[uvicorn.Server, int]:
    """Start the dashboard, scanning past ports Windows reserves or others hold.

    The socket is bound here (clean OSError on a bad port) and handed to
    uvicorn — letting uvicorn bind would end in sys.exit(3) on failure, which
    would tear down the whole pool process.
    """
    app = build_app(state)
    for candidate in range(port, port + 20):
        try:
            sock = socket.create_server(("127.0.0.1", candidate))
        except OSError:
            continue
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        task = asyncio.create_task(_serve_guarded(server, sock))
        while not server.started and not task.done():
            await asyncio.sleep(0.05)
        if task.done():
            sock.close()
            continue
        return server, candidate
    raise OSError(f"no usable dashboard port in {port}..{port + 19}")


async def _serve_guarded(server: uvicorn.Server, sock: socket.socket) -> None:
    try:
        await server.serve(sockets=[sock])
    except SystemExit:  # uvicorn startup failure must not kill the pool process
        log.warning("dashboard server exited during startup")
    except asyncio.CancelledError:  # normal teardown when the pool closes
        pass
    finally:
        sock.close()
