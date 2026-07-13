"""Entry point for the ``onepool`` command."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import urllib.error
import urllib.request

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

import onepool
from onepool.dash.app import DEFAULT_PORT as DASH_PORT
from onepool.dash.app import serve as dash_serve
from onepool.dash.app import shutdown as dash_shutdown
from onepool.discovery import PoolAdvertisement, find_pool
from onepool.hw.doctor import diagnose
from onepool.hw.probe import NodeSpec, probe
from onepool.net.client import JoinRejected, PoolClient
from onepool.net.server import PoolHost
from onepool.session import SessionCode

app = typer.Typer(
    name="onepool",
    help="Turn the laptops in the room into one pool of ML compute.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_BACKEND_LABELS = {
    "cuda": "CUDA",
    "rocm": "ROCm",
    "mps": "MPS",
    "directml": "DirectML",
    "cpu": "CPU",
}


@app.command()
def version() -> None:
    """Print the onepool version."""
    console.print(f"onepool {onepool.__version__}")


@app.command()
def doctor() -> None:
    """Check this machine's hardware and PyTorch setup for pool readiness."""
    with console.status("Probing hardware..."):
        spec = probe()
    _print_node_card(spec)

    diag = diagnose(spec)
    style = {"ready": "green", "cpu-only": "yellow"}.get(diag.status, "red")
    console.print(Panel(diag.headline, title="diagnosis", border_style=style))

    if diag.install_command:
        console.print("[bold]To fix, run:[/bold]")
        console.print(f"  [cyan]{diag.install_command}[/cyan]")
    for note in diag.notes or []:
        console.print(f"[dim]  - {note}[/dim]")


@app.command()
def up(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show network debug logs."),
) -> None:
    """Start a pool on this machine and wait for others to join."""
    _setup_logging(verbose)
    with console.status("Probing hardware..."):
        spec = probe()

    try:
        asyncio.run(_run_host(spec))
    except KeyboardInterrupt:
        console.print("\n[dim]pool closed. nothing left running.[/dim]")


@app.command()
def join(
    code: str = typer.Argument(..., help="Session code shown by 'onepool up', e.g. amber-fox-73"),
    host: str = typer.Option(
        None,
        "--host",
        help="Direct HOST:PORT of the pool, for networks that block mDNS.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show network debug logs."),
) -> None:
    """Join a pool on the local network using its session code."""
    _setup_logging(verbose)
    try:
        session = SessionCode.parse(code)
    except ValueError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(1) from None

    with console.status("Probing hardware..."):
        spec = probe()

    try:
        asyncio.run(_run_client(session, spec, host))
    except KeyboardInterrupt:
        console.print("\n[dim]left the pool.[/dim]")


@app.command()
def train(
    config: str = typer.Argument(..., help="Path to a job YAML (see examples/)."),
    pool: bool = typer.Option(
        False, "--pool", help="Host a pool and train across every machine that joins."
    ),
    nodes: int = typer.Option(
        0, "--nodes", help="With --pool: wait for this many workers before starting."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show training debug logs."),
) -> None:
    """Run a LoRA fine-tuning job — on this machine, or across a pool with --pool."""
    _setup_logging(verbose)
    from onepool.jobs import TrainJob
    from onepool.train import require_training_stack

    try:
        job = TrainJob.from_yaml(config)
    except (OSError, ValueError) as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(1) from None
    try:
        require_training_stack()
    except ImportError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(1) from None

    if pool:
        with console.status("Probing hardware..."):
            spec = probe()
        try:
            asyncio.run(_run_pool_train(job, spec, min_workers=nodes))
        except KeyboardInterrupt:
            console.print("\n[dim]training pool closed.[/dim]")
        return

    try:
        asyncio.run(_run_solo_train(job))
    except KeyboardInterrupt:
        console.print("\n[dim]training stopped.[/dim]")


@app.command()
def status() -> None:
    """Show the pool this machine is hosting (reads the local dashboard API)."""
    for port in range(DASH_PORT, DASH_PORT + 10):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/pool", timeout=2) as resp:
                state = json.load(resp)
            break
        except (urllib.error.URLError, OSError):
            continue
    else:
        console.print("[yellow]no pool is hosted on this machine (run 'onepool up').[/yellow]")
        raise typer.Exit(1)

    table = Table(title=f"pool: {state['session_code']}", title_justify="left")
    table.add_column("node", style="bold")
    table.add_column("os")
    table.add_column("hardware")
    table.add_column("role")
    for m in state["members"]:
        gpus = ", ".join(
            f"{g['name']} ({'ready' if g['usable'] else 'detected'})" for g in m["gpus"]
        )
        hw = gpus or f"CPU x{m['cores']}, {m['ram_gb']} GB RAM"
        table.add_row(m["hostname"], m["os"], hw, "host" if m["is_host"] else "worker")
    console.print(table)


async def _run_host(spec: NodeSpec) -> None:
    session = SessionCode.generate()
    host = PoolHost(session=session, spec=spec)
    await host.start()
    advert = PoolAdvertisement(session.code_id, host.port, host.fingerprint)
    await advert.start()
    dash_server, dash_port = await dash_serve(host.state)

    console.print(
        Panel(
            f"session code:  [bold green]{session.code}[/bold green]\n"
            f"dashboard:     [cyan]http://localhost:{dash_port}[/cyan]\n"
            f"direct joins:  [dim]onepool join {session.code} --host <this-ip>:{host.port}[/dim]",
            title="pool is up",
            border_style="green",
        )
    )
    console.print("[dim]others on this network can now run:[/dim] "
                  f"[bold]onepool join {session.code}[/bold]")
    console.print("[dim]Ctrl-C to close the pool.[/dim]\n")

    def on_change() -> None:
        n = len(host.state.members)
        console.print(f"[green]pool now has {n} node{'s' if n != 1 else ''}[/green]")

    host.state.on_change(on_change)

    try:
        await asyncio.Event().wait()  # run until Ctrl-C
    finally:
        await dash_shutdown(dash_server)
        with contextlib.suppress(Exception):
            await advert.stop()
        await host.stop()


async def _run_solo_train(job) -> None:
    """Single-machine training, with the same live dashboard a pool gets."""
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from onepool.pool import Member, PoolState
    from onepool.train.local import pick_device, run_local

    with console.status("Probing hardware..."):
        spec = probe()
    state = PoolState("local training")
    state.add(Member.from_spec(spec, is_host=True))
    dash_server, dash_port = await dash_serve(state)

    choice = pick_device(job.precision)
    console.print(
        Panel(
            f"model:     [bold]{job.model}[/bold]\n"
            f"dataset:   {job.dataset}\n"
            f"device:    {choice.name} ({choice.device}, {str(choice.dtype).split('.')[-1]})\n"
            f"plan:      {job.steps} steps in rounds of {job.inner_steps} "
            f"(batch {job.batch_size} x accum {job.grad_accum}, seq {job.seq_len})\n"
            f"dashboard: [cyan]http://localhost:{dash_port}[/cyan]",
            title="training job",
            border_style="cyan",
        )
    )

    loop = asyncio.get_running_loop()
    loss_history: list[float] = []
    total_rounds = len(job.rounds)
    state.update_job(
        model=job.model, round=0, total_rounds=total_rounds, loss_history=[], workers=0
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} steps"),
        TextColumn("loss {task.fields[loss]:.4f}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("training", total=job.steps, loss=float("nan"))

        def on_step(step: int, loss: float) -> None:  # runs in the training thread
            progress.update(bar, completed=step, loss=loss)

        def on_round(stats) -> None:  # runs in the training thread
            loss_history.append(round(stats.mean_loss, 4))
            snapshot = list(loss_history)
            loop.call_soon_threadsafe(
                lambda: state.update_job(round=stats.round_index + 1, loss_history=snapshot)
            )
            progress.console.print(
                f"[dim]round {stats.round_index}: loss {stats.mean_loss:.4f}, "
                f"{stats.tokens_per_second:.0f} tok/s — checkpoint saved[/dim]"
            )

        try:
            stats = await asyncio.to_thread(run_local, job, on_step, on_round)
        finally:
            await dash_shutdown(dash_server)

    final = stats[-1] if stats else None
    console.print(
        Panel(
            (f"final loss:  [bold]{final.mean_loss:.4f}[/bold]\n" if final else "")
            + f"adapters:    [cyan]{job.output_dir}/final[/cyan]",
            title="done",
            border_style="green",
        )
    )


async def _run_pool_train(job, spec: NodeSpec, min_workers: int) -> None:
    from onepool.train.distributed import Coordinator, round_plan

    session = SessionCode.generate()
    host = PoolHost(session=session, spec=spec)
    await host.start()
    advert = PoolAdvertisement(session.code_id, host.port, host.fingerprint)
    await advert.start()
    dash_server, dash_port = await dash_serve(host.state)

    console.print(
        Panel(
            f"session code:  [bold green]{session.code}[/bold green]\n"
            f"dashboard:     [cyan]http://localhost:{dash_port}[/cyan]\n"
            f"workers join:  [bold]onepool join {session.code}[/bold]"
            f"  [dim](or --host <this-ip>:{host.port})[/dim]",
            title="training pool is up",
            border_style="green",
        )
    )

    if min_workers > 0:
        with console.status(f"waiting for {min_workers} worker(s) to join..."):
            while len(host.state.members) - 1 < min_workers:
                await asyncio.sleep(0.5)
    n_workers = len(host.state.members) - 1
    console.print(f"[green]starting with {n_workers} worker(s) + this machine[/green]\n")

    loss_history: list[float] = []
    total_rounds = len(round_plan(job))  # includes the short calibration round 0
    host.state.update_job(
        model=job.model, round=0, total_rounds=total_rounds, loss_history=[], workers=n_workers
    )

    def on_round(rnd: int, loss: float, participants: int) -> None:
        loss_history.append(round(loss, 4))
        host.state.update_job(
            round=rnd + 1, loss_history=loss_history, workers=participants - 1
        )
        label = " (calibration)" if rnd == 0 else ""
        console.print(
            f"round {rnd + 1}/{total_rounds}{label}: loss [bold]{loss:.4f}[/bold] "
            f"({participants} node{'s' if participants != 1 else ''})"
        )

    coordinator = Coordinator(
        host=host,
        job=job,
        on_round=on_round,
        on_log=lambda text: console.print(f"[dim]{text}[/dim]"),
    )
    try:
        stats = await coordinator.run()
        final = stats[-1].mean_loss if stats else float("nan")
        console.print(
            Panel(
                f"final loss:  [bold]{final:.4f}[/bold]\n"
                f"adapters:    [cyan]{job.output_dir}/final[/cyan]",
                title="done",
                border_style="green",
            )
        )
    finally:
        await dash_shutdown(dash_server)
        with contextlib.suppress(Exception):
            await advert.stop()
        await host.stop()


async def _run_client(session: SessionCode, spec: NodeSpec, direct: str | None) -> None:
    if direct:
        host_addr, _, port_text = direct.partition(":")
        if not port_text.isdigit():
            console.print("[red]--host must be HOST:PORT[/red]")
            raise typer.Exit(1)
        host_ip, port, fp = host_addr, int(port_text), None
    else:
        with console.status(f"Looking for pool [bold]{session.code}[/bold] on this network..."):
            location = await find_pool(session.code_id)
        if location is None:
            console.print(
                "[red]no pool found for that code.[/red]\n"
                "[dim]- is 'onepool up' running on the host machine?\n"
                "- same WiFi/LAN? some networks block mDNS: ask the host for its IP and use\n"
                "  onepool join CODE --host IP:PORT (shown on the host's screen)[/dim]"
            )
            raise typer.Exit(1)
        host_ip, port, fp = location.host, location.port, location.fingerprint

    client = PoolClient(session=session, spec=spec)
    try:
        await client.connect(host_ip, port, fp)
    except JoinRejected as e:
        console.print(f"[red]join failed: {escape(str(e))}[/red]")
        raise typer.Exit(1) from None
    except (ConnectionError, OSError, asyncio.TimeoutError):
        console.print(f"[red]could not reach pool at {host_ip}:{port}[/red]")
        raise typer.Exit(1) from None

    console.print(
        Panel(
            f"joined pool [bold green]{session.code}[/bold green] at {host_ip}:{port}\n"
            f"this node:  [bold]{spec.hostname}[/bold] (member {client.member_id})\n"
            f"pool size:  {len(client.members)} node(s)",
            title="connected",
            border_style="green",
        )
    )
    console.print("[dim]contributing to the pool. Ctrl-C to leave.[/dim]\n")

    client.on_members_changed = lambda members: console.print(
        f"[green]pool now has {len(members)} node{'s' if len(members) != 1 else ''}[/green]"
    )

    from onepool.train.distributed import worker_loop

    worker = asyncio.create_task(
        worker_loop(client, on_status=lambda text: console.print(f"[cyan]{text}[/cyan]"))
    )
    try:
        await client.run()
        console.print("[yellow]pool host went away — session over.[/yellow]")
    finally:
        worker.cancel()
        await client.leave()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def _print_node_card(spec: NodeSpec) -> None:
    table = Table(title=f"node: {spec.hostname}", title_justify="left")
    table.add_column("component", style="bold")
    table.add_column("details")

    table.add_row("os", f"{spec.os_name} {spec.os_version} ({spec.arch})")
    table.add_row("python", spec.python_version)
    table.add_row(
        "cpu",
        f"{spec.cpu_name}  "
        f"[dim]{spec.cpu_cores_physical}c/{spec.cpu_cores_logical}t[/dim]",
    )
    table.add_row("ram", f"{spec.ram_gb} GB")

    if spec.torch:
        runtime = spec.torch.cuda_version or spec.torch.hip_version
        suffix = f" (+{'hip' if spec.torch.hip_version else 'cuda'} {runtime})" if runtime else ""
        table.add_row("pytorch", f"{spec.torch.version}{suffix}")
    else:
        table.add_row("pytorch", "[red]not installed[/red]")

    if spec.accelerators:
        for a in spec.accelerators:
            vram = f"{a.vram_gb} GB" if a.vram_gb else "unknown VRAM"
            usable = "[green]usable[/green]" if a.via == "torch" else "[yellow]detected[/yellow]"
            backend = _BACKEND_LABELS.get(a.backend, a.backend)
            table.add_row(
                f"gpu {a.index}",
                f"{a.name}  [dim]{backend}, {vram}[/dim]  {usable}",
            )
    else:
        table.add_row("gpu", "[dim]none detected[/dim]")

    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
