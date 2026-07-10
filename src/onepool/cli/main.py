"""Entry point for the ``onepool`` command."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import onepool
from onepool.hw.doctor import diagnose
from onepool.hw.probe import NodeSpec, probe

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
def up() -> None:
    """Start this node and join (or form) the pool on the local network."""
    with console.status("Probing hardware..."):
        spec = probe()
    _print_node_card(spec)
    console.print(
        Panel(
            "Cluster networking lands in the next milestone (M1: discovery + join).\n"
            "For now, [cyan]onepool doctor[/cyan] verifies this node is ready to pool.",
            title="coming up",
            border_style="yellow",
        )
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
