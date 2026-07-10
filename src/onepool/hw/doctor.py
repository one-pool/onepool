"""Diagnose this node's readiness and recommend the right PyTorch install.

The torch wheel a machine needs depends on its GPU vendor and OS — the single
biggest onboarding hurdle for a heterogeneous-hardware tool. ``onepool doctor``
turns that matrix into one concrete command for this machine.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

from onepool.hw.probe import NodeSpec, gpu_vendor

PYTORCH_URL = "https://pytorch.org/get-started/locally/"
AMD_WINDOWS_URL = "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/"


@dataclass
class Diagnosis:
    status: str  # "ready" | "needs-torch" | "wrong-torch" | "cpu-only"
    headline: str
    install_command: str | None = None
    notes: list[str] | None = None


def diagnose(spec: NodeSpec) -> Diagnosis:
    usable = spec.usable_accelerators
    system_only = [a for a in spec.accelerators if a.via == "system"]
    best_vendor = next(
        (v for a in system_only if (v := gpu_vendor(a.name)) in ("nvidia", "amd", "apple")),
        None,
    )

    if usable:
        names = ", ".join(a.name for a in usable)
        return Diagnosis(
            status="ready",
            headline=f"Ready. PyTorch {spec.torch.version} can drive: {names}",  # type: ignore[union-attr]
            notes=_leftover_notes(system_only),
        )

    if spec.torch is None:
        cmd, notes = _install_for(best_vendor, spec.os_name)
        return Diagnosis(
            status="needs-torch",
            headline="PyTorch is not installed.",
            install_command=cmd,
            notes=notes,
        )

    if best_vendor in ("nvidia", "amd"):
        cmd, notes = _install_for(best_vendor, spec.os_name)
        return Diagnosis(
            status="wrong-torch",
            headline=(
                f"PyTorch {spec.torch.version} is installed but cannot see your "
                f"{best_vendor.upper()} GPU (likely a CPU-only build or driver issue)."
            ),
            install_command=cmd,
            notes=notes,
        )

    return Diagnosis(
        status="cpu-only",
        headline=f"No usable GPU found. This node can still join a pool as a CPU worker "
        f"({spec.cpu_cores_logical} threads, {spec.ram_gb} GB RAM).",
    )


def _install_for(vendor: str | None, os_name: str) -> tuple[str, list[str]]:
    notes = [f"Exact wheel versions change; cross-check at {PYTORCH_URL}"]

    if vendor == "nvidia":
        return (
            "pip install torch --index-url https://download.pytorch.org/whl/cu128",
            notes + ["Requires a recent NVIDIA driver (no separate CUDA toolkit needed)."],
        )
    if vendor == "amd":
        if os_name == "Linux":
            return (
                "pip install torch --index-url https://download.pytorch.org/whl/rocm6.4",
                notes,
            )
        return (
            "pip install torch  # CPU fallback",
            notes
            + [
                "AMD on Windows: RDNA3/RDNA4 GPUs have a native PyTorch preview — "
                f"see {AMD_WINDOWS_URL}",
                "Older AMD GPUs on Windows fall back to CPU for now.",
            ],
        )
    if vendor == "apple" or platform.system() == "Darwin":
        return ("pip install torch  # includes Apple Silicon (MPS) support", notes)
    return ("pip install torch  # CPU build", notes)


def _leftover_notes(system_only: list) -> list[str]:
    notes = []
    for a in system_only:
        vendor = gpu_vendor(a.name)
        if vendor in ("nvidia", "amd"):
            notes.append(f"Detected but not usable by torch yet: {a.name}")
    return notes
