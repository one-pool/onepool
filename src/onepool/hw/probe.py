"""Detect the compute hardware available on this node.

Two detection paths:

1. Through PyTorch, when it is installed — this is authoritative, because a GPU
   is only useful to onepool if torch can actually drive it.
2. Through the operating system (nvidia-smi, WMI, system_profiler, lspci) when
   torch is missing or blind to a device — used by ``onepool doctor`` to tell
   the user what *could* work with the right torch build installed.

Every accelerator records which path found it in ``via`` ("torch" / "system").
"""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field

import psutil

_GPU_VENDORS = {
    "nvidia": ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla"),
    "amd": ("amd", "radeon", "instinct", "firepro"),
    "intel": ("intel", "iris", "uhd graphics", "hd graphics", "arc"),
    "apple": ("apple",),
}


@dataclass
class Accelerator:
    backend: str  # "cuda" | "rocm" | "mps" | "directml" | "cpu"
    name: str
    vram_gb: float | None
    index: int = 0
    via: str = "torch"  # "torch" = usable now; "system" = present, torch can't drive it yet


@dataclass
class TorchInfo:
    version: str
    cuda_version: str | None
    hip_version: str | None


@dataclass
class NodeSpec:
    hostname: str
    os_name: str
    os_version: str
    arch: str
    python_version: str
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_gb: float
    machine_id: str = ""  # stable per-machine fingerprint; guards against self-joins
    accelerators: list[Accelerator] = field(default_factory=list)
    torch: TorchInfo | None = None

    @property
    def usable_accelerators(self) -> list[Accelerator]:
        return [a for a in self.accelerators if a.via == "torch"]


def probe() -> NodeSpec:
    """Build the full hardware spec for this machine."""
    torch_info, torch_accels = _probe_torch()
    accels = list(torch_accels)

    # Fill in devices the OS can see but torch can't (or torch is absent).
    torch_names = {a.name.lower() for a in accels}
    for sys_accel in _probe_system():
        if sys_accel.name.lower() not in torch_names:
            accels.append(sys_accel)
    for i, accel in enumerate(accels):
        accel.index = i

    return NodeSpec(
        hostname=socket.gethostname(),
        machine_id=machine_fingerprint(),
        os_name=platform.system(),
        os_version=platform.release(),
        arch=platform.machine(),
        python_version=platform.python_version(),
        cpu_name=_cpu_name(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        ram_gb=round(psutil.virtual_memory().total / 2**30, 1),
        accelerators=accels,
        torch=torch_info,
    )


def machine_fingerprint() -> str:
    """Stable, non-reversible identifier for this physical machine.

    Hostname + primary MAC, hashed. Two onepool processes on the same box get
    the same fingerprint, which is how the host detects (and refuses) a
    machine joining a pool it is already part of.
    """
    import hashlib
    import uuid

    raw = f"{socket.gethostname()}|{uuid.getnode()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def gpu_vendor(name: str) -> str | None:
    lowered = name.lower()
    for vendor, markers in _GPU_VENDORS.items():
        if any(m in lowered for m in markers):
            return vendor
    return None


# --- torch path ---------------------------------------------------------------


def _probe_torch() -> tuple[TorchInfo | None, list[Accelerator]]:
    try:
        import torch
    except ImportError:
        return None, []

    info = TorchInfo(
        version=torch.__version__,
        cuda_version=getattr(torch.version, "cuda", None),
        hip_version=getattr(torch.version, "hip", None),
    )
    accels: list[Accelerator] = []

    if torch.cuda.is_available():
        # A ROCm build of torch exposes AMD GPUs through the same torch.cuda API.
        backend = "rocm" if info.hip_version else "cuda"
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            accels.append(
                Accelerator(
                    backend=backend,
                    name=props.name,
                    vram_gb=round(props.total_memory / 2**30, 1),
                    index=i,
                )
            )

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        # MPS shares unified memory with the CPU; report total system RAM.
        accels.append(
            Accelerator(
                backend="mps",
                name="Apple Silicon GPU",
                vram_gb=round(psutil.virtual_memory().total / 2**30, 1),
            )
        )

    return info, accels


# --- system path --------------------------------------------------------------


def _probe_system() -> list[Accelerator]:
    accels = _probe_nvidia_smi()
    seen = {a.name.lower() for a in accels}

    system = platform.system()
    if system == "Windows":
        others = _probe_windows_video_controllers()
    elif system == "Darwin":
        others = _probe_macos_displays()
    elif system == "Linux":
        others = _probe_lspci()
    else:
        others = []

    for accel in others:
        vendor = gpu_vendor(accel.name)
        if vendor == "nvidia" and accel.name.lower() in seen:
            continue  # already reported with better data by nvidia-smi
        accels.append(accel)
    return accels


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _probe_nvidia_smi() -> list[Accelerator]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if not out:
        return []
    accels = []
    for i, line in enumerate(filter(None, (ln.strip() for ln in out.splitlines()))):
        name, _, mem = line.rpartition(",")
        try:
            vram = round(float(mem.strip()) / 1024, 1)
        except ValueError:
            continue
        accels.append(
            Accelerator(backend="cuda", name=name.strip(), vram_gb=vram, index=i, via="system")
        )
    return accels


def _probe_windows_video_controllers() -> list[Accelerator]:
    out = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name, AdapterRAM | ConvertTo-Json",
        ],
        timeout=20.0,
    )
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]

    accels = []
    for i, gpu in enumerate(data):
        name = (gpu.get("Name") or "").strip()
        if not name:
            continue
        ram = gpu.get("AdapterRAM")
        # AdapterRAM is a 32-bit field: wrong or 0 for cards with >4GB VRAM.
        vram = round(ram / 2**30, 1) if isinstance(ram, (int, float)) and ram > 0 else None
        vendor = gpu_vendor(name)
        backend = "cuda" if vendor == "nvidia" else "directml"
        accels.append(Accelerator(backend=backend, name=name, vram_gb=vram, index=i, via="system"))
    return accels


def _probe_macos_displays() -> list[Accelerator]:
    out = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=20.0)
    if not out:
        return []
    try:
        gpus = json.loads(out).get("SPDisplaysDataType", [])
    except json.JSONDecodeError:
        return []
    accels = []
    for i, gpu in enumerate(gpus):
        name = gpu.get("sppci_model") or gpu.get("_name")
        if not name:
            continue
        backend = "mps" if gpu_vendor(name) == "apple" else "directml"
        accels.append(Accelerator(backend=backend, name=name, vram_gb=None, index=i, via="system"))
    return accels


def _probe_lspci() -> list[Accelerator]:
    if not shutil.which("lspci"):
        return []
    out = _run(["lspci"])
    if not out:
        return []
    accels = []
    i = 0
    for line in out.splitlines():
        if "VGA compatible controller" not in line and "3D controller" not in line:
            continue
        name = line.split(":", 2)[-1].strip()
        vendor = gpu_vendor(name)
        backend = {"nvidia": "cuda", "amd": "rocm"}.get(vendor or "", "cpu")
        accels.append(Accelerator(backend=backend, name=name, vram_gb=None, index=i, via="system"))
        i += 1
    return accels


def _cpu_name() -> str:
    system = platform.system()
    if system == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


if __name__ == "__main__":  # pragma: no cover
    import pprint

    pprint.pprint(probe())
    sys.exit(0)
