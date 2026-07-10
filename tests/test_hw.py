"""Tests for hardware probing and diagnosis."""

from onepool.hw.doctor import diagnose
from onepool.hw.probe import Accelerator, NodeSpec, TorchInfo, gpu_vendor, probe


def _spec(**overrides) -> NodeSpec:
    base = dict(
        hostname="test-node",
        os_name="Windows",
        os_version="11",
        arch="AMD64",
        python_version="3.12.0",
        cpu_name="Test CPU",
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        ram_gb=16.0,
    )
    base.update(overrides)
    return NodeSpec(**base)


def test_probe_returns_sane_spec():
    spec = probe()
    assert spec.hostname
    assert spec.cpu_cores_logical >= spec.cpu_cores_physical >= 1
    assert spec.ram_gb > 0
    for a in spec.accelerators:
        assert a.backend in ("cuda", "rocm", "mps", "directml", "cpu")
        assert a.via in ("torch", "system")


def test_gpu_vendor_detection():
    assert gpu_vendor("NVIDIA GeForce RTX 3050") == "nvidia"
    assert gpu_vendor("GTX 1650") == "nvidia"
    assert gpu_vendor("AMD Radeon RX 7800 XT") == "amd"
    assert gpu_vendor("Intel(R) Iris(R) Xe Graphics") == "intel"
    assert gpu_vendor("Apple M2") == "apple"
    assert gpu_vendor("Some Unknown Device") is None


def test_diagnose_ready_with_usable_gpu():
    spec = _spec(
        torch=TorchInfo(version="2.5.0+cu124", cuda_version="12.4", hip_version=None),
        accelerators=[Accelerator("cuda", "NVIDIA GeForce RTX 3050", 6.0, via="torch")],
    )
    diag = diagnose(spec)
    assert diag.status == "ready"
    assert diag.install_command is None


def test_diagnose_missing_torch_with_nvidia_gpu():
    spec = _spec(
        torch=None,
        accelerators=[Accelerator("cuda", "NVIDIA GeForce GTX 1650", 4.0, via="system")],
    )
    diag = diagnose(spec)
    assert diag.status == "needs-torch"
    assert "cu128" in diag.install_command


def test_diagnose_cpu_torch_with_nvidia_gpu_is_wrong_torch():
    spec = _spec(
        torch=TorchInfo(version="2.5.0", cuda_version=None, hip_version=None),
        accelerators=[Accelerator("cuda", "NVIDIA GeForce RTX 3050", 6.0, via="system")],
    )
    diag = diagnose(spec)
    assert diag.status == "wrong-torch"
    assert diag.install_command is not None


def test_diagnose_amd_on_linux_recommends_rocm():
    spec = _spec(
        os_name="Linux",
        torch=None,
        accelerators=[Accelerator("rocm", "AMD Radeon RX 7800 XT", None, via="system")],
    )
    diag = diagnose(spec)
    assert diag.status == "needs-torch"
    assert "rocm" in diag.install_command


def test_diagnose_no_gpu_is_cpu_only():
    spec = _spec(
        torch=TorchInfo(version="2.5.0", cuda_version=None, hip_version=None),
        accelerators=[],
    )
    diag = diagnose(spec)
    assert diag.status == "cpu-only"


def test_usable_accelerators_filters_system_only():
    spec = _spec(
        accelerators=[
            Accelerator("cuda", "RTX 3050", 6.0, via="torch"),
            Accelerator("directml", "Intel UHD", None, via="system"),
        ]
    )
    assert len(spec.usable_accelerators) == 1
    assert spec.usable_accelerators[0].name == "RTX 3050"
