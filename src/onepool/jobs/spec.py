"""Training job specification, loaded from a small YAML file.

Design rule: every knob has a default that works on a 4GB laptop GPU, so a
minimal job file is just a model and a dataset:

    model: Qwen/Qwen2.5-0.5B
    dataset: data/my_notes.txt
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class TrainJob:
    model: str
    dataset: str
    text_field: str = "text"  # column to read when the dataset has several
    dataset_split: str = "train"
    output_dir: str = "checkpoints"

    seq_len: int = 512
    batch_size: int = 4
    grad_accum: int = 1
    lr: float = 2e-4
    warmup_steps: int = 20
    steps: int = 200  # total optimizer steps
    inner_steps: int = 100  # H: steps per round — the DiLoCo sync unit

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    outer_lr: float = 0.7  # DiLoCo outer Nesterov-SGD (paper defaults)
    outer_momentum: float = 0.9
    sync_timeout: float = 600.0  # seconds the coordinator waits for a round's results

    precision: str = "auto"  # auto | fp32 | fp16 | bf16
    seed: int = 0

    extra: dict = field(default_factory=dict)  # unrecognized keys, kept for forward-compat

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainJob:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping of job settings")
        known = {f.name for f in fields(cls)} - {"extra"}
        missing = {"model", "dataset"} - raw.keys()
        if missing:
            raise ValueError(f"{path}: missing required keys: {', '.join(sorted(missing))}")
        args = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        job = cls(**args, extra=extra)
        job.validate()
        return job

    def validate(self) -> None:
        if self.precision not in ("auto", "fp32", "fp16", "bf16"):
            raise ValueError(f"precision must be auto/fp32/fp16/bf16, got {self.precision!r}")
        for name in ("seq_len", "batch_size", "grad_accum", "steps", "inner_steps", "lora_r"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")

    @property
    def rounds(self) -> list[int]:
        """Step counts per round, e.g. steps=250, inner=100 -> [100, 100, 50]."""
        full, rest = divmod(self.steps, self.inner_steps)
        return [self.inner_steps] * full + ([rest] if rest else [])
