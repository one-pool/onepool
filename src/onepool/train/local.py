"""Single-node LoRA fine-tuning, structured as DiLoCo rounds.

Training runs as a sequence of *rounds* of ``inner_steps`` optimizer steps.
On one machine the rounds simply run back to back — but the round boundary is
exactly where distributed mode (M3) will synchronize pseudo-gradients, so the
loop shape doesn't change when the pool grows past one node.

Heavy imports (torch, transformers) happen inside functions: the CLI must stay
fast for machines that only host or monitor pools.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from onepool.jobs.spec import TrainJob

log = logging.getLogger(__name__)


@dataclass
class RoundStats:
    round_index: int
    steps: int
    mean_loss: float
    tokens_per_second: float
    steps_per_second: float = 0.0  # drives speed-proportional work assignment


@dataclass
class DeviceChoice:
    device: str  # "cuda" | "mps" | "cpu"
    dtype: Any  # torch dtype for model weights / autocast
    use_scaler: bool  # fp16 needs gradient scaling; bf16 and fp32 don't
    name: str


def pick_device(precision: str = "auto") -> DeviceChoice:
    import torch

    if torch.cuda.is_available():
        bf16_ok = torch.cuda.is_bf16_supported()
        name = torch.cuda.get_device_name(0)
        if precision == "fp32":
            return DeviceChoice("cuda", torch.float32, False, name)
        if precision == "bf16" or (precision == "auto" and bf16_ok):
            return DeviceChoice("cuda", torch.bfloat16, False, name)
        # fp16 path for pre-Ampere cards (e.g. GTX 1650)
        return DeviceChoice("cuda", torch.float16, True, name)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and precision in ("auto", "fp16"):
        return DeviceChoice("mps", torch.float32, False, "Apple Silicon GPU")
    return DeviceChoice("cpu", torch.float32, False, "CPU")


class LocalTrainer:
    """Owns the model, data, and optimizer for this node's share of a job."""

    def __init__(
        self, job: TrainJob, on_step=None, shard_index: int = 0, num_shards: int = 1
    ) -> None:
        self.job = job
        self.on_step = on_step  # callback(step, loss) for progress display
        self.global_step = 0
        self.shard_index = shard_index
        self.num_shards = num_shards

    def setup(self) -> None:
        _quiet_third_party_noise()
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(self.job.seed)
        self.choice = pick_device(self.job.precision)
        log.info("training on %s (%s)", self.choice.device, self.choice.name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.job.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.job.model,
            dtype=self.choice.dtype if self.choice.device == "cuda" else None,
        )
        lora = LoraConfig(
            task_type="CAUSAL_LM",
            r=self.job.lora_r,
            lora_alpha=self.job.lora_alpha,
            lora_dropout=self.job.lora_dropout,
            target_modules="all-linear",
        )
        self.model = get_peft_model(model, lora).to(self.choice.device)
        self.model.train()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=self.job.lr)
        self.scheduler = _warmup_cosine(self.optimizer, self.job.warmup_steps, self.job.steps)
        self.scaler = torch.amp.GradScaler("cuda") if self.choice.use_scaler else None
        self._batches = self._batch_iterator()

        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.model.parameters())
        log.info("trainable params: %.2fM of %.2fM total", n_train / 1e6, n_total / 1e6)

    def run_round(self, steps: int, round_index: int = 0) -> RoundStats:
        """Run one DiLoCo round: `steps` local optimizer steps."""
        import torch

        losses: list[float] = []
        tokens = 0
        started = time.time()

        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(self.job.grad_accum):
                batch = next(self._batches)
                batch = {k: v.to(self.choice.device) for k, v in batch.items()}
                tokens += int(batch["attention_mask"].sum())
                if self.choice.device == "cuda" and self.choice.use_scaler:
                    with torch.autocast("cuda", dtype=self.choice.dtype):
                        loss = self.model(**batch).loss / self.job.grad_accum
                    self.scaler.scale(loss).backward()
                else:
                    loss = self.model(**batch).loss / self.job.grad_accum
                    loss.backward()
                step_loss += loss.item()

            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            self.scheduler.step()

            self.global_step += 1
            losses.append(step_loss)
            if self.on_step:
                self.on_step(self.global_step, step_loss)

        elapsed = max(time.time() - started, 1e-6)
        return RoundStats(
            round_index=round_index,
            steps=steps,
            mean_loss=sum(losses) / len(losses),
            tokens_per_second=tokens / elapsed,
            steps_per_second=steps / elapsed,
        )

    # --- distributed hooks (DiLoCo) ---------------------------------------

    def get_adapter_state(self) -> dict:
        """Trainable (LoRA) parameters as named float32 numpy arrays."""
        import torch

        return {
            name: p.detach().to("cpu", dtype=torch.float32).numpy()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

    def set_adapter_state(self, state: dict) -> None:
        """Load canonical weights from the coordinator into the live model."""
        import torch

        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, arr in state.items():
                p = params[name]
                p.copy_(torch.from_numpy(arr).to(device=p.device, dtype=p.dtype))

    def set_shard(self, shard_index: int, num_shards: int) -> None:
        """Re-shard the data stream (e.g. after pool membership changed)."""
        if (shard_index, num_shards) != (self.shard_index, self.num_shards):
            self.shard_index = shard_index
            self.num_shards = num_shards
            self._batches = self._batch_iterator()

    def save(self, tag: str = "final") -> Path:
        out = Path(self.job.output_dir) / tag
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        return out

    # --- data ------------------------------------------------------------

    def _batch_iterator(self) -> Iterator[dict]:
        import torch
        from torch.utils.data import DataLoader

        dataset = _load_texts(self.job)
        if self.num_shards > 1:
            dataset = dataset.shard(num_shards=self.num_shards, index=self.shard_index)

        def tokenize(example: dict) -> dict:
            return self.tokenizer(
                example[self.job.text_field],
                truncation=True,
                max_length=self.job.seq_len,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
        tokenized = tokenized.filter(lambda e: sum(e["attention_mask"]) >= 8)
        if len(tokenized) == 0:
            raise ValueError("dataset produced no usable examples (all too short?)")

        def collate(items: list[dict]) -> dict:
            input_ids = torch.tensor([i["input_ids"] for i in items])
            attention = torch.tensor([i["attention_mask"] for i in items])
            labels = input_ids.clone()
            labels[attention == 0] = -100  # don't learn to predict padding
            return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

        loader = DataLoader(
            tokenized, batch_size=self.job.batch_size, shuffle=True, collate_fn=collate
        )
        while True:  # cycle epochs until the step budget is spent
            yield from loader


def run_local(job: TrainJob, on_step=None, on_round=None) -> list[RoundStats]:
    """Train on this machine only. The distributed path (M3) reuses LocalTrainer."""
    trainer = LocalTrainer(job, on_step=on_step)
    trainer.setup()

    stats: list[RoundStats] = []
    for i, steps in enumerate(job.rounds):
        result = trainer.run_round(steps, round_index=i)
        stats.append(result)
        trainer.save(tag=f"round-{i:03d}")
        if on_round:
            on_round(result)
        log.info(
            "round %d: %d steps, loss %.4f, %.0f tok/s",
            i, result.steps, result.mean_loss, result.tokens_per_second,
        )
    trainer.save("final")
    return stats


def _load_texts(job: TrainJob):
    from datasets import load_dataset

    path = Path(job.dataset)
    if path.exists():
        kind = "json" if path.suffix in (".json", ".jsonl") else "text"
        data = load_dataset(kind, data_files=str(path), split="train")
    else:
        data = load_dataset(job.dataset, split=job.dataset_split)
    if job.text_field not in data.column_names:
        raise ValueError(
            f"dataset has no {job.text_field!r} column "
            f"(available: {', '.join(data.column_names)}) — set text_field in the job file"
        )
    return data


def _quiet_third_party_noise() -> None:
    """Silence advisory chatter from the ML stack that drowns our own output.

    Each of these is a known, harmless message: the HF anonymous-request
    notice, the peft Conv1D fan_in_fan_out auto-correction, and the datasets
    fingerprint hash fallback (our tokenize closure isn't picklable — caching
    is irrelevant for a stream we re-tokenize per run anyway).
    """
    import warnings

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*fan_in_fan_out.*")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except ImportError:
        pass


def _warmup_cosine(optimizer, warmup: int, total: int):
    from torch.optim.lr_scheduler import LambdaLR

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, factor)
