"""Distributed DiLoCo training across a pool.

Roles:

- **Coordinator** — runs inside the ``onepool train --pool`` process (which is
  also the pool host). Owns the canonical adapter weights and the outer
  optimizer, assigns data shards, trains its own share, and folds worker
  pseudo-gradients in at each round boundary.
- **Worker loop** — runs inside every ``onepool join`` process. Waits for
  TRAIN_START, then alternates: H local steps → send pseudo-gradient →
  receive fresh weights.

Fault model (session-grade, by design):
- Worker misses the round deadline → its delta is skipped; it resyncs on the
  next ROUND_UPDATE. Worker disconnects → dropped, shards rebalance.
- Worker joins mid-job → enrolled at the next round boundary with current
  weights.
- Coordinator dies → the job dies (workers return to idle pool).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field, fields

from onepool.jobs.spec import TrainJob
from onepool.net import protocol
from onepool.net.client import PoolClient
from onepool.net.server import PoolHost
from onepool.train.local import LocalTrainer, RoundStats
from onepool.train.outer import NesterovOuter
from onepool.train.tensors import pack_state, state_delta, unpack_state, weighted_average

log = logging.getLogger(__name__)


@dataclass
class WorkerSlot:
    member_id: str
    hostname: str
    enrolled_round: int
    last_loss: float | None = None
    last_tok_s: float | None = None
    steps_per_second: float | None = None
    missed_rounds: int = 0


CALIBRATION_STEPS = 5
MAX_SHIPPED_DATASET = 50 * 1024 * 1024  # local dataset files up to this size travel with the job


def dataset_payload(job: TrainJob) -> dict | None:
    """The job's dataset as wire bytes, when it's a shippable local file.

    Workers must not need any files on disk — the coordinator ships local
    corpora with the job. HF dataset ids return None (each node downloads
    from the hub itself), as do oversized files.
    """
    from pathlib import Path

    p = Path(job.dataset)
    if p.is_file() and p.stat().st_size <= MAX_SHIPPED_DATASET:
        return {"name": p.name, "data": p.read_bytes()}
    return None


def materialize_dataset(payload: dict) -> str:
    """Write a shipped dataset to a temp file; returns the path to train from."""
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="onepool-data-"))
    dest = tmp / payload["name"]
    dest.write_bytes(payload["data"])
    return str(dest)


def round_plan(job: TrainJob) -> list[int]:
    """A short calibration round first, then the job's own rounds.

    Round 0 runs a handful of steps on every node purely to measure real
    steps/sec — without it, the first full round stalls on the slowest
    machine (a 2-core laptop can take minutes for what a GPU does in
    seconds), and speed-proportional scaling has no data to work with.
    """
    return [min(CALIBRATION_STEPS, job.inner_steps)] + job.rounds


def scale_steps(base_steps: int, speed: float | None, fastest: float) -> int:
    """Speed-proportional inner steps so a slow node doesn't stall every round.

    A node at 40% of the fastest speed runs 40% of the steps and finishes the
    round at roughly the same wall time. Floor of 10% keeps every node
    contributing; sample-weighted averaging keeps the math honest.
    """
    if not speed or fastest <= 0:
        return base_steps
    fraction = max(0.1, min(1.0, speed / fastest))
    return max(1, round(base_steps * fraction))


@dataclass
class Coordinator:
    host: PoolHost
    job: TrainJob
    on_round: callable = None  # callback(round_index, mean_loss, workers)
    on_step: callable = None  # local trainer progress passthrough
    on_log: callable = None  # callback(text) for slow-path feedback (barrier waits)

    workers: dict[str, WorkerSlot] = field(init=False, default_factory=dict)

    async def run(self) -> list[RoundStats]:
        trainer = LocalTrainer(self.job, on_step=self.on_step)
        await asyncio.to_thread(trainer.setup)
        weights = trainer.get_adapter_state()
        outer = NesterovOuter(weights, self.job.outer_lr, self.job.outer_momentum)
        self._host_sps = 0.0

        stats: list[RoundStats] = []
        rounds = round_plan(self.job)
        for rnd, steps in enumerate(rounds):
            await self._enroll_new_members(rnd, steps, outer.weights)
            start_state = {k: v.copy() for k, v in outer.weights.items()}

            # the host is a node like any other: it also gets speed-scaled steps
            fastest = self._fastest_sps()
            host_steps = scale_steps(steps, self._host_sps or None, fastest)
            host_task = asyncio.create_task(
                asyncio.to_thread(trainer.run_round, host_steps, rnd)
            )
            worker_results = await self._collect_results(rnd, host_task)
            host_stats: RoundStats = await host_task

            deltas = [state_delta(start_state, trainer.get_adapter_state())]
            sample_weights = [float(host_stats.steps * self.job.batch_size * self.job.grad_accum)]
            losses = [(host_stats.mean_loss, sample_weights[0])]
            for member_id, msg in worker_results.items():
                deltas.append(unpack_state(msg["delta"]))
                sample_weights.append(float(msg["samples"]))
                losses.append((msg["loss"], float(msg["samples"])))
                slot = self.workers[member_id]
                slot.last_loss, slot.last_tok_s = msg["loss"], msg.get("tok_s")
                slot.steps_per_second = msg.get("sps") or slot.steps_per_second
            self._host_sps = host_stats.steps_per_second

            new_weights = outer.step(weighted_average(deltas, sample_weights))
            trainer.set_adapter_state(new_weights)

            mean_loss = sum(loss * w for loss, w in losses) / sum(w for _, w in losses)
            stats.append(
                RoundStats(
                    round_index=rnd,
                    steps=steps,
                    mean_loss=mean_loss,
                    tokens_per_second=host_stats.tokens_per_second
                    + sum(s.last_tok_s or 0 for s in self.workers.values()),
                )
            )
            await asyncio.to_thread(trainer.save, f"round-{rnd:03d}")
            if self.on_round:
                self.on_round(rnd, mean_loss, 1 + len(worker_results))

            if rnd + 1 < len(rounds):
                await self._broadcast_round_update(rnd + 1, rounds[rnd + 1], new_weights)

        await self._broadcast(protocol.TRAIN_END, {"reason": "complete"})
        await asyncio.to_thread(trainer.save, "final")
        return stats

    # --- membership -------------------------------------------------------

    def _pool_worker_ids(self) -> list[str]:
        return [m.member_id for m in self.host.state.members.values() if not m.is_host]

    def _fastest_sps(self) -> float:
        speeds = [s.steps_per_second for s in self.workers.values() if s.steps_per_second]
        return max(speeds + [self._host_sps])

    def _shard_layout(self) -> dict[str, int]:
        """Host is always shard 0; workers get 1..N in stable id order."""
        return {mid: i + 1 for i, mid in enumerate(sorted(self.workers))}

    async def _enroll_new_members(self, rnd: int, steps: int, weights) -> None:
        for member_id in self._pool_worker_ids():
            if member_id in self.workers:
                continue
            member = self.host.state.members[member_id]
            self.workers[member_id] = WorkerSlot(
                member_id=member_id,
                hostname=member.summary()["hostname"],
                enrolled_round=rnd,
            )
        layout = self._shard_layout()
        num_shards = len(self.workers) + 1
        if not hasattr(self, "_dataset_cache"):
            self._dataset_cache = dataset_payload(self.job)
        for member_id, slot in list(self.workers.items()):
            if slot.enrolled_round == rnd:
                ok = await self.host.send_to(
                    member_id,
                    {
                        "t": protocol.TRAIN_START,
                        "job": asdict(self.job),
                        "weights": pack_state(weights),
                        "dataset_file": self._dataset_cache,
                        "round": rnd,
                        "steps": steps,
                        "shard": layout[member_id],
                        "shards": num_shards,
                    },
                )
                if not ok:
                    del self.workers[member_id]

    async def _broadcast_round_update(self, rnd: int, steps: int, weights) -> None:
        layout = self._shard_layout()
        packed = pack_state(weights)  # canonical weights always full precision
        num_shards = len(self.workers) + 1
        fastest = self._fastest_sps()
        for member_id in list(self.workers):
            slot = self.workers[member_id]
            ok = await self.host.send_to(
                member_id,
                {
                    "t": protocol.ROUND_UPDATE,
                    "round": rnd,
                    "steps": scale_steps(steps, slot.steps_per_second, fastest),
                    "weights": packed,
                    "shard": layout[member_id],
                    "shards": num_shards,
                },
            )
            if not ok:
                log.info("worker %s unreachable, dropping from job", member_id)
                del self.workers[member_id]

    async def _broadcast(self, msg_type: str, payload: dict) -> None:
        for member_id in list(self.workers):
            await self.host.send_to(member_id, {"t": msg_type, **payload})

    # --- round collection ---------------------------------------------------

    async def _collect_results(self, rnd: int, host_task=None) -> dict[str, dict]:
        """Wait for this round's pseudo-gradients from every enrolled worker."""
        expected = {
            mid for mid, slot in self.workers.items() if slot.enrolled_round <= rnd
        }
        results: dict[str, dict] = {}
        started = time.monotonic()
        deadline = started + self.job.sync_timeout
        last_log = started

        while expected - results.keys():
            # let the user see why nothing is printing during a slow round
            now = time.monotonic()
            if self.on_log and now - last_log >= 30 and (host_task is None or host_task.done()):
                missing = len(expected - results.keys())
                names = ", ".join(
                    self.workers[m].hostname for m in expected - results.keys()
                )
                self.on_log(
                    f"waiting for {missing} worker(s) [{names}] — "
                    f"{int(now - started)}s into round {rnd} "
                    f"(slow nodes get fewer steps from the next round)"
                )
                last_log = now

            # workers that left the pool aren't coming back this round
            alive = set(self._pool_worker_ids())
            for gone in expected - alive:
                expected.discard(gone)
                self.workers.pop(gone, None)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                for member_id in expected - results.keys():
                    slot = self.workers[member_id]
                    slot.missed_rounds += 1
                    log.warning("worker %s missed round %d deadline", member_id, rnd)
                break
            try:
                member_id, msg = await asyncio.wait_for(
                    self.host.inbox.get(), timeout=min(remaining, 5.0)
                )
            except TimeoutError:
                continue

            if msg["t"] == protocol.TRAIN_ERR:
                log.warning("worker %s failed: %s — removing from job",
                            member_id, msg.get("error"))
                expected.discard(member_id)
                self.workers.pop(member_id, None)
            elif msg["t"] == protocol.ROUND_RESULT:
                if msg["round"] == rnd and member_id in expected:
                    results[member_id] = msg
                    self.workers[member_id].missed_rounds = 0
                # stale results from missed rounds are dropped silently

        return results


# --- worker side --------------------------------------------------------------


async def worker_loop(client: PoolClient, on_status=None) -> None:
    """Participate in training jobs for as long as we're in the pool.

    Runs inside ``onepool join``. Blocks on the train inbox; TRAIN_START turns
    this node into a worker until TRAIN_END.
    """

    def status(text: str) -> None:
        if on_status:
            on_status(text)

    while True:
        msg = await client.train_inbox.get()
        if msg["t"] != protocol.TRAIN_START:
            continue  # stray frame outside a job

        try:
            from onepool.train import require_training_stack

            require_training_stack()
        except ImportError as e:
            await client.send({"t": protocol.TRAIN_ERR, "error": str(e)})
            continue

        job = _job_from_wire(msg["job"])
        status(f"training job received: {job.model} (round {msg['round']})")
        if msg.get("dataset_file"):
            job.dataset = materialize_dataset(msg["dataset_file"])
            status(f"dataset received from coordinator ({msg['dataset_file']['name']})")
        trainer = LocalTrainer(job, shard_index=msg["shard"], num_shards=msg["shards"])
        start_state = unpack_state(msg["weights"])
        try:
            status("preparing model + data (first time may download the model)...")
            await asyncio.to_thread(trainer.setup)
            trainer.set_adapter_state(start_state)
        except Exception as e:  # model/dataset load can fail in many ways
            log.exception("worker setup failed")
            status(f"setup failed: {e}")
            await client.send({"t": protocol.TRAIN_ERR, "error": f"setup failed: {e}"})
            continue

        rnd, steps = msg["round"], msg["steps"]
        while True:
            status(f"round {rnd}: training {steps} steps...")
            stats = await asyncio.to_thread(trainer.run_round, steps, rnd)
            samples = stats.steps * job.batch_size * job.grad_accum
            delta = state_delta(start_state, trainer.get_adapter_state())
            await client.send(
                {
                    "t": protocol.ROUND_RESULT,
                    "round": rnd,
                    "steps": stats.steps,
                    "samples": samples,
                    "loss": stats.mean_loss,
                    "tok_s": stats.tokens_per_second,
                    "sps": stats.steps_per_second,
                    "delta": pack_state(delta, job.sync_compression),
                }
            )
            status(f"round {rnd}: loss {stats.mean_loss:.4f}, {stats.tokens_per_second:.0f} tok/s")

            msg = await client.train_inbox.get()
            if msg["t"] == protocol.TRAIN_END:
                status("job complete")
                break
            if msg["t"] != protocol.ROUND_UPDATE:
                continue
            start_state = unpack_state(msg["weights"])
            trainer.set_adapter_state(start_state)
            trainer.set_shard(msg["shard"], msg["shards"])
            rnd, steps = msg["round"], msg["steps"]


def _job_from_wire(raw: dict) -> TrainJob:
    known = {f.name for f in fields(TrainJob)}
    return TrainJob(**{k: v for k, v in raw.items() if k in known})
