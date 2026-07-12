# onepool [![PyPI Downloads](https://static.pepy.tech/personalized-badge/onepool?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/onepool)

**Turn the laptops in the room into one pool of ML compute.**

onepool makes any group of computers on the same local network — Windows, Linux, or macOS; NVIDIA, AMD, integrated graphics, or plain CPUs — into a single ad-hoc cluster for machine learning. Not just inference: **real distributed training over ordinary WiFi** is the headline feature.

```
pip install onepool           # every machine, any OS
onepool doctor                # tells you the exact PyTorch install for your GPU
pip install "onepool[train]"  # training stack (transformers, peft, datasets)

# machine A — whoever has the job
onepool train run.yaml --pool --nodes 2   # hosts a pool, prints a session code

# machines B, C — anyone in the room
onepool join amber-fox-42                 # zero config, auto-discovered, starts working

# Ctrl-C → clean teardown. Nothing persists. No accounts. No daemons.
```

A live dashboard (session code, nodes, loss curve) serves at `http://localhost:7070` while the pool runs. No pool handy? `onepool train run.yaml` alone trains on just your machine.

## Why

- **Existing tools don't do this.** exo (archived) was inference-only and Apple-first. hivemind targets internet-scale volunteer computing. Ray and Horovod need DevOps expertise and homogeneous clusters. Nothing lets a study group pool three mismatched laptops for an evening of fine-tuning.
- **Heterogeneous is the default, not the edge case.** A 6GB RTX 3050 and a 4GB GTX 1650 in the same pool, each doing work proportional to what it can actually handle — measured, not assumed.
- **WiFi is enough.** Training uses a DiLoCo-style low-communication algorithm (many local steps, rare synchronization of small pseudo-gradients) plus LoRA adapters and int8-quantized sync, so a sync round is single-digit megabytes, not gigabytes.
- **Session-based by design.** A pool exists while its processes run and vanishes when they stop. Disposable clusters, not infrastructure.

## Status

Pre-alpha, built in the open. Roadmap:

| Milestone | Scope | Status |
|---|---|---|
| M0 | CLI skeleton, hardware probe, `onepool doctor` | ✅ done |
| M1 | Zero-config discovery, join/leave, live dashboard | ✅ done |
| M2 | Single-node LoRA fine-tuning path | ✅ done |
| M3 | Distributed training (DiLoCo) across mixed hardware over WiFi | ✅ done |
| M4 | Sync compression, speed-proportional rounds | ✅ done |
| v0.2 | Distributed inference | planned |

**Want to try it on your machines right now?** Follow [TESTING.md](TESTING.md) —
a 15-minute walkthrough from `pip install` to a two-laptop distributed training run.

## Try it now

```
git clone https://github.com/one-pool/onepool && cd onepool
uv sync
uv run onepool doctor    # what can this machine contribute to a pool?
```

`onepool doctor` detects your GPU(s) and tells you the exact PyTorch install command for your hardware — the one genuinely fiddly step on heterogeneous machines.

## Design in one paragraph

Every node runs the same process. mDNS (plus a UDP-broadcast fallback) advertises presence; a human-readable session code carries a shared secret so only invited machines join. The node that submits a job coordinates it: it benchmarks each worker on the actual model, shards data proportional to measured throughput, and runs DiLoCo — each worker takes ~100 local AdamW steps, then the pool all-reduces int8-quantized pseudo-gradients (for LoRA runs, single-digit megabytes) into an outer Nesterov-SGD step. Workers that vanish mid-round are dropped and their shard redistributed; workers that appear are folded in at the next sync. The dashboard is a local web page served by the coordinator.

## License

[AGPL-3.0-or-later](LICENSE). Use it, fork it, learn from it — but derivatives stay open, including network services built on it. The **onepool** name and logo are project trademarks; see [TRADEMARK.md](TRADEMARK.md).
