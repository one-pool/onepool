# Testing onepool

A hands-on walkthrough that takes two (or more) computers from nothing to a
distributed training run. Every step shows what success looks like. Total time:
~15 minutes plus model downloads.

**What you need**

- 1+ computers on the same WiFi/LAN (Windows, Linux, or macOS in any mix)
- Python 3.10+ on each (`python --version`)
- ~2GB free disk per machine (PyTorch + a small model)
- A GPU is nice but not required — CPU nodes work, just slower

Throughout: **machine A** = the one that runs the job (coordinator).
**Machine B, C, …** = everyone else (workers).

---

## Step 1 — Install (every machine)

```bash
pip install onepool
onepool version        # expect: onepool 0.4.0 (or newer)
```

## Step 2 — Hardware check (every machine)

```bash
onepool doctor
```

You get a node card (OS, CPU, RAM, GPUs) and a diagnosis:

- **"PyTorch is not installed."** → run the exact `pip install torch ...`
  command it prints (it picks the right build for your GPU vendor and OS).
- **"...cannot see your NVIDIA GPU"** → you have a CPU-only torch; run the
  printed command to replace it.
- **"No usable GPU found..."** → fine, this machine joins as a CPU worker.

Then install the training stack:

```bash
pip install "onepool[train]"
```

Re-run `onepool doctor` until it says **Ready** (green) or **cpu-only**
(yellow). Red = follow the printed fix.

## Step 3 — Pool formation (no training yet)

On **machine A**:

```bash
onepool up
```

Expect a green panel:

```
session code:  amber-fox-73
dashboard:     http://localhost:7070
```

- Windows will show a firewall prompt the first time — click **Allow**.
- Open the dashboard URL in a browser: you should see machine A as a node card.

On **machine B**:

```bash
onepool join amber-fox-73        # use YOUR code from machine A's screen
```

Expect within ~5 seconds:

```
joined pool amber-fox-73 at 192.168.x.x:PORT
pool size:  2 node(s)
```

Machine A prints "pool now has 2 nodes" and the dashboard updates live.

**Checks to try**

- `onepool status` on machine A → table of both nodes.
- Wrong code (`onepool join wrong-code-99`) → clean "no pool found" error.
- Ctrl-C on machine B → machine A reports the pool shrank. Rejoin works.
- Ctrl-C on machine A → machine B reports "pool host went away".

**If join says "no pool found"** — your network probably blocks mDNS
(common on guest/corporate WiFi). Use the direct route shown on machine A's
screen:

```bash
onepool join amber-fox-73 --host 192.168.1.4:57944
```

## Step 4 — Training test files (machine A only)

Make a folder with a tiny corpus and a job file:

```bash
mkdir pooltest && cd pooltest
python -c "open('corpus.txt','w').write(('To pool or not to pool, that is the question.\n'*3+'\n')*400)"
```

Create `job.yaml`:

```yaml
model: distilgpt2
dataset: corpus.txt
seq_len: 256
batch_size: 4        # drop to 2 on 4GB GPUs, 1 on CPU-only
steps: 40
inner_steps: 20
output_dir: checkpoints/test
```

## Step 5 — Single-node training (machine A)

```bash
onepool train job.yaml
```

First run downloads distilgpt2 (~350MB). A live dashboard URL is printed too —
open it for the loss curve. Then a progress bar with live loss.
**Success = loss falls** (roughly 6.0 → below 5.0 in 40 steps) and:

```
final loss:  4.8xxx
adapters:    checkpoints/test/final
```

## Step 6 — Distributed training (the whole point)

On **machine A** (in the `pooltest` folder):

```bash
onepool train job.yaml --pool --nodes 1
```

It prints a session code and waits for 1 worker.

On **machine B**:

```bash
onepool join <that-code>
```

Machine B prints its progress stage by stage — no silent phases:

```
training job received: distilgpt2 (round 0)
dataset received from coordinator (corpus.txt)     <- workers need no files
preparing model + data (first time may download the model)...
round 0: training 5 steps...
round 0: loss 6.1xxx, xx tok/s
```

Round 0 is a short **calibration round** (5 steps) that measures each node's
speed; from round 1, slower nodes automatically get proportionally fewer steps
so nobody stalls the pool. On machine A expect:

```
round 1/3 (calibration): loss 6.1xxx (2 nodes)
round 2/3: loss 5.9xxx (2 nodes)
round 3/3: loss 4.7xxx (2 nodes)
final loss:  4.7xxx
```

If a round is slow, machine A prints `waiting for 1 worker(s) [NAME] — 30s
into round...` instead of sitting silent.

Machine B shows its own per-round loss and ends with `job complete`.

Open machine A's dashboard during the run: live loss curve + both node cards.

**What just happened:** each machine ran 20 local optimizer steps on its own
half of the data, shipped an int8-compressed pseudo-gradient (~1MB) over TLS,
the coordinator averaged them and broadcast fresh weights — DiLoCo over your
WiFi.

## Step 7 — Fault tolerance (optional but fun)

Start the same distributed run again, and **Ctrl-C machine B mid-round**.
Machine A logs a warning, finishes the round without B, and completes the job
solo. No hang, no crash.

Also try joining machine B *after* training started — it gets enrolled at the
next round boundary.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no pool found for that code` | Same WiFi? Host still running? Else use `--host IP:PORT` (mDNS blocked) |
| Windows firewall prompt | Allow once; private networks only is fine |
| CUDA out of memory | Lower `batch_size` (2, then 1), or `seq_len: 128` |
| Painfully slow on a machine | It's on CPU — check `onepool doctor`; pool still works, that node just does fewer steps per round |
| `missing: torch...` on join during a job | That worker needs Step 2 completed; it declines the job gracefully, pool continues |
| Model download slow/blocked | Set `HF_TOKEN` env var (free huggingface.co account) for faster downloads |
| Dashboard port busy | onepool scans 7070–7089 automatically; check the printed URL |

## Reporting results

Useful bug reports include: OS + GPU of each machine, `onepool version`,
the full console output of both sides (`--verbose` flag adds detail), and
whether Steps 3/5/6 each passed. Issues: https://github.com/one-pool/onepool/issues
