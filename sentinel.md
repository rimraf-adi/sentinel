# Sentinel — ML Training GUI & Framework Plan

*Style: caveman. Simple technical English. Short words. Short sentences.*

## 1. Problem

ML/DL training work very repeat. Same loop write again and again. Same sweep run again. Same run compare by hand. Same config copy-paste.

Goal: make machine do repeat work. Human do thinking work.

Want:

- GUI for train, track, monitor.
- Rust if can.
- Support PyTorch and Spark (Spark later).
- Standard tricks built-in and tested.
- Fast hyperparameter tuning from GUI.

## 2. Scope, Users, Non-Goals

All core choices decided:

| Question | Locked Answer | What it change |
| --- | --- | --- |
| Name | **Sentinel** (`sentinel`) | Brand, CLI, packages, repo files |
| Who use it? | Solo or small team | Token auth, simple sharing, direct SSH |
| How many GPUs? | One box, 1–8 GPUs, or remote nodes | Dynamic GPU worker pool, NVML pack |
| What models? | All domains (vision, tabular, text, LLM) | Placeholder schema upfront, recipes step by step |
| Where train run? | Local machine + Remote SSH / Cloud VM | Outbound Agent on remote box, connect direct or SSH tunnel |
| What is pain most? | **All of them** | Thin Slice tackles loop, configs, sweeps, and remote monitor together |
| Start from scratch or fine-tune? | Both | Pretrained weights, LoRA / PEFT, `init_from` config |
| Own code? | Yes, custom scripts | 3 levels: Command mode, SDK mode, Framework mode |

**Non-goals (we do NOT build):**

- New deep learning framework.
- Own training engine in Rust.
- Spark in MVP (core loop first).
- AutoML magic that hide what happen.
- Cloud billing system.

## 3. Big Idea

**Rust is boss. Python is worker.**

Rust run GUI, queue, store, monitor. Python run training. Why? All good ML tools live in Python (PyTorch, Lightning, HuggingFace, DDP, FSDP). Rust train frameworks (burn, tch-rs) small. If train in Rust, lose big ecosystem. Bad trade.

So:

- Train code stay Python.
- Rust control, watch, show.
- Rust daemon from Day 1.

### Product shape

**IDE + scheduler + monitor + templates = less repeat work.** The whole process becomes a template. User pick a template, change a few fields, press run.

| Pillar | What it do | How we build it |
| --- | --- | --- |
| Workspace (Web UI) | Ultra-light web editor for configs and scripts. Open read-only by default. Click to edit. Idle drop back to read-only. Near-zero RAM. | Pure Rust Leptos / Dioxus via WASM |
| Scheduler | Queue, priority, GPU fit, retry, sweeps, pipelines. Dynamic GPU worker pool. | Rust daemon from Day 1 |
| Monitor | Live curves, logs, GPU stats, alerts. Watch jobs on remote machines from laptop. | Rust daemon + small Rust agent on each machine |
| Templates | Reusable recipe of the whole process: data, model, train, tune, eval, export. | Hydra / OmegaConf configs + template registry |

Templates are the glue. Workspace edit them. Scheduler run them. Monitor watch them.

### Positioning: it is just a git repo

**One line: a git repo for ML, with a GUI and better configs.** Like `create-next-app`, but for training runs. One command make a ready project.

- **Repo is the product.** Project = git repo. Code, templates, configs, run specs are plain files in it.
- **GUI is a view.** It read and write those files. Close the GUI, repo still works.
- **Zero lock-in.** `sentinel export script` generate **pure bare PyTorch code** (strip both Sentinel and Lightning). Anyone can run `python train.py` anywhere.
- **Git hold:** code, templates, configs, machine list (no secrets), run specs, `.dvc` pointers.
- **Git not hold:** metrics, logs, checkpoints, big data. These live in `mlruns/` (MLflow format), DVC cache, or object store.
- **Strict MLflow storage:** Metrics and runs write directly to `mlruns/` directory or talk to MLflow tracking server API. Existing tools read it with zero friction.
- **Better configs:** Hydra / OmegaConf YAML hierarchy. Composable, typed, validated before launch.
- **Data handling:** DVC (Data Version Control) + local paths. Versioned pointers in git, auto-cache near GPU.

**Tagline ideas:**

- "The watchful guardian for your training runs."
- "Rust is boss. Python is worker."
- "create-next-app, but for PyTorch."
- "Stop rewriting the same training loop."

### Framework model (like Next.js)

**Decision: it is a framework, not a template dump.** Next.js works like this: you write small files in agreed folders, the framework run, build and serve, and you upgrade by bumping a version.

| Next.js | Sentinel |
| --- | --- |
| `create-next-app` | `create-sentinel-app` (or `uvx sentinel new`) |
| `app/` folders (file-based routing) | `tasks/`, `models/`, `datasets/`, `templates/`, `configs/` |
| `next.config.js` | `sentinel.yaml` (human settings) |
| `package-lock.json` | `sentinel.lock` (tool version, template hashes, answers) |
| `next dev` | `sentinel dev`: open web GUI, live reload of configs |
| `next build` and `next start` | `sentinel run`, `sentinel sweep`, `sentinel export` |
| Defaults for everything | Defaults for optimizer, AMP, checkpoints, logging |
| Middleware and plugins | `hooks/` callbacks (`on_step`, `on_epoch`, `on_eval`) |
| `npm i next@latest` + codemods | `uv add sentinel@latest` + `sentinel upgrade` |

**Framework rules:**

- User write only what is specific: model, dataset, metric, configs. Framework own the training loop, AMP, DDP, checkpoint, resume, logging.
- A file in a convention folder is registered by its file name. Name is the id in templates and GUI.
- **Override by file.** Put a file with the same name in your repo (like `train_step.py`) and Sentinel use yours instead of built-in one.
- Command mode (Part D) stay for scripts that do not fit the framework.
- **No lock-in.** `sentinel export script` spit out pure bare PyTorch.
- Upgrade = bump version + 3-way git merge on isolated branch (Part G).

## 4. Parts

### Part A: Python SDK (thin)

Small library wrap PyTorch / Lightning. Give tested recipes for repeat work:

- Mixed precision (AMP)
- Gradient accumulation
- Gradient clipping
- LR scheduler
- Early stop
- Save and resume checkpoint
- Fix random seed
- Multi-GPU setup (DDP, FSDP)
- Standard metrics
- Log git hash, env, config
- Safe tracking: all calls wrapped in try/except. Never crash training.

SDK send events to Rust daemon or write straight to `mlruns/`.

### Part B: Rust Daemon (core)

Built in Rust from Day 1:

- **Single binary.** Fast startup, zero Python runtime dependency for supervisor.
- **Job supervisor.** Start, stop, restart process.
- **Dynamic GPU Worker Pool.** Check GPU memory and compute using `nvml-wrapper`. Pack trials onto free GPUs.
- **Job queue.** Local first. SSH remote VMs second. Slurm and Kubernetes later.
- **Storage driver.** Direct read/write to `mlruns/` (MLflow format).
- **Downsample.** Millions of metric points shrink on server before sending to UI.
- **HPO boss.** Drive Optuna sidecar with ASHA / Hyperband pruning.
- **API.** `axum` for HTTP, WebSocket, and serving WASM UI.

### Part C: GUI & Dashboard

**Decision: Pure Rust frontend with Leptos / Dioxus via WebAssembly.**

- **100% Rust full-stack:** Daemon and Web UI share Rust types and serialization.
- **Web only:** Open in any browser. No heavy desktop wrapper in Phase 1.
- **Ultra-low latency and RAM profile:**
  - Open **read-only** by default.
  - Switch to edit mode only when user explicitly clicks.
  - Drop back to read-only when user is idle (shed event listeners, stop polling).
  - Very minimal overhead on the host machine.

### Part D: Run Modes, Custom Scripts, Backends

Support all 3 levels of custom script adoption:

| Level | What user do | What user get |
| --- | --- | --- |
| **1. Command mode** | Give existing command only (`python train.py --lr 3e-4`) | Zero code changes. Run, queue, logs, GPU stats, git diff capture, remote launch. |
| **2. SDK mode** | Add 2 lines of Python (`import sentinel; sentinel.init()`) | Live metrics, checkpoints, pruning, auto-resume, MLflow run sync. |
| **3. Framework mode** | Put model and data in convention folders + use Hydra configs | Full automation: tested AMP, DDP, auto-resume, GUI form generation. |

Custom scripts work in sweeps. Tool pass trial values by CLI args, env vars, or YAML config override.

**Backends: same job spec everywhere.**

| Backend | How it start job | When |
| --- | --- | --- |
| Local | Subprocess on same machine | First |
| Remote SSH | Small Rust agent on cloud VM / remote box | Second |
| Cloud API | Daemon start/stop VMs via provider API | Later |
| Cluster | Slurm, Kubernetes | Later |

Remote rules:

- **Code:** Send git commit + uncommitted diff, or rsync.
- **Environment:** Container image or uv lockfile.
- **Data:** Local path or DVC pull near GPU.
- **Agent:** Outbound-only connection. Works behind NAT and firewall.
- **Results:** Written to `mlruns/` and synced back.

### Part E: Remote Monitoring From Your Laptop

Watch and control jobs on other machines from your laptop. No manual SSH into 5 boxes.

- **Outbound Agent:** Small Rust agent on remote machine. Connects out via WebSocket or SSH tunnel. No open inbound ports required.
- **Direct or Tunnel:** Connect directly or over SSH tunnel. Works behind firewall/NAT.
- **Offline safe:** Agent buffers metric events on disk. Syncs when network returns.
- **Adoption:** Agent can attach to existing PID started outside tool and tail GPU/logs.
- **Single screen:** Shows all boxes, GPU temperatures, VRAM, disk space, and job ETAs.

### Part F: Create Command, Packaging & Repo Layout

**Packaging Plan (uvx first, All 4 targets):**

Build single Rust binary in CI for 4 platforms:
1. `x86_64-unknown-linux-gnu` (Linux GPU servers)
2. `aarch64-unknown-linux-gnu` (ARM Linux / GH200)
3. `aarch64-apple-darwin` (Apple Silicon Mac)
4. `x86_64-pc-windows-msvc` (Windows)

Packaged into platform wheels published to PyPI. Single command to run:

```bash
uvx create-sentinel-app my-project
cd my-project
sentinel dev                     # start daemon & open Leptos WASM UI
sentinel run configs/train.yaml  # launch run
sentinel sweep configs/sweep.yaml # launch parallel sweep
sentinel export script           # export pure bare PyTorch loop
```

**Generated repo layout:**

```text
my-project/
  sentinel.yaml     human project settings (like Cargo.toml)
  sentinel.lock     machine hashes, versions, template answers (like Cargo.lock)
  tasks/            task definitions & metric rules
  models/           model architectures (one file per model)
  datasets/         data loaders & DVC pointers
  templates/        recipes inheriting built-ins
  configs/          Hydra / OmegaConf YAML hierarchy
  hooks/            callbacks (on_step, on_epoch, on_eval)
  machines.yaml     local and remote SSH machines (no secrets)
  mlruns/           strict MLflow tracking directory
  tests/            smoke tests for templates
  pyproject.toml    pinned dependencies (sentinel, torch, etc.)
  .sentinel/        local daemon cache & buffers (gitignored)
```

### Part G: Upgrade Command

Upgrades must never destroy user work.

**Rules:**

- **Separate files:** `sentinel.yaml` is for humans. `sentinel.lock` is for the machine.
- **3-way git merge:** `sentinel upgrade` creates an isolated branch `sentinel-upgrade-<version>`.
- Merges old template, new template, and user edits using git 3-way merge.
- If conflicts happen, standard git conflict markers appear. Main branch is never touched directly.
- User reviews diff in GUI or git, runs smoke tests, then merges or aborts.

## 5. Config System

Config system uses **Hydra / OmegaConf YAML hierarchy** (standard PyTorch research style).

**Rules:**

- **YAML in git is truth:** No hidden settings in database.
- **Hierarchy & composition:** Config groups for `model`, `dataset`, `optimizer`, `training`. Example: `model=resnet50 dataset=cifar10 lr=3e-4`.
- **Fast validation:** Validate configs before launching job. Fail in 1 second, not after 2 hours in queue.
- **Freeze per run:** Full resolved Hydra YAML saved with the run in `mlruns/`.

## 6. Templates: All Domains Upfront

Templates hold the recipe for the whole process.

**Domain placeholders in schema from Day 1:**

- `vision`: classification, detection, segmentation
- `tabular`: classification, regression
- `text`: classification, embedding
- `llm-finetune`: LoRA / PEFT, SFT
- `blank`: custom script template

Recipes implemented incrementally. All placeholder types valid in schema immediately.

## 7. Reproducibility & Data Handling

Save with every run:

- Git commit hash + uncommitted diff.
- Resolved Hydra config YAML.
- Random seeds (Python, NumPy, PyTorch, CUDA).
- Data pointer: **DVC (Data Version Control) hash + local path**.
- Environment lockfile / container image.
- Hardware info (GPU model, driver, CUDA version).
- One-click action: **"Re-run this run"**.

## 8. Failure Handling & Safe Tracking

- **Tracker safety:** Tracker NEVER crashes training. All SDK calls wrapped in try/except.
- **Local disk buffer:** If daemon restarts, events buffer locally and sync later.
- **OOM detection:** Catch CUDA out-of-memory error, suggest smaller batch size, auto-retry with gradient accumulation.
- **NaN / Inf stop:** Halt run immediately on NaN loss, save last valid checkpoint, fire alert.
- **Heartbeat watch:** If remote node drops, re-queue job or resume from checkpoint.
- **Alerts:** Webhooks to Slack, Discord, email.

## 9. Fast Hyperparameter Tuning & Sweeps

Speed comes from smart scheduling:

- **Dynamic GPU Worker Pool:** Rust daemon checks GPU VRAM and compute via NVML. Automatically packs trials onto available GPUs.
- **Early pruning:** Optuna sidecar driving ASHA / Hyperband. Kill losing trials fast.
- **Zero wasted compute:** Pruned trials terminate immediately and free GPU memory for the next trial in queue.
- **Visual sweeps:** Parallel-coordinates plot and hyperparameter importance curves rendered in Leptos WASM UI.

## 10. Spark (Later)

- Spark is a Non-Goal for Phase 1.
- Core loop and multi-GPU PyTorch first.
- Spark Connect integration considered for future enterprise data prep phase.

## 11. Zero Lock-in & Storage Compatibility

- **Strict MLflow storage:** Stores all runs directly in `mlruns/` format or talks to MLflow tracking server.
- **Bare PyTorch export:** `sentinel export script` generates standalone, pure PyTorch code without Sentinel or Lightning dependencies.

## 12. Testing the Recipes

No recipe ships without a test:

- CPU smoke tests (tiny model, 20 steps, loss decreases).
- Numeric match test (recipe vs raw PyTorch loop: exact loss parity).
- Resume test (checkpoint at step 10, resume, verify identical final metric).
- Daemon load test (thousands of events, WebSocket reconnect resilience).

## 13. Security and Access

- Bind to localhost by default.
- Remote access via SSH tunnel or reverse proxy with token auth.
- Outbound-only agent connection on remote GPU boxes.
- Secrets (SSH keys, cloud tokens) stay in environment or secret store, never in git YAMLs.

## 14. Success Metrics

Measure repeat work baseline before and after:

- Time from idea to first running job.
- Time to configure and start a sweep.
- GPU-hours wasted on dead trials (pruning reduces this).
- Runs lost to crashes without resume (target: zero).
- 100% of runs reproducible via git hash + DVC hash + config.

## 15. Implementation Phases

- **Phase 1 (Thin Slice):**
  - CLI `create-sentinel-app` / `sentinel`.
  - Rust daemon supervisor + Axum API.
  - Leptos / Dioxus WASM web UI (ultra-low RAM, read-only idle drop).
  - Hydra config system + domain template placeholders.
  - Command mode + SDK mode.
  - Local + Remote SSH execution via outbound Agent.
  - Strict MLflow `mlruns/` storage.
  - Optuna ASHA pruning with Dynamic GPU Worker Pool.
  - `sentinel export script` (pure bare PyTorch).
  - Cross-platform wheels for all 4 targets (Linux x86_64, Linux aarch64, macOS arm64, Windows).
- **Phase 2 (MVP Polish):**
  - Live curves downsampling, GPU dial gauges, alerts (Slack/webhook).
  - DVC auto-pull integration.
  - Auto-resume on crash.
- **Phase 3 (GUI Launch & Workspaces):**
  - GUI form generation from Hydra schemas.
  - 3-way merge `sentinel upgrade` flow.
  - Save run as template.
- **Phase 4 (Advanced Tuning):**
  - Parallel-coordinates, auto batch-size finder, learning rate finder.
- **Phase 5 (Cloud & Clusters):**
  - Slurm, Kubernetes, cloud provider on-demand VM spinning.

## 16. Rules to Remember

1. **Rust is boss. Python is worker.**
2. **Repo is the product. GUI is a view.**
3. **Pure Rust full-stack:** Rust daemon + Leptos / Dioxus WASM UI.
4. **Web UI is lightweight:** Open read-only, edit on click, drop to read-only when idle.
5. **Zero lock-in:** `sentinel export script` outputs pure bare PyTorch.
6. **Strict MLflow storage:** Write to `mlruns/` layout.
7. **Configs in Hydra YAML:** Composable, validated before queueing.
8. **Tracker never crashes training.**
9. **Agents connect outbound only.**
10. **Dynamic GPU Worker Pool:** Pack runs and prune early via Optuna ASHA.
11. **DVC + local for data.**
12. **Safe upgrades:** 3-way merge on isolated branch.
13. **Small MVP. Thin slice first.**
