# Sentinel

**Enterprise-Grade Machine Learning Training Supervisor, Dynamic Scheduler, and Operations Plane**

[![Rust Version](https://img.shields.io/badge/Rust-2021_Edition-orange?logo=rust)](https://www.rust-lang.org)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org)
[![Package Manager](https://img.shields.io/badge/Environment-uv-purple?logo=astral)](https://github.com/astral-sh/uv)
[![Storage Standard](https://img.shields.io/badge/Storage-MLflow_3.x_Compliant-0194E2?logo=mlflow)](https://mlflow.org)
[![Build Status](https://img.shields.io/badge/Tests-31%20Passed-brightgreen)](https://github.com/rimraf-adi/sentinel)
[![Default Port](https://img.shields.io/badge/Port-2512-gray)](http://localhost:2512)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](LICENSE)

---

## 1. System Overview

Sentinel is a high-performance, dual-engine machine learning infrastructure platform engineered to decouple experiment orchestration from deep learning execution. Traditional machine learning workflows often suffer from boilerplate proliferation, fragile Python-level process supervision, unvalidated configuration drift, and restrictive framework lock-in.

Sentinel addresses these structural bottlenecks through an asynchronous, bifurcated control topology:

* **Control & Supervision Plane (Rust):** A standalone, statically compiled native binary responsible for asynchronous process supervision, non-colliding dynamic GPU allocation, hardware telemetry ingestion (NVML), WebSocket event streaming, declarative component resolution, and an ultra-low-overhead web management console.
* **Execution & Compute Plane (Python):** Pure, unencumbered Python runtimes executing standard frameworks (PyTorch, torchvision, Hugging Face, Optuna). Workloads execute without proprietary base classes, non-standard runtime wrappers, or foreign abstraction layers.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SENTINEL CONTROL PLANE                          │
│                      (High-Performance Native Rust)                    │
│                                                                        │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌───────────────────┐  │
│  │ Web UI Console   │  │ Process Supervisor  │  │ Dynamic GPU Pool  │  │
│  │ (Port 2512)      │  │ (SIGTERM / Draining)│  │ (NVML Allocation) │  │
│  └─────────┬────────┘  └──────────┬──────────┘  └─────────┬─────────┘  │
│            │                      │                       │            │
│            ▼                      ▼                       ▼            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │             Axum REST API & WebSocket Telemetry Hub              │  │
│  └────────────────────────────────┬─────────────────────────────────┘  │
└───────────────────────────────────┼────────────────────────────────────┘
                                    │ (CUDA_VISIBLE_DEVICES, Pipes, IPC)
┌───────────────────────────────────┼────────────────────────────────────┐
│                                   ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                `uv run` Managed Python Subprocesses              │  │
│  │                                                                  │  │
│  │   ┌─────────────────────┐              ┌──────────────────────┐  │  │
│  │   │ PyTorch Model Loop  │              │ Optuna ASHA Sweeps   │  │  │
│  │   └──────────┬──────────┘              └──────────┬───────────┘  │  │
│  │              │                                    │              │  │
│  │              ▼                                    ▼              │  │
│  │   ┌───────────────────────────────────────────────────────────┐  │  │
│  │   │            `sentinel-sdk` (Crash-Immune Client)           │  │  │
│  │   └─────────────────────────────┬─────────────────────────────┘  │  │
│  └─────────────────────────────────┼────────────────────────────────┘  │
│                                    ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              Strict MLflow-Compliant Store (`mlruns/`)           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                         EXECUTION & DATA PLANE                         │
└────────────────────────────────────────────────────────────────────────┘
```

---


## Block hierarchy

Models are built from blocks in two tiers, so common structures are defined once and reused.

| Tier | What it is | Lives in |
|---|---|---|
| **Fundamental** | A native block: one PyTorch layer (`linear`, `conv2d`, `lstm`, `attention`, `add`, ...). | the Rust catalog (`sentinel component blocks`) |
| **Compound** | A reusable subgraph of fundamental blocks *or other compound blocks*, with its own ports and hyperparameters (`resnet_block`, `residual_mlp_block`, ...). | `blocks/<family>/<name>.yaml` (shipped ones are embedded in the binary) |
| **Model** | A whole graph using blocks from either tier, with a training recipe. | `components/` |

```yaml
# blocks/core/residual_mlp_block.yaml: built from the compound mlp_block
compound: residual_mlp_block
params:
  hidden: {default: 64}
  width:  {}                  # no default = required
graph:
  inputs: {in: [B, "${width}"]}
  nodes:
    mlp:  {block: mlp_block, params: {width: "${hidden}"}}
    proj: {block: linear, params: {out: "${width}"}}
    skip: {block: add}
  edges: ["in -> mlp", "mlp -> proj", "proj -> skip.a", "in -> skip.b"]
  outputs: {out: skip}
```

- A compound block's input shapes come from whatever is wired to it, so one definition works at any size; errors name every
  compound on the path (`in compound block 'residual_mlp_block' (node 'r1'): ...`).
- Compounds compile to **nested PyTorch classes** (one per distinct configuration, dependencies first), so exported code reads like
  hand-written modules. Weight sharing, multi-output ports and keyword port names (`in`) all work.
- A compound block can never contain itself, directly or through other blocks.
- In the Builder, the palette has a **Fundamental / Compound** switch, compound nodes show what they are built from (recursively,
  with live shapes), and **Save as -> Compound block** turns the canvas into a new palette block.
- `sentinel component show <model>` prints the whole tree; `sentinel component blocks` lists both tiers.

Shipped compound blocks: `mlp_block`, `residual_mlp_block`, `conv_block`, `resnet_block`, `transformer_block`, `unet_down`, `unet_up`,
`vae_encoder`, `vae_decoder`. Shipped models built from them include `vision/vit-tiny`, `vision/unet`, `vision/resnet-mini`,
`generative/vae` (trained with reconstruction + beta * KL) and `tabular/deep-residual-mlp`.

## Export and predict

Every graph-component run saves, next to its metrics (`mlruns/<exp>/<run>/artifacts/`): `model.pt` (weights), `model_script.pt`
(TorchScript, loads with `torch.jit.load` and needs no Sentinel code), `model.onnx` (best effort), `meta.json` (task, feature
names, classes, input shape, test metric) and `train.py` (the exact script that produced it).

```bash
sentinel model list                                   # runs that exported a model
sentinel model predict b5a43acb -i "5.0,3.4,1.5,0.2"  # -> setosa (100.0%); repeat -i for several rows, --json for machines
sentinel model export  b5a43acb -f onnx -o flowers.onnx   # torchscript | onnx | weights | meta | script
```

The same is available in the dashboard (Experiments, select a run: download chips and a "Try it" form) and over HTTP
(`GET /api/tracking/runs/:id/artifacts`, `POST /api/predict {run_id, inputs}`). Run ids accept any unambiguous prefix.
GAN runs (`generative/gan@v1`) export the generator the same way; `sentinel model generate <run> -c 8` and the dashboard's
Generate button draw samples. A GAN component defines two graphs plus a training recipe (`gan: {generator, discriminator,
training}`), both built from the same block hierarchy, with two optimizers and an alternating update loop.

Predictions run in a short-lived Python process, so the daemon itself never loads PyTorch; batch size is free.

## Model statistics

Every model reports its size before you train it: learnable **parameters** (per node, per compound block, and in total, with
weight sharing counted once), memory as float32, **MACs per sample** in the matrix and convolution layers, how deep the block
hierarchy goes, and which block types hold the weights.

```
$ sentinel component show vision/vit-tiny@v1
...
MODEL
  parameters   112,906 (441.0 KB as float32)
  compute      1.39 MMACs per sample (matrix and convolution layers)
  structure    24 fundamental blocks · 4 compound blocks · 2 level(s) of nesting
  weights in   linear 30% · attention 29% · positional_encoding 29% · patch_embed 11% · layernorm 1%
```

The counts are computed analytically from shapes, and the test suite checks them against PyTorch: parameter counts equal
`sum(p.numel())` for every shipped component and every parameterised block (including multi-layer bidirectional LSTM/GRU, GAT and
transformer layers), and MACs equal PyTorch's FLOP counter (as 2 x MACs) on MLP, CNN, ResNet and U-Net models. In the Builder
the same numbers appear in a live bar, as a badge on each node, and in the inspector; component cards show the size of the
default configuration; and each training run records `n_params`.

## Tasks, losses and training strategies

A component's `training:` section picks a task, a loss, an optimizer and optional strategies. Run
`sentinel component training` (or `--task ranking`) for the full catalog; the Builder form and
`GET /api/training/catalog` use the same data. The task is inferred from the loss; set `task:` to override.

```yaml
training:
  loss: {name: focal, gamma: 2.0, class_weights: balanced}
  optimizer: {name: sgd, momentum: 0.9, nesterov: true}
  lr: 0.05
  epochs: 100
  scheduler: {name: cosine, warmup_epochs: 5}
  clip_grad: 1.0
  accumulate: 4
  amp: bf16
  early_stopping: {patience: 10, monitor: test_loss}
  save_best: true
  data: {csv: data/d.csv, features: [a, b], target: label}
```

- **Tasks:** classification, binary, multilabel, regression, ranking (needs `data.group`), reconstruction, vae, custom.
- **Losses:** cross-entropy (label smoothing, class weights), focal, multi-margin, BCE (pos weight), MSE, MAE, Huber, smooth-L1, quantile, Poisson, log-cosh, RankNet, margin-rank, ListNet, VAE.
- **Custom:** `loss: custom` with a `custom:` section holding inline `source:` or a project-relative `file:`, plus `loss:` / `metric:` function names.
- **Strategies:** weight decay, gradient clipping, accumulation, mixed precision, early stopping, best-epoch restore, layer freezing, `init_from` a saved `model.pt`, device selection.

Mistakes are rejected before anything runs, with a specific message (for example, a loss that does not fit the task).

## Footprint

Sentinel's own cost, measured by `tests/bench_footprint.py` on a release build (Apple M1, macOS; run it on your machine with
`cargo build --release -p sentinel-cli && python3 tests/bench_footprint.py`). Training runs are not included.

| Scenario | RSS | CPU (of one core) |
|---|---|---|
| daemon idle, no dashboard | 15 MB | 0.16% |
| dashboard open | 15 MB | 0.74% |
| 3 dashboards polling | 16 MB | 0.91% |
| ~100 API requests/s | 16 MB | 2.7% |
| watching one running job | 18 MB | 1.5% |

The daemon samples once a second only while a dashboard is connected or a job is running, otherwise every five seconds. All
open dashboards share one sampler, so extra tabs cost almost nothing. The binary is 7 MB with the dashboard embedded.
`sentinel doctor` peaks near 200 MB only because it imports PyTorch in a child process to check it.

## 2. Core Capabilities

### 2.1 Asynchronous Process Supervision & GPU Allocation
* **Deterministic Process Lifecycle:** Subprocess execution is managed asynchronously via `tokio::process`. Processes receive POSIX `SIGTERM` signals for graceful termination, followed by automated stdout/stderr buffer drainage and exit code capture.
* **Dynamic GPU Worker Pool:** The platform queries NVML to monitor physical GPU utilization, compute load, temperature, and VRAM occupancy. Device slots are dynamically assigned to jobs via `CUDA_VISIBLE_DEVICES` injection to prevent memory contention. Transparent CPU fallback is provided when compute devices are unavailable.

### 2.2 Declarative Recipe Components with Inheritance
* **Strict Parameter Typing & Bounds:** Experiment configurations are defined through typed YAML schemas (`int`, `float`, `string`, `choice`) with strict numeric bounds (`[min, max]`) and choice enumerations.
* **Hierarchical Inheritance:** Components compose hierarchically via `extends:` directives, allowing base configurations to be safely specialized across tasks and architectures.
* **Framework Invariant Protection:** Critical production configurations (e.g., `tracking.backend: mlflow`, `precision: bf16-mixed`) can be marked as `locked`, preventing unintended override during runtime execution.

### 2.3 Zero-Lock-In Standalone Script Compilation
* **Unencumbered Code Generation:** The compilation command (`sentinel export script`) resolves component hierarchies and parameter dictionaries into standalone, production-grade PyTorch scripts (`train.py`).
* **Complete Framework Decoupling:** Generated scripts have zero dependencies on Sentinel, Lightning, or proprietary runtime packages, ensuring workloads can execute directly in arbitrary environments.

### 2.4 Distributed Hyperparameter Optimization (Optuna ASHA)
* **Automated Trial Scheduling:** Integrated with Optuna's Asynchronous Successive Halving Algorithm (`SuccessiveHalvingPruner`).
* **VRAM Recovery through Early Pruning:** Diverging or underperforming candidate models are terminated early, releasing GPU memory slots back into the dynamic pool for concurrent trials.

### 2.5 Resilient, Crash-Immune Telemetry SDK
* **Zero Overhead Isolation:** The `sentinel-sdk` Python client writes metrics and parameters directly to the filesystem following strict MLflow 2.x and 3.x specifications (`mlruns/`).
* **Non-Blocking Telemetry Ingestion:** All SDK calls are wrapped in non-panicking failsafe mechanisms. Disk write anomalies, network drops, or supervisor outages **never disrupt active training loops**.

### 2.6 Outbound Edge Agent for Remote Nodes
* **Firewall and NAT Traversing:** The `sentinel-agent` service establishes outbound-only WebSocket links to the primary supervisor daemon, eliminating the requirement for exposed inbound ports.
* **Durable Disk Buffering:** In the event of network disruption, telemetry events are serialized to an append-only on-disk ledger (`.sentinel/agent_buffer.jsonl`) and replayed deterministically upon reconnection.

### 2.7 Operations Management Console (Default Port 2512)
* **Geist-Inspired Monochrome Interface:** High-contrast pitch-black (`#000000`) and pure white (`#ffffff`) operational dashboard engineered for continuous operations monitoring.
* **Strict Read-Only Code Isolation:** Project source code and training recipes are displayed in a read-only browser to eliminate accidental client-side mutation while keeping memory overhead negligible.
* **Interactive Control Planes:** Recipe parameters, sweep boundaries, and process supervision controls remain fully interactive and editable.
* **Graceful Empty-State Fallbacks:** System cards gracefully display operational standby states when experiments or external repositories are uninitialized.

### 2.8 Flexible Dataset Mounting & External Storage Paths
* **Zero Repo Bloat:** Large ML datasets (e.g., ImageNet, OpenWebText, Parquet tables) do not need to reside within the repository.
* **External Path Mounts:** Supports arbitrary local filesystem paths (e.g. `/mnt/nvme/data`, `~/datasets/cifar10`), network shares, or in-repo data files.
* **Python Resolution (`sentinel.get_dataset_path`):** Recipes and training scripts resolve mounted datasets automatically across environments without hardcoded paths.
* **Live Storage Inspection:** The web console automatically validates dataset existence, total byte size, file count, and mount type (`EXTERNAL MOUNT` vs `IN REPO`).

---

## 3. Workspace Architecture

The platform is structured as a Cargo multi-crate workspace accompanied by a root Python workspace managed via `uv`:

```
sentinel/
├── Cargo.toml                       # Workspace manifest defining Rust members and dependencies
├── sentinel.lock                    # Machine-managed lockfile (tool version, components & invariants)
├── sentinel.yaml                    # Workspace Sentinel configuration (project name, defaults, runners)
├── machines.yaml                    # Machine & worker node inventory (local & remote SSH)
├── pyproject.toml                   # Root Python project specification (uv workspace mode)
├── uv.lock                          # Cryptographically pinned Python dependencies
├── sentinel.md                      # Technical specifications and architectural design document
├── README.md                        # Master platform documentation
├── guide.md                         # Operational guide for autonomous agents
├── FILES.md                         # Complete codebase inventory and handoff documentation
├── .agents/
│   └── skills/
│       └── sentinel-integration/    # Antigravity operational skill specification
├── crates/
│   ├── sentinel-core/               # Domain primitives, Hydra configuration, component engine
│   ├── sentinel-daemon/             # Process supervisor, dynamic GPU pool, Axum REST/WebSocket
│   ├── sentinel-agent/              # Remote node agent with offline disk-buffer replay
│   ├── sentinel-cli/                # Unified binary CLI (`sentinel dev`, `component`, `export`)
├── ui/                              # React dashboard (embedded into the binary)
├── components/                      # Component library (graph components + recipes)
├── packages/
│   └── sentinel-sdk/                # Python client SDK with crash-immune MLflow tracking
├── recipes/
│   ├── vision_train.py              # Synthetic PyTorch vision classification loop
│   └── sweep_orchestrator.py        # Distributed Optuna ASHA hyperparameter sweep engine
├── components/
│   ├── vision/                      # Computer vision image classification component
│   ├── llm/                         # Large Language Model PEFT/LoRA fine-tuning component
│   ├── tabular/                     # Multi-layer perceptron tabular classification component
│   └── custom/                      # Starter custom recipes and arbitrary command adapters
└── tests/
    └── e2e_lifecycle.sh             # Automated end-to-end user lifecycle integration test
```

---

## 4. Installation & Deployment

### 4.1 System Prerequisites

* **Rust Toolchain:** Version 1.75 or higher (`cargo`, `rustc`)
* **Python Runtime:** Version 3.10 or higher
* **Package Management:** Strictly managed via **`uv`**

### 4.2 Building from Source

```bash
# Clone the repository
git clone https://github.com/rimraf-adi/sentinel.git
cd sentinel

# Synchronize the locked Python environment
uv sync

# Compile the unified release binary
cargo build --release -p sentinel-cli
```

The compiled binary will be located at `target/release/sentinel`.

### 4.3 Starting the Control Daemon

By default, the platform initializes the server on port **`2512`**:

```bash
# Start daemon and launch the operations console
./target/release/sentinel dev
```

Output:
```text
🚀 Sentinel Daemon active at http://localhost:2512
   API health: http://localhost:2512/api/health
   WebSocket telemetry: ws://localhost:2512/ws/live
```

To configure a custom port:
```bash
./target/release/sentinel dev --port 3000
```

---

## 5. Command-Line Interface (CLI) Specification

```text
Usage: sentinel <COMMAND>

Commands:
  dev       Start daemon and launch the low-overhead Web UI [default port: 2512]
  init      Convert/initialize an existing directory into a Sentinel repository
  new       Scaffold a new project repository with conventions and configs
  run       Run a training job from a component or config
  sweep     Launch Optuna ASHA hyperparameter sweep
  component  Component management subcommands (list, new, check, save-run)
  export    Export run as standalone pure bare PyTorch script
  help      Print this message or the help of the given subcommand(s)
```

### Detailed Command Index

| Command | Arguments / Flags | Description |
| :--- | :--- | :--- |
| `sentinel dev` | `[--port <PORT>]` (default: `2512`) | Starts the Axum HTTP REST and WebSocket telemetry daemon. |
| `sentinel new <NAME>` | `<NAME>` | Scaffolds a new ML project structure with default components and configs. |
| `sentinel component list` | None | Discovers and displays built-in and custom components. |
| `sentinel component new <NAME>` | `[--domain <DOMAIN>]` `[--extends <PARENT>]` | Scaffolds a valid custom component configuration. |
| `sentinel component check <PATH>` | `<PATH>` | Validates syntax, parameter ranges, choices, and locked attributes. |
| `sentinel component save-run <ID>` | `<ID>` `--name <NAME>` | Freezes parameters from an executed experiment into a reusable component. |
| `sentinel export script` | `[--component <ID>]` `[--output <PATH>]` | Transpiles a component configuration into pure PyTorch code. |
| `sentinel sweep <CONFIG>` | `<CONFIG>` | Executes parallel hyperparameter sweeps with automated early stopping. |

---

## 6. Python Telemetry SDK Reference

The `sentinel-sdk` provides an idiomatic, non-blocking telemetry interface:

```python
import time
import sentinel

# Initialize experiment context (creates MLflow-compliant layout in mlruns/)
run = sentinel.init(
    experiment_name="cifar10-benchmark",
    run_name="resnet18-baseline"
)

# Record immutable hyperparameters
run.log_params({
    "model": "resnet18",
    "learning_rate": 0.0003,
    "batch_size": 64,
    "optimizer": "AdamW"
})

# Training Loop
for epoch in range(1, 11):
    train_loss = 2.5 / epoch
    val_accuracy = 0.70 + (0.02 * epoch)

    # Log step-indexed metric telemetry
    # Safe Execution Guarantee: Failures in logging never disrupt training.
    run.log({
        "train/loss": train_loss,
        "val/accuracy": val_accuracy
    }, step=epoch)

    time.sleep(0.05)

# Finalize run metadata and status
run.finish()
```

---

## 7. Verification, Quality Assurance & Testing

The platform enforces strict automated quality gates across systems programming and machine learning disciplines:

```bash
# 1. Execute complete Rust workspace test suite (31 tests)
cargo test --workspace

# 2. Execute Python SDK compliance & crash-immunity tests
uv run pytest packages/sentinel-sdk/tests/test_sdk.py
uv run python packages/sentinel-sdk/tests/test_rigorous.py

# 3. Execute End-to-End User Lifecycle Integration Test
bash tests/e2e_lifecycle.sh
```

### Automated Test Coverage Summary

* **`sentinel-core` (15 tests):** Multi-level component inheritance resolution (`extends:`), numerical parameter boundary enforcement, categorical choice validation, locked-parameter invariance, and bare script compilation.
* **`sentinel-daemon` (9 tests):** Subprocess signal termination (`SIGTERM`), pipe drain integrity, dynamic GPU allocation and contention handling, high-throughput log streaming (10,000+ lines), and one-million-point server-side time-series downsampling (<15ms).
* **`sentinel-agent` (4 tests):** Offline telemetry buffering, disk queue replay upon reconnection, and corrupted record skipping.
* **Component graphs (17 tests):** DAG validation (cycles, ports, shapes, weight sharing), shape inference, and real PyTorch forward passes for every shipped graph component.
* **`sentinel-sdk`:** Strict bit-for-bit parity with official `mlflow.MlflowClient` and multithreaded logging safety.

---

## 8. REST & WebSocket API Specification

When running on default port **`2512`**, Sentinel exposes the following HTTP and WebSocket interfaces:

| Endpoint | Method | Payload / Response | Description |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | HTML (`text/html`) | Minimalist monochrome operations console. |
| `/api/health` | `GET` | `{"status": "ok", "service": "sentinel-daemon"}` | Service health verification. |
| `/api/jobs` | `GET` | `JobRecord[]` | Lists active and historical job records. |
| `/api/jobs` | `POST` | `{"name": "...", "command": "..."}` | Submits arbitrary command for execution. |
| `/api/jobs/run-component`| `POST` | `{"component_id": "...", "overrides": {...}}` | Launches a job from a resolved component. |
| `/api/jobs/:id/logs` | `GET` | `{"job_id": "...", "logs": [...]}` | Retrieves captured stdout/stderr lines. |
| `/api/jobs/:id/stop` | `POST` | `{"status": "stopped"}` | Dispatches `SIGTERM` to active job process. |
| `/api/hardware` | `GET` | `HardwareStats` | Real-time CPU, RAM, and NVML telemetry snapshot. |
| `/api/components` | `GET` | `Component[]` | Catalog of loaded declarative recipes. |
| `/api/files/content` | `GET` | `{"content": "..."}` | Serves read-only source code for UI viewer. |
| `/ws/live` | `WebSocket` | Live Telemetry Stream | Real-time bidirectional WebSocket metric push. |

---

## 9. License

Sentinel is distributed under the dual licensing model standard in the Rust ecosystem:

* **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
* **MIT License** ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))

Contributions submitted for inclusion in the platform are dual-licensed under these terms.
