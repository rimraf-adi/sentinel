# Sentinel — Codebase File Tree & Handoff Summary

> **Agentic Handoff Context:** This document provides a complete directory map, architectural role, and concise summary of every file in the Sentinel repository to orient autonomous agents and developers immediately.

---

## Architecture at a Glance

* **Philosophy:** *"Rust is boss. Python is worker."*
* **Core Daemon & CLI:** Single Rust binary built from a Cargo workspace across 5 modular crates.
* **Web UI:** Pure Rust frontend with low-power idle controller served directly by the Axum daemon.
* **Python Runtime:** Managed **strictly using `uv`** (`packages/sentinel-sdk`).
* **Experiment Storage:** Strict **MLflow-native** format (`mlruns/` layout readable by standard MLflow tooling).
* **Zero Lock-in Export:** `sentinel export script` generates self-contained bare PyTorch code.

---

## Directory Tree

```text
sentinel/
├── sentinel.lock                       # Machine-managed lockfile (tool version, components & invariants)
├── sentinel.yaml                       # Workspace Sentinel configuration (project name, defaults, runners)
├── machines.yaml                       # Machine & worker node inventory (local & remote SSH)
├── datasets.yaml                       # Dataset mounts & external filesystem paths inventory
├── .github/
│   └── workflows/
│       └── release.yml                 # CI workflow building platform wheels for 4 OS/arch targets
├── crates/
│   ├── sentinel-core/                  # Core domain models, component engine, config parser & exporter
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs                  # Module exports for sentinel-core
│   │   │   ├── models.rs               # Core types: JobSpec, JobRecord, HardwareStats, MetricPoint
│   │   │   ├── config.rs               # Hydra / OmegaConf YAML hierarchy & dot-path override engine
│   │   │   ├── components.rs           # Declarative component schema, 3-tier registry, validation & inheritance
│   │   │   ├── blocks.rs               # Fundamental block catalog & shape inference
│   │   │   ├── blocks/
│   │   │   │   └── more.rs             # Extended tensor ops, norms, pools, 3D convolutions & generative blocks
│   │   │   ├── compound.rs             # Compound block hierarchy & nested module generator
│   │   │   ├── graph.rs                # Graph DAG validation, cycle detection & shape propagation
│   │   │   ├── authoring.rs            # Visual Builder serialization & parameter tieing
│   │   │   ├── scaffold.rs             # Project generator for `sentinel new`
│   │   │   ├── export.rs               # Zero lock-in pure PyTorch script generator
│   │   │   └── storage.rs              # Storage interfaces & MLflow driver definitions
│   │   └── tests/
│   │       ├── test_edge_cases.rs      # 3-level inheritance, strict type, range & malformed YAML tests
│   │       └── test_more_blocks.rs     # 72 fundamental block PyTorch validation tests
│   ├── sentinel-daemon/                # Process supervisor, scheduler, NVML monitor & Axum API
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs                  # Module exports for sentinel-daemon
│   │   │   ├── supervisor.rs           # Async process supervisor, stdout/stderr drain & signal handling
│   │   │   ├── scheduler.rs            # Dynamic GPU Worker Pool & HardwareMonitor (sysinfo/NVML)
│   │   │   └── server.rs               # Axum REST API, WebSocket telemetry & 3-tier component reloading
│   │   └── tests/
│   │       └── test_stress.rs          # Concurrency, 5,000-line log stress & GPU contention tests
│   ├── sentinel-agent/                 # Outbound WebSocket remote agent for remote GPU nodes
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs                  # Module exports for sentinel-agent
│   │   │   └── client.rs               # Outbound WebSocket client & offline disk event buffer
│   │   └── tests/
│   │       └── test_agent_stress.rs    # 1,000-event offline buffer & corrupted line recovery tests
│   ├── sentinel-cli/                   # `sentinel` CLI binary & Maturin wheel packaging
│   │   ├── Cargo.toml
│   │   ├── pyproject.toml              # Maturin build configuration for PyPI/uv wheels
│   │   └── src/
│   │       └── main.rs                 # CLI entry point (dev, run, sweep, component, template, export, model, new)
├── ui/                                 # React dashboard (Vite + TypeScript + React Flow), embedded into the binary
│   ├── package.json
│   ├── vite.config.ts
│   └── src/                            # App.tsx, api.ts, pages/*, builder/* (drag & drop component graph)
├── components/                         # Component library: graph components (ANN, LSTM, CNN, ...) and recipes
├── packages/
│   └── sentinel-sdk/                   # Thin Python SDK with safe-tracking wrapper
│       ├── pyproject.toml              # Hatchling package metadata for sentinel-sdk
│       ├── README.md                   # Quickstart guide for Python developers
│       ├── src/
│       │   └── sentinel/
│       │       └── __init__.py         # init(), log(), log_params(), finish() & safe try/except wrappers
│       └── tests/
│           ├── test_sdk.py             # Basic SDK logging & official MLflowClient verification test
│           └── test_rigorous.py        # Multi-threaded concurrent logging & crash-immunity tests
├── recipes/
│   ├── vision_train.py                 # Fast CPU/GPU synthetic vision training smoke test
│   └── sweep_orchestrator.py           # Optuna ASHA dynamic hyperparameter sweep orchestrator
├── components/                          # Reference & starter components
│   ├── vision/
│   │   └── image-classification.yaml   # Vision classification reference (ResNet/ViT, AMP, cosine LR)
│   ├── custom/
│   │   ├── command-runner.yaml         # Zero-touch wrapper for existing user commands (`python train.py`)
│   │   └── starter.yaml                # Clean starter component for custom recipes
│   ├── llm/
│   │   └── lora-finetune.yaml          # LLM LoRA/PEFT parameter-efficient fine-tuning placeholder
│   └── tabular/
│       └── mlp-classifier.yaml         # Structured tabular MLP classifier placeholder
├── tests/
│   └── e2e_lifecycle.sh                # Automated end-to-end user lifecycle integration test script
├── Cargo.lock                          # Pinned Cargo workspace dependencies lockfile
├── Cargo.toml                          # Workspace root manifest declaring all 5 crates
├── pyproject.toml                      # Root uv workspace configuration (`package = false`)
├── uv.lock                             # Pinned Python dependencies lockfile (torch, mlflow, optuna)
├── sentinel.md                         # Formal technical specification and architectural blueprint
├── README.md                           # Master project documentation, quickstart, and CLI guide
├── guide.md                            # Comprehensive operational guide for autonomous agents
├── FILES.md                            # Directory tree and file-by-file agentic handoff summary (this file)
├── .agents/
│   └── skills/
│       └── sentinel-integration/
│           └── SKILL.md                # Antigravity skill for agents to hook into Sentinel codebases
└── .gitignore                          # Ignores build artifacts (target/, .venv/, dist/, mlruns/)
```

---

## File-by-File Summary

### 1. Root Configuration & Documentation
* **`Cargo.toml`**: Declares Cargo workspace members (`sentinel-core`, `sentinel-daemon`, `sentinel-agent`, `sentinel-cli`) and shared workspace dependencies.
* **`pyproject.toml`**: Root Python project configuration declaring `package = false` so `uv` treats the repo as a workspace for recipes, tests, and tools.
* **`uv.lock`**: Pinned lockfile for Python packages (PyTorch, torchvision, MLflow, Optuna, PyYAML).
* **`sentinel.md`**: Master architecture document containing the locked specification, philosophy, and roadmaps.
* **`README.md`**: Master user-facing documentation, quickstart, feature overview, CLI reference, and architecture guide.
* **`guide.md`**: Operational guide for autonomous AI agents instructing how to ground, inspect, launch, monitor, and export Sentinel-supervised workflows.
* **`.agents/skills/sentinel-integration/SKILL.md`**: Antigravity workspace skill packaging operational commands and protocols for AI agents interacting with Sentinel.
* **`FILES.md`**: This file.
* **`.github/workflows/release.yml`**: GitHub Actions matrix workflow building cross-platform wheels via `maturin-action` for:
  1. `x86_64-unknown-linux-gnu` (Linux x86_64)
  2. `aarch64-unknown-linux-gnu` (Linux ARM64 / GH200)
  3. `aarch64-apple-darwin` (macOS Apple Silicon)
  4. `x86_64-pc-windows-msvc` (Windows x86_64)

---

### 2. `crates/sentinel-core` (Data Models, Components & Exporter)
* **`src/models.rs`**: Canonical domain structs: `JobSpec` (execution parameters), `JobRecord` (live status & exit codes), `JobStatus`, `RunMode` (`Command`, `Sdk`, `Framework`), `HardwareStats`, `GpuStats`, `MetricPoint`, and `MachineSpec`.
* **`src/config.rs`**: Implements `HydraConfig` supporting hierarchical YAML composition and arbitrary dot-path overrides (e.g. `model.backbone.encoder.layers=12`).
* **`src/components.rs`**: Core declarative component engine. Handles YAML parsing, parameter schema validation (types, ranges, choice options), locked fields protection, component inheritance resolution (`extends:`), custom component starter creation, and freezing runs into components.
* **`src/scaffold.rs`**: `ProjectScaffolder` engine used by `sentinel new` to generate a production-ready repository with `sentinel.yaml` (human settings), `sentinel.lock` (machine state), `machines.yaml`, and folder conventions.
* **`src/export.rs`**: `ScriptExporter` translates component specs into standalone, pure bare PyTorch scripts with zero runtime dependencies on Sentinel or Lightning.
* **`src/storage.rs`**: Storage abstractions for tracking runs in MLflow-native layout.
* **`tests/test_edge_cases.rs`**: Integration test suite verifying 3-level component inheritance, numeric range enforcement, choice validation, locked parameter barriers, and malformed YAML handling.

---

### 3. `crates/sentinel-daemon` (Supervisor, Scheduler & API Server)
* **`src/supervisor.rs`**: `ProcessSupervisor` handles asynchronous subprocess execution (`tokio::process::Command`), environment variable injection (`CUDA_VISIBLE_DEVICES`), real-time stdout/stderr streaming, process termination via SIGTERM, and complete pipe draining upon process exit.
* **`src/scheduler.rs`**: Contains `HardwareMonitor` (gathers host CPU/RAM and GPU telemetry) and `DynamicGpuPool` (allocates and releases physical GPU device IDs to prevent device collisions).
* **`src/server.rs`**: Axum server serving:
  * `GET /` — Serves low-overhead dashboard HTML.
  * `GET /api/health` — Daemon health check.
  * `GET /api/jobs`, `POST /api/jobs` — Job management and submission.
  * `POST /api/jobs/:id/stop` — Cancels and terminates running jobs.
  * `GET /api/hardware` — Real-time telemetry snapshot.
  * `GET /api/components` — Returns component catalog.
  * `GET /ws/live` — WebSocket stream pushing live hardware telemetry.
  * `Server::downsample_series()` — High-performance server-side metric downsampler (capable of downsampling 1M points in <15ms).
* **`tests/test_stress.rs`**: High-concurrency integration tests verifying 5 parallel jobs, 5,000-line log streaming without drop, process failure exit code capture, GPU contention recovery, and 1M-point downsampling.

---

### 4. `crates/sentinel-agent` (Remote Node Agent)
* **`src/client.rs`**: `AgentClient` connects outbound via WebSocket to the central daemon. Implements `EventBuffer`, a persistent disk buffer (`.sentinel/agent_buffer.jsonl`) that queues telemetry during network disconnections and drains/replays upon reconnection.
* **`tests/test_agent_stress.rs`**: Stress tests verifying buffering 1,000 events during network downtime and recovering valid events when corrupted lines are injected.

---

### 5. `crates/sentinel-cli` (Command Line Interface & Wheel Packaging)
* **`src/main.rs`**: CLI entry point with subcommands:
  * `sentinel dev` — Starts Axum daemon and serves the web dashboard.
  * `sentinel new <name>` — Scaffolds a new project repository.
  * `sentinel component list` — Lists built-in and user custom components.
  * `sentinel component new <name>` — Generates a custom component starter.
  * `sentinel component check <path>` — Validates component YAML syntax and parameters.
  * `sentinel export script` — Generates a standalone bare PyTorch script.
  * `sentinel sweep <config>` — Spawns parallel Optuna ASHA hyperparameter sweeps.
* **`pyproject.toml`**: Configures `maturin` to build platform wheels containing the compiled `sentinel` binary for PyPI/uv distribution.

---

### 6. `ui/` (React Dashboard)
* **Stack:** Vite + React + TypeScript; [React Flow](https://reactflow.dev) powers the component-graph canvas. `npm run build` writes `ui/dist`, which `crates/sentinel-daemon/src/assets.rs` embeds into the binary with `rust-embed`, so the shipped `sentinel` executable is self-contained. See `ui/README.md`.
* **Pages:** Overview, Runs (live logs), Sweeps, Components (hyperparameter forms), **Builder**, Datasets, Machines (live telemetry over `/ws/live`), Code (read-only).
* **Builder:** drag blocks from the palette, wire output ports to input ports (cycles are refused), live shape inference, undo/redo, minimap, View PyTorch, and Run. Component graphs round-trip through `/api/graph/{validate,compose,run}`.
* **Build integration:** `crates/sentinel-daemon/build.rs` builds the UI with npm when `ui/dist` is missing or stale (skip with `SENTINEL_SKIP_UI=1`). Release CI builds it once and passes it to the per-target wheel builds.

---

### 7. `packages/sentinel-sdk` (Python Telemetry SDK)
* **`src/sentinel/__init__.py`**: Python client providing:
  * `init()` — Initializes a run and creates the standard MLflow `mlruns/` structure.
  * `log(metrics, step)` — Writes metrics in `<timestamp> <value> <step>` format.
  * `log_params(params)` — Writes hyperparameters.
  * `log_artifact(path)` — Copies artifacts to `artifacts/`.
  * `finish()` — Updates `meta.yaml` with end time and completion status.
  * `_safe_call` — Safety decorator ensuring logging errors **never raise exceptions or halt user training**.
* **`tests/test_sdk.py`**: Validates basic logging and tests that the official `mlflow.MlflowClient` reads Sentinel-generated runs bit-for-bit.
* **`tests/test_rigorous.py`**: Verifies multithreaded concurrent logging across 4 threads and crash immunity when passed circular references or `NaN`/`Inf` floats.

---

### 8. `recipes/` (Reference Python Execution Workers)
* **`recipes/vision_train.py`**: Synthetic PyTorch convnet classification smoke test. Runs on CPU or GPU without external network access to verify convergence.
* **`recipes/sweep_orchestrator.py`**: Parallel hyperparameter tuning engine powered by Optuna with `SuccessiveHalvingPruner` (ASHA). Dynamically evaluates trials and terminates diverging trials early.

---

### 9. `components/` (Reference & Starter Components)
* **`components/vision/image-classification.yaml`**: Standard image classification recipe with ResNet/ViT, AMP precision, and cosine annealing scheduler.
* **`components/custom/command-runner.yaml`**: Zero-touch adapter wrapping legacy scripts (`python train.py`).
* **`components/custom/starter.yaml`**: Base component for user-defined custom recipes.
* **`components/llm/lora-finetune.yaml`**: Large Language Model LoRA/PEFT fine-tuning specification.
* **`components/tabular/mlp-classifier.yaml`**: Multi-layer perceptron tabular classification specification.

---

### 10. `tests/` (End-to-End Test Suite)
* **`tests/e2e_lifecycle.sh`**: Automated shell script testing project scaffolding, component creation, validation, script export, and standalone PyTorch execution.

---

## Developer Quick Reference

### Building & Running Rust Workspace
```bash
# Check compilation across all 5 crates
cargo check --workspace

# Run all 23 unit and integration tests
cargo test --workspace

# Build sentinel CLI binary
cargo build -p sentinel-cli
```

### Python Environment & Tests (`uv`)
```bash
# Run SDK & MLflow compatibility tests
uv run python packages/sentinel-sdk/tests/test_sdk.py
uv run python packages/sentinel-sdk/tests/test_rigorous.py

# Run synthetic vision training recipe
uv run python recipes/vision_train.py

# Run parallel Optuna ASHA sweep
uv run python recipes/sweep_orchestrator.py --trials 10 --workers 2
```

### End-to-End Lifecycle Verification
```bash
# Run complete lifecycle validation test
./tests/e2e_lifecycle.sh
```

### Building Platform Wheels Locally
```bash
# Build standalone Python wheel containing sentinel CLI
uvx maturin build -m crates/sentinel-cli/Cargo.toml --out dist

# Install and test wheel in virtual environment
uv pip install dist/*.whl
sentinel --help
```
