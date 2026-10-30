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
├── sentinel.lock                       # Machine-managed lockfile (tool version, templates & invariants)
├── sentinel.yaml                       # Workspace Sentinel configuration (project name, defaults, runners)
├── machines.yaml                       # Machine & worker node inventory (local & remote SSH)
├── .github/
│   └── workflows/
│       └── release.yml                 # CI workflow building platform wheels for 4 OS/arch targets
├── crates/
│   ├── sentinel-core/                  # Core domain models, template engine, config parser & exporter
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs                  # Module exports for sentinel-core
│   │   │   ├── models.rs               # Core types: JobSpec, JobRecord, HardwareStats, MetricPoint
│   │   │   ├── config.rs               # Hydra / OmegaConf YAML hierarchy & dot-path override engine
│   │   │   ├── templates.rs            # Declarative template schema, registry, validation & inheritance
│   │   │   ├── scaffold.rs             # Project generator for `sentinel new`
│   │   │   ├── export.rs               # Zero lock-in pure PyTorch script generator
│   │   │   └── storage.rs              # Storage interfaces & MLflow driver definitions
│   │   └── tests/
│   │       └── test_edge_cases.rs      # 3-level inheritance, strict type, range & malformed YAML tests
│   ├── sentinel-daemon/                # Process supervisor, scheduler, NVML monitor & Axum API
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs                  # Module exports for sentinel-daemon
│   │   │   ├── supervisor.rs           # Async process supervisor, stdout/stderr drain & signal handling
│   │   │   ├── scheduler.rs            # Dynamic GPU Worker Pool & HardwareMonitor (sysinfo/NVML)
│   │   │   └── server.rs               # Axum REST API, WebSocket telemetry & metric downsampler
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
│   │       └── main.rs                 # CLI entry point (dev, run, sweep, template, export, new)
│   └── sentinel-ui/                    # Lightweight Web UI & state transition controller
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs                  # Module exports for sentinel-ui
│           ├── state.rs                # Dashboard state engine (ReadOnly -> Editing -> Idle transitions)
│           └── view.rs                 # Standalone HTML/WASM view renderer with gauges & template cards
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
├── templates/                          # Reference & starter templates
│   ├── vision/
│   │   └── image-classification.yaml   # Vision classification reference (ResNet/ViT, AMP, cosine LR)
│   ├── custom/
│   │   ├── command-runner.yaml         # Zero-touch wrapper for existing user commands (`python train.py`)
│   │   └── starter.yaml                # Clean starter template for custom recipes
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
* **`Cargo.toml`**: Declares Cargo workspace members (`sentinel-core`, `sentinel-daemon`, `sentinel-agent`, `sentinel-cli`, `sentinel-ui`) and shared workspace dependencies.
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

### 2. `crates/sentinel-core` (Data Models, Templates & Exporter)
* **`src/models.rs`**: Canonical domain structs: `JobSpec` (execution parameters), `JobRecord` (live status & exit codes), `JobStatus`, `RunMode` (`Command`, `Sdk`, `Framework`), `HardwareStats`, `GpuStats`, `MetricPoint`, and `MachineSpec`.
* **`src/config.rs`**: Implements `HydraConfig` supporting hierarchical YAML composition and arbitrary dot-path overrides (e.g. `model.backbone.encoder.layers=12`).
* **`src/templates.rs`**: Core declarative template engine. Handles YAML parsing, parameter schema validation (types, ranges, choice options), locked fields protection, template inheritance resolution (`extends:`), custom template starter creation, and freezing runs into templates.
* **`src/scaffold.rs`**: `ProjectScaffolder` engine used by `sentinel new` to generate a production-ready repository with `sentinel.yaml` (human settings), `sentinel.lock` (machine state), `machines.yaml`, and folder conventions.
* **`src/export.rs`**: `ScriptExporter` translates template specs into standalone, pure bare PyTorch scripts with zero runtime dependencies on Sentinel or Lightning.
* **`src/storage.rs`**: Storage abstractions for tracking runs in MLflow-native layout.
* **`tests/test_edge_cases.rs`**: Integration test suite verifying 3-level template inheritance, numeric range enforcement, choice validation, locked parameter barriers, and malformed YAML handling.

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
  * `GET /api/templates` — Returns template catalog.
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
  * `sentinel template list` — Lists built-in and user custom templates.
  * `sentinel template new <name>` — Generates a custom template starter.
  * `sentinel template check <path>` — Validates template YAML syntax and parameters.
  * `sentinel export script` — Generates a standalone bare PyTorch script.
  * `sentinel sweep <config>` — Spawns parallel Optuna ASHA hyperparameter sweeps.
* **`pyproject.toml`**: Configures `maturin` to build platform wheels containing the compiled `sentinel` binary for PyPI/uv distribution.

---

### 6. `crates/sentinel-ui` (Web Dashboard & State Engine)
* **`src/state.rs`**: Manages dashboard mode transitions (`ReadOnly` $\to$ `Editing` $\to$ `Idle`). Features an automated idle detector that transitions to low-power `Idle` mode after 15 seconds of inactivity to shed event listeners and slow down polling.
* **`src/view.rs`**: Standalone HTML view generator rendering a minimalist, clean black-and-white interface inspired by Vercel's Geist design system. Includes:
  * Left sidebar navigation (`Overview`, `Runs`, `Sweeps`, `Templates`, `Datasets`, `Machines`, `Code (Read-Only)`).
  * Recent run hero card with live status pills, duration, worker info, progress bar, and elegant empty-state handling.
  * Git Repos / Projects overview with clean empty-state fallback when no external projects are configured.
  * Fully interactive & editable template configurator modal and custom command runner.
  * Strictly read-only IDE code viewer for project files and recipes with zero memory bloat.

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

### 9. `templates/` (Reference & Starter Templates)
* **`templates/vision/image-classification.yaml`**: Standard image classification recipe with ResNet/ViT, AMP precision, and cosine annealing scheduler.
* **`templates/custom/command-runner.yaml`**: Zero-touch adapter wrapping legacy scripts (`python train.py`).
* **`templates/custom/starter.yaml`**: Base template for user-defined custom recipes.
* **`templates/llm/lora-finetune.yaml`**: Large Language Model LoRA/PEFT fine-tuning specification.
* **`templates/tabular/mlp-classifier.yaml`**: Multi-layer perceptron tabular classification specification.

---

### 10. `tests/` (End-to-End Test Suite)
* **`tests/e2e_lifecycle.sh`**: Automated shell script testing project scaffolding, template creation, validation, script export, and standalone PyTorch execution.

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
