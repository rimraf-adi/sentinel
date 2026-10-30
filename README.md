# Sentinel 🛡️

> **The watchful guardian for your training runs.**  
> *"Rust is boss. Python is worker."*

[![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange?logo=rust)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org)
[![Package Manager](https://img.shields.io/badge/uv-strictly_enforced-purple?logo=astral)](https://github.com/astral-sh/uv)
[![Storage](https://img.shields.io/badge/MLflow-3.x_Native-0194E2?logo=mlflow)](https://mlflow.org)
[![Tests](https://img.shields.io/badge/Tests-31%20Passed-brightgreen)](https://github.com/rimraf-adi/sentinel)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](LICENSE)

Sentinel is a high-performance machine learning supervisor, job scheduler, and experiment control plane built with a dual-engine architecture:

* **Rust is Boss:** A single, lightweight Rust binary manages process supervision, dynamic GPU worker pool allocation, NVML telemetry, WebSocket streaming, and a zero-overhead Vercel-style web interface.
* **Python is Worker:** Pure Python processes execute deep learning models with PyTorch, torchvision, HuggingFace, and Optuna without proprietary training wrappers or framework lock-in.

---

## Key Pillars & Features

### 1. Minimalist Monochrome Dashboard (Vercel Style)
* **Geist-Inspired Design:** Clean, high-contrast pitch-black (`#000000`) and pure white (`#ffffff`) aesthetic with thin crisp borders (`#222222`).
* **Sidebar Navigation:** Workspace overview, live runs, hyperparameter sweeps, template catalog, dataset mounts, machine telemetry, and in-browser code viewer.
* **Clean Empty States:** Unobtrusive, minimal fallback states when no training runs or external projects are registered.

### 2. Strict Read-Only Code View (Zero RAM Bloat)
* **Code / IDE View is Strictly Read-Only:** Inspect recipes, configs, and project source files in the browser with ultra-low latency and zero memory overhead. Code cannot be accidentally mutated in the browser.
* **Editable Dashboards & Controls:** All other components—recipe parameter forms, custom command execution, sweep budgets, and job lifecycle controls—remain 100% interactive and editable.

### 3. Declarative Template Engine & Inheritance
* **Structured Schemas:** Hierarchical configuration with typed parameters (`int`, `float`, `string`, `choice`), range bounds (`[min, max]`), and default values.
* **Locked Framework Safeguards:** Protect production parameters (e.g. `tracking.backend: mlflow`, `precision: bf16-mixed`) from accidental runtime overrides.
* **Zero Lock-In Exporter (`sentinel export script`):** Translates declarative template configurations into standalone, pure bare PyTorch code stripping both Sentinel and Lightning.

### 4. Dynamic GPU Pool & Process Supervisor
* **Zero-Collision GPU Packing:** `DynamicGpuPool` tracks physical GPU slots, automatically packs jobs onto available devices via `CUDA_VISIBLE_DEVICES`, and falls back to CPU when needed.
* **Process Lifecycle:** Asynchronous subprocess management with SIGTERM termination, log streaming, exit code capture, and post-exit pipe draining.
* **Live Telemetry:** Real-time host CPU, host RAM, GPU utilization, temperature, and VRAM monitoring via NVML.

### 5. Optuna ASHA Sweeps Orchestrator
* Automated hyperparameter optimization powered by Optuna with `SuccessiveHalvingPruner` (ASHA).
* Concurrently evaluates trials and terminates diverging candidates early to free VRAM for promising runs.

### 6. Crash-Immune Python SDK (`sentinel-sdk`)
* Standard MLflow-native file logging (`mlruns/`) with zero dependency on an active daemon.
* Failsafe decorators guarantee that telemetry or disk errors **never halt or crash model training**.

### 7. Outbound Remote Node Agent (`sentinel-agent`)
* Lightweight remote agent connecting outbound via WebSocket to the central supervisor (firewall and NAT friendly).
* Persistent on-disk queue (`.sentinel/agent_buffer.jsonl`) buffers events during network outages and replays them upon reconnection.

---

## Repository Architecture

```
uxML/
├── Cargo.toml                       # Rust workspace configuration (5 crates)
├── pyproject.toml                   # uv workspace & maturin wheel config
├── uv.lock                          # Pinned Python dependencies
├── sentinel.md                      # Master architecture specification
├── FILES.md                         # Detailed codebase inventory for agent handoffs
├── crates/
│   ├── sentinel-core/               # Domain models, Hydra config, template engine, script exporter
│   ├── sentinel-daemon/             # Process supervisor, dynamic GPU pool, NVML monitor, Axum server
│   ├── sentinel-agent/              # Outbound agent with offline persistent disk buffer
│   ├── sentinel-cli/                # Unified binary CLI (`sentinel dev`, `template`, `export`, `sweep`)
│   └── sentinel-ui/                 # Standalone Vercel-style HTML/WASM dashboard & state engine
├── packages/
│   └── sentinel-sdk/                # Python client SDK with crash-immune MLflow 3.x tracking
├── recipes/
│   ├── vision_train.py              # Reference synthetic PyTorch image classification loop
│   └── sweep_orchestrator.py        # Optuna ASHA multi-trial hyperparameter sweep engine
├── templates/
│   ├── vision/                      # Computer vision image classification template
│   ├── llm/                         # LLM PEFT/LoRA fine-tuning template
│   ├── tabular/                     # Multi-layer perceptron tabular classification template
│   └── custom/                      # Starter and custom command runner templates
└── tests/
    └── e2e_lifecycle.sh             # End-to-end integration test script
```

---

## Quickstart

### Prerequisites

* **Rust:** 1.75+ (`cargo`, `rustc`)
* **Python:** 3.10+ managed strictly via **`uv`**

```bash
# Clone the repository
git clone https://github.com/rimraf-adi/sentinel.git
cd sentinel

# Install Python dependencies using uv
uv sync
```

### Build & Run the Dashboard

```bash
# Compile the Sentinel CLI
cargo build -p sentinel-cli

# Start the local daemon and web interface (defaults to port 2512)
target/debug/sentinel dev
```

Open your browser at **`http://localhost:2512`** to view the Vercel-style training control dashboard.

---

## CLI Reference

| Command | Description |
| :--- | :--- |
| `sentinel dev [--port 2512]` | Starts the Axum daemon REST/WebSocket server and serves the web UI (default: 2512) |
| `sentinel new <project-name>` | Scaffolds a new project repository with conventions and configs |
| `sentinel template list` | Lists all built-in and custom templates in the registry |
| `sentinel template new <name> [--extends <parent>]` | Generates a custom template starter YAML |
| `sentinel template check <path>` | Validates YAML syntax, parameter types, ranges, and locked fields |
| `sentinel template save-run <run-id>` | Freezes an experiment run's parameters into a reusable template |
| `sentinel export script [--template <id>]` | Generates a pure bare PyTorch standalone script |
| `sentinel sweep <config>` | Runs an automated Optuna ASHA hyperparameter sweep |

---

## Python SDK Example

```python
import time
import sentinel

# Initialize experiment (creates MLflow 3.x structure in mlruns/)
run = sentinel.init(experiment_name="mnist-classifier", run_name="baseline-run")
run.log_params({"lr": 0.001, "batch_size": 64, "optimizer": "AdamW"})

for epoch in range(1, 10):
    loss = 1.0 / epoch
    acc = 0.8 + (0.02 * epoch)
    
    # Safe logging: disk or daemon errors will NEVER crash training
    run.log({"train_loss": loss, "val_acc": acc}, step=epoch)
    time.sleep(0.1)

run.finish()
```

---

## Running Verification Tests

Sentinel includes a rigorous testing suite covering edge cases, process failures, concurrency, and MLflow parity:

```bash
# Run all 31 Rust unit & integration tests
cargo test --workspace

# Run Python SDK tests with uv
uv run pytest packages/sentinel-sdk/tests/test_sdk.py
uv run python packages/sentinel-sdk/tests/test_rigorous.py

# Run full end-to-end user lifecycle test
bash tests/e2e_lifecycle.sh
```

---

## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
