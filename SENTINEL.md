# SENTINEL.md — Operational Manual for Autonomous Agents & Developers

> **Target Audience:** Autonomous coding agents, LLM background runners, and ML engineers operating inside Sentinel codebases.  
> **Core Axiom:** *Rust is the control plane supervisor; Python is the unencumbered execution worker.*

---

## 1. System Architecture & Mental Model

Sentinel decouples model topology, training supervision, and hardware resource allocation from deep learning execution:

```
┌──────────────────────────────────────────────────────────────────┐
│                   SENTINEL CONTROL PLANE (Rust)                  │
│                     Daemon: Port 2512 (default)                  │
│                                                                  │
│  • Process Supervisor (SIGTERM/SIGKILL, PID tracking, exit codes)│
│  • Dynamic GPU Pool (NVML query, CUDA_VISIBLE_DEVICES assignment)│
│  • Static Topology & Shape Inference Engine (Zero-cost DAG check)│
│  • Axum HTTP REST Engine & WebSocket Telemetry Streamer          │
│  • Embedded React Dashboard & Native Static Asset Server         │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ IPC / Pipes / Env Vars
┌─────────────────────────────────▼────────────────────────────────┐
│                  PYTHON COMPUTE PLANE (uv / PyTorch)             │
│                                                                  │
│  • Pure PyTorch Training Loops (No wrappers, no base classes)    │
│  • Optuna ASHA Distributed Hyperparameter Search Trials          │
│  • `sentinel-sdk` Non-blocking Telemetry Client (Crash-immune)   │
│  • Storage Layer: Strict MLflow 2.x/3.x Directory (`mlruns/`)    │
└──────────────────────────────────────────────────────────────────┘
```

### Agent Invariants
1. **Never write training loops in Rust.** Rust only supervises, monitors, allocates GPUs, and validates DAG topologies.
2. **Never wrap Python in proprietary base classes.** All training code compiles to standard PyTorch modules or raw standalone scripts.
3. **Always use `uv run` for Python execution.** Avoid unmanaged host python or pip commands that pollute the global environment.
4. **Fail before launch.** Always validate shapes, bounds, and YAML connectivity with `sentinel component check` before scheduling runs.

---

## 2. Environment Discovery & Health Protocol

When an agent enters a workspace, execute this discovery routine:

```bash
# 1. Verify Sentinel binary availability
which sentinel || test -f ./target/release/sentinel && echo "Sentinel binary OK"

# 2. Verify Python, PyTorch, NVML, and uv environment
sentinel doctor

# 3. Check if Sentinel Daemon is running
curl -s http://localhost:2512/api/health | jq .

# 4. Check hardware availability
curl -s http://localhost:2512/api/hardware | jq .
```

### Expected Doctor Output
`sentinel doctor` checks and outputs JSON/structured diagnostics:
- **Rust Runtime:** $\ge$ 1.75.0
- **Python Environment:** $\ge$ 3.10 via `uv`
- **PyTorch:** Version, CUDA availability, and device count
- **NVML Drivers:** Per-GPU status, PCIe bus IDs, and memory limit
- **Templates Directory:** User and project component roots

---

## 3. Installation & Daemon Lifecycle

### Building Sentinel from Source
```bash
# Clone and build native release binary
git clone https://github.com/rimraf-adi/sentinel.git
cd sentinel
uv sync
cargo build --release -p sentinel-cli

# Symlink or copy to PATH
cp ./target/release/sentinel /usr/local/bin/sentinel
```

### Starting & Managing the Daemon
The daemon runs the background supervisor, GPU scheduler, API, and embedded web UI:

```bash
# Start in foreground
sentinel dev

# Start on custom port in background
sentinel dev --port 2512 > /dev/null 2>&1 &

# Scaffold new Sentinel workspace in current folder
sentinel init

# Scaffold a new named project
sentinel new my-ml-project
```

---

## 4. Components & 3-Tier Architecture System

Sentinel resolves templates and architectures through a 3-tier precedence hierarchy:

| Tier | Origin | Location | Mutability | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | `builtin` | Embedded in binary | Immutable | Production baselines (ViT, U-Net, ResNet, VAE, GAN) |
| **Tier 2** | `user` | `~/.sentinel/templates/` | User-wide | Reusable modules shared across all projects |
| **Tier 3** | `project` | `./components/` | Local Git repo | Project-specific models and configurations |

### Discovery & Inspection Commands
```bash
# List all components across all tiers
sentinel component list

# Inspect component graph, parameter count, FLOPs, and memory footprint
sentinel component show vision/vit-tiny@v1

# List all available fundamental and compound blocks
sentinel component blocks

# List available training recipes, loss functions, optimizers, and schedulers
sentinel component training
```

### Validating a Component
Always validate before submitting jobs:
```bash
sentinel component check components/tabular/residual-mlp.yaml
```
Validation asserts:
- Syntax and schema conform to version `1.0`.
- Shape propagation matches across all ports and connections.
- Parameter bounds satisfy ranges (`min` $\le$ `val` $\le$ `max`).
- No circular dependencies or self-referential compound blocks exist.

---

## 5. Declarative Block Specification (YAML)

### Compound Block Schema (`blocks/<family>/<name>.yaml`)
Compound blocks define reusable subgraphs with typed parameters and dynamic tensor port shapes:

```yaml
compound: residual_mlp_block
params:
  hidden: {default: 64, type: int}
  width:  {}                        # No default = mandatory required parameter
graph:
  inputs:
    in: [B, "${width}"]             # Dynamic tensor dimension interpolation
  nodes:
    mlp:  {block: mlp_block, params: {width: "${hidden}"}}
    proj: {block: linear, params: {out: "${width}"}}
    skip: {block: add}
  edges:
    - "in -> mlp"
    - "mlp -> proj"
    - "proj -> skip.a"
    - "in -> skip.b"
  outputs:
    out: skip
```

### Full Model Component Schema (`components/<domain>/<name>.yaml`)
Combines model graph wiring with execution hyperparameters:

```yaml
schema_version: "1.0"
component_id: "tabular/residual-mlp@v1"
name: "Residual MLP"
domain: "tabular"

parameters:
  width: {type: int, default: 32, range: [4, 1024]}
  hidden: {type: int, default: 64, range: [4, 2048]}
  classes: {type: int, default: 2, range: [1, 1000]}

graph:
  inputs:
    features: [B, "${width}"]
  nodes:
    block1: {block: residual_mlp_block, params: {width: "${width}", hidden: "${hidden}"}}
    norm:   {block: layernorm}
    head:   {block: linear, params: {out: "${classes}"}}
  edges:
    - "features -> block1"
    - "block1 -> norm"
    - "norm -> head"
  outputs:
    logits: head

training:
  loss: {name: cross_entropy}
  optimizer: {name: adamw, lr: 0.001, weight_decay: 0.01}
  scheduler: {name: cosine, warmup_epochs: 3}
  epochs: 50
  batch_size: 64
  clip_grad: 1.0
  accumulate: 1
  amp: bf16
  early_stopping: {patience: 10, monitor: "val_loss"}
  save_best: true
  data:
    csv: "data/train.csv"
    features: ["f1", "f2", "f3"]
    target: "label"
```

---

## 6. Execution, Sweeps & Telemetry

### Running Experiments
```bash
# Run using the default project component
sentinel run

# Run with parameter overrides
sentinel run --component tabular/residual-mlp@v1 --lr 0.0005 --batch-size 128

# Programmatic submission via REST API
curl -X POST http://localhost:2512/api/jobs/run-component \
  -H "Content-Type: application/json" \
  -d '{
    "component_id": "tabular/residual-mlp@v1",
    "overrides": {"lr": 0.0005, "epochs": 20}
  }'
```

### Running Optuna ASHA Sweeps
Sentinel orchestrates asynchronous parallel hyperparameter sweeps with automatic early-stopping:

```yaml
# sweep.yaml
study:
  name: "tabular-lr-search"
  direction: "minimize"
  metric: "val_loss"
  max_trials: 20
  parallel_jobs: 4
  pruner:
    type: "asha"
    min_resource: 2
    reduction_factor: 3

parameters:
  lr: {type: float, low: 1e-5, high: 1e-2, log: true}
  hidden: {type: categorical, choices: [32, 64, 128, 256]}
```

```bash
# Launch sweep
sentinel sweep sweep.yaml
```

---

## 7. Python SDK Protocol (`sentinel-sdk`)

The Python SDK communicates telemetry and metrics back to the Rust supervisor. It is **crash-immune**: disk write errors, network partitions, or supervisor downtime never crash the Python training loop.

```python
import sentinel

# 1. Initialize run tracking
run = sentinel.init(
    experiment_name="cifar10-benchmark",
    run_name="vit-tiny-run-1"
)

# 2. Log configuration hyperparameters
run.log_params({
    "model": "vit-tiny",
    "learning_rate": 3e-4,
    "batch_size": 64,
    "optimizer": "adamw"
})

# 3. Training loop with non-blocking step logging
for epoch in range(1, 11):
    train_loss = train_epoch()
    val_acc = evaluate()
    
    # Non-blocking telemetry ingestion
    run.log({
        "train/loss": train_loss,
        "val/accuracy": val_acc
    }, step=epoch)

# 4. Save artifacts and finalize run
run.log_artifact("weights/best_model.pt")
run.finish()
```

---

## 8. Standalone Script Export & Model Artifacts

### Zero-Lock-In Standalone Export
Sentinel can compile any component DAG into a clean, human-readable PyTorch training script with **zero Sentinel dependencies**:

```bash
# Export standalone train.py
sentinel export script --component vision/vit-tiny@v1 --output train.py

# Verify the script runs independently
uv run python train.py
```

### Predictions & Weight Exports
```bash
# List all recorded runs
sentinel model list

# Run batch or single inference using unambiguous run ID prefix
sentinel model predict 8a2f -i "5.1,3.5,1.4,0.2"

# Export model artifacts
sentinel model export 8a2f -f onnx -o model.onnx
sentinel model export 8a2f -f torchscript -o model_script.pt
sentinel model export 8a2f -f weights -o weights.pt
```

---

## 9. HTTP REST & WebSocket API Reference

The Sentinel daemon exposes an asynchronous JSON API on port 2512:

| Endpoint | Method | Payload / Query | Description |
| :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | None | Returns supervisor status, version, and uptime |
| `/api/hardware` | `GET` | None | Real-time CPU, RAM, and per-GPU metrics (NVML) |
| `/api/jobs` | `GET` | None | List running, failed, and completed jobs |
| `/api/jobs` | `POST` | `{"command": "..."}` | Submit arbitrary command for supervision |
| `/api/jobs/run-component`| `POST` | `{"component_id": "...", "overrides": {...}}` | Launch training from component definition |
| `/api/jobs/:id/logs` | `GET` | None | Stream stdout/stderr for specific job ID |
| `/api/jobs/:id/stop` | `POST` | None | Dispatches `SIGTERM` followed by drain grace |
| `/api/components` | `GET` | None | Catalog of all available components |
| `/api/training/catalog` | `GET` | None | Catalog of available tasks, losses, optimizers |
| `/api/predict` | `POST` | `{"run_id": "...", "inputs": [...]}` | Execute stateless prediction |
| `/ws/live` | `WebSocket` | Telemetry packet | Real-time bi-directional telemetry stream |

---

## 10. Agent Decision Matrix & Troubleshooting

| Symptom / Error | Root Cause | Agent Action Required |
| :--- | :--- | :--- |
| `Address already in use (os error 48)` | Port 2512 occupied by previous daemon instance | Run `lsof -i :2512` and terminate stale process, or start with `sentinel dev --port <NEW_PORT>`. |
| `ComponentCheckError: shape mismatch` | Tensor dimension incompatibility between connected ports | Run `sentinel component show <id>` to check inferred port shapes. Reconcile projection layer dimensions. |
| `NVML library not found` | Host machine has no NVIDIA GPU or CUDA drivers | Normal behavior on Apple Silicon / CPU machines. Sentinel automatically falls back to CPU compute mode. |
| `Job killed with signal 15 (SIGTERM)` | Supervisor stopped the run or trial was pruned by ASHA | Expected behavior during early-stopping sweeps. Check Optuna study logs. |
| `Missing dependency in Python environment` | `uv` environment out of sync with project lockfile | Execute `uv sync` in the project root to restore virtualenv dependencies. |

---

## 11. Testing & Verification Suite

Before submitting pull requests or concluding agentic tasks, run the full verification matrix:

```bash
# 1. Rust workspace unit & integration tests
cargo test --workspace

# 2. Python SDK integration tests
uv run pytest packages/sentinel-sdk/tests/test_sdk.py

# 3. Python rigorous stress tests
uv run python packages/sentinel-sdk/tests/test_rigorous.py

# 4. End-to-end full lifecycle validation
bash tests/e2e_lifecycle.sh
```
