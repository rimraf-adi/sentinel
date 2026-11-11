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

### Single-Command Installation
```bash
# macOS & Linux
curl -fsSL https://raw.githubusercontent.com/rimraf-adi/sentinel/main/install.sh | bash

# Or install directly via Cargo from git (all OS, including Windows):
cargo install --git https://github.com/rimraf-adi/sentinel.git sentinel-cli
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/rimraf-adi/sentinel/main/install.ps1 | iex
```

### Cross-Compiling for Windows from macOS ARM / Linux
```bash
# 1. Install mingw-w64 (or `cargo install cargo-xwin` for MSVC targets)
brew install mingw-w64

# 2. Cross-compile target/x86_64-pc-windows-gnu/release/sentinel.exe (with embedded UI)
./scripts/build-windows.sh
```
The resulting `sentinel.exe` embeds the dashboard and links only standard Windows system DLLs, so it can be copied directly to a Windows machine without Rust or Node installed.

### Manual Build from Source (Optional)
```bash
git clone https://github.com/rimraf-adi/sentinel.git
cd sentinel
uv sync
cargo build --release -p sentinel-cli

# macOS & Linux:
cp ./target/release/sentinel /usr/local/bin/sentinel
```
```powershell
# Windows (PowerShell): copy target\release\sentinel.exe into a directory on PATH
Copy-Item .\target\release\sentinel.exe C:\sentinel\sentinel.exe
```

### Windows GPU Setup (CUDA + PyTorch)
On Windows GPU machines, verify NVIDIA driver access and install a CUDA-enabled PyTorch wheel before launching jobs:

```powershell
# 1. Confirm NVIDIA driver / NVML visibility
nvidia-smi

# 2. Create a Python 3.10+ virtual environment with CUDA PyTorch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# (Optional) If running Sentinel without activating .venv:
$env:SENTINEL_PYTHON = "$PWD\.venv\Scripts\python.exe"

# 3. Verify Python, CUDA availability, and GPU detection
sentinel doctor
sentinel gpus
```

### Starting & Managing the Daemon
The daemon runs the background supervisor, GPU scheduler, API, and embedded web UI (default bind: `127.0.0.1:2512`):

```bash
# Start in foreground
sentinel dev

# Start on custom port in background (macOS & Linux)
sentinel dev --port 2512 > /dev/null 2>&1 &

# Scaffold new Sentinel workspace in current folder
sentinel init

# Scaffold a new named project
sentinel new my-ml-project
```

### Remote VPS Access (Tailscale / SSH Tunnel)
Sentinel's API can start jobs and has no built-in authentication, so keep it bound to `127.0.0.1` and expose it privately rather than opening `--host 0.0.0.0` or `tailscale funnel` to the public internet:

```powershell
# Recommended on a remote Windows GPU VPS: Tailscale HTTPS proxy (private to your tailnet)
sentinel dev
tailscale serve --bg http://127.0.0.1:2512
tailscale serve status
```

```bash
# Or forward port 2512 over SSH from macOS/Linux
ssh -L 2512:127.0.0.1:2512 <user>@<vps-ip>
```

---

## 4. Components & Templates Guide (3-Tier Architecture System)

Sentinel decouples neural-network **Blocks** (reusable layers and subgraphs under `builtin/blocks/`), **Components** (full runnable graph + training specs), and **Templates** (shareable cross-project component blueprints) across a 3-tier precedence hierarchy (`crates/sentinel-core/src/components.rs`):

| Tier | Origin | Location | Mutability | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | `builtin` | Embedded in binary (`crates/sentinel-core/builtin/components/` & `builtin/blocks/`) | Immutable (cannot be shadowed) | Built-in baselines across tabular, sequence, vision, generative, and graph domains |
| **Tier 2** | `user` | `~/.sentinel/templates/` (or `$SENTINEL_TEMPLATES_DIR`) | Mutable | User-global templates shared across all projects |
| **Tier 3** | `project` | `./components/` | Mutable (overrides `user` tier) | Project-local components, customizations, and frozen runs |

### Component Commands (`sentinel component`)
```bash
# List all components across built-in, user, and project tiers
sentinel component list

# Inspect execution order, inferred tensor shapes on every port, parameter count, and MACs
sentinel component show vision/vit-tiny@v1
sentinel component show tabular/residual-mlp@v1 --set hidden=64 --set classes=5

# Scaffold a new project component YAML in ./components/ (optionally extending an existing ID)
sentinel component new my-mlp --domain tabular
sentinel component new tuned-mlp --domain tabular --extends tabular/residual-mlp@v1

# Validate schema, port shapes, parameter ranges, and acyclic compound expansion
sentinel component check components/tabular/my-mlp.yaml

# Freeze an existing trained run into a reusable component
sentinel component save-run <RUN_ID> my-saved-model

# Compile a graph component directly to standalone PyTorch code (or model class only)
sentinel component compose tabular/residual-mlp@v1 --output train.py --set lr=0.001
sentinel component compose vision/vit-tiny@v1 --output model.py --model-only

# List all fundamental + compound blocks, or training catalog options
sentinel component blocks
sentinel component training --task classification
```

### Template Commands (`sentinel template`, `add`, `push`)
Templates reuse the exact component YAML format in Tier 2 (`~/.sentinel/templates/`) or install from Sentinel Hub into `./components/`:

```bash
# Print the user-global template directory (~/.sentinel/templates or $SENTINEL_TEMPLATES_DIR)
sentinel template dir

# List user-global templates
sentinel template list

# Install a validated component YAML into Tier 2 (--force overwrites an existing user template; built-ins cannot be shadowed)
sentinel template add components/tabular/my-mlp.yaml
sentinel template add components/tabular/my-mlp.yaml --force

# Remove a user-global template by component_id
sentinel template remove tabular/my-mlp@v1

# Install a module from Sentinel Hub (with built-in offline fallback) into ./components/ or ~/.sentinel/templates/
sentinel add vit-tiny
sentinel add vit-tiny --global

# Publish a validated component YAML to sentinel-hub
sentinel push components/tabular/my-mlp.yaml
sentinel publish components/tabular/my-mlp.yaml
```

### Component Validation Rules
Run `sentinel component check <path>` before committing or launching jobs. Validation verifies:
- `schema_version` (`"1.0"`), `component_id`, `name`, `domain`, and typed `parameters` (`int`, `float`, `choice`, `string`).
- Tensor shape propagation matches across every input, node, and output port.
- Parameter overrides stay within declared `range` bounds (`min` $\le$ `val` $\le$ `max`) or `options`.
- No circular graph edges or self-referential compound block definitions exist.

---

## 5. Declarative Block & Component Specification (YAML)

### Compound Block Schema (`crates/sentinel-core/builtin/blocks/<family>/<name>.yaml`)
Compound blocks define reusable subgraphs with typed parameters and dynamic tensor port shapes:

```yaml
compound: residual_mlp_block
family: core
description: "An MLP block with a skip connection: out = in + project(mlp(in)). Built from the compound mlp_block"
params:
  hidden: {default: 64, help: "Width inside the block (the skip keeps the input width)"}
  width:  {help: "Input and output features (must equal the input's last dimension)"}
  dropout: {default: 0.0, help: "Dropout probability"}
graph:
  inputs:
    in: [B, "${width}"]             # Dynamic tensor dimension interpolation
  nodes:
    mlp:  {block: mlp_block, params: {width: "${hidden}", dropout: "${dropout}"}}
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

### Full Model Component / Template Schema (`./components/<name>.yaml` or `~/.sentinel/templates/<domain>/<slug>.yaml`)
Combines parameterized graph wiring with the `graph.training` recipe (supports `extends:` + `locked:` for inheritance):

```yaml
schema_version: "1.0"
component_id: "tabular/residual-mlp@v1"
name: "Residual MLP (Skip Connection)"
description: "MLP with a skip connection (add block) - the smallest real DAG"
domain: "tabular"

parameters:
  features:     {type: int, default: 2, range: [1, 4096], help: "Input feature count"}
  classes:      {type: int, default: 3, range: [1, 1000], help: "Output size: classes or 1 for regression"}
  hidden:       {type: int, default: 32, range: [2, 1024], help: "Hidden width"}
  lr:           {type: float, default: 0.01, range: [0.00001, 0.5], help: "Learning rate"}
  epochs:       {type: int, default: 200, range: [1, 5000], help: "Training epochs"}
  csv:          {type: string, default: "data/demo.csv", help: "Training CSV"}
  feature_cols: {type: string, default: "x1,x2", help: "Comma-separated numeric feature columns"}
  loss:         {type: choice, default: "cross_entropy", options: ["cross_entropy", "mse", "bce_logits"], help: "Loss function"}
  target:       {type: string, default: "label", help: "Target column"}

graph:
  inputs: {x: [B, "${features}"]}
  nodes:
    embed: {block: linear, params: {out: "${hidden}"}}
    fc:    {block: linear, params: {out: "${hidden}"}}
    act:   {block: activation, params: {kind: relu}}
    skip:  {block: add}
    head:  {block: linear, params: {out: "${classes}"}}
  edges:
    - "x -> embed"
    - "embed -> fc"
    - "fc -> act"
    - "act -> skip.a"
    - "embed -> skip.b"
    - "skip -> head"
  outputs: {logits: head}
  training:
    loss: "${loss}"
    optimizer: adam
    lr: "${lr}"
    epochs: "${epochs}"
    data: {csv: "${csv}", features: "${feature_cols}", target: "${target}"}
```

---

## 6. Execution, Sweeps & Telemetry

### Running Experiments
```bash
# Run a component by ID or YAML path
sentinel run tabular/residual-mlp@v1

# Run with parameter overrides (--set KEY=VALUE, repeatable)
sentinel run tabular/residual-mlp@v1 --set lr=0.0005 --set epochs=50 --set csv=data/train.csv

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
