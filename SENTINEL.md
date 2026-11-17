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

## 3. Installation, Upgrades & Daemon Lifecycle

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

### Updating Sentinel to the Latest Version
Keep Sentinel up to date with the latest features, block catalogs, and bug fixes:

```bash
# macOS & Linux (re-run installer to overwrite binary with the latest release)
curl -fsSL https://raw.githubusercontent.com/rimraf-adi/sentinel/main/install.sh | bash

# Cargo (force reinstallation of latest main branch)
cargo install --force --git https://github.com/rimraf-adi/sentinel.git sentinel-cli
```

```powershell
# Windows (PowerShell - re-run installer to fetch latest release)
irm https://raw.githubusercontent.com/rimraf-adi/sentinel/main/install.ps1 | iex
```

#### Updating Python Dependencies & SDK in a Project
```bash
# Update Python virtual environment & lockfile to latest packages
uv sync --upgrade

# Update or reinstall sentinel-sdk
uv add sentinel-sdk
```

#### Verifying Version & Diagnostics
```bash
# Confirm installed CLI version
sentinel --version

# Run full health check of Rust runtime, Python, CUDA, and GPU hardware
sentinel doctor
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

---

## 4. Repository Structure & Agent Directory Conventions

> [!IMPORTANT]
> **Strict Agent Invariant (Anti-Deviation Rule):**  
> Autonomous coding agents must respect Sentinel's canonical repository layout. **Do NOT deviate, invent custom top-level folder structures (e.g. `src/`, `trainers/`, `custom_framework/`), or create proprietary runner scripts.** Everything in Sentinel belongs in a designated canonical location.

### Canonical Repository Layout
When a repository is initialized (`sentinel init` or `sentinel new`), Sentinel sets up the following structure:

```
my-sentinel-project/
├── sentinel.yaml            # Project configuration (framework, tracking, runner defaults) - DO NOT delete
├── sentinel.lock            # Machine-managed lockfile (ownership, versions, hashes) - DO NOT hand-edit
├── machines.yaml            # Hardware execution targets (local workstation + remote Tailscale nodes)
├── SENTINEL.md              # Operational guide for coding agents & developers (this file)
├── pyproject.toml           # uv-managed Python project definition & dependencies
├── uv.lock                  # Deterministic Python dependency lockfile
├── .gitignore               # Ignores .venv/, mlruns/, .sentinel/, __pycache__/, large weights
│
├── configs/                 # Hierarchical YAML configurations (Hydra-compatible)
│   ├── base.yaml            # Master experiment config (composes model, dataset, hyperparameters)
│   ├── model/               # Model-specific configs (e.g. resnet18.yaml, vit.yaml)
│   ├── dataset/             # Dataset-specific configs (e.g. synthetic.yaml, cifar10.yaml)
│   └── experiment/          # Experiment recipes (e.g. baseline.yaml, ablation_lr.yaml)
│
├── experiments/             # Isolated directories per experiment ("One experiment = One directory")
│   └── <experiment_name>/   # E.g. experiments/baseline/, experiments/exp01_resnet/
│       ├── config.yaml      # Snapshot of exact config used for the run
│       ├── results/         # Evaluation metrics, predictions, summary.json, metrics.json
│       ├── plots/           # Loss curves, ROC curves, confusion matrices, figures
│       ├── checkpoints/     # Trained model weights (weights.pt, model.onnx)
│       └── logs/            # Execution logs, stdout/stderr, telemetry
│
├── models/                  # Pure PyTorch neural network architectures
│   └── example_model.py     # Standard nn.Module implementations (no proprietary wrappers)
│
├── datasets/                # Dataset loaders, feature engineering, and CSV data files
│
├── components/              # Project-local declarative components (Tier 3)
│   └── custom/              # Custom YAML component definitions and training DAGs
│
├── hooks/                   # Lifecycle hooks for training loops (early stopping, custom metrics)
│
├── tests/                   # Unit, integration, and shape validation test suites
│
└── mlruns/                  # MLflow experiment tracking store (auto-generated, gitignored)
```

### Purpose & Conventions by Directory

| Directory / File | Ownership | Content & Agent Rules |
| :--- | :--- | :--- |
| `sentinel.yaml` | Human / Managed | Project settings, framework selection (`pytorch`), tracking backend (`mlflow`), precision. Keep clean. |
| `sentinel.lock` | Machine Managed | Generated by `sentinel init`/`scaffold`. Contains schema version and file manifest. Never delete. |
| `machines.yaml` | Managed | Definitions of local and remote GPU servers (Tailscale nodes, SSH boxes). No secrets or passwords. |
| `SENTINEL.md` | Managed | Operational manual and agent instructions. Placed in every initialized repo. Follow strictly. |
| `configs/` | Developer / Agent | All hyperparameter configs belong here (`configs/model/`, `configs/dataset/`, `configs/experiment/`). |
| `experiments/` | Machine / Agent | Every experiment has its own directory. All outputs, results, plots, and checkpoints stay inside. |
| `models/` | Developer / Agent | Pure PyTorch models inheriting standard `torch.nn.Module`. No custom runner boilerplate. |
| `datasets/` | Developer / Agent | Data files (CSVs), dataset loaders, and preprocessing scripts. |
| `components/custom/`| Developer / Agent | Declarative component recipes (`Tier 3`). Always validate with `sentinel component check`. |
| `hooks/` | Developer / Agent | Python hook scripts called by the training engine (e.g. validation callbacks). |
| `tests/` | Developer / Agent | Automated tests. Agents must ensure tests pass before completing tasks. |
| `mlruns/` | Machine Generated | MLflow tracking artifacts, run directories, and metric stores. Never check into git. |

### Strict "Configs & Experiments" Architectural Invariant
1. **One Experiment, One Directory:**  
   Every single training run or experiment trial MUST write strictly to its own dedicated subfolder under `experiments/<experiment_name>/` (or the SDK run directory `mlruns/<experiment_id>/<run_id>/`). Never dump outputs into the project root or mix outputs from different experiments.
2. **Standard Experiment Directory Structure:**  
   Each experiment folder MUST adhere to this internal layout:
   - `config.yaml`: Frozen snapshot of the exact hyperparameters and dataset specs used.
   - `results/`: Evaluation metrics (`metrics.json`), test scores, evaluation predictions, and summary statistics.
   - `plots/`: All figures, loss/accuracy curves, ROC curves, and confusion matrices.
   - `checkpoints/`: Model weights (`weights.pt`), TorchScript exports, and ONNX models.
   - `logs/`: Execution logs and supervisor telemetry.
3. **ZERO STRAY RESULT FILES (Strict Policy):**  
   - **No loose output files anywhere in root:** Never save `.png`, `.jpg`, `.pt`, `.onnx`, `.json`, `.csv`, or `.log` files in the repository root, `configs/`, or `models/`.
   - Sentinel's Python SDK automatically intercepts `plt.savefig()` and redirects output to `<run_dir>/artifacts/plots/` to prevent root directory pollution.
   - If running custom scripts, the agent must explicitly point all outputs to `experiments/<experiment_name>/` and verify after execution that the root directory remains pristine.

### Strict Agent Invariants (How to Avoid Deviating)
1. **Never invent custom runner frameworks or bash launchers:**  
   Sentinel's Rust daemon (`sentinel dev`) supervises execution, tracks PIDs, and monitors GPU resources. Agents do not need to create background process scripts or custom daemon wrappers.
2. **Never wrap PyTorch in proprietary or framework-specific base classes:**  
   Code in `models/` must be vanilla PyTorch (`nn.Module`). Anyone should be able to run it with standard PyTorch.
3. **Always use `uv run` for executing Python scripts:**  
   Never run `python ...` or `pip install ...` against global Python. Always run through `uv run`.
4. **Validate components before execution:**  
   Before running a component YAML, always execute `sentinel component check <path>`. Fix tensor shape mismatches and parameter range violations before launching.
5. **Always isolate all results inside `experiments/`:**  
   Ensure every result, plot, and model checkpoint is cleanly organized under `experiments/<experiment_name>/`. Zero stray files.

---

## 5. Remote Operations & Distributed Execution with Tailscale

Sentinel is built for remote GPU environments (cloud VPS, Lambda Labs, RunPod, home server, Windows GPU workstation). Because Sentinel's REST API can spawn jobs and has no built-in password authentication, **never expose port 2512 directly to `0.0.0.0` or public internet**. Use **Tailscale** for private, encrypted zero-config networking.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PRIVATE TAILSCALE MESH (Tailnet)                │
│                                                                        │
│   Local Machine (Laptop/Agent)             Remote GPU VPS / Workstation│
│   100.x.y.1                                100.x.y.2 (gpu-box)         │
│                                                                        │
│   ┌─────────────────────┐                  ┌─────────────────────────┐ │
│   │ Sentinel CLI / Web  │                  │ Sentinel Daemon (:2512) │ │
│   │ tailscale ssh       ├─────────────────►│ tailscale serve (HTTPS) │ │
│   │ machines.yaml entry │  Encrypted Mesh  │ PyTorch / NVIDIA NVML   │ │
│   └─────────────────────┘                  └─────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### Setting Up Sentinel with Tailscale on Remote VPS

#### Step 1: Install & Connect Tailscale on Remote Machine
```bash
# Linux / macOS:
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Windows:
# Install from https://tailscale.com/download/windows, then run:
tailscale up
```

#### Step 2: Start Sentinel Daemon on Remote Machine
Start the Sentinel daemon bound safely to `127.0.0.1:2512`:
```bash
# Foreground:
sentinel dev

# Or background on Linux/macOS:
sentinel dev --port 2512 > /dev/null 2>&1 &
```

#### Step 3: Expose Privately Over Tailnet with `tailscale serve`
`tailscale serve` creates an authenticated, HTTPS-encrypted reverse proxy to port 2512 accessible only to devices on your Tailnet:

```bash
# Expose daemon privately on your Tailnet via HTTPS
tailscale serve --bg http://127.0.0.1:2512

# Check status (provides your MagicDNS URL, e.g. https://gpu-box.tailnet.ts.net)
tailscale serve status
```

> [!CAUTION]
> **NEVER USE `tailscale funnel`:**  
> `tailscale serve` restricts access strictly to authorized machines inside your private Tailnet. `tailscale funnel` opens traffic to the entire public internet, which would expose Sentinel's unauthenticated execution API to the world.

### Connecting Local Sentinel Commands to Tailscale Machines

#### Method A: Configuring Remote Tailscale Node in `machines.yaml`
Add the remote Tailscale node to `machines.yaml` in your project root using either its Tailscale 100.x.y.z IP or MagicDNS hostname:

```yaml
# machines.yaml
machines:
  - id: "local"
    name: "Local Workstation"
    host: "127.0.0.1"
    port: 2512
    is_local: true
    gpus: "auto"

  - id: "remote-a100"
    name: "Cloud A100 VPS via Tailscale"
    host: "gpu-box.tailnet.ts.net" # or Tailscale IP: "100.x.y.z"
    user: "ubuntu"
    port: 22
    is_local: false
    gpus: "auto"
```

Dispatch jobs directly to the remote Tailscale node:
```bash
sentinel run tabular/residual-mlp@v1 --machine remote-a100
```

#### Method B: SSH Tunneling over Tailscale
Forward the remote daemon's port 2512 to your local machine:

```bash
# Using standard SSH over Tailscale:
ssh -L 2512:127.0.0.1:2512 ubuntu@gpu-box.tailnet.ts.net

# Or using native Tailscale SSH:
tailscale ssh -L 2512:127.0.0.1:2512 ubuntu@gpu-box
```
Once forwarded, local CLI commands (`sentinel doctor`, `sentinel component list`, `sentinel run`) and the web dashboard (`http://localhost:2512`) interact directly with the remote GPU machine.

#### Method C: Querying Remote API Directly Over Tailnet
Because `tailscale serve` provides HTTPS with a valid certificate, you can query remote health and telemetry directly:

```bash
# Health query
curl -s https://gpu-box.tailnet.ts.net/api/health | jq .

# Hardware & GPU telemetry (query remote VRAM and GPU utilization)
curl -s https://gpu-box.tailnet.ts.net/api/hardware | jq .

# Submit training job to remote machine
curl -X POST https://gpu-box.tailnet.ts.net/api/jobs/run-component \
  -H "Content-Type: application/json" \
  -d '{
    "component_id": "vision/resnet-mini@v1",
    "overrides": {"lr": 0.001, "epochs": 20}
  }'
```

#### Teardown / Resetting Tailscale Serve
When finished with the remote session:
```bash
tailscale serve --reset
```

---

## 6. Components & Templates Guide (3-Tier Architecture System)

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

## 7. Declarative Block & Component Specification (YAML)

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

## 8. Execution, Sweeps & Telemetry

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

## 9. Python SDK Protocol (`sentinel-sdk`)

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

## 10. Standalone Script Export & Model Artifacts

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

## 11. HTTP REST & WebSocket API Reference

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

## 12. Agent Decision Matrix & Troubleshooting

| Symptom / Error | Root Cause | Agent Action Required |
| :--- | :--- | :--- |
| `Address already in use (os error 48)` | Port 2512 occupied by previous daemon instance | Run `lsof -i :2512` and terminate stale process, or start with `sentinel dev --port <NEW_PORT>`. |
| `ComponentCheckError: shape mismatch` | Tensor dimension incompatibility between connected ports | Run `sentinel component show <id>` to check inferred port shapes. Reconcile projection layer dimensions. |
| `NVML library not found` | Host machine has no NVIDIA GPU or CUDA drivers | Normal behavior on Apple Silicon / CPU machines. Sentinel automatically falls back to CPU compute mode. |
| `Job killed with signal 15 (SIGTERM)` | Supervisor stopped the run or trial was pruned by ASHA | Expected behavior during early-stopping sweeps. Check Optuna study logs. |
| `Missing dependency in Python environment` | `uv` environment out of sync with project lockfile | Execute `uv sync` in the project root to restore virtualenv dependencies. |

---

## 13. Testing & Verification Suite

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
