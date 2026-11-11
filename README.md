# Sentinel

Training supervisor and ML operations plane for coding agents and ML teams. One small Rust binary: templates to start from, modular blocks to build with, and basic observability out of the box.

Training code stays plain PyTorch. Models are YAML graphs that export to a standalone `train.py` with no Sentinel dependency.

## Highlights

* **Templates.** Start from shipped models (ViT, U-Net, ResNet, VAE, GAN, residual MLPs) or save your own, at built-in, user, or project level.
* **Modular.** Models are graphs of reusable blocks, and blocks can contain other blocks. Loss, optimizer, and scheduler are independent settings.
* **Basic observability.** Live metrics, job logs, and CPU/RAM/GPU stats in a dashboard, stored in MLflow-compatible format.
* **No lock-in.** Runs are ordinary Python. No base classes, no wrappers.
* **Validated before running.** Shape, bounds, and loss/task errors are caught up front and name the offending block.
* **Lightweight and agent-first.** About 15 MB RAM at idle. Every action is a CLI command or HTTP call.

## Quick start

Install with a single command:

```bash
# macOS & Linux
curl -fsSL https://raw.githubusercontent.com/rimraf-adi/sentinel/main/install.sh | bash
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/rimraf-adi/sentinel/main/install.ps1 | iex
```

Or install via Cargo directly:

```bash
cargo install --git https://github.com/rimraf-adi/sentinel.git sentinel-cli
```

Cross-compile `sentinel.exe` for Windows (`x86_64-pc-windows-gnu` or `msvc`) from macOS/Linux:

```bash
brew install mingw-w64   # macOS ARM/Intel
./scripts/build-windows.sh
```

Once installed, launch the control plane & dashboard:

```bash
sentinel dev # dashboard active at http://localhost:2512
```

Use `--port` to change the port. Run `sentinel doctor` to verify your Python and PyTorch setup.

### Windows GPU & Remote VPS Setup

1. **Install Python 3.10+ & CUDA PyTorch** (matching your NVIDIA driver):

   ```powershell
   nvidia-smi                                 # verify GPU driver works
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

   *(If you run `sentinel` without activating `.venv`, point it to that interpreter: `$env:SENTINEL_PYTHON = "$PWD\.venv\Scripts\python.exe"`.)*

2. **Initialize & verify CUDA telemetry:**

   ```powershell
   sentinel init
   sentinel doctor
   sentinel gpus
   ```

3. **Access the dashboard safely from another machine:**

   Sentinel binds `127.0.0.1:2512` by default because its API can spawn jobs and has no built-in auth. Expose it privately over **Tailscale** or an **SSH tunnel**:

   ```powershell
   # Tailscale HTTPS proxy (recommended on a remote Windows GPU VPS; private to your tailnet)
   sentinel dev
   tailscale serve --bg http://127.0.0.1:2512
   ```

   ```bash
   # Or SSH local port forward from macOS/Linux
   ssh -L 2512:127.0.0.1:2512 <user>@<vps-ip>
   ```

## Architecture

| Plane | Language | Responsibilities |
| :--- | :--- | :--- |
| **Control** | Rust | Process supervision, GPU allocation (NVML), telemetry streaming, component resolution, dashboard, REST/WebSocket API |
| **Execution** | Python | Training loops and sweeps (PyTorch, Hugging Face, Optuna), launched with `uv run`, logging through `sentinel-sdk` |

The daemon assigns GPUs through `CUDA_VISIBLE_DEVICES` and falls back to CPU when none are available. Runs are written to `mlruns/`.

## Templates

Models and recipes are discovered across three tiers:

| Tier | Location | Behavior |
| :--- | :--- | :--- |
| **Built-in** | Embedded in the binary | Immutable |
| **User** | `~/.sentinel/templates/` or `$SENTINEL_TEMPLATES_DIR` | Shared across projects |
| **Project** | `./components/` | Local. Overrides user templates. |

```bash
sentinel component list                               # all tiers
sentinel template add path/to/model.yaml              # install a user template
sentinel template list | remove | dir
sentinel component save-run <ID> --name <NAME>        # freeze a run into a component
```

## Modular blocks

| Tier | Description | Location |
| :--- | :--- | :--- |
| **Fundamental** | A single PyTorch layer (72 available) | Rust catalog |
| **Compound** | A reusable subgraph with its own ports and parameters | `blocks/<family>/<name>.yaml` |
| **Model** | A full graph plus a training recipe | `components/` |

```yaml
compound: residual_mlp_block
params:
  hidden: {default: 64}
  width: {} # required
graph:
  inputs: {in: [B, "${width}"]}
  nodes:
    mlp: {block: mlp_block, params: {width: "${hidden}"}}
    proj: {block: linear, params: {out: "${width}"}}
    skip: {block: add}
  edges: ["in -> mlp", "mlp -> proj", "proj -> skip.a", "in -> skip.b"]
  outputs: {out: skip}
```

* Input shapes are inferred from wiring, so one definition works at any size.
* Compounds compile to nested PyTorch classes and read like hand-written modules.
* Errors name every compound on the path. Recursive definitions are rejected.

```bash
sentinel component blocks                   # list all blocks and parameters
sentinel component show vision/vit-tiny@v1  # block tree and model statistics
```

Every model reports parameters, memory, MACs per sample, and nesting depth before training. These counts are tested against PyTorch.

## Training configuration

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

Tasks: `classification`, `binary`, `multilabel`, `regression`, `ranking`, `reconstruction`, `vae`, `custom`. The task is inferred from the loss unless set explicitly. Custom losses accept inline source or a project file. Run `sentinel component training` for the full list of losses, optimizers, and strategies.

Hyperparameter sweeps use Optuna ASHA (`sentinel sweep <config>`). Underperforming trials are stopped early and their GPUs are reassigned.

## Observability

* **Metrics:** step-indexed, logged with the SDK, shown live in the dashboard.
* **Logs:** stdout/stderr per job.
* **Hardware:** CPU, RAM, and per-GPU utilization, temperature, and VRAM.
* **Stream:** live telemetry at `ws://localhost:2512/ws/live`.
* **Storage:** MLflow 2.x/3.x layout in `mlruns/`.
* **Model view:** DAG visualizer with parameter and compute counts.

## Python SDK

```python
import sentinel

run = sentinel.init(experiment_name="cifar10-benchmark", run_name="resnet18-baseline")
run.log_params({"model": "resnet18", "learning_rate": 3e-4, "batch_size": 64})

for epoch in range(1, 11):
    run.log({"train/loss": 2.5 / epoch, "val/accuracy": 0.70 + 0.02 * epoch}, step=epoch)

run.finish()
```

SDK calls never raise into your training loop. Disk errors or a stopped supervisor are absorbed.

## Export and prediction

Each run saves `model.pt`, `model_script.pt` (TorchScript), `model.onnx` (best effort), `meta.json`, and the exact `train.py`.

```bash
sentinel model list
sentinel model predict <run> -i "5.0,3.4,1.5,0.2"
sentinel model export <run> -f onnx -o model.onnx  # torchscript | onnx | weights | meta | script
sentinel model generate <gan-run> -c 8
sentinel export script --component <id> --output train.py
```

Run IDs accept any unambiguous prefix. Predictions run in a short-lived Python process, so the daemon never loads PyTorch.

## Datasets and remote nodes

Datasets mount from any local path or network share via `datasets.yaml`, and resolve in code with `sentinel.get_dataset_path("name")`. The dashboard reports size, file count, and mount type.

Remote nodes run `sentinel-agent`, which connects outbound over WebSocket. No inbound ports are required. Events buffer to `.sentinel/agent_buffer.jsonl` during outages and replay on reconnect.

## CLI

| Command | Purpose |
| :--- | :--- |
| `dev [--port N]` | Start the daemon and dashboard |
| `new <NAME> / init` | Scaffold a project / adopt an existing directory |
| `run / sweep <CONFIG>` | Run a job / run a hyperparameter sweep |
| `component …` | `list`, `show`, `blocks`, `training`, `new`, `check`, `save-run` |
| `template …` | `list`, `add`, `remove`, `dir` |
| `model …` | `list`, `predict`, `export`, `generate` |
| `export script` | Export a standalone PyTorch script |
| `doctor` | Check the Python/PyTorch environment |

## HTTP API

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | GET | Health check |
| `/api/jobs` | GET, POST | List jobs, submit a command |
| `/api/jobs/run-component` | POST | Launch from `{component_id, overrides}` |
| `/api/jobs/:id/logs` | GET | Job output |
| `/api/jobs/:id/stop` | POST | Send SIGTERM |
| `/api/hardware` | GET | CPU, RAM, GPU snapshot |
| `/api/components` | GET | Component catalog |
| `/api/training/catalog` | GET | Tasks, losses, optimizers, strategies |
| `/api/tracking/runs/:id/artifacts` | GET | Run artifacts |
| `/api/predict` | POST | `{run_id, inputs}` |
| `/ws/live` | WebSocket | Live telemetry |

## Footprint

Measured with `tests/bench_footprint.py` on a release build (Apple M1, macOS), excluding training.

| Scenario | RSS | CPU (one core) |
| :--- | :--- | :--- |
| **Idle** | 15 MB | 0.16% |
| **Dashboard open** | 15 MB | 0.74% |
| **Watching one running job** | 18 MB | 1.5% |
| **~100 API requests/s** | 16 MB | 2.7% |

The binary is 7 MB with the dashboard embedded.

## Repository layout

```
crates/       sentinel-core, sentinel-daemon, sentinel-agent, sentinel-cli
ui/           React dashboard (embedded in the binary)
blocks/       compound block library
components/   shipped models and recipes
packages/     sentinel-sdk (Python)
recipes/      training loop, sweep orchestrator
tests/        end-to-end and benchmark scripts
guide.md      operating guide for agents
SENTINEL.md   operational manual and specification for agents
```

## Testing

```bash
cargo test --workspace
uv run pytest packages/sentinel-sdk/tests/test_sdk.py
uv run python packages/sentinel-sdk/tests/test_rigorous.py
bash tests/e2e_lifecycle.sh
```

## License

Dual-licensed under Apache 2.0 or MIT, at your option.
