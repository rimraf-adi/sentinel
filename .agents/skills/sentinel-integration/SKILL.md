---
name: sentinel-integration
description: >-
  Operational guide and skill for autonomous agents to hook into Sentinel-powered
  machine learning codebases. Covers environment detection, component lifecycle,
  job supervision, REST/WebSocket APIs, crash-immune logging with sentinel-sdk,
  and Optuna ASHA hyperparameter sweeps.
---

# Sentinel Integration Skill

This skill provides autonomous AI coding agents with the instructions, tools, and protocols required to inspect, configure, run, and supervise machine learning workflows in Sentinel-powered repositories.

## Core Tenets

1. **"Rust is boss. Python is worker."**
   - Rust manages process supervision, dynamic GPU allocation, hardware telemetry, and the web interface.
   - Python executes the deep learning models using standard tools (PyTorch, torchvision, HuggingFace, Optuna).
2. **Strict Environment Rule (`uv`):**
   - All Python commands must be invoked using `uv run python ...`.
   - Never use global `pip` or unmanaged `python` that could corrupt the environment.
3. **Reproducibility via Components:**
   - Always prefer defining or extending declarative components in `components/` over ad-hoc scripts.
4. **Zero Lock-In:**
   - Any component can be converted into pure bare PyTorch code at any time with `sentinel export script`.

---

## Agent Detection & Initialization

When an agent enters a workspace, execute the following detection routine:

```bash
# Check if Sentinel is present
if [ -f "sentinel.yaml" ] || [ -d "components" ]; then
    echo "Sentinel codebase detected."
fi

# Ensure local daemon is running (default port: 2512)
curl -s http://localhost:2512/api/health || {
    echo "Starting Sentinel daemon..."
    target/debug/sentinel dev &
}
```

---

## Common Agent Workflows

### 1. Inspecting & Validating Components

Before starting a run, check the component catalog and validate parameter ranges:

```bash
# List available components (shows [built-in], [user template], and [project] badges)
sentinel component list

# Validate component parameters and locked fields
sentinel component check components/vision/image-classification.yaml
```

### 1b. Managing Custom Templates (3-Tier Discovery)

Sentinel discovers templates through a 3-tier hierarchy:
1. **Built-in:** Compiled into binary (immutable; cannot be shadowed).
2. **User-Global (`~/.sentinel/templates/`):** Shared across all projects on the machine.
3. **Project-Local (`components/`):** Workspace-specific components (can override user templates).

```bash
# View or manage user-global templates
sentinel template dir
sentinel template list
sentinel template add path/to/template.yaml [--force]
sentinel template remove <component-id>
```

### 2. Launching an Experiment

To launch a training run, call the Axum REST API:

```bash
curl -X POST http://localhost:2512/api/jobs/run-component \
  -H "Content-Type: application/json" \
  -d '{
    "component_id": "vision/image-classification@v1",
    "overrides": {
      "training.lr": 0.0003,
      "training.batch_size": 64,
      "training.epochs": 10
    }
  }'
```

### 3. Monitoring & Tailing Logs

Query active jobs and retrieve execution logs:

```bash
# List jobs
curl -s http://localhost:2512/api/jobs | jq .

# Tail logs for a specific job
curl -s http://localhost:2512/api/jobs/<JOB_ID>/logs | jq -r '.logs[]'
```

### 4. Stopping an Experiment

To abort a diverging or failing training run:

```bash
curl -X POST http://localhost:2512/api/jobs/<JOB_ID>/stop
```

### 5. Running Hyperparameter Sweeps (Optuna ASHA)

When asked to optimize hyperparameters:

```bash
uv run python recipes/sweep_orchestrator.py --trials 15 --max-epochs 10
```

### 6. Exporting to Standalone Pure PyTorch

If the user wants code that runs without Sentinel installed:

```bash
sentinel export script --component vision/image-classification@v1 > standalone_train.py
uv run python standalone_train.py
```

---

## Python Telemetry Instrumentation (`sentinel-sdk`)

When writing new Python models, always instrument them with `sentinel`:

```python
import sentinel

run = sentinel.init(experiment_name="experiment-name", run_name="run-name")
run.log_params({"lr": 0.001, "batch_size": 64})

for epoch in range(1, epochs + 1):
    loss = train_one_epoch(...)
    # Safe logging: will never crash training even if daemon disconnects
    run.log({"loss": loss}, step=epoch)

# All experimental plots and results are isolated in that run directory by default:
sentinel.save_plot("confusion_matrix.png", fig)
sentinel.save_results({"val_acc": 0.942, "f1": 0.938}, "metrics.json")
# Even plain `plt.savefig("loss.png")` is automatically redirected to `<run_dir>/artifacts/plots/`

run.finish()
```

### Run Directory Isolation Policy (Default Setting)
- **Zero Root Pollution**: All experimental plots (`artifacts/plots/`), evaluation summaries (`artifacts/results/`), checkpoints (`artifacts/checkpoints/`), and telemetry are stored exclusively inside the active run directory (`mlruns/<experiment_id>/<run_id>/artifacts/`).
- **Automatic Plot Redirection**: When a Sentinel run is active, relative `plt.savefig(...)` calls are automatically intercepted and routed to the run's `artifacts/plots/` folder.
- **Environment Context**: Subprocesses and tools receive `SENTINEL_RUN_DIR`, `SENTINEL_ARTIFACT_DIR`, `SENTINEL_PLOTS_DIR`, and `SENTINEL_RESULTS_DIR`.

---

## Agent Verification Protocol

Before finishing a task:
1. Run `cargo test --workspace` to ensure no Rust regressions.
2. Run `uv run pytest packages/sentinel-sdk/tests/` to verify Python telemetry.
3. Check `curl -s http://localhost:2512/api/hardware` to verify all GPU slots have been released.
4. Ensure git commit messages are simple and descriptive.
