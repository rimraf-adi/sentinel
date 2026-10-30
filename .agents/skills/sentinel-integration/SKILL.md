---
name: sentinel-integration
description: >-
  Operational guide and skill for autonomous agents to hook into Sentinel-powered
  machine learning codebases. Covers environment detection, template lifecycle,
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
3. **Reproducibility via Templates:**
   - Always prefer defining or extending declarative templates in `templates/` over ad-hoc scripts.
4. **Zero Lock-In:**
   - Any template can be converted into pure bare PyTorch code at any time with `sentinel export script`.

---

## Agent Detection & Initialization

When an agent enters a workspace, execute the following detection routine:

```bash
# Check if Sentinel is present
if [ -f "sentinel.yaml" ] || [ -d "templates" ]; then
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

### 1. Inspecting & Validating Templates

Before starting a run, check the template catalog and validate parameter ranges:

```bash
# List available templates
sentinel template list

# Validate template parameters and locked fields
sentinel template check templates/vision/image-classification.yaml
```

### 2. Launching an Experiment

To launch a training run, call the Axum REST API:

```bash
curl -X POST http://localhost:2512/api/jobs/run-template \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "vision/image-classification@v1",
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
sentinel export script --template vision/image-classification@v1 > standalone_train.py
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

run.finish()
```

---

## Agent Verification Protocol

Before finishing a task:
1. Run `cargo test --workspace` to ensure no Rust regressions.
2. Run `uv run pytest packages/sentinel-sdk/tests/` to verify Python telemetry.
3. Check `curl -s http://localhost:2512/api/hardware` to verify all GPU slots have been released.
4. Ensure git commit messages are simple and descriptive.
