# Agent Guide: Integrating with Sentinel-Powered Codebases

> **Audience:** Autonomous AI coding agents, background agents, and developers building agentic workflows on top of Sentinel.  
> **Core Principle:** *"Rust is boss. Python is worker."*

---

## 1. Mental Model & Architectural Boundaries

When you (the agent) are operating within a Sentinel-powered repository, keep these boundaries strictly in mind:

| Component | Role | What You Should Do | What You Should NOT Do |
| :--- | :--- | :--- | :--- |
| **Rust Daemon (`sentinel-daemon`)** | Supervisor, dynamic GPU allocator, NVML telemetry, API server. | Interact via CLI (`sentinel dev`, `sentinel run`) or REST/WebSocket endpoints. | Do not rewrite training logic in Rust. Rust only manages and supervises. |
| **Python Worker (`uv`)** | Training loop, PyTorch, MLflow tracking, Optuna sweeps. | Execute Python scripts strictly using `uv run python ...`. | Never run raw unmanaged `pip install` or `python` commands that escape the environment. |
| **Declarative Templates (`templates/`)** | Single source of truth for experiment recipes. | Create, extend, override parameters, and validate templates with `sentinel template check`. | Do not bypass templates to hardcode unversioned flags in ad-hoc bash scripts. |
| **Storage (`mlruns/`)** | MLflow-native experiment tracking. | Read experiment results using standard MLflow or Sentinel Python SDK. | Do not create proprietary metrics formats; adhere strictly to MLflow 3.x layout. |
| **Web Dashboard (`sentinel-ui`)** | Minimalist Vercel-style UI. | Code view in UI is **strictly read-only**; parameter forms and run controls are **editable**. | Do not attempt to add code editing inputs to the web dashboard. |

---

## 2. Detection & Grounding Protocol

When entering a new workspace, follow this discovery routine to detect if Sentinel is present:

```bash
# 1. Check for Sentinel configuration or template directory
test -f sentinel.yaml || test -d templates/ && echo "Sentinel codebase detected"

# 2. Check if Sentinel CLI binary is available
which sentinel || test -f target/debug/sentinel && echo "Sentinel binary found"

# 3. Check if Sentinel Daemon is running
curl -s http://localhost:8080/api/health
```

If the daemon is offline, launch it as a background process:
```bash
# Start Sentinel daemon on port 8080
sentinel dev --port 8080 &
# Or via debug target:
target/debug/sentinel dev --port 8080 &
```

---

## 3. Working with Templates (The Core ML Workflow)

Sentinel enforces declarative, reproducible experiments via YAML templates with inheritance (`extends:`), type validation, and locked fields.

### Listing Templates
```bash
sentinel template list
# Or via API:
curl -s http://localhost:8080/api/templates | jq .
```

### Validating a Template Before Execution
Always run validation before attempting to start a training run:
```bash
sentinel template check templates/vision/image-classification.yaml
```
*Validation checks:*
* Syntax correctness (YAML composition).
* Parameter ranges (`min` $\le$ value $\le$ `max`).
* Choice options (`value in options`).
* Locked parameter integrity (asserts that locked settings like `tracking.backend: mlflow` have not been violated).

### Creating a New Custom Recipe Template
When the user asks to train a new model or create a new recipe:
```bash
sentinel template new my-model-recipe --extends templates/vision/image-classification.yaml
```

---

## 4. Launching & Supervising Training Runs

### Launching via REST API (Recommended for Agents)
Send a `POST` request to `/api/jobs/run-template` with parameter overrides:
```bash
curl -X POST http://localhost:8080/api/jobs/run-template \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "vision/image-classification@v1",
    "overrides": {
      "training.lr": 0.0005,
      "training.batch_size": 32,
      "training.epochs": 5
    }
  }'
```
Response returns the `JobRecord` with unique `spec.id` and worker `pid`.

### Launching Custom Scripts (Command Mode)
For non-template arbitrary Python scripts, submit via `/api/jobs`:
```bash
curl -X POST http://localhost:8080/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "bert-finetune",
    "command": "uv run python recipes/my_script.py --lr 3e-5",
    "run_mode": "Standalone"
  }'
```

### Monitoring Active Jobs & Fetching Logs
```bash
# List all jobs
curl -s http://localhost:8080/api/jobs | jq .

# Stream or fetch logs for a specific job
curl -s http://localhost:8080/api/jobs/<JOB_ID>/logs | jq .logs
```

### Terminating a Run
If a job is diverging, stuck, or requested to stop:
```bash
curl -X POST http://localhost:8080/api/jobs/<JOB_ID>/stop
```
The daemon sends `SIGTERM`, waits for child drain, and frees GPU slots in the dynamic pool.

---

## 5. Instrumenting Python Scripts with `sentinel-sdk`

When writing new Python training code, integrate `sentinel-sdk`. It guarantees **crash immunity** (logging failures will never interrupt training):

```python
import sentinel

# 1. Initialize run (creates mlruns/ experiment structure)
run = sentinel.init(
    experiment_name="cifar10-benchmark",
    run_name="vit-baseline"
)

# 2. Log hyperparameters
run.log_params({
    "lr": 0.0003,
    "batch_size": 64,
    "optimizer": "AdamW",
    "weight_decay": 0.05
})

# 3. Training Loop
for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch(...)
    val_loss, val_acc = evaluate(...)

    # Crash-safe metric logging: safe to call even if daemon is stopped
    run.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc
    }, step=epoch)

# 4. Finish run
run.finish()
```

---

## 6. Hyperparameter Sweeps (Optuna ASHA)

When tasked with tuning hyperparameters, use Sentinel's Optuna ASHA orchestrator:

```bash
# Launch multi-trial sweep with early pruning
uv run python recipes/sweep_orchestrator.py --trials 15 --max-epochs 10
```
*How it works:*
1. Samples parameters (e.g. log-uniform learning rate).
2. Spawns worker trials on available GPU pool slots.
3. Reports intermediate metric to Optuna `SuccessiveHalvingPruner`.
4. Early-terminates underperforming trials to recycle VRAM.

---

## 7. Zero Lock-In Export

If the user requests to run training on a cluster or VM without Sentinel installed:
```bash
# Export to pure bare PyTorch code
sentinel export script --template vision/image-classification@v1 > standalone_train.py

# Verify the script runs standalone with pure torch
uv run python standalone_train.py
```

---

## 8. Agent Checklist & Protocols

Before declaring an experiment or feature task complete, verify:

- [ ] **Environment Consistency:** Strictly executed via `uv run`.
- [ ] **Daemon Health:** Verified `/api/health` returns `200 OK`.
- [ ] **Template Validation:** Passed `sentinel template check`.
- [ ] **GPU Deallocation:** Checked `/api/hardware` to verify GPU slots were released after job completion.
- [ ] **Log Verification:** Checked `/api/jobs/:id/logs` to ensure no uncaught Python tracebacks.
- [ ] **MLflow Integrity:** Confirmed `mlruns/` contains valid `meta.yaml`, `params/`, and `metrics/`.
