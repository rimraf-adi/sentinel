"""
Sentinel Python SDK
Thin, safe client for training telemetry, MLflow-native storage, and daemon supervision.
"""

import functools
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure MLflow filesystem backend is permitted and clean
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

logger = logging.getLogger("sentinel")

def _safe_call(default_return=None):
    """Decorator ensuring Sentinel tracking calls NEVER crash user training."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"[Sentinel SDK Warning] Error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator

class Run:
    def __init__(
        self,
        experiment_id: str = "0",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        tracking_dir: str = "mlruns",
    ):
        self.run_id = uuid.uuid4().hex
        self.experiment_id = experiment_id
        self.run_name = run_name or f"run_{self.run_id[:8]}"
        self.tracking_dir = Path(tracking_dir)
        self.start_time_ms = int(time.time() * 1000)
        self.is_active = True
        self.current_step = 0

        # Create MLflow-compatible directory structure
        self.run_dir = self.tracking_dir / self.experiment_id / self.run_id
        self.metrics_dir = self.run_dir / "metrics"
        self.params_dir = self.run_dir / "params"
        self.tags_dir = self.run_dir / "tags"
        self.artifacts_dir = self.run_dir / "artifacts"

        self._setup_mlflow_dirs()
        self._write_run_meta(status="RUNNING")

        # Record standard tags
        default_tags = {
            "mlflow.runName": self.run_name,
            "mlflow.user": os.environ.get("USER", "unknown"),
            "mlflow.source.name": sys.argv[0] if hasattr(sys, "argv") and sys.argv else "interactive",
            "mlflow.source.type": "LOCAL",
            "sentinel.version": "0.1.0",
        }
        if tags:
            default_tags.update(tags)
        for k, v in default_tags.items():
            self._write_tag(k, str(v))

        # Record params
        if config:
            self._log_params_recursive(config)

    def _setup_mlflow_dirs(self):
        exp_dir = self.tracking_dir / self.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_meta = exp_dir / "meta.yaml"
        if not exp_meta.exists():
            exp_meta.write_text(
                f"artifact_location: {exp_dir.resolve().as_uri()}\n"
                f"creation_time: {self.start_time_ms}\n"
                f"experiment_id: '{self.experiment_id}'\n"
                f"last_update_time: {self.start_time_ms}\n"
                f"lifecycle_stage: active\n"
                f"name: Default\n"
            )

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.tags_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _write_run_meta(self, status: str = "RUNNING", end_time_ms: Optional[int] = None):
        meta_file = self.run_dir / "meta.yaml"
        content = (
            f"artifact_uri: {self.artifacts_dir.resolve().as_uri()}\n"
            f"end_time: {end_time_ms if end_time_ms is not None else 'null'}\n"
            f"entry_point_name: ''\n"
            f"experiment_id: '{self.experiment_id}'\n"
            f"lifecycle_stage: active\n"
            f"run_id: {self.run_id}\n"
            f"run_name: {self.run_name}\n"
            f"run_uuid: {self.run_id}\n"
            f"source_name: ''\n"
            f"source_type: 4\n"
            f"source_version: ''\n"
            f"start_time: {self.start_time_ms}\n"
            f"status: {3 if status == 'FINISHED' else (4 if status == 'FAILED' else 1)}\n"
            f"tags: []\n"
            f"user_id: '{os.environ.get('USER', 'unknown')}'\n"
        )
        meta_file.write_text(content)

    def _write_tag(self, key: str, value: str):
        safe_key = key.replace("/", ".")
        tag_file = self.tags_dir / safe_key
        tag_file.parent.mkdir(parents=True, exist_ok=True)
        tag_file.write_text(str(value))

    def _log_params_recursive(self, params: Dict[str, Any], prefix: str = ""):
        for k, v in params.items():
            full_key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                self._log_params_recursive(v, prefix=full_key)
            else:
                param_file = self.params_dir / full_key
                param_file.parent.mkdir(parents=True, exist_ok=True)
                param_file.write_text(str(v))

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        if not self.is_active:
            return
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        now_ms = int(time.time() * 1000)

        for key, value in metrics.items():
            metric_file = self.metrics_dir / key
            metric_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metric_file, "a", encoding="utf-8") as f:
                f.write(f"{now_ms} {float(value)} {self.current_step}\n")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        src = Path(local_path)
        if not src.exists():
            return
        dest_dir = self.artifacts_dir / (artifact_path or "")
        dest_dir.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dest_dir / src.name)
        else:
            shutil.copytree(src, dest_dir / src.name, dirs_exist_ok=True)

    def finish(self, status: str = "FINISHED"):
        if not self.is_active:
            return
        end_time_ms = int(time.time() * 1000)
        self._write_run_meta(status=status, end_time_ms=end_time_ms)
        self.is_active = False

import sys

_current_run: Optional[Run] = None

@_safe_call(default_return=None)
def init(
    project: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    tracking_dir: str = "mlruns",
) -> Optional[Run]:
    global _current_run
    _current_run = Run(
        experiment_id="0",
        run_name=run_name,
        config=config,
        tags=tags,
        tracking_dir=tracking_dir,
    )
    return _current_run

@_safe_call(default_return=None)
def log(metrics: Dict[str, float], step: Optional[int] = None):
    global _current_run
    if _current_run is not None:
        _current_run.log(metrics, step=step)

@_safe_call(default_return=None)
def log_params(params: Dict[str, Any]):
    global _current_run
    if _current_run is not None:
        _current_run._log_params_recursive(params)

@_safe_call(default_return=None)
def log_artifact(local_path: str, artifact_path: Optional[str] = None):
    global _current_run
    if _current_run is not None:
        _current_run.log_artifact(local_path, artifact_path=artifact_path)

@_safe_call(default_return=None)
def finish(status: str = "FINISHED"):
    global _current_run
    if _current_run is not None:
        _current_run.finish(status=status)
        _current_run = None

@_safe_call(default_return=None)
def get_run() -> Optional[Run]:
    global _current_run
    return _current_run

# ---------------------------------------------------------
# Sentinel Built-In Checkpointing & Seamless Resume
# ---------------------------------------------------------

@_safe_call(default_return=None)
def save_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    mode: str = "latest",
    checkpoint_dir: Optional[str] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Saves a training checkpoint safely with atomic file replacement.
    
    Args:
        model: PyTorch nn.Module instance
        optimizer: Optional torch.optim.Optimizer
        scheduler: Optional learning rate scheduler
        epoch: Current training epoch index
        step: Current global training step
        metrics: Dictionary of metric values at this checkpoint
        mode: "latest" (default) overwrites checkpoint_latest.pt atomically.
              "all" keeps checkpoint_epoch_{epoch}_step_{step}.pt and updates checkpoint_latest.pt.
        checkpoint_dir: Target directory. Defaults to active run's artifacts/checkpoints,
                        or "./checkpoints" if no run is active.
        extra_state: Optional dictionary of arbitrary extra state.
        
    Returns:
        Path string to the saved checkpoint, or None on failure.
    """
    import torch

    global _current_run

    if checkpoint_dir is not None:
        target_dir = Path(checkpoint_dir)
    elif _current_run is not None:
        target_dir = _current_run.artifacts_dir / "checkpoints"
    else:
        target_dir = Path(os.environ.get("SENTINEL_CHECKPOINT_DIR", "checkpoints"))

    target_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else model,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None and hasattr(optimizer, "state_dict") else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
        "metrics": metrics or {},
        "extra_state": extra_state or {},
        "timestamp": time.time(),
        "sentinel_version": "0.1.0",
    }

    # Save to atomic temp file first to prevent corruption if interrupted
    tmp_name = f".tmp_ckpt_{uuid.uuid4().hex[:8]}.pt"
    tmp_path = target_dir / tmp_name
    torch.save(payload, tmp_path)

    latest_path = target_dir / "checkpoint_latest.pt"

    if mode == "all":
        ep_tag = f"epoch_{epoch}" if epoch is not None else "epoch_none"
        st_tag = f"_step_{step}" if step is not None else ""
        all_name = f"checkpoint_{ep_tag}{st_tag}.pt"
        shutil.copy2(tmp_path, target_dir / all_name)

    # Atomic rename replaces checkpoint_latest.pt cleanly
    os.replace(tmp_path, latest_path)

    # Note in Sentinel run if active
    if _current_run is not None:
        _current_run._write_tag("sentinel.last_checkpoint", str(latest_path.resolve()))
        if epoch is not None:
            _current_run._write_tag("sentinel.last_checkpoint_epoch", str(epoch))

    return str(latest_path.resolve())


@_safe_call(default_return=None)
def get_latest_checkpoint(checkpoint_dir: Optional[str] = None) -> Optional[Path]:
    """Finds the latest valid checkpoint file."""
    global _current_run

    if checkpoint_dir is not None:
        target_dir = Path(checkpoint_dir)
    elif _current_run is not None:
        target_dir = _current_run.artifacts_dir / "checkpoints"
    else:
        target_dir = Path(os.environ.get("SENTINEL_CHECKPOINT_DIR", "checkpoints"))

    if not target_dir.exists():
        return None

    latest_file = target_dir / "checkpoint_latest.pt"
    if latest_file.exists():
        return latest_file

    # Fallback: scan for any checkpoint_*.pt files sorted by modification time
    ckpts = sorted(target_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]

    return None


@_safe_call(default_return=False)
def has_checkpoint(checkpoint_dir: Optional[str] = None) -> bool:
    """Returns True if a valid checkpoint is available for resumption."""
    return get_latest_checkpoint(checkpoint_dir) is not None


@_safe_call(default_return=None)
def load_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    map_location: Optional[Any] = None,
    strict: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Seamlessly resumes training by loading the latest (or specified) checkpoint.
    
    Args:
        model: PyTorch nn.Module to load weights into
        optimizer: Optional optimizer to restore momentum/state
        scheduler: Optional scheduler to restore step count
        path: Explicit path to checkpoint file (if None, discovers latest checkpoint)
        checkpoint_dir: Directory where checkpoints reside (used if path is None)
        map_location: Device mapping (e.g., 'cpu', 'cuda', torch.device)
        strict: Whether to strictly enforce key matching in model state_dict
        
    Returns:
        Checkpoint dictionary with keys ('epoch', 'step', 'metrics', 'extra_state', etc.),
        or None if no checkpoint was found or loading failed.
    """
    import torch

    ckpt_path = Path(path) if path is not None else get_latest_checkpoint(checkpoint_dir)
    if ckpt_path is None or not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # 1. Restore model state
    if "model_state_dict" in ckpt and hasattr(model, "load_state_dict"):
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    elif hasattr(model, "load_state_dict") and isinstance(ckpt, dict) and not any(k in ckpt for k in ["epoch", "step"]):
        # Raw state dict format
        model.load_state_dict(ckpt, strict=strict)

    # 2. Restore optimizer state
    if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            logger.warning(f"[Sentinel Checkpoint] Failed to restore optimizer state: {e}")

    # 3. Restore scheduler state
    if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"[Sentinel Checkpoint] Failed to restore scheduler state: {e}")

    logger.info(
        f"[Sentinel Checkpoint] Resumed successfully from {ckpt_path.name} "
        f"(epoch: {ckpt.get('epoch')}, step: {ckpt.get('step')})"
    )

    return ckpt

@_safe_call(default_return=None)
def get_dataset_path(name: str) -> Optional[str]:
    """Resolve a dataset path from datasets.yaml, sentinel.yaml, or local datasets/ directory.
    
    Supports external filesystem paths (e.g., /data/imagenet, ~/datasets/cifar10),
    environment variables (e.g. $DATA_DIR), and in-repo paths.
    """
    import yaml
    
    # 1. Search configuration files
    for config_name in ["datasets.yaml", "sentinel.yaml"]:
        p = Path(config_name)
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = yaml.safe_load(f) or {}
                datasets = data.get("datasets", [])
                if isinstance(datasets, list):
                    for d in datasets:
                        if isinstance(d, dict) and d.get("name") == name:
                            raw_path = d.get("path", "")
                            expanded = os.path.expanduser(os.path.expandvars(str(raw_path)))
                            return str(Path(expanded).resolve())
                elif isinstance(datasets, dict) and name in datasets:
                    raw_path = datasets[name].get("path") if isinstance(datasets[name], dict) else datasets[name]
                    expanded = os.path.expanduser(os.path.expandvars(str(raw_path)))
                    return str(Path(expanded).resolve())
            except Exception:
                pass
                
    # 2. Check local repo datasets directory
    local = Path("datasets") / name
    if local.exists():
        return str(local.resolve())

    # Fallback if path itself exists
    if Path(name).exists():
        return str(Path(name).resolve())

    return None


class ModelFlowWatcher:
    """Attaches forward and backward hooks to PyTorch modules to monitor activations and gradients."""
    def __init__(self, model, log_freq: int = 10, out_dir: str = ".sentinel"):
        self.model = model
        self.log_freq = log_freq
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.activations: Dict[str, Any] = {}
        self.gradients: Dict[str, Any] = {}
        self.hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return

        for name, module in self.model.named_modules():
            if list(module.children()):
                continue
            has_params = any(p.requires_grad for p in module.parameters())
            if not has_params and not isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                continue

            layer_name = name or module.__class__.__name__

            def make_fwd_hook(l_name):
                def hook(mod, inp, out):
                    if not getattr(self.model, 'training', True):
                        return
                    with torch.no_grad():
                        t = out if isinstance(out, torch.Tensor) else out[0] if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor) else None
                        if t is not None:
                            val = t.detach()
                            mean_act = float(val.abs().mean().item())
                            std_act = float(val.std().item()) if val.numel() > 1 else 0.0
                            dead_pct = float((val <= 0).float().mean().item()) * 100.0 if val.numel() > 0 else 0.0
                            self.activations[l_name] = {
                                "mean": mean_act,
                                "std": std_act,
                                "dead_pct": dead_pct,
                                "shape": list(val.shape),
                            }
                return hook

            def make_bwd_hook(l_name):
                def hook(mod, grad_input, grad_output):
                    if not getattr(self.model, 'training', True):
                        return
                    with torch.no_grad():
                        g = grad_output[0] if grad_output and isinstance(grad_output[0], torch.Tensor) else None
                        if g is not None:
                            norm = float(g.detach().norm(2).item())
                            mean_g = float(g.detach().abs().mean().item())
                            self.gradients[l_name] = {
                                "norm": norm,
                                "mean": mean_g,
                            }
                        self._maybe_flush()
                return hook

            self.hooks.append(module.register_forward_hook(make_fwd_hook(layer_name)))
            self.hooks.append(module.register_full_backward_hook(make_bwd_hook(layer_name)))

    def _maybe_flush(self):
        self.step += 1
        if self.step % self.log_freq != 0:
            return
        layers_data = []
        grad_norms = []
        burst_detected = False
        burst_layer = None
        burst_norm = 0.0
        clipped_layers = []

        clip_limit = 1.0
        burst_limit = 5.0

        for l_name, act_data in self.activations.items():
            g_data = self.gradients.get(l_name, {"norm": 0.0, "mean": 0.0})
            norm = g_data["norm"]
            grad_norms.append(norm)

            dead_pct = act_data.get("dead_pct", 0.0)
            status = "healthy"
            is_clipped = norm > clip_limit
            is_burst = norm > burst_limit

            if is_burst:
                burst_detected = True
                burst_layer = l_name
                burst_norm = max(burst_norm, norm)
                status = "exploding"
            elif norm < 1e-5:
                status = "vanishing"
            elif dead_pct > 80:
                status = "dead"

            if is_clipped:
                clipped_layers.append(l_name)

            # Stability index: high variance in activation + balanced gradients = high stability
            stability = max(0.0, min(100.0, (1.0 - (dead_pct / 100.0)) * (1.0 / (1.0 + abs(norm - 0.5))) * 100.0))

            layers_data.append({
                "name": l_name,
                "activation_mean": act_data["mean"],
                "activation_std": act_data["std"],
                "dead_pct": dead_pct,
                "gradient_norm": norm,
                "gradient_mean": g_data["mean"],
                "status": status,
                "stability": round(stability, 1),
                "is_clipped": is_clipped,
                "is_burst": is_burst,
                "shape": act_data.get("shape", []),
            })

        vanishing_ratio = (grad_norms[0] / max(grad_norms[-1], 1e-8)) if len(grad_norms) >= 2 else 1.0
        total_norm = (sum(n * n for n in grad_norms) ** 0.5) if grad_norms else 0.0

        payload = {
            "step": self.step,
            "timestamp": time.time(),
            "vanishing_ratio": round(vanishing_ratio, 3),
            "total_grad_norm": round(total_norm, 3),
            "exploding": burst_detected,
            "burst": {
                "detected": burst_detected,
                "layer": burst_layer,
                "norm": round(burst_norm, 3),
                "threshold": burst_limit,
            },
            "clipping": {
                "active": len(clipped_layers) > 0,
                "clipped_layers": clipped_layers,
                "threshold": clip_limit,
                "scale_factor": round(clip_limit / max(total_norm, 1e-6), 3) if total_norm > clip_limit else 1.0,
            },
            "layers": layers_data,
        }
        flow_file = self.out_dir / "live_flow.json"
        try:
            flow_file.write_text(json.dumps(payload))
        except Exception:
            pass

    def remove(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()


@_safe_call(default_return=None)
def watch_model(model, log_freq: int = 10, out_dir: str = ".sentinel") -> Optional[ModelFlowWatcher]:
    """Attach forward and backward gradient/activation hooks for live telemetry in Sentinel."""
    return ModelFlowWatcher(model, log_freq=log_freq, out_dir=out_dir)


# ---------------------------------------------------------
# Sentinel Native Explainable AI (xAI) Exports
# ---------------------------------------------------------
try:
    from .xai import (
        integrated_gradients,
        spectral_rank,
        explain,
        log_explanation,
        grad_cam,
        attention_rollout,
        loss_landscape,
    )
except ImportError:
    pass

try:
    from . import hooks
except (ImportError, AttributeError):
    hooks = None



