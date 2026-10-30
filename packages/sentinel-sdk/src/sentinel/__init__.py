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
