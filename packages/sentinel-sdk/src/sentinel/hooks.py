"""
Sentinel Dynamic Integration Hooks Loader
Automatically discovers, imports, and invokes Python companion hooks installed in .sentinel/integrations/
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

logger = logging.getLogger("sentinel.hooks")

class HookRegistry:
    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir)
        self.hooks_dir = self.workspace_dir / ".sentinel" / "integrations"
        self._loaded_modules: Dict[str, Any] = {}
        self._discover_and_load()

    def _discover_and_load(self):
        if not self.hooks_dir.exists():
            return

        # Check protocol manifest for disabled modules if present
        enabled_modules = None
        proto_file = self.workspace_dir / "sentinel.protocol.yaml"
        if not proto_file.exists():
            proto_file = self.workspace_dir / ".sentinel" / "sentinel.protocol.yaml"

        if proto_file.exists():
            try:
                import yaml
                with open(proto_file, "r") as f:
                    proto = yaml.safe_load(f)
                    if isinstance(proto, dict) and "integrations" in proto:
                        enabled_modules = {
                            i["name"].lower() for i in proto["integrations"] if i.get("enabled", True)
                        }
            except Exception:
                pass

        for entry in self.hooks_dir.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name.lower()
            if enabled_modules is not None and name not in enabled_modules:
                continue

            hook_file = entry / "hook.py"
            if not hook_file.exists():
                continue

            try:
                module_name = f"sentinel_integration_{name}"
                spec = importlib.util.spec_from_file_location(module_name, str(hook_file))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)
                    self._loaded_modules[name] = mod
                    logger.info(f"Loaded Sentinel integration runtime hook: {name}")
            except Exception as e:
                logger.warning(f"Failed to load hook for {name}: {e}")

    def dispatch(self, hook_name: str, *args, **kwargs):
        """Invoke a lifecycle hook across all active integrations."""
        for mod_name, mod in self._loaded_modules.items():
            func = getattr(mod, hook_name, None)
            if callable(func):
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in {mod_name}.{hook_name}: {e}")

_GLOBAL_REGISTRY = None

def get_registry(workspace_dir: str = ".") -> HookRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = HookRegistry(workspace_dir)
    return _GLOBAL_REGISTRY

def on_train_begin(model, optimizer=None, config=None, workspace_dir: str = "."):
    get_registry(workspace_dir).dispatch("on_train_begin", model=model, optimizer=optimizer, config=config)

def on_step_end(step: int, loss: float, model=None, workspace_dir: str = ".", **kwargs):
    get_registry(workspace_dir).dispatch("on_step_end", step=step, loss=loss, model=model, **kwargs)

def on_epoch_end(epoch: int, val_loss: float, model=None, workspace_dir: str = ".", **kwargs):
    get_registry(workspace_dir).dispatch("on_epoch_end", epoch=epoch, val_loss=val_loss, model=model, **kwargs)

def on_train_end(model=None, run_id: str = None, workspace_dir: str = "."):
    get_registry(workspace_dir).dispatch("on_train_end", model=model, run_id=run_id)
