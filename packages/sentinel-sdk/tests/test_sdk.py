"""
Tests for Sentinel Python SDK and strict MLflow layout compatibility.
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import sys
import shutil
from pathlib import Path

# Add sentinel-sdk/src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import sentinel
import mlflow

def test_sentinel_mlflow_end_to_end():
    test_tracking_dir = Path("/tmp/sentinel_test_mlflow")
    if test_tracking_dir.exists():
        shutil.rmtree(test_tracking_dir)

    # 1. Initialize Sentinel run
    config = {"lr": 0.0003, "batch_size": 64, "optimizer": {"name": "AdamW", "weight_decay": 0.01}}
    run = sentinel.init(
        run_name="test-cifar-run",
        config=config,
        tags={"custom_tag": "v1_test"},
        tracking_dir=str(test_tracking_dir),
    )
    assert run is not None
    run_id = run.run_id

    # 2. Log metrics across steps
    for step in range(1, 4):
        sentinel.log({"train/loss": 1.0 / step, "train/acc": 0.2 * step}, step=step)

    # 3. Log an artifact
    dummy_artifact = test_tracking_dir / "sample_model.txt"
    dummy_artifact.write_text("dummy model weights")
    sentinel.log_artifact(str(dummy_artifact), artifact_path="checkpoints")

    # 4. Finish run
    sentinel.finish(status="FINISHED")

    # 5. VERIFY with native MLflow client
    client = mlflow.MlflowClient(tracking_uri=f"file://{test_tracking_dir.resolve()}")
    mlflow_run = client.get_run(run_id)

    assert mlflow_run.info.run_id == run_id
    assert mlflow_run.info.status == "FINISHED"
    assert mlflow_run.data.tags.get("mlflow.runName") == "test-cifar-run"
    assert mlflow_run.data.tags.get("custom_tag") == "v1_test"
    assert float(mlflow_run.data.params.get("lr")) == 0.0003
    assert mlflow_run.data.params.get("optimizer.name") == "AdamW"

    # Verify metric history
    loss_history = client.get_metric_history(run_id, "train/loss")
    assert len(loss_history) == 3
    assert [m.step for m in loss_history] == [1, 2, 3]
    assert [m.value for m in loss_history] == [1.0, 0.5, 1.0 / 3]

    print("✅ All Sentinel SDK + MLflow-native compatibility assertions PASSED!")

def test_safe_tracking_never_crashes():
    # Intentionally corrupt internal state or pass invalid data
    sentinel.log({"corrupt": "not_a_float"})
    sentinel.log_params({"bad": object()})
    sentinel.log_artifact("/path/that/does/not/exist")
    # Must reach here without throwing
    print("✅ Safe tracking wrapper verified: 0 exceptions raised on bad input!")

def test_get_dataset_path_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    # 1. Test dataset in datasets.yaml
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text("datasets:\n  - name: my_external_data\n    path: /tmp\n")
    
    resolved = sentinel.get_dataset_path("my_external_data")
    assert resolved is not None
    assert resolved == str(Path("/tmp").resolve())

    # 2. Test in-repo datasets/ directory
    (tmp_path / "datasets" / "local_split").mkdir(parents=True)
    resolved_local = sentinel.get_dataset_path("local_split")
    assert resolved_local is not None
    assert "local_split" in resolved_local

    # 3. Non-existent returns None
    assert sentinel.get_dataset_path("non_existent_dataset_123") is None

if __name__ == "__main__":
    test_sentinel_mlflow_end_to_end()
    test_safe_tracking_never_crashes()
