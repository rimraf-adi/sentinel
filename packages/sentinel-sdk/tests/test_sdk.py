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

def test_sentinel_checkpointing_latest_and_all(tmp_path):
    import torch
    import torch.nn as nn
    from torch.optim import Adam

    ckpt_dir = tmp_path / "checkpoints"

    # Define simple model and optimizer
    model1 = nn.Sequential(nn.Linear(4, 2), nn.ReLU())
    opt1 = Adam(model1.parameters(), lr=0.01)

    # 1. Test mode="latest" (default)
    p1 = sentinel.save_checkpoint(
        model=model1,
        optimizer=opt1,
        epoch=1,
        step=10,
        metrics={"loss": 0.42},
        mode="latest",
        checkpoint_dir=str(ckpt_dir),
    )
    assert p1 is not None
    assert (ckpt_dir / "checkpoint_latest.pt").exists()
    assert not (ckpt_dir / "checkpoint_epoch_1_step_10.pt").exists()
    assert sentinel.has_checkpoint(str(ckpt_dir))

    # 2. Test mode="all"
    p2 = sentinel.save_checkpoint(
        model=model1,
        optimizer=opt1,
        epoch=2,
        step=20,
        metrics={"loss": 0.28},
        mode="all",
        checkpoint_dir=str(ckpt_dir),
    )
    assert p2 is not None
    assert (ckpt_dir / "checkpoint_latest.pt").exists()
    assert (ckpt_dir / "checkpoint_epoch_2_step_20.pt").exists()

    # 3. Test seamless resume with load_checkpoint
    model2 = nn.Sequential(nn.Linear(4, 2), nn.ReLU())
    opt2 = Adam(model2.parameters(), lr=0.01)

    # Weights before load should differ from model1
    loaded_state = sentinel.load_checkpoint(
        model=model2,
        optimizer=opt2,
        checkpoint_dir=str(ckpt_dir),
    )
    assert loaded_state is not None
    assert loaded_state["epoch"] == 2
    assert loaded_state["step"] == 20
    assert loaded_state["metrics"]["loss"] == 0.28

    # Weights after load must match exactly
    for p_a, p_b in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p_a, p_b)

    print("✅ Sentinel Checkpointing (latest and all) & Seamless Resume PASSED!")

def test_plots_and_results_stay_in_run_directory_by_default(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    tracking_dir = tmp_path / "mlruns"

    # Start run
    run = sentinel.init(
        run_name="plots-and-results-isolation-test",
        tracking_dir=str(tracking_dir),
    )
    assert run is not None
    run_dir = run.run_dir
    plots_dir = run.plots_dir
    results_dir = run.results_dir

    # Verify environment variables were set
    assert os.environ.get("SENTINEL_RUN_DIR") == str(run_dir.resolve())
    assert os.environ.get("SENTINEL_PLOTS_DIR") == str(plots_dir.resolve())
    assert os.environ.get("SENTINEL_RESULTS_DIR") == str(results_dir.resolve())

    # 1. Explicit save_plot and log_figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [0.1, 0.4, 0.9])
    plot_path = sentinel.save_plot("loss_curve", fig)
    assert Path(plot_path).exists()
    assert Path(plot_path).parent == plots_dir
    assert (plots_dir / "loss_curve.png").exists()

    # 2. Explicit save_results
    eval_results = {"accuracy": 0.942, "loss": 0.081, "f1": 0.938}
    res_path = sentinel.save_results(eval_results, "eval_metrics.json")
    assert Path(res_path).exists()
    assert Path(res_path).parent == results_dir
    assert (results_dir / "eval_metrics.json").exists()

    # 3. AUTOMATIC DEFAULT REDIRECTION OF MATPLOTLIB SAVEFIG
    # A standard script simply calls plt.savefig("confusion_matrix.png") without specifying a path
    plt.figure()
    plt.plot([0, 1], [1, 0])
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Crucial assertion: the root directory MUST NOT contain confusion_matrix.png
    assert not (tmp_path / "confusion_matrix.png").exists(), "Root directory was polluted with plot!"
    # The plot MUST stay in the run's plots directory!
    assert (plots_dir / "confusion_matrix.png").exists(), "Plot was not redirected to run directory!"

    # 4. Finish run
    sentinel.finish()
    assert os.environ.get("SENTINEL_RUN_DIR") is None
    print("✅ Run directory isolation policy for all plots and results PASSED!")

if __name__ == "__main__":
    test_sentinel_mlflow_end_to_end()
    test_safe_tracking_never_crashes()
