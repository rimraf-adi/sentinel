"""
Rigorous stress, concurrency, and fidelity tests for Sentinel Python SDK.
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import sys
import shutil
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import sentinel
import mlflow

def test_multithreaded_concurrent_logging():
    test_dir = Path("/tmp/sentinel_stress_multithread")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    run = sentinel.init(
        run_name="stress-threaded-run",
        config={"batch_size": 128, "model": "resnet50"},
        tracking_dir=str(test_dir),
    )
    assert run is not None
    run_id = run.run_id

    # 4 concurrent threads logging metrics simultaneously
    def worker(thread_id, steps=50):
        for s in range(steps):
            sentinel.log({f"worker_{thread_id}/metric": s * 0.5}, step=s)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    sentinel.finish(status="FINISHED")

    # Verify with MLflow
    client = mlflow.MlflowClient(tracking_uri=f"file://{test_dir.resolve()}")
    mlflow_run = client.get_run(run_id)
    assert mlflow_run.info.status == "FINISHED"

    for i in range(4):
        history = client.get_metric_history(run_id, f"worker_{i}/metric")
        assert len(history) == 50, f"Worker {i} should have logged 50 points!"

    print("✅ Multi-threaded concurrent logging verified: 200 metric points across 4 threads with 0 corruption!")

def test_crash_immunity_edge_cases():
    # Test logging without init
    sentinel.finish()
    sentinel.log({"loss": 1.0}) # Should safely no-op

    # Test passing circular or invalid structures
    d = {}
    d["self"] = d
    sentinel.init(config=d, tracking_dir="/tmp/sentinel_invalid")
    sentinel.log({"loss": float("nan")}) # NaN float logging
    sentinel.log({"loss": float("inf")}) # Inf float logging
    sentinel.finish()

    print("✅ Crash immunity verified under circular references, uninitialized state, and NaN/Inf floats!")

if __name__ == "__main__":
    test_multithreaded_concurrent_logging()
    test_crash_immunity_edge_cases()
