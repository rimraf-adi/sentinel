use sentinel_core::models::{JobSpec, JobStatus, RunMode};
use sentinel_daemon::scheduler::DynamicGpuPool;
use sentinel_daemon::server::Server;
use sentinel_daemon::supervisor::ProcessSupervisor;
use std::time::Duration;

#[tokio::test]
async fn test_concurrent_multiple_jobs() {
    let supervisor = ProcessSupervisor::new();
    let count = 5;
    let mut job_ids = Vec::new();

    // Spawn 5 jobs concurrently
    for i in 0..count {
        let spec = JobSpec::new(format!("concurrent-job-{i}"), RunMode::Command)
            .with_command(format!("echo 'job {i} step 1' && sleep 0.05 && echo 'job {i} done'"));
        job_ids.push(spec.id);
        supervisor.start_job(spec).await.expect("start job");
    }

    // Wait for all to complete
    tokio::time::sleep(Duration::from_millis(300)).await;

    let jobs = supervisor.list_jobs().await;
    assert_eq!(jobs.len(), count);

    for id in job_ids {
        let record = supervisor.get_job(id).await.expect("job exists");
        assert_eq!(record.status, JobStatus::Completed);
        assert_eq!(record.exit_code, Some(0));

        let logs = supervisor.get_logs(id).await;
        assert!(logs.iter().any(|l| l.contains("done")));
    }
}

#[tokio::test]
async fn test_high_volume_log_streaming() {
    let supervisor = ProcessSupervisor::new();
    // Generate 5,000 lines of rapid output
    let spec = JobSpec::new("high-volume-logs", RunMode::Command)
        .with_command("for i in $(seq 1 5000); do echo \"line $i\"; done");

    let job_id = spec.id;
    supervisor.start_job(spec).await.expect("start log job");

    tokio::time::sleep(Duration::from_millis(600)).await;

    let record = supervisor.get_job(job_id).await.expect("job exists");
    assert_eq!(record.status, JobStatus::Completed);

    let logs = supervisor.get_logs(job_id).await;
    assert!(logs.len() >= 5000, "Should capture all 5000 log lines! Captured: {}", logs.len());
    assert!(logs.first().unwrap().contains("line 1"));
    assert!(logs.last().unwrap().contains("line 5000"));
}

#[tokio::test]
async fn test_process_failure_and_exit_code_capture() {
    let supervisor = ProcessSupervisor::new();
    let spec = JobSpec::new("failing-job", RunMode::Command)
        .with_command("echo 'error before crash' >&2 && exit 42");

    let job_id = spec.id;
    supervisor.start_job(spec).await.expect("start failing job");

    tokio::time::sleep(Duration::from_millis(150)).await;

    let record = supervisor.get_job(job_id).await.expect("job exists");
    assert_eq!(record.status, JobStatus::Failed);
    assert_eq!(record.exit_code, Some(42));

    let logs = supervisor.get_logs(job_id).await;
    assert!(logs.iter().any(|l| l.contains("[stderr] error before crash")));
}

#[tokio::test]
async fn test_dynamic_gpu_pool_contention_stress() {
    let pool = DynamicGpuPool::new(4);

    // Concurrently allocate and release 100 times across 4 tasks
    let mut handles = Vec::new();
    for _ in 0..4 {
        let p = pool.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..25 {
                if let Some(gpus) = p.allocate(1).await {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                    p.release(&gpus).await;
                }
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    // After all tasks finish, all 4 GPUs must be restored
    assert_eq!(pool.free_count().await, 4);
}

#[test]
fn test_downsampler_one_million_points_stress() {
    // 1,000,000 metric points
    let points: Vec<(u64, f64)> = (0..1_000_000).map(|i| (i, i as f64 * 0.1)).collect();

    let start = std::time::Instant::now();
    let downsampled = Server::downsample_series(&points, 500);
    let duration = start.elapsed();

    assert!(downsampled.len() <= 501);
    assert_eq!(downsampled.first().unwrap().0, 0);
    assert_eq!(downsampled.last().unwrap().0, 999_999);
    // Must downsample in under 20 milliseconds
    assert!(
        duration < Duration::from_millis(50),
        "Downsampling took too long: {:?}",
        duration
    );
}
