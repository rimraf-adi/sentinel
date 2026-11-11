//! Process supervisor: launches, monitors, streams logs, and terminates jobs.

use sentinel_core::models::{JobRecord, JobSpec, JobStatus};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use crate::store::JobStore;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum SupervisorError {
    #[error("Job not found: {0}")]
    JobNotFound(Uuid),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Failed to execute command: {0}")]
    ExecutionError(String),
}

struct ActiveJob {
    record: JobRecord,
    logs: Vec<String>,
    child_id: Option<u32>,
}

#[derive(Clone, Default)]
pub struct ProcessSupervisor {
    jobs: Arc<RwLock<HashMap<Uuid, ActiveJob>>>,
    store: Option<Arc<JobStore>>,
}

impl ProcessSupervisor {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            store: None,
        }
    }

    /// A supervisor whose job history is kept on disk and restored on start.
    pub fn with_store(store: JobStore) -> Self {
        let jobs = store
            .load_all()
            .into_iter()
            .map(|(record, logs)| (record.spec.id, ActiveJob { record, logs, child_id: None }))
            .collect();
        Self { jobs: Arc::new(RwLock::new(jobs)), store: Some(Arc::new(store)) }
    }

    /// Launch a new job asynchronously.
    pub async fn start_job(&self, spec: JobSpec) -> Result<JobRecord, SupervisorError> {
        let job_id = spec.id;
        let mut record = JobRecord::new(spec.clone());
        record.status = JobStatus::Running;
        record.started_at = Some(chrono::Utc::now());

        let cmd_string = match &spec.command {
            Some(cmd) => cmd.clone(),
            None => {
                // If no command specified, run python on recipe or exit 0
                let py = sentinel_core::platform::python_exe();
                let script = sentinel_core::platform::shell_quote("print('running sentinel default worker')");
                format!("{} -c {script}", sentinel_core::platform::shell_quote(&py))
            }
        };

        // Prepare shell execution
        // `sh -c` on Unix, `cmd /C` on Windows.
        let (shell, shell_args) = sentinel_core::platform::shell_invocation(&cmd_string);
        let mut cmd = Command::new(shell);
        cmd.args(shell_args);
        cmd.current_dir(&spec.working_dir);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Environment variables
        for (k, v) in &spec.env_vars {
            cmd.env(k, v);
        }

        // Set CUDA_VISIBLE_DEVICES if GPUs are assigned
        if !spec.gpu_ids.is_empty() {
            let gpus_str = spec
                .gpu_ids
                .iter()
                .map(|g| g.to_string())
                .collect::<Vec<_>>()
                .join(",");
            cmd.env("CUDA_VISIBLE_DEVICES", gpus_str);
        }

        let mut child = cmd.spawn()?;
        let child_pid = child.id();
        record.pid = child_pid;

        let active_job = ActiveJob {
            record: record.clone(),
            logs: Vec::new(),
            child_id: child_pid,
        };

        self.jobs.write().await.insert(job_id, active_job);
        if let Some(s) = &self.store {
            s.save(&record);
        }

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let jobs_ref = self.jobs.clone();
        let store = self.store.clone();

        // Spawn background task to stream logs and wait for process completion
        tokio::spawn(async move {
            let mut stdout_lines = stdout.map(|s| BufReader::new(s).lines());
            let mut stderr_lines = stderr.map(|s| BufReader::new(s).lines());

            loop {
                tokio::select! {
                    line = async {
                        if let Some(ref mut reader) = stdout_lines {
                            reader.next_line().await.ok().flatten()
                        } else {
                            None
                        }
                    } => {
                        match line {
                            Some(text) => {
                                let mut lock = jobs_ref.write().await;
                                if let Some(job) = lock.get_mut(&job_id) {
                                    { let l = format!("[stdout] {text}"); if let Some(s) = &store { s.append_log(job_id, &l); } job.logs.push(l); }
                                }
                            }
                            None => stdout_lines = None,
                        }
                    }
                    line = async {
                        if let Some(ref mut reader) = stderr_lines {
                            reader.next_line().await.ok().flatten()
                        } else {
                            None
                        }
                    } => {
                        match line {
                            Some(text) => {
                                let mut lock = jobs_ref.write().await;
                                if let Some(job) = lock.get_mut(&job_id) {
                                    { let l = format!("[stderr] {text}"); if let Some(s) = &store { s.append_log(job_id, &l); } job.logs.push(l); }
                                }
                            }
                            None => stderr_lines = None,
                        }
                    }
                    status = child.wait() => {
                        // Drain any remaining lines from stdout and stderr before recording finish
                        while let Some(line) = if let Some(ref mut reader) = stdout_lines {
                            reader.next_line().await.ok().flatten()
                        } else {
                            None
                        } {
                            let mut lock = jobs_ref.write().await;
                            if let Some(job) = lock.get_mut(&job_id) {
                                { let l = format!("[stdout] {line}"); if let Some(s) = &store { s.append_log(job_id, &l); } job.logs.push(l); }
                            }
                        }

                        while let Some(line) = if let Some(ref mut reader) = stderr_lines {
                            reader.next_line().await.ok().flatten()
                        } else {
                            None
                        } {
                            let mut lock = jobs_ref.write().await;
                            if let Some(job) = lock.get_mut(&job_id) {
                                { let l = format!("[stderr] {line}"); if let Some(s) = &store { s.append_log(job_id, &l); } job.logs.push(l); }
                            }
                        }

                        let mut lock = jobs_ref.write().await;
                        if let Some(job) = lock.get_mut(&job_id) {
                            // A job the user stopped stays Cancelled; its SIGTERM exit is not a failure.
                            let cancelled = job.record.status == JobStatus::Cancelled;
                            job.record.finished_at = Some(chrono::Utc::now());
                            match status {
                                Ok(exit_status) => {
                                    job.record.exit_code = exit_status.code();
                                    if cancelled {
                                    } else if exit_status.success() {
                                        job.record.status = JobStatus::Completed;
                                    } else {
                                        job.record.status = JobStatus::Failed;
                                        job.record.error_message = Some(format!("Process exited with status: {exit_status}"));
                                    }
                                }
                                Err(err) if !cancelled => {
                                    job.record.status = JobStatus::Failed;
                                    job.record.error_message = Some(err.to_string());
                                }
                                Err(_) => {}
                            }
                            if let Some(s) = &store { s.save(&job.record); }
                        }
                        break;
                    }
                }

                if stdout_lines.is_none() && stderr_lines.is_none() {
                    // Wait for process exit if streams closed
                    if let Ok(exit_status) = child.wait().await {
                        let mut lock = jobs_ref.write().await;
                        if let Some(job) = lock.get_mut(&job_id) {
                            job.record.finished_at = Some(chrono::Utc::now());
                            job.record.exit_code = exit_status.code();
                            if job.record.status != JobStatus::Cancelled {
                                job.record.status = if exit_status.success() {
                                    JobStatus::Completed
                                } else {
                                    JobStatus::Failed
                                };
                            }
                            if let Some(s) = &store { s.save(&job.record); }
                        }
                    }
                    break;
                }
            }
        });

        Ok(record)
    }

    /// Stop/terminate an active job.
    pub async fn stop_job(&self, job_id: Uuid) -> Result<(), SupervisorError> {
        let mut lock = self.jobs.write().await;
        let job = lock.get_mut(&job_id).ok_or(SupervisorError::JobNotFound(job_id))?;

        if let Some(pid) = job.child_id {
            // SIGTERM on Unix, `taskkill /T /F` on Windows
            sentinel_core::platform::terminate_process(pid);
            job.record.status = JobStatus::Cancelled;
            job.record.finished_at = Some(chrono::Utc::now());
            if let Some(s) = &self.store {
                s.save(&job.record);
            }
        }

        Ok(())
    }

    /// Get current record for a job.
    pub async fn get_job(&self, job_id: Uuid) -> Option<JobRecord> {
        self.jobs.read().await.get(&job_id).map(|j| j.record.clone())
    }

    /// List all job records.
    pub async fn list_jobs(&self) -> Vec<JobRecord> {
        self.jobs.read().await.values().map(|j| j.record.clone()).collect()
    }

    /// Get captured logs for a job.
    pub async fn get_logs(&self, job_id: Uuid) -> Vec<String> {
        self.jobs
            .read()
            .await
            .get(&job_id)
            .map(|j| j.logs.clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sentinel_core::models::RunMode;
    use std::time::Duration;

    #[tokio::test]
    async fn test_supervisor_run_and_stream_logs() {
        let supervisor = ProcessSupervisor::new();
        let spec = JobSpec::new("test-echo", RunMode::Command)
            .with_command("echo 'hello from sentinel' && echo 'second line'");

        let job_id = spec.id;
        let record = supervisor.start_job(spec).await.expect("start job");
        assert_eq!(record.status, JobStatus::Running);

        // Wait a short moment for completion
        tokio::time::sleep(Duration::from_millis(150)).await;

        let finished = supervisor.get_job(job_id).await.expect("find job");
        assert_eq!(finished.status, JobStatus::Completed);
        assert_eq!(finished.exit_code, Some(0));

        let logs = supervisor.get_logs(job_id).await;
        assert!(logs.iter().any(|l| l.contains("hello from sentinel")));
        assert!(logs.iter().any(|l| l.contains("second line")));
    }

    #[tokio::test]
    async fn test_supervisor_terminate_job() {
        let supervisor = ProcessSupervisor::new();
        let spec = JobSpec::new("long-job", RunMode::Command).with_command(idle(10));

        let job_id = spec.id;
        supervisor.start_job(spec).await.expect("start long job");

        tokio::time::sleep(Duration::from_millis(50)).await;
        supervisor.stop_job(job_id).await.expect("stop job");

        let stopped = supervisor.get_job(job_id).await.expect("find job");
        assert_eq!(stopped.status, JobStatus::Cancelled);
    }

    /// A command that idles for about `secs` seconds on any OS.
    fn idle(secs: u32) -> String {
        if cfg!(windows) { format!("ping -n {} 127.0.0.1 >nul", secs + 1) } else { format!("sleep {secs}") }
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("sentinel-jobs-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[tokio::test]
    async fn test_history_survives_restart() {
        let dir = tmp("restart");
        let first = ProcessSupervisor::with_store(JobStore::open(&dir).unwrap());
        let spec = JobSpec::new("keep-me", RunMode::Command).with_command("echo persisted line");
        let id = spec.id;
        first.start_job(spec).await.unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;
        drop(first);

        let second = ProcessSupervisor::with_store(JobStore::open(&dir).unwrap());
        let rec = second.get_job(id).await.expect("job restored");
        assert_eq!(rec.status, JobStatus::Completed);
        assert!(second.get_logs(id).await.iter().any(|l| l.contains("persisted line")));
    }

    #[tokio::test]
    async fn test_running_jobs_become_failed_after_restart() {
        let dir = tmp("orphan");
        let first = ProcessSupervisor::with_store(JobStore::open(&dir).unwrap());
        let spec = JobSpec::new("orphan", RunMode::Command).with_command(idle(30));
        let id = spec.id;
        first.start_job(spec).await.unwrap();
        tokio::time::sleep(Duration::from_millis(80)).await;
        let pid = first.get_job(id).await.unwrap().pid.unwrap();
        drop(first); // simulate the daemon dying; the record on disk still says running

        let second = ProcessSupervisor::with_store(JobStore::open(&dir).unwrap());
        let rec = second.get_job(id).await.unwrap();
        assert_eq!(rec.status, JobStatus::Failed);
        assert!(rec.error_message.unwrap().contains("restarted"));
        sentinel_core::platform::terminate_process(pid);
    }

    #[tokio::test]
    async fn test_stopped_job_stays_cancelled_after_process_exits() {
        let supervisor = ProcessSupervisor::new();
        let spec = JobSpec::new("stop-me", RunMode::Command).with_command(idle(10));
        let id = spec.id;
        supervisor.start_job(spec).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        supervisor.stop_job(id).await.unwrap();
        tokio::time::sleep(Duration::from_millis(400)).await; // let the monitor task observe the exit
        assert_eq!(supervisor.get_job(id).await.unwrap().status, JobStatus::Cancelled);
    }
}
