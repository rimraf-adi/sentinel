//! Process supervisor: launches, monitors, streams logs, and terminates jobs.

use sentinel_core::models::{JobRecord, JobSpec, JobStatus};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
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
}

impl ProcessSupervisor {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
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
                "python -c 'print(\"running sentinel default worker\")'".to_string()
            }
        };

        // Prepare shell execution
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(&cmd_string);
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

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let jobs_ref = self.jobs.clone();

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
                                    job.logs.push(format!("[stdout] {text}"));
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
                                    job.logs.push(format!("[stderr] {text}"));
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
                                job.logs.push(format!("[stdout] {line}"));
                            }
                        }

                        while let Some(line) = if let Some(ref mut reader) = stderr_lines {
                            reader.next_line().await.ok().flatten()
                        } else {
                            None
                        } {
                            let mut lock = jobs_ref.write().await;
                            if let Some(job) = lock.get_mut(&job_id) {
                                job.logs.push(format!("[stderr] {line}"));
                            }
                        }

                        let mut lock = jobs_ref.write().await;
                        if let Some(job) = lock.get_mut(&job_id) {
                            job.record.finished_at = Some(chrono::Utc::now());
                            match status {
                                Ok(exit_status) => {
                                    job.record.exit_code = exit_status.code();
                                    if exit_status.success() {
                                        job.record.status = JobStatus::Completed;
                                    } else {
                                        job.record.status = JobStatus::Failed;
                                        job.record.error_message = Some(format!("Process exited with status: {exit_status}"));
                                    }
                                }
                                Err(err) => {
                                    job.record.status = JobStatus::Failed;
                                    job.record.error_message = Some(err.to_string());
                                }
                            }
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
                            job.record.status = if exit_status.success() {
                                JobStatus::Completed
                            } else {
                                JobStatus::Failed
                            };
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
            // Send SIGTERM to process
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
            job.record.status = JobStatus::Cancelled;
            job.record.finished_at = Some(chrono::Utc::now());
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
        let spec = JobSpec::new("long-job", RunMode::Command).with_command("sleep 10");

        let job_id = spec.id;
        supervisor.start_job(spec).await.expect("start long job");

        tokio::time::sleep(Duration::from_millis(50)).await;
        supervisor.stop_job(job_id).await.expect("stop job");

        let stopped = supervisor.get_job(job_id).await.expect("find job");
        assert_eq!(stopped.status, JobStatus::Cancelled);
    }
}
