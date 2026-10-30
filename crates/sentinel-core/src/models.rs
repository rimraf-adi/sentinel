//! Core domain models for Sentinel.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// How a job is executed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    #[default]
    Command,
    Sdk,
    Framework,
}

/// Lifecycle status of a training job.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    #[default]
    Pending,
    Running,
    Completed,
    Failed,
    Pruned,
    Cancelled,
}

/// Execution specification for a single training job or trial.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobSpec {
    pub id: Uuid,
    pub name: String,
    pub run_mode: RunMode,
    pub command: Option<String>,
    pub template_id: Option<String>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub env_vars: HashMap<String, String>,
    pub gpu_ids: Vec<u32>,
    pub working_dir: String,
    pub created_at: DateTime<Utc>,
}

impl JobSpec {
    pub fn new(name: impl Into<String>, run_mode: RunMode) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            run_mode,
            command: None,
            template_id: None,
            parameters: HashMap::new(),
            env_vars: HashMap::new(),
            gpu_ids: Vec::new(),
            working_dir: ".".to_string(),
            created_at: Utc::now(),
        }
    }

    pub fn with_command(mut self, cmd: impl Into<String>) -> Self {
        self.command = Some(cmd.into());
        self
    }

    pub fn with_template(mut self, template_id: impl Into<String>) -> Self {
        self.template_id = Some(template_id.into());
        self
    }

    pub fn with_gpus(mut self, gpus: Vec<u32>) -> Self {
        self.gpu_ids = gpus;
        self
    }
}

/// State of a running or historical job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobRecord {
    pub spec: JobSpec,
    pub status: JobStatus,
    pub pid: Option<u32>,
    pub exit_code: Option<i32>,
    pub error_message: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl JobRecord {
    pub fn new(spec: JobSpec) -> Self {
        Self {
            spec,
            status: JobStatus::Pending,
            pid: None,
            exit_code: None,
            error_message: None,
            started_at: None,
            finished_at: None,
        }
    }
}

/// Hardware telemetry point (GPU, CPU, Memory).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareStats {
    pub timestamp: DateTime<Utc>,
    pub machine_id: String,
    pub cpu_util_percent: f32,
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub gpus: Vec<GpuStats>,
}

/// Per-GPU telemetry metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuStats {
    pub index: u32,
    pub name: String,
    pub vram_used_bytes: u64,
    pub vram_total_bytes: u64,
    pub gpu_util_percent: f32,
    pub temperature_c: f32,
}

/// Single metric point logged during training.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricPoint {
    pub run_id: String,
    pub key: String,
    pub value: f64,
    pub step: u64,
    pub epoch: Option<u64>,
    pub timestamp: DateTime<Utc>,
}

impl MetricPoint {
    pub fn new(run_id: impl Into<String>, key: impl Into<String>, value: f64, step: u64) -> Self {
        Self {
            run_id: run_id.into(),
            key: key.into(),
            value,
            step,
            epoch: None,
            timestamp: Utc::now(),
        }
    }
}

/// Target machine specification for local or remote SSH execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MachineSpec {
    pub id: String,
    pub name: String,
    pub host: String,
    pub user: Option<String>,
    pub port: u16,
    pub is_local: bool,
    pub agent_connected: bool,
}

impl MachineSpec {
    pub fn local() -> Self {
        Self {
            id: "local".to_string(),
            name: "Local Machine".to_string(),
            host: "127.0.0.1".to_string(),
            user: None,
            port: 8080,
            is_local: true,
            agent_connected: true,
        }
    }

    pub fn remote(id: impl Into<String>, name: impl Into<String>, host: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            host: host.into(),
            user: Some("ubuntu".to_string()),
            port: 22,
            is_local: false,
            agent_connected: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_spec_serde_roundtrip() {
        let mut spec = JobSpec::new("cifar10-baseline", RunMode::Framework)
            .with_template("vision/image-classification@v1")
            .with_gpus(vec![0, 1]);
        spec.parameters
            .insert("lr".to_string(), serde_json::json!(0.0003));
        spec.env_vars
            .insert("CUDA_VISIBLE_DEVICES".to_string(), "0,1".to_string());

        let json = serde_json::to_string(&spec).expect("serialize JSON");
        let parsed: JobSpec = serde_json::from_str(&json).expect("deserialize JSON");
        assert_eq!(spec.id, parsed.id);
        assert_eq!(spec.name, parsed.name);
        assert_eq!(spec.gpu_ids, vec![0, 1]);

        let yaml = serde_yaml::to_string(&spec).expect("serialize YAML");
        let parsed_yaml: JobSpec = serde_yaml::from_str(&yaml).expect("deserialize YAML");
        assert_eq!(spec.name, parsed_yaml.name);
    }

    #[test]
    fn test_metric_point_creation() {
        let point = MetricPoint::new("run-123", "train/loss", 0.425, 100);
        assert_eq!(point.key, "train/loss");
        assert_eq!(point.value, 0.425);
        assert_eq!(point.step, 100);
    }

    #[test]
    fn test_machine_spec_defaults() {
        let local = MachineSpec::local();
        assert!(local.is_local);
        assert!(local.agent_connected);

        let remote = MachineSpec::remote("gpu-box-1", "Cloud A100", "192.168.1.50");
        assert!(!remote.is_local);
        assert_eq!(remote.port, 22);
    }
}
