//! Sentinel Project Generator (`sentinel new` / `create-sentinel-app`).

use std::fs;
use std::path::Path;

pub struct ProjectScaffolder;

impl ProjectScaffolder {
    /// Scaffold a complete Sentinel repository layout.
    pub fn scaffold(target_dir: &Path, project_name: &str) -> anyhow::Result<()> {
        fs::create_dir_all(target_dir)?;

        // Subdirectories
        let subdirs = [
            "configs/model",
            "configs/dataset",
            "tasks",
            "models",
            "datasets",
            "templates/custom",
            "hooks",
            "tests",
        ];

        for sub in &subdirs {
            fs::create_dir_all(target_dir.join(sub))?;
        }

        // 1. sentinel.yaml (Human-configured project settings)
        let sentinel_yaml = format!(
            r#"# Sentinel Project Configuration
schema_version: "1.0"
project_name: "{project_name}"
version: "0.1.0"

defaults:
  framework: "pytorch"
  tracking_backend: "mlflow"
  precision: "16-mixed"

# Hardware and machine defaults
runner:
  default_backend: "local"
  gpus: "auto" # or [0, 1]
"#
        );
        fs::write(target_dir.join("sentinel.yaml"), sentinel_yaml)?;

        // 2. sentinel.lock (Machine-managed versions, hashes, scaffold answers)
        let sentinel_lock = format!(
            r#"# Sentinel Machine Lockfile (Auto-generated)
tool_version: "0.1.0"
template_id: "vision/image-classification@v1"
template_version: 1
created_at: "{}"
answers:
  framework: "pytorch"
  package_manager: "uv"
  tracking: "mlflow"
files:
  sentinel.yaml: {{ owner: "managed" }}
  machines.yaml: {{ owner: "managed" }}
  configs/base.yaml: {{ owner: "scaffolded" }}
  models/example_model.py: {{ owner: "scaffolded" }}
"#,
            chrono::Utc::now().to_rfc3339()
        );
        fs::write(target_dir.join("sentinel.lock"), sentinel_lock)?;

        // 3. machines.yaml (Local and remote machines, no secrets)
        let machines_yaml = r#"# Machine Definitions (Local & Remote SSH)
machines:
  - id: "local"
    name: "Local Workstation"
    host: "127.0.0.1"
    is_local: true
    gpus: "auto"

  # Example remote GPU server:
  # - id: "cloud-gpu-1"
  #   name: "Cloud A100 Box"
  #   host: "192.168.1.100"
  #   user: "ubuntu"
  #   port: 22
  #   is_local: false
"#;
        fs::write(target_dir.join("machines.yaml"), machines_yaml)?;

        // 4. configs/base.yaml (Hydra hierarchical config)
        let base_yaml = r#"# Hydra Base Configuration
defaults:
  - model: resnet18
  - dataset: synthetic
  - _self_

training:
  lr: 0.0003
  epochs: 10
  batch_size: 64
  seed: 42

tracking:
  experiment_name: "default-experiment"
  mlflow_dir: "mlruns"
"#;
        fs::write(target_dir.join("configs/base.yaml"), base_yaml)?;

        // 5. configs/model/resnet18.yaml
        let model_yaml = r#"# Model configuration
name: "resnet18"
pretrained: true
num_classes: 10
"#;
        fs::write(target_dir.join("configs/model/resnet18.yaml"), model_yaml)?;

        // 6. configs/dataset/synthetic.yaml
        let dataset_yaml = r#"# Dataset configuration
name: "synthetic"
num_samples: 128
image_size: [3, 32, 32]
num_classes: 10
"#;
        fs::write(
            target_dir.join("configs/dataset/synthetic.yaml"),
            dataset_yaml,
        )?;

        // 7. models/example_model.py
        let model_py = r#"import torch
import torch.nn as nn

class ExampleModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))
"#;
        fs::write(target_dir.join("models/example_model.py"), model_py)?;

        // 8. pyproject.toml (uv-ready)
        let pyproject_toml = format!(
            r#"[project]
name = "{project_name}"
version = "0.1.0"
description = "Machine learning project built with Sentinel"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "mlflow>=2.10.0",
    "optuna>=3.5.0",
    "pyyaml>=6.0",
]

[tool.uv]
package = false
"#
        );
        fs::write(target_dir.join("pyproject.toml"), pyproject_toml)?;

        // 9. .gitignore
        let gitignore = r#"# Generated project gitignore
.venv/
__pycache__/
*.py[cod]
.sentinel/
mlruns/
.dvc/cache/
.DS_Store
"#;
        fs::write(target_dir.join(".gitignore"), gitignore)?;

        // 10. templates/custom/my_template.yaml
        let custom_template_yaml = r#"schema_version: "1.0"
template_id: "custom/my-experiment@v1"
name: "My Custom Experiment"
description: "Custom recipe scaffolded for this repository"
domain: "custom"
defaults:
  - _self_
parameters:
  training.lr:
    type: float
    default: 0.0003
    range: [0.00001, 0.1]
    help: "Learning rate"
locked:
  tracking.backend: "mlflow"
"#;
        fs::write(
            target_dir.join("templates/custom/my_template.yaml"),
            custom_template_yaml,
        )?;

        Ok(())
    }

    /// Check whether a directory is an initialized Sentinel repository (identifies by `sentinel.lock`).
    pub fn is_sentinel_repo(path: &Path) -> bool {
        path.join("sentinel.lock").exists()
    }

    /// Convert/initialize an existing directory into a Sentinel repository in-place
    /// without overwriting existing user files.
    pub fn init(target_dir: &Path, project_name: &str) -> anyhow::Result<()> {
        fs::create_dir_all(target_dir)?;

        let subdirs = [
            "configs",
            "models",
            "datasets",
            "templates/custom",
            "hooks",
        ];
        for sub in &subdirs {
            let _ = fs::create_dir_all(target_dir.join(sub));
        }

        // 1. Write sentinel.yaml if it does not already exist
        let yaml_path = target_dir.join("sentinel.yaml");
        if !yaml_path.exists() {
            let sentinel_yaml = format!(
                r#"# Sentinel Project Configuration
schema_version: "1.0"
project_name: "{project_name}"
version: "0.1.0"

defaults:
  framework: "pytorch"
  tracking_backend: "mlflow"
  precision: "16-mixed"

runner:
  default_backend: "local"
  gpus: "auto"
"#
            );
            fs::write(yaml_path, sentinel_yaml)?;
        }

        // 2. Write sentinel.lock (The Machine-Managed Identifier Lockfile)
        let lock_path = target_dir.join("sentinel.lock");
        if !lock_path.exists() {
            let sentinel_lock = format!(
                r#"# Sentinel Machine Lockfile (Auto-generated)
tool_version: "0.1.0"
template_id: "custom/starter@v1"
template_version: 1
created_at: "{}"
answers:
  framework: "pytorch"
  package_manager: "uv"
  tracking: "mlflow"
files:
  sentinel.yaml: {{ owner: "managed" }}
  sentinel.lock: {{ owner: "managed" }}
"#,
                chrono::Utc::now().to_rfc3339()
            );
            fs::write(lock_path, sentinel_lock)?;
        }

        // 3. Write machines.yaml if missing
        let machines_path = target_dir.join("machines.yaml");
        if !machines_path.exists() {
            let machines_yaml = r#"# Machine Definitions (Local & Remote SSH)
machines:
  - id: "local"
    name: "Local Workstation"
    host: "127.0.0.1"
    port: 2512
    is_local: true
    gpus: "auto"
"#;
            fs::write(machines_path, machines_yaml)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaffolder_generates_all_files() {
        let temp_dir = std::env::temp_dir().join(format!("sentinel_test_{}", uuid::Uuid::new_v4()));
        ProjectScaffolder::scaffold(&temp_dir, "test-ml-project").expect("scaffold project");

        assert!(temp_dir.join("sentinel.yaml").exists());
        assert!(temp_dir.join("sentinel.lock").exists());
        assert!(temp_dir.join("machines.yaml").exists());
        assert!(temp_dir.join("configs/base.yaml").exists());
        assert!(temp_dir.join("configs/model/resnet18.yaml").exists());
        assert!(temp_dir.join("configs/dataset/synthetic.yaml").exists());
        assert!(temp_dir.join("models/example_model.py").exists());
        assert!(temp_dir.join("templates/custom/my_template.yaml").exists());
        assert!(temp_dir.join("pyproject.toml").exists());
        assert!(temp_dir.join(".gitignore").exists());

        // Validate YAML parsing of generated files
        let sentinel_content =
            fs::read_to_string(temp_dir.join("sentinel.yaml")).expect("read sentinel.yaml");
        let parsed_sentinel: serde_yaml::Value =
            serde_yaml::from_str(&sentinel_content).expect("parse sentinel.yaml");
        assert_eq!(parsed_sentinel["project_name"], "test-ml-project");

        let base_content =
            fs::read_to_string(temp_dir.join("configs/base.yaml")).expect("read base.yaml");
        let parsed_base: serde_yaml::Value =
            serde_yaml::from_str(&base_content).expect("parse base.yaml");
        assert!(parsed_base["training"]["lr"].as_f64().is_some());

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_is_sentinel_repo_and_init() {
        let temp_dir = std::env::temp_dir().join(format!("sentinel_init_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        // 1. Initially, sentinel.lock does not exist
        assert!(!ProjectScaffolder::is_sentinel_repo(&temp_dir));

        // 2. Initialize in-place
        ProjectScaffolder::init(&temp_dir, "my-existing-project").expect("init repo");

        // 3. Now sentinel.lock exists and is recognized as a Sentinel repo
        assert!(ProjectScaffolder::is_sentinel_repo(&temp_dir));
        assert!(temp_dir.join("sentinel.lock").exists());
        assert!(temp_dir.join("sentinel.yaml").exists());

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
