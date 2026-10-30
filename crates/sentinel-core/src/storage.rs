//! Storage drivers and dataset registry (local and external dataset paths).

use crate::models::DatasetSpec;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Default)]
pub struct MlflowStorage {
    pub root_dir: PathBuf,
}

/// Helper to discover, inspect, and manage datasets (supports local and external mounts).
pub struct DatasetRegistry;

impl DatasetRegistry {
    /// Load all datasets registered in `datasets.yaml`, `sentinel.yaml`, or found in `datasets/`
    pub fn load_all(workspace_root: &Path) -> Vec<DatasetSpec> {
        let mut list = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // 1. Check datasets.yaml
        let datasets_yaml_path = workspace_root.join("datasets.yaml");
        if datasets_yaml_path.exists() {
            if let Ok(content) = fs::read_to_string(&datasets_yaml_path) {
                if let Ok(val) = serde_yaml::from_str::<serde_yaml::Value>(&content) {
                    if let Some(items) = val.get("datasets").and_then(|d| d.as_sequence()) {
                        for item in items {
                            if let Ok(mut spec) = serde_yaml::from_value::<DatasetSpec>(item.clone()) {
                                spec.inspect(workspace_root);
                                seen.insert(spec.name.clone());
                                list.push(spec);
                            }
                        }
                    }
                }
            }
        }

        // 2. Check sentinel.yaml for datasets section
        let sentinel_yaml_path = workspace_root.join("sentinel.yaml");
        if sentinel_yaml_path.exists() {
            if let Ok(content) = fs::read_to_string(&sentinel_yaml_path) {
                if let Ok(val) = serde_yaml::from_str::<serde_yaml::Value>(&content) {
                    if let Some(items) = val.get("datasets").and_then(|d| d.as_sequence()) {
                        for item in items {
                            if let Ok(mut spec) = serde_yaml::from_value::<DatasetSpec>(item.clone()) {
                                if !seen.contains(&spec.name) {
                                    spec.inspect(workspace_root);
                                    seen.insert(spec.name.clone());
                                    list.push(spec);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Scan local repo datasets/ folder for any subfolders, files, or symlinks
        let local_dir = workspace_root.join("datasets");
        if local_dir.exists() && local_dir.is_dir() {
            if let Ok(entries) = fs::read_dir(&local_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with('.') || seen.contains(&name) {
                        continue;
                    }
                    let p = entry.path();
                    let format = if p.is_dir() { "folder" } else {
                        p.extension().and_then(|ext| ext.to_str()).unwrap_or("file")
                    };
                    let mut spec = DatasetSpec::new(&name, format!("datasets/{}", name), format);
                    spec.inspect(workspace_root);
                    seen.insert(name);
                    list.push(spec);
                }
            }
        }

        list
    }

    /// Add or update a dataset in `datasets.yaml`
    pub fn add_dataset(workspace_root: &Path, spec: DatasetSpec) -> anyhow::Result<()> {
        let datasets_yaml_path = workspace_root.join("datasets.yaml");
        let mut list = Self::load_all(workspace_root);

        // Remove existing with same name if updating
        list.retain(|d| d.name != spec.name);
        list.push(spec);

        let yaml = serde_yaml::to_string(&serde_json::json!({
            "datasets": list
        }))?;
        fs::write(datasets_yaml_path, yaml)?;
        Ok(())
    }

    /// Remove a dataset from `datasets.yaml`
    pub fn remove_dataset(workspace_root: &Path, name: &str) -> anyhow::Result<()> {
        let datasets_yaml_path = workspace_root.join("datasets.yaml");
        let mut list = Self::load_all(workspace_root);
        list.retain(|d| d.name != name);

        let yaml = serde_yaml::to_string(&serde_json::json!({
            "datasets": list
        }))?;
        fs::write(datasets_yaml_path, yaml)?;
        Ok(())
    }
}
