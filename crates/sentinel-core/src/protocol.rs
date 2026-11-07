use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const DEFAULT_SPACE: &str = "https://raw.githubusercontent.com/rimraf-adi/sentinel-hub/main";
pub const PROTOCOL_FILENAME: &str = "sentinel.protocol.yaml";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntegrationConfig {
    #[serde(default)]
    pub params: HashMap<String, serde_yaml::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntegrationSpec {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_source")]
    pub source: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub config: HashMap<String, serde_yaml::Value>,
}

fn default_version() -> String {
    "1.0.0".to_string()
}

fn default_source() -> String {
    "sentinel-hub".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicySpec {
    #[serde(default)]
    pub min_representation_rank: Option<f64>,
    #[serde(default)]
    pub max_gradient_norm: Option<f64>,
    #[serde(default)]
    pub enforce_remediation: Option<bool>,
}

impl Default for PolicySpec {
    fn default() -> Self {
        Self {
            min_representation_rank: Some(0.25),
            max_gradient_norm: Some(10.0),
            enforce_remediation: Some(true),
        }
    }
}

/// The declarative Sentinel Protocol manifest defining active spaces,
/// dynamic capabilities/integrations, and organizational policies.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProtocolManifest {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    #[serde(default = "default_space_url")]
    pub space: String,
    #[serde(default = "default_org")]
    pub org: String,
    #[serde(default)]
    pub integrations: Vec<IntegrationSpec>,
    #[serde(default)]
    pub policies: PolicySpec,
}

fn default_schema_version() -> String {
    "1.0".to_string()
}

fn default_space_url() -> String {
    DEFAULT_SPACE.to_string()
}

fn default_org() -> String {
    "community".to_string()
}

impl Default for ProtocolManifest {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            space: default_space_url(),
            org: default_org(),
            integrations: vec![
                IntegrationSpec {
                    name: "xai".to_string(),
                    version: "1.0.0".to_string(),
                    source: "sentinel-hub".to_string(),
                    description: Some("Axiomatic Integrated Gradients, SVD Rank, Loss Landscape & Remediation".to_string()),
                    enabled: true,
                    config: HashMap::new(),
                },
                IntegrationSpec {
                    name: "gradient-flow".to_string(),
                    version: "1.0.0".to_string(),
                    source: "sentinel-hub".to_string(),
                    description: Some("Live forward and backward gradient flow inspection".to_string()),
                    enabled: true,
                    config: HashMap::new(),
                },
            ],
            policies: PolicySpec::default(),
        }
    }
}

impl ProtocolManifest {
    /// Locate `sentinel.protocol.yaml` in `root` or `.sentinel/`, or fallback to default.
    pub fn load_or_default(root: &Path) -> Self {
        let direct = root.join(PROTOCOL_FILENAME);
        if direct.exists() {
            if let Ok(content) = std::fs::read_to_string(&direct) {
                if let Ok(manifest) = serde_yaml::from_str::<ProtocolManifest>(&content) {
                    return manifest;
                }
            }
        }

        let dot_sentinel = root.join(".sentinel").join(PROTOCOL_FILENAME);
        if dot_sentinel.exists() {
            if let Ok(content) = std::fs::read_to_string(&dot_sentinel) {
                if let Ok(manifest) = serde_yaml::from_str::<ProtocolManifest>(&content) {
                    return manifest;
                }
            }
        }

        Self::default()
    }

    /// Save the protocol manifest to `root/sentinel.protocol.yaml`.
    pub fn save(&self, root: &Path) -> Result<PathBuf> {
        let dest = root.join(PROTOCOL_FILENAME);
        let serialized = serde_yaml::to_string(self)
            .context("failed to serialize protocol manifest to YAML")?;
        std::fs::write(&dest, serialized)
            .with_context(|| format!("failed to write {}", dest.display()))?;
        Ok(dest)
    }

    /// Check if a specific integration is active and enabled.
    pub fn has_integration(&self, name: &str) -> bool {
        self.integrations
            .iter()
            .any(|i| i.name.eq_ignore_ascii_case(name) && i.enabled)
    }

    /// Enable or add an integration.
    pub fn add_integration(&mut self, spec: IntegrationSpec) {
        if let Some(existing) = self.integrations.iter_mut().find(|i| i.name.eq_ignore_ascii_case(&spec.name)) {
            existing.enabled = true;
            existing.version = spec.version;
            if spec.description.is_some() {
                existing.description = spec.description;
            }
        } else {
            self.integrations.push(spec);
        }
    }

    /// Remove or disable an integration.
    pub fn remove_integration(&mut self, name: &str) -> bool {
        let prev_len = self.integrations.len();
        self.integrations.retain(|i| !i.name.eq_ignore_ascii_case(name));
        self.integrations.len() < prev_len
    }

    /// Toggle enabled status of an integration.
    pub fn toggle_integration(&mut self, name: &str) -> Option<bool> {
        if let Some(i) = self.integrations.iter_mut().find(|i| i.name.eq_ignore_ascii_case(name)) {
            i.enabled = !i.enabled;
            Some(i.enabled)
        } else {
            None
        }
    }
}
