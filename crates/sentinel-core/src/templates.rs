//! Sentinel Template Engine & Custom Template Architecture.

use crate::config::HydraConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TemplateError {
    #[error("Template not found: {0}")]
    NotFound(String),
    #[error("Failed to parse template YAML: {0}")]
    ParseError(#[from] serde_yaml::Error),
    #[error("Template validation failed: {0}")]
    ValidationError(String),
    #[error("Parameter '{0}' is locked by template and cannot be overridden")]
    LockedParameter(String),
    #[error("Parameter '{0}' value {1} out of allowed range [{2}, {3}]")]
    OutOfRange(String, f64, f64, f64),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Parameter data type definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParamType {
    String,
    Int,
    Float,
    Bool,
    Choice,
}

/// A single exposed parameter specification for UI and CLI form generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParamSpec {
    #[serde(rename = "type")]
    pub param_type: ParamType,
    pub default: serde_yaml::Value,
    #[serde(default)]
    pub options: Vec<String>,
    #[serde(default)]
    pub range: Option<(f64, f64)>,
    #[serde(default)]
    pub help: String,
}

/// Sweep configuration in template.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TemplateSweep {
    pub optimizer: Option<String>,
    pub pruner: Option<String>,
    pub max_trials: Option<u32>,
    pub parallel_workers: Option<u32>,
    #[serde(default)]
    pub params: HashMap<String, serde_yaml::Value>,
}

/// Complete declarative Sentinel template.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Template {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub template_id: String,
    pub name: String,
    pub description: String,
    pub domain: String,
    pub extends: Option<String>,
    #[serde(default)]
    pub defaults: Vec<serde_yaml::Value>,
    #[serde(default)]
    pub parameters: HashMap<String, ParamSpec>,
    #[serde(default)]
    pub locked: HashMap<String, serde_yaml::Value>,
    #[serde(default)]
    pub sweep: Option<TemplateSweep>,
    #[serde(default)]
    pub smoke_test: HashMap<String, serde_yaml::Value>,
}

fn default_schema_version() -> String {
    "1.0".to_string()
}

impl Template {
    pub fn from_yaml(content: &str) -> Result<Self, TemplateError> {
        let t: Self = serde_yaml::from_str(content)?;
        t.validate()?;
        Ok(t)
    }

    pub fn to_yaml(&self) -> Result<String, TemplateError> {
        let yaml = serde_yaml::to_string(self)?;
        Ok(yaml)
    }

    /// Basic structural validation.
    pub fn validate(&self) -> Result<(), TemplateError> {
        if self.template_id.trim().is_empty() {
            return Err(TemplateError::ValidationError(
                "template_id cannot be empty".to_string(),
            ));
        }
        if self.name.trim().is_empty() {
            return Err(TemplateError::ValidationError(
                "name cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate user overrides against template parameters and locked fields.
    pub fn validate_overrides(
        &self,
        overrides: &HashMap<String, serde_yaml::Value>,
    ) -> Result<(), TemplateError> {
        for (key, val) in overrides {
            // Check if parameter is locked
            if self.locked.contains_key(key) {
                return Err(TemplateError::LockedParameter(key.clone()));
            }

            // Check range if parameter is defined and numeric
            if let Some(spec) = self.parameters.get(key) {
                if let Some((min, max)) = spec.range {
                    if let Some(num) = val.as_f64() {
                        if num < min || num > max {
                            return Err(TemplateError::OutOfRange(key.clone(), num, min, max));
                        }
                    }
                }
                if spec.param_type == ParamType::Choice && !spec.options.is_empty() {
                    if let Some(s) = val.as_str() {
                        if !spec.options.iter().any(|opt| opt == s) {
                            return Err(TemplateError::ValidationError(format!(
                                "Value '{s}' is not a valid option for parameter '{key}'. Allowed: {:?}",
                                spec.options
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Resolve effective HydraConfig for a run by applying template defaults, locked fields, and user overrides.
    pub fn resolve_config(
        &self,
        overrides: &HashMap<String, serde_yaml::Value>,
    ) -> Result<HydraConfig, TemplateError> {
        self.validate_overrides(overrides)?;

        let mut config = HydraConfig {
            defaults: self.defaults.clone(),
            values: HashMap::new(),
        };

        // 1. Insert default parameter values
        for (k, spec) in &self.parameters {
            config.apply_override(k, spec.default.clone());
        }

        // 2. Insert locked fields
        for (k, val) in &self.locked {
            config.apply_override(k, val.clone());
        }

        // 3. Insert user overrides
        for (k, val) in overrides {
            config.apply_override(k, val.clone());
        }

        Ok(config)
    }
}

/// Registry to discover, load, and manage built-in and custom user templates.
#[derive(Debug, Default)]
pub struct TemplateRegistry {
    templates: HashMap<String, Template>,
}

impl TemplateRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            templates: HashMap::new(),
        };
        registry.register_builtins();
        registry
    }

    /// Register default reference templates.
    pub fn register_builtins(&mut self) {
        let vision_yaml = r#"
schema_version: "1.0"
template_id: "vision/image-classification@v1"
name: "Vision Classification"
description: "Image classification with ResNet/ViT, AMP, and cosine scheduling"
domain: "vision"
defaults:
  - model: resnet18
  - dataset: cifar10
  - _self_
parameters:
  model.name:
    type: choice
    default: "resnet18"
    options: ["resnet18", "resnet50", "vit_b_16"]
    help: "Model architecture"
  training.lr:
    type: float
    default: 0.0003
    range: [0.00001, 0.1]
    help: "Peak learning rate"
  training.batch_size:
    type: int
    default: 64
    range: [8, 512]
    help: "Batch size"
  training.epochs:
    type: int
    default: 20
    range: [1, 500]
    help: "Total epochs"
locked:
  training.precision: "16-mixed"
  tracking.backend: "mlflow"
sweep:
  optimizer: "optuna"
  pruner: "asha"
  max_trials: 20
  parallel_workers: 2
  params:
    training.lr:
      type: "float_log"
      min: 0.00001
      max: 0.01
smoke_test:
  training.epochs: 1
  training.limit_train_batches: 5
  training.limit_val_batches: 2
  device: "cpu"
"#;
        if let Ok(t) = Template::from_yaml(vision_yaml) {
            self.templates.insert(t.template_id.clone(), t);
        }

        let custom_runner_yaml = r#"
schema_version: "1.0"
template_id: "custom/command-runner@v1"
name: "Custom Command Runner"
description: "Zero-touch adapter for existing scripts like 'python train.py'"
domain: "custom"
defaults: []
parameters:
  command:
    type: string
    default: "python train.py"
    help: "Command to execute"
locked:
  tracking.backend: "mlflow"
smoke_test:
  command: "python -c 'print(\"smoke test passed\")'"
"#;
        if let Ok(t) = Template::from_yaml(custom_runner_yaml) {
            self.templates.insert(t.template_id.clone(), t);
        }
    }

    /// Load custom templates from a filesystem directory.
    pub fn load_from_dir(&mut self, dir: &Path) -> Result<usize, TemplateError> {
        if !dir.exists() || !dir.is_dir() {
            return Ok(0);
        }

        let mut count = 0;
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "yaml" || ext == "yml" {
                        let content = std::fs::read_to_string(&path)?;
                        if let Ok(t) = Template::from_yaml(&content) {
                            self.templates.insert(t.template_id.clone(), t);
                            count += 1;
                        }
                    }
                }
            } else if path.is_dir() {
                // Recursively check subdirectories (e.g. templates/custom/)
                count += self.load_from_dir(&path)?;
            }
        }
        Ok(count)
    }

    pub fn get(&self, template_id: &str) -> Option<&Template> {
        self.templates.get(template_id)
    }

    pub fn list(&self) -> Vec<&Template> {
        let mut list: Vec<&Template> = self.templates.values().collect();
        list.sort_by(|a, b| a.template_id.cmp(&b.template_id));
        list
    }

    /// Resolve inheritance if template extends another.
    pub fn resolve_inheritance(&self, template: &Template) -> Result<Template, TemplateError> {
        if let Some(ref parent_id) = template.extends {
            let parent = self
                .get(parent_id)
                .ok_or_else(|| TemplateError::NotFound(parent_id.clone()))?;

            let parent_resolved = self.resolve_inheritance(parent)?;

            let mut merged = parent_resolved.clone();
            merged.template_id = template.template_id.clone();
            merged.name = template.name.clone();
            merged.description = template.description.clone();
            merged.domain = template.domain.clone();
            merged.extends = None;

            // Merge defaults
            for d in &template.defaults {
                merged.defaults.push(d.clone());
            }

            // Merge parameters
            for (k, v) in &template.parameters {
                merged.parameters.insert(k.clone(), v.clone());
            }

            // Merge locked fields
            for (k, v) in &template.locked {
                merged.locked.insert(k.clone(), v.clone());
            }

            if template.sweep.is_some() {
                merged.sweep = template.sweep.clone();
            }

            for (k, v) in &template.smoke_test {
                merged.smoke_test.insert(k.clone(), v.clone());
            }

            Ok(merged)
        } else {
            Ok(template.clone())
        }
    }

    /// Generate starter template content for user authoring.
    pub fn create_custom_starter(
        name: &str,
        domain: &str,
        parent_id: Option<&str>,
    ) -> Result<String, TemplateError> {
        let clean_id = name.to_lowercase().replace(' ', "-");
        let template_id = format!("custom/{clean_id}@v1");

        let mut starter = Template {
            schema_version: "1.0".to_string(),
            template_id,
            name: name.to_string(),
            description: format!("Custom template for {name}"),
            domain: domain.to_string(),
            extends: parent_id.map(|s| s.to_string()),
            defaults: vec![serde_yaml::Value::String("_self_".to_string())],
            parameters: HashMap::new(),
            locked: HashMap::new(),
            sweep: None,
            smoke_test: HashMap::new(),
        };

        starter.parameters.insert(
            "training.lr".to_string(),
            ParamSpec {
                param_type: ParamType::Float,
                default: serde_yaml::Value::Number(serde_yaml::Number::from(0.0001)),
                options: Vec::new(),
                range: Some((0.000001, 1.0)),
                help: "Learning rate".to_string(),
            },
        );

        starter.to_yaml()
    }

    /// Freeze a winning run into a reusable custom template.
    pub fn save_run_as_template(
        template_name: &str,
        domain: &str,
        resolved_config: &HydraConfig,
    ) -> Result<String, TemplateError> {
        let clean_id = template_name.to_lowercase().replace(' ', "-");
        let template_id = format!("custom/{clean_id}@v1");

        let mut t = Template {
            schema_version: "1.0".to_string(),
            template_id,
            name: template_name.to_string(),
            description: format!("Frozen template captured from run"),
            domain: domain.to_string(),
            extends: None,
            defaults: resolved_config.defaults.clone(),
            parameters: HashMap::new(),
            locked: HashMap::new(),
            sweep: None,
            smoke_test: HashMap::new(),
        };

        for (k, v) in &resolved_config.values {
            t.locked.insert(k.clone(), v.clone());
        }

        t.to_yaml()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_templates_registered() {
        let registry = TemplateRegistry::new();
        assert!(registry.get("vision/image-classification@v1").is_some());
        assert!(registry.get("custom/command-runner@v1").is_some());
    }

    #[test]
    fn test_override_locked_parameter_fails() {
        let registry = TemplateRegistry::new();
        let template = registry
            .get("vision/image-classification@v1")
            .expect("find vision template");

        let mut overrides = HashMap::new();
        overrides.insert(
            "training.precision".to_string(),
            serde_yaml::Value::String("32-true".to_string()),
        );

        let err = template.validate_overrides(&overrides);
        assert!(matches!(err, Err(TemplateError::LockedParameter(_))));
    }

    #[test]
    fn test_override_out_of_range_fails() {
        let registry = TemplateRegistry::new();
        let template = registry
            .get("vision/image-classification@v1")
            .expect("find vision template");

        let mut overrides = HashMap::new();
        overrides.insert(
            "training.lr".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(10.0)),
        ); // max is 0.1

        let err = template.validate_overrides(&overrides);
        assert!(matches!(err, Err(TemplateError::OutOfRange(_, _, _, _))));
    }

    #[test]
    fn test_custom_template_starter_and_resolution() {
        let yaml = TemplateRegistry::create_custom_starter(
            "My Fine Tuner",
            "llm",
            Some("vision/image-classification@v1"),
        )
        .expect("create custom starter");

        let custom = Template::from_yaml(&yaml).expect("parse custom template");
        assert_eq!(custom.template_id, "custom/my-fine-tuner@v1");
        assert_eq!(
            custom.extends,
            Some("vision/image-classification@v1".to_string())
        );

        let registry = TemplateRegistry::new();
        let resolved = registry
            .resolve_inheritance(&custom)
            .expect("resolve inheritance");
        assert_eq!(resolved.extends, None);
        // Inherited parameters from parent
        assert!(resolved.parameters.contains_key("model.name"));
        // Has its own parameters
        assert!(resolved.parameters.contains_key("training.lr"));
    }

    #[test]
    fn test_save_run_as_template() {
        let mut config = HydraConfig::default();
        config.defaults.push(serde_yaml::Value::String("model: resnet50".to_string()));
        config.values.insert(
            "training.lr".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(0.0005)),
        );

        let yaml = TemplateRegistry::save_run_as_template("Winning Sweep Run", "vision", &config)
            .expect("save run as template");
        let parsed = Template::from_yaml(&yaml).expect("parse frozen template");
        assert_eq!(parsed.template_id, "custom/winning-sweep-run@v1");
        assert!(parsed.locked.contains_key("training.lr"));
    }
}
