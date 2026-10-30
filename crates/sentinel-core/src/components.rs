//! Sentinel Component Engine & Custom Component Architecture.

use crate::config::HydraConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ComponentError {
    #[error("Component not found: {0}")]
    NotFound(String),
    #[error("Failed to parse component YAML: {0}")]
    ParseError(#[from] serde_yaml::Error),
    #[error("Component validation failed: {0}")]
    ValidationError(String),
    #[error("Parameter '{0}' is locked by component and cannot be overridden")]
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

/// Sweep configuration in component.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ComponentSweep {
    pub optimizer: Option<String>,
    pub pruner: Option<String>,
    pub max_trials: Option<u32>,
    pub parallel_workers: Option<u32>,
    #[serde(default)]
    pub params: HashMap<String, serde_yaml::Value>,
}

/// Complete declarative Sentinel component.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Component {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub component_id: String,
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
    pub sweep: Option<ComponentSweep>,
    #[serde(default)]
    pub smoke_test: HashMap<String, serde_yaml::Value>,
    /// Optional DAG of blocks; `${param}` placeholders resolve from `parameters`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<serde_yaml::Value>,
}

fn default_schema_version() -> String {
    "1.0".to_string()
}

impl Component {
    pub fn from_yaml(content: &str) -> Result<Self, ComponentError> {
        let t: Self = serde_yaml::from_str(content)?;
        t.validate()?;
        Ok(t)
    }

    pub fn to_yaml(&self) -> Result<String, ComponentError> {
        let yaml = serde_yaml::to_string(self)?;
        Ok(yaml)
    }

    /// Basic structural validation.
    pub fn validate(&self) -> Result<(), ComponentError> {
        if self.component_id.trim().is_empty() {
            return Err(ComponentError::ValidationError(
                "component_id cannot be empty".to_string(),
            ));
        }
        if self.name.trim().is_empty() {
            return Err(ComponentError::ValidationError(
                "name cannot be empty".to_string(),
            ));
        }
        if self.graph.is_some() {
            self.build_graph(&HashMap::new())
                .map_err(|e| ComponentError::ValidationError(format!("graph: {e}")))?;
        }
        Ok(())
    }

    /// Resolve `${param}` placeholders (overrides > locked > defaults) and validate the graph.
    pub fn build_graph(
        &self,
        overrides: &HashMap<String, serde_yaml::Value>,
    ) -> Result<crate::graph::ValidatedGraph, crate::graph::GraphError> {
        let raw = self
            .graph
            .as_ref()
            .ok_or_else(|| crate::graph::GraphError::Parse("component has no graph".into()))?;
        for k in overrides.keys() {
            if !self.parameters.contains_key(k) && !self.locked.contains_key(k) {
                let mut known: Vec<&str> = self.parameters.keys().map(|s| s.as_str()).collect();
                known.sort();
                return Err(crate::graph::GraphError::Parse(format!(
                    "unknown parameter '{k}' for {} (known: {})",
                    self.component_id,
                    known.join(", ")
                )));
            }
        }
        let mut vars = std::collections::BTreeMap::new();
        for (k, spec) in &self.parameters {
            vars.insert(k.clone(), spec.default.clone());
        }
        for (k, v) in &self.locked {
            vars.insert(k.clone(), v.clone());
        }
        for (k, v) in overrides {
            vars.insert(k.clone(), v.clone());
        }
        let resolved = crate::graph::substitute(raw, &vars);
        crate::graph::GraphSpec::from_value(resolved)?.validate()
    }

    /// Validate user overrides against component parameters and locked fields.
    pub fn validate_overrides(
        &self,
        overrides: &HashMap<String, serde_yaml::Value>,
    ) -> Result<(), ComponentError> {
        for (key, val) in overrides {
            // Check if parameter is locked
            if self.locked.contains_key(key) {
                return Err(ComponentError::LockedParameter(key.clone()));
            }

            // Check range if parameter is defined and numeric
            if let Some(spec) = self.parameters.get(key) {
                if let Some((min, max)) = spec.range {
                    if let Some(num) = val.as_f64() {
                        if num < min || num > max {
                            return Err(ComponentError::OutOfRange(key.clone(), num, min, max));
                        }
                    }
                }
                if spec.param_type == ParamType::Choice && !spec.options.is_empty() {
                    if let Some(s) = val.as_str() {
                        if !spec.options.iter().any(|opt| opt == s) {
                            return Err(ComponentError::ValidationError(format!(
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

    /// Resolve effective HydraConfig for a run by applying component defaults, locked fields, and user overrides.
    pub fn resolve_config(
        &self,
        overrides: &HashMap<String, serde_yaml::Value>,
    ) -> Result<HydraConfig, ComponentError> {
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

/// Registry to discover, load, and manage built-in and custom user components.
#[derive(Debug, Default, Clone)]
pub struct ComponentRegistry {
    components: HashMap<String, Component>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            components: HashMap::new(),
        };
        registry.register_builtins();
        registry.register_graph_builtins();
        registry
    }

    /// Register default reference components.
    pub fn register_builtins(&mut self) {
        let vision_yaml = r#"
schema_version: "1.0"
component_id: "vision/image-classification@v1"
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
        if let Ok(t) = Component::from_yaml(vision_yaml) {
            self.components.insert(t.component_id.clone(), t);
        }

        let custom_runner_yaml = r#"
schema_version: "1.0"
component_id: "custom/command-runner@v1"
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
        if let Ok(t) = Component::from_yaml(custom_runner_yaml) {
            self.components.insert(t.component_id.clone(), t);
        }
    }

    /// Graph components shipped inside the binary so every project has the block library.
    pub fn register_graph_builtins(&mut self) {
        const SHIPPED: &[&str] = &[
            include_str!("../../../components/tabular/ann-classifier.yaml"),
            include_str!("../../../components/tabular/residual-mlp.yaml"),
            include_str!("../../../components/tabular/deep-residual-mlp.yaml"),
            include_str!("../../../components/sequence/lstm-classifier.yaml"),
            include_str!("../../../components/sequence/transformer-classifier.yaml"),
            include_str!("../../../components/vision/cnn-classifier.yaml"),
            include_str!("../../../components/graph/gcn-classifier.yaml"),
        ];
        for y in SHIPPED {
            let c = Component::from_yaml(y).expect("shipped graph component must be valid");
            self.components.insert(c.component_id.clone(), c);
        }
    }

    /// Load custom components from a filesystem directory.
    pub fn load_from_dir(&mut self, dir: &Path) -> Result<usize, ComponentError> {
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
                        if let Ok(t) = Component::from_yaml(&content) {
                            self.components.insert(t.component_id.clone(), t);
                            count += 1;
                        }
                    }
                }
            } else if path.is_dir() {
                // Recursively check subdirectories (e.g. components/custom/)
                count += self.load_from_dir(&path)?;
            }
        }
        Ok(count)
    }

    pub fn get(&self, component_id: &str) -> Option<&Component> {
        self.components.get(component_id)
    }

    pub fn list(&self) -> Vec<&Component> {
        let mut list: Vec<&Component> = self.components.values().collect();
        list.sort_by(|a, b| a.component_id.cmp(&b.component_id));
        list
    }

    /// Resolve inheritance if component extends another.
    pub fn resolve_inheritance(&self, component: &Component) -> Result<Component, ComponentError> {
        if let Some(ref parent_id) = component.extends {
            let parent = self
                .get(parent_id)
                .ok_or_else(|| ComponentError::NotFound(parent_id.clone()))?;

            let parent_resolved = self.resolve_inheritance(parent)?;

            let mut merged = parent_resolved.clone();
            merged.component_id = component.component_id.clone();
            merged.name = component.name.clone();
            merged.description = component.description.clone();
            merged.domain = component.domain.clone();
            merged.extends = None;

            // Merge defaults
            for d in &component.defaults {
                merged.defaults.push(d.clone());
            }

            // Merge parameters
            for (k, v) in &component.parameters {
                merged.parameters.insert(k.clone(), v.clone());
            }

            // Merge locked fields
            for (k, v) in &component.locked {
                merged.locked.insert(k.clone(), v.clone());
            }

            if component.sweep.is_some() {
                merged.sweep = component.sweep.clone();
            }

            for (k, v) in &component.smoke_test {
                merged.smoke_test.insert(k.clone(), v.clone());
            }

            if component.graph.is_some() {
                merged.graph = component.graph.clone();
            }

            Ok(merged)
        } else {
            Ok(component.clone())
        }
    }

    /// Generate starter component content for user authoring.
    pub fn create_custom_starter(
        name: &str,
        domain: &str,
        parent_id: Option<&str>,
    ) -> Result<String, ComponentError> {
        let clean_id = name.to_lowercase().replace(' ', "-");
        let component_id = format!("custom/{clean_id}@v1");

        let mut starter = Component {
            schema_version: "1.0".to_string(),
            component_id,
            name: name.to_string(),
            description: format!("Custom component for {name}"),
            domain: domain.to_string(),
            extends: parent_id.map(|s| s.to_string()),
            defaults: vec![serde_yaml::Value::String("_self_".to_string())],
            parameters: HashMap::new(),
            locked: HashMap::new(),
            sweep: None,
            smoke_test: HashMap::new(),
            graph: None,
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

    /// Freeze a winning run into a reusable custom component.
    pub fn save_run_as_component(
        component_name: &str,
        domain: &str,
        resolved_config: &HydraConfig,
    ) -> Result<String, ComponentError> {
        let clean_id = component_name.to_lowercase().replace(' ', "-");
        let component_id = format!("custom/{clean_id}@v1");

        let mut t = Component {
            schema_version: "1.0".to_string(),
            component_id,
            name: component_name.to_string(),
            description: format!("Frozen component captured from run"),
            domain: domain.to_string(),
            extends: None,
            defaults: resolved_config.defaults.clone(),
            parameters: HashMap::new(),
            locked: HashMap::new(),
            sweep: None,
            smoke_test: HashMap::new(),
            graph: None,
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
    fn test_builtin_components_registered() {
        let registry = ComponentRegistry::new();
        assert!(registry.get("vision/image-classification@v1").is_some());
        assert!(registry.get("custom/command-runner@v1").is_some());
    }

    #[test]
    fn test_override_locked_parameter_fails() {
        let registry = ComponentRegistry::new();
        let component = registry
            .get("vision/image-classification@v1")
            .expect("find vision component");

        let mut overrides = HashMap::new();
        overrides.insert(
            "training.precision".to_string(),
            serde_yaml::Value::String("32-true".to_string()),
        );

        let err = component.validate_overrides(&overrides);
        assert!(matches!(err, Err(ComponentError::LockedParameter(_))));
    }

    #[test]
    fn test_override_out_of_range_fails() {
        let registry = ComponentRegistry::new();
        let component = registry
            .get("vision/image-classification@v1")
            .expect("find vision component");

        let mut overrides = HashMap::new();
        overrides.insert(
            "training.lr".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(10.0)),
        ); // max is 0.1

        let err = component.validate_overrides(&overrides);
        assert!(matches!(err, Err(ComponentError::OutOfRange(_, _, _, _))));
    }

    #[test]
    fn test_custom_component_starter_and_resolution() {
        let yaml = ComponentRegistry::create_custom_starter(
            "My Fine Tuner",
            "llm",
            Some("vision/image-classification@v1"),
        )
        .expect("create custom starter");

        let custom = Component::from_yaml(&yaml).expect("parse custom component");
        assert_eq!(custom.component_id, "custom/my-fine-tuner@v1");
        assert_eq!(
            custom.extends,
            Some("vision/image-classification@v1".to_string())
        );

        let registry = ComponentRegistry::new();
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
    fn test_save_run_as_component() {
        let mut config = HydraConfig::default();
        config.defaults.push(serde_yaml::Value::String("model: resnet50".to_string()));
        config.values.insert(
            "training.lr".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(0.0005)),
        );

        let yaml = ComponentRegistry::save_run_as_component("Winning Sweep Run", "vision", &config)
            .expect("save run as component");
        let parsed = Component::from_yaml(&yaml).expect("parse frozen component");
        assert_eq!(parsed.component_id, "custom/winning-sweep-run@v1");
        assert!(parsed.locked.contains_key("training.lr"));
    }
}
