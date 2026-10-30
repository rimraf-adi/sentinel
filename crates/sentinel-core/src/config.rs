//! Hydra / OmegaConf YAML hierarchy and config composition.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A composable Hydra-style configuration file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct HydraConfig {
    #[serde(default)]
    pub defaults: Vec<serde_yaml::Value>,
    #[serde(flatten)]
    pub values: HashMap<String, serde_yaml::Value>,
}

impl HydraConfig {
    pub fn from_yaml(content: &str) -> anyhow::Result<Self> {
        let config: Self = serde_yaml::from_str(content)?;
        Ok(config)
    }

    pub fn to_yaml(&self) -> anyhow::Result<String> {
        let yaml = serde_yaml::to_string(self)?;
        Ok(yaml)
    }

    /// Apply key-value overrides (e.g. `training.lr=0.001` or `model=resnet50`).
    pub fn apply_override(&mut self, dot_path: &str, value: serde_yaml::Value) {
        let parts: Vec<&str> = dot_path.split('.').collect();
        if parts.len() == 1 {
            self.values.insert(parts[0].to_string(), value);
        } else {
            // Traverse or create nested mapping
            let mut current = self
                .values
                .entry(parts[0].to_string())
                .or_insert_with(|| serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));

            for part in &parts[1..parts.len() - 1] {
                if !current.is_mapping() {
                    *current = serde_yaml::Value::Mapping(serde_yaml::Mapping::new());
                }
                let mapping = current.as_mapping_mut().unwrap();
                let key = serde_yaml::Value::String(part.to_string());
                current = mapping
                    .entry(key)
                    .or_insert_with(|| serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
            }

            if let Some(mapping) = current.as_mapping_mut() {
                let final_key = serde_yaml::Value::String(parts.last().unwrap().to_string());
                mapping.insert(final_key, value);
            }
        }
    }

    /// Extract resolved value by dotted path.
    pub fn get_value(&self, dot_path: &str) -> Option<&serde_yaml::Value> {
        let parts: Vec<&str> = dot_path.split('.').collect();
        let mut current = self.values.get(parts[0])?;

        for part in &parts[1..] {
            if let Some(mapping) = current.as_mapping() {
                let key = serde_yaml::Value::String(part.to_string());
                current = mapping.get(&key)?;
            } else {
                return None;
            }
        }

        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hydra_config_parse_and_override() {
        let yaml = r#"
defaults:
  - model: resnet18
  - dataset: cifar10
  - _self_

training:
  lr: 0.0003
  epochs: 20
  batch_size: 64
"#;

        let mut config = HydraConfig::from_yaml(yaml).expect("parse hydra config");
        assert_eq!(config.defaults.len(), 3);

        // Check value
        let lr = config.get_value("training.lr").expect("find training.lr");
        assert_eq!(lr.as_f64(), Some(0.0003));

        // Apply override
        config.apply_override(
            "training.lr",
            serde_yaml::Value::Number(serde_yaml::Number::from(0.001)),
        );
        let updated_lr = config.get_value("training.lr").expect("find updated lr");
        assert_eq!(updated_lr.as_f64(), Some(0.001));

        // Add new nested field
        config.apply_override(
            "training.optimizer.weight_decay",
            serde_yaml::Value::Number(serde_yaml::Number::from(0.01)),
        );
        let wd = config
            .get_value("training.optimizer.weight_decay")
            .expect("find nested wd");
        assert_eq!(wd.as_f64(), Some(0.01));
    }
}
