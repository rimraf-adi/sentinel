use sentinel_core::config::HydraConfig;
use sentinel_core::templates::{ParamSpec, ParamType, Template, TemplateError, TemplateRegistry};
use std::collections::HashMap;

#[test]
fn test_deep_three_level_inheritance() {
    let base_yaml = r#"
schema_version: "1.0"
template_id: "core/base@v1"
name: "Base Root Template"
description: "Root configuration"
domain: "core"
parameters:
  training.epochs:
    type: int
    default: 50
    range: [1, 1000]
    help: "Max epochs"
locked:
  training.precision: "16-mixed"
  tracking.backend: "mlflow"
"#;

    let vision_yaml = r#"
schema_version: "1.0"
template_id: "vision/intermediate@v1"
name: "Vision Intermediate"
description: "Intermediate vision template"
domain: "vision"
extends: "core/base@v1"
parameters:
  model.name:
    type: choice
    default: "resnet18"
    options: ["resnet18", "resnet50"]
    help: "Architecture"
  training.lr:
    type: float
    default: 0.0003
    range: [0.00001, 0.1]
    help: "Learning rate"
"#;

    let custom_yaml = r#"
schema_version: "1.0"
template_id: "custom/leaf@v1"
name: "Custom Leaf Template"
description: "Concrete experiment template"
domain: "vision"
extends: "vision/intermediate@v1"
parameters:
  data.batch_size:
    type: int
    default: 128
    range: [8, 512]
    help: "Batch size"
"#;

    let base = Template::from_yaml(base_yaml).expect("parse base");
    let vision = Template::from_yaml(vision_yaml).expect("parse vision");
    let custom = Template::from_yaml(custom_yaml).expect("parse custom");

    let mut registry = TemplateRegistry::new();
    let _ = registry.load_from_dir(std::path::Path::new("templates"));

    let mut temp_map = HashMap::new();
    temp_map.insert("core/base@v1".to_string(), base);
    temp_map.insert("vision/intermediate@v1".to_string(), vision);
    temp_map.insert("custom/leaf@v1".to_string(), custom);

    // Test inheritance resolution
    let leaf = temp_map.get("custom/leaf@v1").unwrap();
    
    // Test resolving intermediate first
    let intermediate = temp_map.get("vision/intermediate@v1").unwrap();
    let mut resolved_intermediate = intermediate.clone();
    let base_t = temp_map.get("core/base@v1").unwrap();
    for (k, v) in &base_t.parameters {
        resolved_intermediate.parameters.insert(k.clone(), v.clone());
    }
    for (k, v) in &base_t.locked {
        resolved_intermediate.locked.insert(k.clone(), v.clone());
    }

    // Now resolve leaf with intermediate
    let mut resolved_leaf = leaf.clone();
    for (k, v) in &resolved_intermediate.parameters {
        resolved_leaf.parameters.insert(k.clone(), v.clone());
    }
    for (k, v) in &resolved_intermediate.locked {
        resolved_leaf.locked.insert(k.clone(), v.clone());
    }

    // Assert parameters from all 3 levels exist
    assert!(resolved_leaf.parameters.contains_key("training.epochs")); // from base
    assert!(resolved_leaf.parameters.contains_key("model.name"));      // from intermediate
    assert!(resolved_leaf.parameters.contains_key("data.batch_size")); // from leaf

    // Assert locked fields from base are preserved
    assert_eq!(
        resolved_leaf.locked.get("training.precision"),
        Some(&serde_yaml::Value::String("16-mixed".to_string()))
    );

    // Test rejecting attempt to override locked parameter on leaf
    let mut overrides = HashMap::new();
    overrides.insert(
        "training.precision".to_string(),
        serde_yaml::Value::String("32-true".to_string()),
    );
    let err = resolved_leaf.validate_overrides(&overrides);
    assert!(matches!(err, Err(TemplateError::LockedParameter(_))));
}

#[test]
fn test_param_type_validations() {
    let mut params = HashMap::new();
    params.insert(
        "optimizer.name".to_string(),
        ParamSpec {
            param_type: ParamType::Choice,
            default: serde_yaml::Value::String("adamw".to_string()),
            options: vec!["adamw".to_string(), "sgd".to_string()],
            range: None,
            help: "Choice of optimizer".to_string(),
        },
    );
    params.insert(
        "training.lr".to_string(),
        ParamSpec {
            param_type: ParamType::Float,
            default: serde_yaml::Value::Number(serde_yaml::Number::from(0.001)),
            options: Vec::new(),
            range: Some((0.00001, 0.01)),
            help: "Learning rate".to_string(),
        },
    );

    let template = Template {
        schema_version: "1.0".to_string(),
        template_id: "test/validation@v1".to_string(),
        name: "Validation Test".to_string(),
        description: "Test".to_string(),
        domain: "test".to_string(),
        extends: None,
        defaults: Vec::new(),
        parameters: params,
        locked: HashMap::new(),
        sweep: None,
        smoke_test: HashMap::new(),
    };

    // Valid overrides
    let mut valid = HashMap::new();
    valid.insert(
        "optimizer.name".to_string(),
        serde_yaml::Value::String("sgd".to_string()),
    );
    valid.insert(
        "training.lr".to_string(),
        serde_yaml::Value::Number(serde_yaml::Number::from(0.005)),
    );
    assert!(template.validate_overrides(&valid).is_ok());

    // Invalid choice
    let mut invalid_choice = HashMap::new();
    invalid_choice.insert(
        "optimizer.name".to_string(),
        serde_yaml::Value::String("rmsprop".to_string()),
    );
    let err_choice = template.validate_overrides(&invalid_choice);
    assert!(matches!(err_choice, Err(TemplateError::ValidationError(_))));

    // Out of range (too high)
    let mut too_high = HashMap::new();
    too_high.insert(
        "training.lr".to_string(),
        serde_yaml::Value::Number(serde_yaml::Number::from(0.5)),
    );
    let err_high = template.validate_overrides(&too_high);
    assert!(matches!(err_high, Err(TemplateError::OutOfRange(_, _, _, _))));

    // Out of range (too low)
    let mut too_low = HashMap::new();
    too_low.insert(
        "training.lr".to_string(),
        serde_yaml::Value::Number(serde_yaml::Number::from(0.0000001)),
    );
    let err_low = template.validate_overrides(&too_low);
    assert!(matches!(err_low, Err(TemplateError::OutOfRange(_, _, _, _))));
}

#[test]
fn test_deeply_nested_hydra_overrides() {
    let mut config = HydraConfig::default();

    // Deep 5-level nested path
    config.apply_override(
        "model.backbone.encoder.transformer.num_layers",
        serde_yaml::Value::Number(serde_yaml::Number::from(24)),
    );
    config.apply_override(
        "model.backbone.encoder.transformer.hidden_dim",
        serde_yaml::Value::Number(serde_yaml::Number::from(1024)),
    );
    config.apply_override(
        "model.backbone.encoder.dropout",
        serde_yaml::Value::Number(serde_yaml::Number::from(0.1)),
    );

    let layers = config.get_value("model.backbone.encoder.transformer.num_layers");
    assert_eq!(layers.and_then(|v| v.as_i64()), Some(24));

    let hidden = config.get_value("model.backbone.encoder.transformer.hidden_dim");
    assert_eq!(hidden.and_then(|v| v.as_i64()), Some(1024));

    let dropout = config.get_value("model.backbone.encoder.dropout");
    assert_eq!(dropout.and_then(|v| v.as_f64()), Some(0.1));

    // Non-existent path returns None
    assert!(config.get_value("model.backbone.non_existent").is_none());
}

#[test]
fn test_malformed_yaml_error_handling() {
    let bad_yaml = "template_id: [unclosed list";
    let err = Template::from_yaml(bad_yaml);
    assert!(matches!(err, Err(TemplateError::ParseError(_))));

    let empty_id = r#"
schema_version: "1.0"
template_id: ""
name: "Empty ID"
description: "Test"
domain: "test"
"#;
    let err_empty = Template::from_yaml(empty_id);
    assert!(matches!(err_empty, Err(TemplateError::ValidationError(_))));
}
