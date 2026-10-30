//! Turn a drawn graph into a reusable component: selected block parameters become named,
//! sweepable hyperparameters (`${name}` placeholders plus a `parameters:` entry).

use crate::components::{Component, ComponentError, ParamSpec, ParamType};
use crate::graph::GraphSpec;
use serde::Deserialize;
use serde_yaml::Value;
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone, Deserialize)]
pub struct ExposeParam {
    pub node: String,
    pub param: String,
    /// Hyperparameter name in the component's `parameters:` map.
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SaveSpec {
    pub name: String,
    pub domain: String,
    #[serde(default)]
    pub description: String,
    pub graph: GraphSpec,
    #[serde(default)]
    pub expose: Vec<ExposeParam>,
}

fn err(msg: impl Into<String>) -> ComponentError {
    ComponentError::ValidationError(msg.into())
}

pub fn slugify(s: &str) -> String {
    let mut out = String::new();
    for c in s.trim().to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn is_ident(s: &str) -> bool {
    let mut cs = s.chars();
    matches!(cs.next(), Some(c) if c.is_ascii_alphabetic() || c == '_') && cs.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn param_spec(v: &Value, help: String, range: Option<(f64, f64)>) -> Result<ParamSpec, ComponentError> {
    let param_type = match v {
        Value::Bool(_) => ParamType::Bool,
        Value::Number(n) if n.is_i64() || n.is_u64() => ParamType::Int,
        Value::Number(_) => ParamType::Float,
        Value::String(_) => ParamType::String,
        _ => return Err(err(format!("only scalar values can become hyperparameters, got {v:?}"))),
    };
    Ok(ParamSpec { param_type, default: v.clone(), options: vec![], range, help })
}

/// Returns `(component_id, yaml)`. The YAML is round-trip checked: it loads and builds with defaults.
pub fn build_component_yaml(spec: &SaveSpec) -> Result<(String, String), ComponentError> {
    let slug = slugify(&spec.name);
    let domain = slugify(&spec.domain);
    if slug.is_empty() || domain.is_empty() {
        return Err(err("name and domain must contain letters or digits"));
    }
    let validated = spec.graph.validate().map_err(|e| err(format!("graph is not valid: {e}")))?;

    let mut graph = serde_yaml::to_value(&spec.graph).map_err(|e| err(e.to_string()))?;
    let mut parameters: HashMap<String, ParamSpec> = HashMap::new();
    let mut seen = BTreeSet::new();

    for ex in &spec.expose {
        if !is_ident(&ex.name) {
            return Err(err(format!("'{}' is not a valid hyperparameter name", ex.name)));
        }
        let resolved = validated
            .params
            .get(&ex.node)
            .and_then(|p| p.get(&ex.param))
            .ok_or_else(|| err(format!("node '{}' has no parameter '{}'", ex.node, ex.param)))?;
        match parameters.get_mut(&ex.name) {
            // Same name again = tie the parameters together (e.g. both sides of a residual add).
            Some(existing) => {
                if &existing.default != resolved {
                    return Err(err(format!(
                        "'{}' ties parameters that currently differ ({:?} vs {:?}); set them equal first",
                        ex.name, existing.default, resolved
                    )));
                }
                existing.help.push_str(&format!(", {}.{}", ex.node, ex.param));
            }
            None => {
                parameters.insert(ex.name.clone(), param_spec(resolved, format!("{}.{}", ex.node, ex.param), None)?);
                seen.insert(ex.name.clone());
            }
        }
        graph["nodes"][ex.node.as_str()]["params"][ex.param.as_str()] = Value::String(format!("${{{}}}", ex.name));
    }

    // Training knobs are always hyperparameters, unless the user already claimed the names.
    if let Some(t) = &spec.graph.training {
        if seen.insert("lr".into()) {
            parameters.insert("lr".into(), param_spec(&Value::from(t.lr), "Learning rate".into(), Some((1e-6, 1.0)))?);
            graph["training"]["lr"] = Value::String("${lr}".into());
        }
        if seen.insert("epochs".into()) {
            parameters.insert("epochs".into(), param_spec(&Value::from(t.epochs as i64), "Training epochs".into(), Some((1.0, 100000.0)))?);
            graph["training"]["epochs"] = Value::String("${epochs}".into());
        }
    }

    let component_id = format!("{domain}/{slug}@v1");
    let component = Component {
        schema_version: "1.0".into(),
        component_id: component_id.clone(),
        name: spec.name.trim().to_string(),
        description: if spec.description.trim().is_empty() { format!("Custom {} component", spec.name.trim()) } else { spec.description.trim().to_string() },
        domain,
        extends: None,
        defaults: vec![],
        parameters,
        locked: HashMap::new(),
        sweep: None,
        smoke_test: HashMap::new(),
        graph: Some(graph),
    };
    let yaml = component.to_yaml()?;
    // Prove the file we are about to write loads and builds on its own.
    Component::from_yaml(&yaml)?;
    Ok((component_id, yaml))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(expose: &str) -> SaveSpec {
        let y = format!(
            r#"
name: My Net
domain: Tabular
graph:
  inputs: {{x: [B, 2]}}
  nodes:
    a: {{block: linear, params: {{out: 8}}}}
    r: {{block: activation, params: {{kind: relu}}}}
    h: {{block: linear, params: {{out: 3}}}}
  edges: ['x -> a', 'a -> r', 'r -> h']
  outputs: {{y: h}}
  training: {{loss: cross_entropy, lr: 0.05, epochs: 30, data: {{csv: d.csv, features: [x1, x2], target: label}}}}
expose: {expose}
"#
        );
        serde_yaml::from_str(&y).unwrap()
    }

    #[test]
    fn exposed_params_become_hyperparameters_that_rebuild_the_graph() {
        let (id, yaml) = build_component_yaml(&spec("[{node: a, param: out, name: width}]")).unwrap();
        assert_eq!(id, "tabular/my-net@v1");
        let c = Component::from_yaml(&yaml).unwrap();
        assert_eq!(c.parameters["width"].default, Value::from(8));
        assert!(c.parameters.contains_key("lr") && c.parameters.contains_key("epochs"));
        let mut ov = HashMap::new();
        ov.insert("width".to_string(), Value::from(32));
        ov.insert("lr".to_string(), Value::from(0.001));
        let g = c.build_graph(&ov).unwrap();
        assert_eq!(crate::graph::fmt_shape(&g.shapes["a.out"]), "[B, 32]");
        assert_eq!(g.spec.training.unwrap().lr, 0.001);
    }

    #[test]
    fn defaults_not_written_explicitly_can_be_exposed() {
        let (_, yaml) = build_component_yaml(&spec("[{node: r, param: kind, name: act}, {node: a, param: bias, name: use_bias}]")).unwrap();
        let c = Component::from_yaml(&yaml).unwrap();
        assert_eq!(c.parameters["use_bias"].default, Value::Bool(true));
        assert_eq!(c.parameters["act"].default, Value::String("relu".into()));
    }

    #[test]
    fn rejects_bad_requests() {
        for (expose, needle) in [
            ("[{node: a, param: out, name: w}, {node: h, param: out, name: w}]", "currently differ"),
            ("[{node: a, param: out, name: '9bad'}]", "valid hyperparameter name"),
            ("[{node: nope, param: out, name: w}]", "no parameter"),
            ("[{node: a, param: nope, name: w}]", "no parameter"),
        ] {
            let e = build_component_yaml(&spec(expose)).unwrap_err().to_string();
            assert!(e.contains(needle), "{expose}: {e}");
        }
    }

    #[test]
    fn rejects_invalid_graphs_and_empty_names() {
        let mut s = spec("[]");
        s.graph.edges.clear();
        assert!(build_component_yaml(&s).unwrap_err().to_string().contains("not valid"));
        let mut s = spec("[]");
        s.name = "!!!".into();
        assert!(build_component_yaml(&s).is_err());
    }

    #[test]
    fn user_named_lr_is_not_overwritten() {
        let (_, yaml) = build_component_yaml(&spec("[{node: a, param: out, name: lr}]")).unwrap();
        let c = Component::from_yaml(&yaml).unwrap();
        assert_eq!(c.parameters["lr"].default, Value::from(8)); // the user's own hyperparameter wins
        assert!(c.parameters.contains_key("epochs"));
    }

    #[test]
    fn equal_parameters_can_be_tied_under_one_name() {
        // Residual add needs both sides the same width: one hyperparameter must drive both.
        let y = r#"
name: Res
domain: tabular
graph:
  inputs: {x: [B, 4]}
  nodes:
    e: {block: linear, params: {out: 16}}
    f: {block: linear, params: {out: 16}}
    s: {block: add}
  edges: ['x -> e', 'e -> f', 'f -> s.a', 'e -> s.b']
  outputs: {y: s}
expose: [{node: e, param: out, name: width}, {node: f, param: out, name: width}]
"#;
        let spec: SaveSpec = serde_yaml::from_str(y).unwrap();
        let (_, yaml) = build_component_yaml(&spec).unwrap();
        let c = Component::from_yaml(&yaml).unwrap();
        assert_eq!(c.parameters.len(), 1);
        assert!(c.parameters["width"].help.contains("e.out") && c.parameters["width"].help.contains("f.out"));
        let mut ov = HashMap::new();
        ov.insert("width".to_string(), Value::from(48));
        let g = c.build_graph(&ov).unwrap();
        assert_eq!(crate::graph::fmt_shape(&g.shapes["s.out"]), "[B, 48]");
    }

    #[test]
    fn slugs() {
        assert_eq!(slugify("  My  Net!! v2 "), "my-net-v2");
        assert_eq!(slugify("***"), "");
    }
}
