//! Turn a drawn graph into a reusable component: selected block parameters become named,
//! sweepable hyperparameters (`${name}` placeholders plus a `parameters:` entry).

use crate::components::{Component, ComponentError, ParamSpec, ParamType};
use crate::compound::{BlockLibrary, CompoundDef};
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
        gan: None,
        origin: None,
    };
    let yaml = component.to_yaml()?;
    // Prove the file we are about to write loads and builds on its own.
    Component::from_yaml(&yaml)?;
    Ok((component_id, yaml))
}

#[derive(Debug, Clone, Deserialize)]
pub struct SaveBlockSpec {
    pub name: String,
    pub family: String,
    #[serde(default)]
    pub description: String,
    pub graph: GraphSpec,
    #[serde(default)]
    pub expose: Vec<ExposeParam>,
}

/// `My Block!` -> `my_block`
pub fn snake(s: &str) -> String {
    slugify(s).replace('-', "_")
}

/// Turn a drawn graph into a compound block: graph inputs and outputs become ports, exposed parameters become
/// the block's parameters. Returns `(block_name, yaml)`; the YAML is proven by instantiating it once.
pub fn build_compound_yaml(spec: &SaveBlockSpec, lib: &BlockLibrary) -> Result<(String, String), ComponentError> {
    let name = snake(&spec.name);
    let family = snake(&spec.family);
    if name.is_empty() || family.is_empty() {
        return Err(err("name and family must contain letters or digits"));
    }
    if !name.chars().next().is_some_and(|c| c.is_ascii_lowercase() || c == '_') {
        return Err(err("a block name must start with a letter"));
    }
    if spec.graph.inputs.is_empty() {
        return Err(err("a compound block needs at least one graph input (they become its input ports)"));
    }
    let mut graph_spec = spec.graph.clone();
    graph_spec.training = None; // a block is a piece of a model, not a training recipe
    let validated = graph_spec.validate_with(lib).map_err(|e| err(format!("graph is not valid: {e}")))?;

    let mut graph = serde_yaml::to_value(&graph_spec).map_err(|e| err(e.to_string()))?;
    if let Some(m) = graph.as_mapping_mut() {
        m.remove("training");
    }
    let mut params = serde_yaml::Mapping::new();
    for ex in &spec.expose {
        if !is_ident(&ex.name) || !ex.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
            return Err(err(format!("'{}' is not a valid block parameter name (lowercase letters, digits, underscores)", ex.name)));
        }
        let resolved = validated
            .params
            .get(&ex.node)
            .and_then(|p| p.get(&ex.param))
            .ok_or_else(|| err(format!("node '{}' has no parameter '{}'", ex.node, ex.param)))?;
        if !matches!(resolved, Value::Bool(_) | Value::Number(_) | Value::String(_)) {
            return Err(err(format!("only scalar values can become block parameters, got {resolved:?}")));
        }
        match params.get(&Value::String(ex.name.clone())) {
            Some(existing) => {
                if existing.get("default") != Some(resolved) {
                    return Err(err(format!("'{}' ties parameters that currently differ; set them equal first", ex.name)));
                }
            }
            None => {
                let mut decl = serde_yaml::Mapping::new();
                decl.insert("default".into(), resolved.clone());
                decl.insert("help".into(), format!("{}.{}", ex.node, ex.param).into());
                params.insert(ex.name.clone().into(), Value::Mapping(decl));
            }
        }
        graph["nodes"][ex.node.as_str()]["params"][ex.param.as_str()] = Value::String(format!("${{{}}}", ex.name));
    }

    let mut doc = serde_yaml::Mapping::new();
    doc.insert("compound".into(), name.clone().into());
    doc.insert("family".into(), family.into());
    doc.insert("description".into(), if spec.description.trim().is_empty() { format!("Custom block {}", spec.name.trim()) } else { spec.description.trim().to_string() }.into());
    doc.insert("params".into(), Value::Mapping(params));
    doc.insert("graph".into(), graph);
    let yaml = serde_yaml::to_string(&Value::Mapping(doc)).map_err(|e| err(e.to_string()))?;

    // Prove it: it parses, and instantiates with its own defaults and declared input shapes.
    let def = CompoundDef::from_yaml(&yaml).map_err(err)?;
    let mut probe = lib.clone();
    probe.insert(def.clone()).map_err(err)?;
    let kind = probe.resolve(&name).expect("just inserted");
    let defaults = kind.resolve_params("probe", &Default::default()).map_err(|e| err(e.to_string()))?;
    let mut ins = std::collections::BTreeMap::new();
    for (k, v) in &spec.graph.inputs {
        ins.insert(k.clone(), v.clone());
    }
    def.instantiate("probe", &defaults, &ins, &probe, &[]).map_err(|e| err(format!("saved block does not instantiate: {e}")))?;
    Ok((name, yaml))
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

    fn block_spec(expose: &str) -> SaveBlockSpec {
        let y = format!(
            r#"
name: My Gate
family: core
graph:
  inputs: {{x: [B, 8]}}
  nodes:
    a: {{block: linear, params: {{out: 8}}}}
    r: {{block: activation, params: {{kind: relu}}}}
    s: {{block: add}}
  edges: ['x -> a', 'a -> r', 'r -> s.a', 'x -> s.b']
  outputs: {{y: s}}
  training: {{loss: mse, data: {{csv: d.csv, features: [x1], target: y}}}}
expose: {expose}
"#
        );
        serde_yaml::from_str(&y).unwrap()
    }

    #[test]
    fn a_drawn_graph_becomes_a_usable_compound_block() {
        let lib = BlockLibrary::builtin();
        let (name, yaml) = build_compound_yaml(&block_spec("[{node: a, param: out, name: width}]"), &lib).unwrap();
        assert_eq!(name, "my_gate");
        assert!(!yaml.contains("training"), "a block carries no training recipe");
        let mut lib2 = lib.clone();
        lib2.insert(CompoundDef::from_yaml(&yaml).unwrap()).unwrap();
        // use it inside a bigger graph at a different width: x [B, 8] keeps the skip valid only when width == 8
        let g = GraphSpec::from_yaml("inputs: {i: [B, 8]}\nnodes: {g: {block: my_gate, params: {width: 8}}}\nedges: ['i -> g']\noutputs: {o: g}").unwrap().validate_with(&lib2).unwrap();
        assert_eq!(crate::graph::fmt_shape(&g.shapes["g.y"]), "[B, 8]");
        let bad = GraphSpec::from_yaml("inputs: {i: [B, 8]}\nnodes: {g: {block: my_gate, params: {width: 4}}}\nedges: ['i -> g']\noutputs: {o: g}").unwrap().validate_with(&lib2);
        assert!(bad.unwrap_err().to_string().contains("my_gate"), "the skip needs width 8");
    }

    #[test]
    fn compound_authoring_rejects_bad_requests() {
        let lib = BlockLibrary::builtin();
        let mut s = block_spec("[]");
        s.name = "linear".into();
        // collides with a fundamental block: caught when the definition is parsed
        assert!(build_compound_yaml(&s, &lib).unwrap_err().to_string().contains("fundamental"));
        let mut s = block_spec("[]");
        s.name = "mlp_block".into();
        assert!(build_compound_yaml(&s, &lib).unwrap_err().to_string().contains("already exists"));
        let mut s = block_spec("[]");
        s.graph.inputs.clear();
        assert!(build_compound_yaml(&s, &lib).is_err());
        assert!(build_compound_yaml(&block_spec("[{node: a, param: out, name: Bad}]"), &lib).unwrap_err().to_string().contains("valid block parameter name"));
        assert!(build_compound_yaml(&block_spec("[{node: zzz, param: out, name: w}]"), &lib).unwrap_err().to_string().contains("no parameter"));
    }
}
