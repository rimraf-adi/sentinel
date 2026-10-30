//! Component graphs: a DAG of typed blocks with named ports, symbolic shape inference,
//! validation, and the data structures shared by codegen, CLI, daemon, and UI.

use crate::compound::{BlockKind, BlockLibrary};
use serde::{Deserialize, Serialize};
use serde_yaml::Value;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum GraphError {
    #[error("unknown block '{block}' in node '{node}'")]
    UnknownBlock { node: String, block: String },
    #[error("bad edge '{0}': expected 'source -> node.port'")]
    BadEdge(String),
    #[error("edge '{edge}': {msg}")]
    EdgeError { edge: String, msg: String },
    #[error("port {node}.{port} is fed by more than one edge")]
    DuplicateInput { node: String, port: String },
    #[error("port {node}.{port} has no incoming edge")]
    MissingInput { node: String, port: String },
    #[error("graph contains a cycle through: {0}")]
    Cycle(String),
    #[error("node '{node}': parameter '{param}': {msg}")]
    BadParam { node: String, param: String, msg: String },
    #[error("shape error at {node}.{port}: {msg}")]
    Shape { node: String, port: String, msg: String },
    #[error("output '{name}': {msg}")]
    BadOutput { name: String, msg: String },
    #[error("node '{node}': cannot share weights with '{target}': {msg}")]
    BadShare { node: String, target: String, msg: String },
    #[error("in compound block '{compound}' (node '{node}'): {msg}")]
    Compound { node: String, compound: String, msg: String },
    #[error("graph has no nodes")]
    Empty,
    #[error("training spec: {0}")]
    Training(String),
    #[error("failed to parse graph: {0}")]
    Parse(String),
}

/// One tensor dimension: a known size or a symbol such as the batch `B`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dim {
    Const(i64),
    Sym(String),
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Const(n) => write!(f, "{n}"),
            Dim::Sym(s) => write!(f, "{s}"),
        }
    }
}

pub type Shape = Vec<Dim>;
pub type Params = BTreeMap<String, Value>;

pub fn fmt_shape(s: &Shape) -> String {
    format!("[{}]", s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeSpec {
    pub block: String,
    #[serde(default)]
    pub params: Params,
    /// Reuse the module (weights) of another node of the same block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub share: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataSpec {
    pub csv: String,
    /// A YAML list, or one comma-separated string (so it can be a single `${param}`).
    #[serde(deserialize_with = "list_or_csv")]
    pub features: Vec<String>,
    /// Target column. Unsupervised losses (`vae`) need none.
    #[serde(default)]
    pub target: String,
    /// Per-sample shape the feature vector is reshaped to (e.g. [2, 1] for a sequence).
    #[serde(default)]
    pub reshape: Option<Vec<i64>>,
    #[serde(default = "default_split")]
    pub test_split: f64,
    #[serde(default)]
    pub seed: u64,
}

fn list_or_csv<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Either {
        List(Vec<String>),
        Csv(String),
    }
    Ok(match Either::deserialize(d)? {
        Either::List(v) => v,
        Either::Csv(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
    })
}

fn default_split() -> f64 {
    0.2
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingSpec {
    /// cross_entropy | mse | bce_logits
    pub loss: String,
    #[serde(default = "default_optimizer")]
    pub optimizer: String,
    #[serde(default = "default_lr")]
    pub lr: f64,
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Weight of the KL term for the `vae` loss.
    #[serde(default = "default_beta")]
    pub beta: f64,
    pub data: DataSpec,
}

fn default_beta() -> f64 {
    1.0
}

fn default_optimizer() -> String {
    "adam".into()
}
fn default_lr() -> f64 {
    1e-3
}
fn default_epochs() -> u32 {
    50
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSpec {
    pub inputs: BTreeMap<String, Shape>,
    pub nodes: BTreeMap<String, NodeSpec>,
    pub edges: Vec<String>,
    pub outputs: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingSpec>,
}

/// Where a tensor comes from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Source {
    Input { name: String },
    Port { node: String, port: String },
}

impl Source {
    pub fn key(&self) -> String {
        match self {
            Source::Input { name } => name.clone(),
            Source::Port { node, port } => format!("{node}.{port}"),
        }
    }
}

/// A graph that passed validation, with resolved parameters and inferred shapes.
#[derive(Debug, Clone)]
pub struct ValidatedGraph {
    pub spec: GraphSpec,
    /// Resolved block (either tier) per node.
    pub kinds: BTreeMap<String, BlockKind>,
    /// For compound nodes: the validated inner graph, used to emit a nested module class.
    pub compounds: BTreeMap<String, Box<ValidatedGraph>>,
    /// Topological execution order.
    pub order: Vec<String>,
    /// Fully resolved (defaults filled) parameters per node.
    pub params: BTreeMap<String, Params>,
    /// (dst node, dst port) -> source.
    pub wires: BTreeMap<(String, String), Source>,
    /// "node.port" or input name -> inferred shape.
    pub shapes: BTreeMap<String, Shape>,
    /// output name -> source
    pub outputs: BTreeMap<String, Source>,
}

impl ValidatedGraph {
    pub fn kind(&self, node: &str) -> &BlockKind {
        &self.kinds[node]
    }

    pub fn shape_of(&self, s: &Source) -> &Shape {
        &self.shapes[&s.key()]
    }
}

impl GraphSpec {
    pub fn from_yaml(s: &str) -> Result<Self, GraphError> {
        serde_yaml::from_str(s).map_err(|e| GraphError::Parse(e.to_string()))
    }

    pub fn from_value(v: Value) -> Result<Self, GraphError> {
        serde_yaml::from_value(v).map_err(|e| GraphError::Parse(e.to_string()))
    }

    /// Validate against the process-wide block library (shipped compounds plus the project's `./blocks`).
    pub fn validate(&self) -> Result<ValidatedGraph, GraphError> {
        self.validate_with(&BlockLibrary::global())
    }

    pub fn validate_with(&self, lib: &BlockLibrary) -> Result<ValidatedGraph, GraphError> {
        self.validate_stack(lib, &[])
    }

    /// `stack` holds the compound blocks currently being expanded, to reject a block that contains itself.
    pub fn validate_stack(&self, lib: &BlockLibrary, stack: &[String]) -> Result<ValidatedGraph, GraphError> {
        if self.nodes.is_empty() {
            return Err(GraphError::Empty);
        }

        // 1. blocks + resolved params
        let mut blocks: BTreeMap<&str, BlockKind> = BTreeMap::new();
        let mut params: BTreeMap<String, Params> = BTreeMap::new();
        for (name, n) in &self.nodes {
            let b = lib.resolve(&n.block).ok_or_else(|| GraphError::UnknownBlock {
                node: name.clone(),
                block: n.block.clone(),
            })?;
            params.insert(name.clone(), b.resolve_params(name, &n.params)?);
            blocks.insert(name, b);
        }

        // 2. edges -> wires
        let mut wires: BTreeMap<(String, String), Source> = BTreeMap::new();
        for e in &self.edges {
            let (lhs, rhs) = e.split_once("->").ok_or_else(|| GraphError::BadEdge(e.clone()))?;
            let (lhs, rhs) = (lhs.trim(), rhs.trim());
            if lhs.is_empty() || rhs.is_empty() {
                return Err(GraphError::BadEdge(e.clone()));
            }
            let err = |msg: String| GraphError::EdgeError { edge: e.clone(), msg };

            let src = self.parse_source(lhs, &blocks).map_err(&err)?;
            let (dn, dp) = parse_dst(rhs, &blocks).map_err(&err)?;
            let key = (dn.clone(), dp.clone());
            if wires.contains_key(&key) {
                return Err(GraphError::DuplicateInput { node: dn, port: dp });
            }
            wires.insert(key, src);
        }
        for (name, b) in &blocks {
            for p in b.inputs() {
                if !wires.contains_key(&(name.to_string(), p.to_string())) {
                    return Err(GraphError::MissingInput {
                        node: name.to_string(),
                        port: p.to_string(),
                    });
                }
            }
        }

        // 3. topological order (Kahn, deterministic)
        let mut indeg: BTreeMap<&str, usize> = self.nodes.keys().map(|k| (k.as_str(), 0)).collect();
        let mut succ: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for ((dn, _), src) in &wires {
            if let Source::Port { node, .. } = src {
                if succ.entry(node.as_str()).or_default().insert(dn.as_str()) {
                    *indeg.get_mut(dn.as_str()).unwrap() += 1;
                }
            }
        }
        let mut queue: VecDeque<&str> = indeg.iter().filter(|(_, d)| **d == 0).map(|(k, _)| *k).collect();
        let mut order: Vec<String> = Vec::new();
        while let Some(n) = queue.pop_front() {
            order.push(n.to_string());
            if let Some(next) = succ.get(n) {
                for m in next {
                    let d = indeg.get_mut(m).unwrap();
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(m);
                    }
                }
            }
        }
        if order.len() != self.nodes.len() {
            let stuck: Vec<&str> = indeg.iter().filter(|(_, d)| **d > 0).map(|(k, _)| *k).collect();
            return Err(GraphError::Cycle(stuck.join(", ")));
        }

        // 4. shape inference
        let mut shapes: BTreeMap<String, Shape> = BTreeMap::new();
        let mut compounds: BTreeMap<String, Box<ValidatedGraph>> = BTreeMap::new();
        for (name, shp) in &self.inputs {
            shapes.insert(name.clone(), shp.clone());
        }
        for node in &order {
            let b = &blocks[node.as_str()];
            let mut ins: BTreeMap<String, Shape> = BTreeMap::new();
            for p in b.inputs() {
                let src = &wires[&(node.clone(), p.to_string())];
                ins.insert(p.to_string(), shapes[&src.key()].clone());
            }
            let (outs, inner) = b.infer(node, &params[node], &ins, lib, stack)?;
            if let Some(inner) = inner {
                compounds.insert(node.clone(), inner);
            }
            for (port, s) in outs {
                shapes.insert(format!("{node}.{port}"), s);
            }
        }

        // 5. weight sharing
        for (name, n) in &self.nodes {
            if let Some(target) = &n.share {
                let err = |msg: &str| GraphError::BadShare {
                    node: name.clone(),
                    target: target.clone(),
                    msg: msg.into(),
                };
                let tn = self.nodes.get(target).ok_or_else(|| err("no such node"))?;
                if target == name {
                    return Err(err("a node cannot share with itself"));
                }
                if tn.share.is_some() {
                    return Err(err("target itself shares weights; point at the original"));
                }
                if tn.block != n.block || params[target] != params[name] {
                    return Err(err("block and parameters must be identical"));
                }
                let mut ins_a = BTreeMap::new();
                let mut ins_b = BTreeMap::new();
                for p in blocks[name.as_str()].inputs() {
                    ins_a.insert(p.to_string(), shapes[&wires[&(name.clone(), p.to_string())].key()].clone());
                    ins_b.insert(p.to_string(), shapes[&wires[&(target.clone(), p.to_string())].key()].clone());
                }
                let b = &blocks[name.as_str()];
                if b.instance_key(name, &params[name], &ins_a) != b.instance_key(target, &params[target], &ins_b) {
                    return Err(err("input shapes imply different module configurations"));
                }
                if b.instance_key(name, &params[name], &ins_a).is_none() {
                    return Err(err("block has no weights to share"));
                }
            }
        }

        // 6. outputs
        let mut outputs = BTreeMap::new();
        if self.outputs.is_empty() {
            return Err(GraphError::BadOutput { name: "*".into(), msg: "graph declares no outputs".into() });
        }
        for (oname, target) in &self.outputs {
            let src = self
                .parse_source(target.trim(), &blocks)
                .map_err(|msg| GraphError::BadOutput { name: oname.clone(), msg })?;
            outputs.insert(oname.clone(), src);
        }

        let kinds = blocks.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        Ok(ValidatedGraph { spec: self.clone(), kinds, compounds, order, params, wires, shapes, outputs })
    }

    fn parse_source(&self, s: &str, blocks: &BTreeMap<&str, BlockKind>) -> Result<Source, String> {
        if let Some((n, p)) = s.split_once('.') {
            let b = blocks.get(n).ok_or_else(|| format!("unknown node '{n}'"))?;
            if !b.outputs().iter().any(|o| o == p) {
                return Err(format!("block '{}' has no output port '{p}' (outputs: {:?})", b.name(), b.outputs()));
            }
            return Ok(Source::Port { node: n.into(), port: p.into() });
        }
        if self.inputs.contains_key(s) {
            return Ok(Source::Input { name: s.into() });
        }
        let b = blocks.get(s).ok_or_else(|| format!("'{s}' is neither a graph input nor a node"))?;
        match b.outputs().as_slice() {
            [only] => Ok(Source::Port { node: s.into(), port: only.clone() }),
            many => Err(format!("node '{s}' has several outputs {many:?}; name one as '{s}.<port>'")),
        }
    }
}

fn parse_dst(s: &str, blocks: &BTreeMap<&str, BlockKind>) -> Result<(String, String), String> {
    if let Some((n, p)) = s.split_once('.') {
        let b = blocks.get(n).ok_or_else(|| format!("unknown node '{n}'"))?;
        if !b.inputs().iter().any(|i| i == p) {
            return Err(format!("block '{}' has no input port '{p}' (inputs: {:?})", b.name(), b.inputs()));
        }
        return Ok((n.into(), p.into()));
    }
    let b = blocks.get(s).ok_or_else(|| format!("unknown node '{s}'"))?;
    match b.inputs().as_slice() {
        [only] => Ok((s.into(), only.clone())),
        many => Err(format!("node '{s}' has several inputs {many:?}; name one as '{s}.<port>'")),
    }
}

/// Replace `${name}` placeholders throughout a YAML tree. A string that is exactly one
/// placeholder takes the parameter's own type; embedded placeholders are interpolated.
pub fn substitute(v: &Value, vars: &BTreeMap<String, Value>) -> Value {
    match v {
        Value::String(s) => {
            if let Some(name) = s.strip_prefix("${").and_then(|r| r.strip_suffix('}')) {
                if !name.contains("${") {
                    if let Some(val) = vars.get(name.trim()) {
                        return val.clone();
                    }
                }
            }
            let mut out = s.clone();
            for (k, val) in vars {
                let pat = format!("${{{k}}}");
                if out.contains(&pat) {
                    let rep = match val {
                        Value::String(x) => x.clone(),
                        other => serde_yaml::to_string(other).unwrap_or_default().trim().to_string(),
                    };
                    out = out.replace(&pat, &rep);
                }
            }
            Value::String(out)
        }
        Value::Sequence(seq) => Value::Sequence(seq.iter().map(|x| substitute(x, vars)).collect()),
        Value::Mapping(m) => {
            let mut nm = serde_yaml::Mapping::new();
            for (k, x) in m {
                nm.insert(k.clone(), substitute(x, vars));
            }
            Value::Mapping(nm)
        }
        other => other.clone(),
    }
}
