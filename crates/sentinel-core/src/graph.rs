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
    /// Target column. Unsupervised tasks (`vae`, `reconstruction`) need none.
    #[serde(default)]
    pub target: String,
    /// Several target columns: multilabel classification and multi-output regression.
    #[serde(default, deserialize_with = "list_or_csv_opt")]
    pub targets: Vec<String>,
    /// Query/group id column for the `ranking` task: rows sharing a value are ranked against each other.
    #[serde(default)]
    pub group: Option<String>,
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

fn list_or_csv_opt<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    list_or_csv(d)
}

fn default_split() -> f64 {
    0.2
}

/// A choice with optional settings: `adam`, or `{name: sgd, momentum: 0.9, nesterov: true}`.
#[derive(Debug, Clone, PartialEq)]
pub struct Named {
    pub name: String,
    pub params: BTreeMap<String, Value>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum NamedRepr {
    Plain(String),
    Full(BTreeMap<String, Value>),
}

impl TryFrom<NamedRepr> for Named {
    type Error = String;
    fn try_from(r: NamedRepr) -> Result<Self, String> {
        match r {
            NamedRepr::Plain(name) => Ok(Named { name, params: BTreeMap::new() }),
            NamedRepr::Full(mut m) => match m.remove("name") {
                Some(Value::String(name)) => Ok(Named { name, params: m }),
                _ => Err("expected a name, or a mapping with a `name` key".into()),
            },
        }
    }
}

impl From<Named> for NamedRepr {
    fn from(n: Named) -> Self {
        if n.params.is_empty() {
            NamedRepr::Plain(n.name)
        } else {
            let mut m = n.params;
            m.insert("name".into(), Value::String(n.name));
            NamedRepr::Full(m)
        }
    }
}

impl Serialize for Named {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        NamedRepr::from(self.clone()).serialize(s)
    }
}

impl<'de> Deserialize<'de> for Named {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Named::try_from(NamedRepr::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

impl Named {
    pub fn plain(name: &str) -> Named {
        Named { name: name.into(), params: BTreeMap::new() }
    }
    pub fn f64(&self, k: &str) -> Option<f64> {
        self.params.get(k).and_then(|v| v.as_f64())
    }
    pub fn i64(&self, k: &str) -> Option<i64> {
        self.params.get(k).and_then(|v| v.as_i64())
    }
    pub fn bool(&self, k: &str) -> Option<bool> {
        self.params.get(k).and_then(|v| v.as_bool())
    }
    pub fn str(&self, k: &str) -> Option<&str> {
        self.params.get(k).and_then(|v| v.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EarlyStopping {
    /// Epochs without improvement before stopping.
    pub patience: u32,
    #[serde(default)]
    pub min_delta: f64,
    /// Metric to watch; defaults to the task's primary metric.
    #[serde(default)]
    pub monitor: Option<String>,
}

/// A task or loss the catalog does not know: a Python function supplied by the project.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CustomSpec {
    /// Python file inside the project (relative, no `..`) defining the functions below; inlined into the script.
    #[serde(default)]
    pub file: Option<String>,
    /// Or the code itself, written in the YAML.
    #[serde(default)]
    pub source: Option<String>,
    /// `def <loss>(output, target) -> scalar tensor`
    #[serde(default = "default_custom_loss")]
    pub loss: String,
    /// Optional `def <metric>(output, target) -> float`, reported as `test_<metric>`.
    #[serde(default)]
    pub metric: Option<String>,
    /// `min` or `max`: which direction of the metric is better (for early stopping and best checkpoints).
    #[serde(default = "default_mode")]
    pub mode: String,
    /// `float` (regression-like targets) or `long` (class labels, mapped to indices).
    #[serde(default = "default_dtype")]
    pub target_dtype: String,
}

fn default_custom_loss() -> String {
    "loss".into()
}
fn default_mode() -> String {
    "min".into()
}
fn default_dtype() -> String {
    "float".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingSpec {
    /// A loss name, or `{name, ...settings}`. See `training::catalog()`.
    pub loss: Named,
    /// classification | binary | multilabel | regression | ranking | reconstruction | vae | custom.
    /// Inferred from the loss when omitted.
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default = "default_optimizer")]
    pub optimizer: Named,
    #[serde(default = "default_lr")]
    pub lr: f64,
    #[serde(default)]
    pub weight_decay: f64,
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    /// Rows per batch (queries per batch for ranking). Default: the whole training set.
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Weight of the KL term for the `vae` loss.
    #[serde(default = "default_beta")]
    pub beta: f64,
    #[serde(default)]
    pub scheduler: Option<Named>,
    /// Gradient-norm clipping threshold.
    #[serde(default)]
    pub clip_grad: Option<f64>,
    /// Batches whose gradients are summed before one optimizer step.
    #[serde(default = "default_accumulate")]
    pub accumulate: u32,
    /// off | auto | bf16 | fp16
    #[serde(default = "default_amp")]
    pub amp: String,
    /// auto | cpu | cuda | mps
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default)]
    pub early_stopping: Option<EarlyStopping>,
    /// Restore the weights from the best epoch (by the monitored metric) when training ends.
    #[serde(default)]
    pub save_best: bool,
    /// Top-level nodes whose weights are frozen (transfer learning).
    #[serde(default)]
    pub freeze: Vec<String>,
    /// Checkpoint mode: "latest" (default) keeps only the latest checkpoint; "all" keeps all checkpoints; "off" disables.
    #[serde(default = "default_checkpoint")]
    pub checkpoint: String,
    /// Automatically resume from checkpoint if present (default: true).
    #[serde(default = "default_resume")]
    pub resume: bool,
    /// Path to a `state_dict` (e.g. an earlier run's `model.pt`) to start from.
    #[serde(default)]
    pub init_from: Option<String>,
    #[serde(default)]
    pub custom: Option<CustomSpec>,
    pub data: DataSpec,
}

fn default_checkpoint() -> String {
    "latest".into()
}

fn default_resume() -> bool {
    true
}

fn default_accumulate() -> u32 {
    1
}
fn default_amp() -> String {
    "off".into()
}
fn default_device() -> String {
    "auto".into()
}

fn default_beta() -> f64 {
    1.0
}

fn default_optimizer() -> Named {
    Named::plain("adam")
}
fn default_lr() -> f64 {
    1e-3
}
fn default_epochs() -> u32 {
    50
}

/// Training recipe for a generative adversarial network: two optimizers and an alternating loop.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GanTraining {
    /// Size of the noise vector fed to the generator.
    pub latent: usize,
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default = "default_gan_lr")]
    pub lr_g: f64,
    #[serde(default = "default_gan_lr")]
    pub lr_d: f64,
    #[serde(default = "default_gan_beta1")]
    pub beta1: f64,
    /// Discriminator updates per generator update.
    #[serde(default = "default_d_steps")]
    pub d_steps: u32,
    pub data: DataSpec,
}

fn default_gan_lr() -> f64 {
    2e-4
}
fn default_gan_beta1() -> f64 {
    0.5
}
fn default_d_steps() -> u32 {
    1
}

/// Two graphs trained against each other.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GanSpec {
    pub generator: GraphSpec,
    pub discriminator: GraphSpec,
    pub training: GanTraining,
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

/// Size and compute of a validated graph, recursively through compound blocks.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct GraphStats {
    /// Learnable parameters; nodes that share weights with another node count once.
    pub params: u64,
    /// Parameter memory in float32.
    pub bytes_fp32: u64,
    /// Multiply-accumulates per sample in matrix/convolution layers; None if some size is symbolic.
    pub macs: Option<u64>,
    /// Fundamental block instances, including those inside compound blocks.
    pub blocks: usize,
    /// Compound block instances at every level.
    pub compounds: usize,
    /// Levels of compound nesting (0 = only fundamental blocks).
    pub depth: usize,
    /// Parameters by fundamental block type: where the weights live.
    pub by_block: BTreeMap<String, u64>,
    pub by_node: BTreeMap<String, NodeStats>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct NodeStats {
    pub params: u64,
    pub macs: Option<u64>,
    /// Reuses another node's weights (its parameters are counted there).
    pub shared: bool,
    pub inner: Option<Box<GraphStats>>,
}

impl ValidatedGraph {
    /// Parameter counts, memory and MACs for the whole graph and every node.
    pub fn stats(&self) -> GraphStats {
        let mut st = GraphStats { macs: Some(0), ..Default::default() };
        for node in &self.order {
            let kind = &self.kinds[node];
            let shared = self.spec.nodes[node].share.is_some();
            let ins: BTreeMap<String, Shape> = kind.inputs().into_iter().map(|p| { let sh = self.shapes[&self.wires[&(node.clone(), p.clone())].key()].clone(); (p, sh) }).collect();
            let outs: BTreeMap<String, Shape> = kind.outputs().into_iter().map(|p| { let sh = self.shapes[&format!("{node}.{p}")].clone(); (p, sh) }).collect();
            let params = &self.params[node];
            let mut ns = NodeStats { shared, ..Default::default() };
            match kind {
                BlockKind::Fundamental(b) => {
                    ns.params = if shared { 0 } else { b.param_count(params, &ins) };
                    ns.macs = b.macs(params, &ins, &outs);
                    st.blocks += 1;
                    *st.by_block.entry(b.name().to_string()).or_default() += ns.params;
                }
                BlockKind::Compound(_) => {
                    let inner = self.compounds[node].stats();
                    ns.params = if shared { 0 } else { inner.params };
                    ns.macs = inner.macs;
                    st.blocks += inner.blocks;
                    st.compounds += 1 + inner.compounds;
                    st.depth = st.depth.max(1 + inner.depth);
                    if !shared {
                        for (k, v) in &inner.by_block {
                            *st.by_block.entry(k.clone()).or_default() += v;
                        }
                    }
                    ns.inner = Some(Box::new(inner));
                }
            }
            st.params += ns.params;
            st.macs = match (st.macs, ns.macs) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            };
            st.by_node.insert(node.clone(), ns);
        }
        st.bytes_fp32 = st.params * 4;
        st
    }

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
