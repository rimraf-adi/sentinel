//! Block library: every neural-network building block (ANN, RNN, CNN, attention, GNN, ...)
//! with its ports, parameters, shape rule, and PyTorch emission.

use crate::graph::{Dim, GraphError, Params, Shape};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Block {
    Linear,
    Activation,
    Dropout,
    LayerNorm,
    BatchNorm,
    Add,
    Concat,
    Lstm,
    Gru,
    Conv1d,
    Conv2d,
    MaxPool2d,
    Flatten,
    Reshape,
    Embedding,
    MeanPool,
    LastStep,
    Attention,
    TransformerEncoder,
    Gcn,
}

const ALL: &[Block] = &[
    Block::Linear,
    Block::Activation,
    Block::Dropout,
    Block::LayerNorm,
    Block::BatchNorm,
    Block::Add,
    Block::Concat,
    Block::Lstm,
    Block::Gru,
    Block::Conv1d,
    Block::Conv2d,
    Block::MaxPool2d,
    Block::Flatten,
    Block::Reshape,
    Block::Embedding,
    Block::MeanPool,
    Block::LastStep,
    Block::Attention,
    Block::TransformerEncoder,
    Block::Gcn,
];

/// (name, default as JSON — empty string means required, help)
type ParamDef = (&'static str, &'static str, &'static str);

#[derive(Serialize)]
pub struct ParamInfo {
    pub name: String,
    pub required: bool,
    pub default: Option<serde_json::Value>,
    pub help: String,
}

#[derive(Serialize)]
pub struct BlockInfo {
    pub name: String,
    /// `fundamental` (native) or `compound` (a subgraph of other blocks)
    pub tier: &'static str,
    pub family: String,
    pub description: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub params: Vec<ParamInfo>,
    /// Compound blocks: the blocks this one is built from.
    pub uses: Vec<String>,
    /// Compound blocks: the raw graph definition, for display.
    pub definition: Option<serde_json::Value>,
}

fn bad(node: &str, param: &str, msg: impl Into<String>) -> GraphError {
    GraphError::BadParam { node: node.into(), param: param.into(), msg: msg.into() }
}

fn shape_err(node: &str, port: &str, msg: impl Into<String>) -> GraphError {
    GraphError::Shape { node: node.into(), port: port.into(), msg: msg.into() }
}

fn geti(p: &Params, k: &str) -> i64 {
    p.get(k).and_then(|v| v.as_i64()).unwrap_or(0)
}
fn getf(p: &Params, k: &str) -> f64 {
    p.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0)
}
fn getb(p: &Params, k: &str) -> bool {
    p.get(k).and_then(|v| v.as_bool()).unwrap_or(false)
}
fn gets(p: &Params, k: &str) -> String {
    p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string()
}

fn last_const(node: &str, port: &str, s: &Shape) -> Result<i64, GraphError> {
    match s.last() {
        Some(Dim::Const(n)) => Ok(*n),
        Some(d) => Err(shape_err(node, port, format!("last dimension must be a known size, found '{d}'"))),
        None => Err(shape_err(node, port, "scalar input has no last dimension")),
    }
}

fn dim_at(node: &str, port: &str, s: &Shape, i: usize) -> Result<i64, GraphError> {
    match s.get(i) {
        Some(Dim::Const(n)) => Ok(*n),
        Some(d) => Err(shape_err(node, port, format!("dimension {i} must be a known size, found '{d}'"))),
        None => Err(shape_err(node, port, format!("expected rank > {i}, got {}", crate::graph::fmt_shape(s)))),
    }
}

fn rank(node: &str, port: &str, s: &Shape, r: usize) -> Result<(), GraphError> {
    if s.len() != r {
        return Err(shape_err(node, port, format!("expected rank {r}, got {}", crate::graph::fmt_shape(s))));
    }
    Ok(())
}

fn conv_len(node: &str, port: &str, d: &Dim, k: i64, s: i64, p: i64) -> Result<Dim, GraphError> {
    match d {
        Dim::Const(n) => {
            if n + 2 * p < k {
                return Err(shape_err(node, port, format!("size {n} (+padding {p}) smaller than kernel {k}")));
            }
            Ok(Dim::Const((n + 2 * p - k) / s + 1))
        }
        Dim::Sym(x) => Ok(Dim::Sym(format!("{x}'"))),
    }
}

fn one(port: &str, s: Shape) -> BTreeMap<String, Shape> {
    BTreeMap::from([(port.to_string(), s)])
}

impl Block {
    pub fn all() -> &'static [Block] {
        ALL
    }

    /// Accepts `lstm` or `lstm@v1`.
    pub fn parse(s: &str) -> Option<Block> {
        let base = s.split('@').next().unwrap_or(s);
        ALL.iter().copied().find(|b| b.name() == base)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Block::Linear => "linear",
            Block::Activation => "activation",
            Block::Dropout => "dropout",
            Block::LayerNorm => "layernorm",
            Block::BatchNorm => "batchnorm",
            Block::Add => "add",
            Block::Concat => "concat",
            Block::Lstm => "lstm",
            Block::Gru => "gru",
            Block::Conv1d => "conv1d",
            Block::Conv2d => "conv2d",
            Block::MaxPool2d => "maxpool2d",
            Block::Flatten => "flatten",
            Block::Reshape => "reshape",
            Block::Embedding => "embedding",
            Block::MeanPool => "mean_pool",
            Block::LastStep => "last_step",
            Block::Attention => "attention",
            Block::TransformerEncoder => "transformer_encoder",
            Block::Gcn => "gcn",
        }
    }

    pub fn family(&self) -> &'static str {
        match self {
            Block::Linear | Block::Activation | Block::Dropout | Block::LayerNorm | Block::BatchNorm => "core",
            Block::Add | Block::Concat | Block::Flatten | Block::Reshape | Block::MeanPool | Block::LastStep => "shape",
            Block::Lstm | Block::Gru | Block::Embedding => "sequence",
            Block::Attention | Block::TransformerEncoder => "attention",
            Block::Conv1d | Block::Conv2d | Block::MaxPool2d => "vision",
            Block::Gcn => "graph",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Block::Linear => "Fully connected layer on the last dimension (the ANN building block)",
            Block::Activation => "Element-wise nonlinearity (relu, gelu, tanh, sigmoid, silu, softmax, leaky_relu)",
            Block::Dropout => "Randomly zeroes activations during training",
            Block::LayerNorm => "Layer normalization over the last dimension",
            Block::BatchNorm => "Batch normalization over the feature/channel dimension",
            Block::Add => "Element-wise sum of two tensors (residual / skip connection)",
            Block::Concat => "Concatenate two tensors along an axis",
            Block::Lstm => "Long short-term memory network over a sequence [B, T, F]",
            Block::Gru => "Gated recurrent unit over a sequence [B, T, F]",
            Block::Conv1d => "1-D convolution over [B, C, L]",
            Block::Conv2d => "2-D convolution over [B, C, H, W]",
            Block::MaxPool2d => "2-D max pooling",
            Block::Flatten => "Flatten everything except the batch dimension",
            Block::Reshape => "Reshape the per-sample dimensions",
            Block::Embedding => "Lookup table mapping integer ids [B, T] to vectors [B, T, dim]",
            Block::MeanPool => "Mean over one dimension (e.g. pool a sequence)",
            Block::LastStep => "Take the last time step of a sequence [B, T, F] -> [B, F]",
            Block::Attention => "Multi-head self-attention over [B, T, E]",
            Block::TransformerEncoder => "Transformer encoder layer (attention + feed-forward) over [B, T, E]",
            Block::Gcn => "Graph convolution: adj @ (x W) over node features [B, N, F] and adjacency [B, N, N]",
        }
    }

    pub fn inputs(&self) -> &'static [&'static str] {
        match self {
            Block::Add | Block::Concat => &["a", "b"],
            Block::Gcn => &["x", "adj"],
            _ => &["in"],
        }
    }

    pub fn outputs(&self) -> &'static [&'static str] {
        match self {
            Block::Lstm | Block::Gru => &["out", "last"],
            _ => &["out"],
        }
    }

    fn param_defs(&self) -> &'static [ParamDef] {
        match self {
            Block::Linear => &[("out", "", "Output features"), ("bias", "true", "Learn an additive bias")],
            Block::Activation => &[("kind", "\"relu\"", "relu | gelu | tanh | sigmoid | silu | softmax | leaky_relu")],
            Block::Dropout => &[("p", "0.1", "Drop probability")],
            Block::Concat => &[("axis", "-1", "Axis to concatenate along")],
            Block::Lstm | Block::Gru => &[
                ("hidden", "", "Hidden units"),
                ("layers", "1", "Stacked layers"),
                ("bidirectional", "false", "Run in both directions"),
                ("dropout", "0.0", "Dropout between layers"),
            ],
            Block::Conv1d | Block::Conv2d => &[
                ("out_channels", "", "Output channels"),
                ("kernel", "3", "Kernel size"),
                ("stride", "1", "Stride"),
                ("padding", "0", "Zero padding"),
            ],
            Block::MaxPool2d => &[("kernel", "2", "Window size"), ("stride", "0", "Stride (0 = kernel)")],
            Block::Reshape => &[("shape", "", "Per-sample target shape, e.g. [2, 1]")],
            Block::Embedding => &[("num_embeddings", "", "Vocabulary size"), ("dim", "", "Embedding size")],
            Block::MeanPool => &[("dim", "1", "Dimension to average")],
            Block::Attention => &[("heads", "4", "Attention heads")],
            Block::TransformerEncoder => &[
                ("heads", "4", "Attention heads"),
                ("ff", "256", "Feed-forward width"),
                ("dropout", "0.1", "Dropout"),
            ],
            Block::Gcn => &[("out", "", "Output features")],
            Block::LayerNorm | Block::BatchNorm | Block::Add | Block::Flatten | Block::LastStep => &[],
        }
    }

    pub fn info(&self) -> BlockInfo {
        BlockInfo {
            name: self.name().into(),
            tier: "fundamental",
            family: self.family().into(),
            description: self.description().into(),
            inputs: self.inputs().iter().map(|s| s.to_string()).collect(),
            outputs: self.outputs().iter().map(|s| s.to_string()).collect(),
            params: self
                .param_defs()
                .iter()
                .map(|(n, d, h)| ParamInfo {
                    name: n.to_string(),
                    required: d.is_empty(),
                    default: if d.is_empty() { None } else { serde_json::from_str(d).ok() },
                    help: h.to_string(),
                })
                .collect(),
            uses: vec![],
            definition: None,
        }
    }

    /// Fill defaults, reject unknown or missing parameters, and type-check.
    pub fn resolve_params(&self, node: &str, given: &Params) -> Result<Params, GraphError> {
        let defs = self.param_defs();
        for k in given.keys() {
            if !defs.iter().any(|(n, _, _)| n == k) {
                return Err(bad(
                    node,
                    k,
                    format!("unknown parameter for '{}' (known: {:?})", self.name(), defs.iter().map(|d| d.0).collect::<Vec<_>>()),
                ));
            }
        }
        let mut out = Params::new();
        for (name, default, _) in defs {
            let v = match given.get(*name) {
                Some(v) => v.clone(),
                None if default.is_empty() => return Err(bad(node, name, "required parameter is missing")),
                None => {
                    let j: serde_json::Value = serde_json::from_str(default).expect("static default");
                    serde_yaml::to_value(j).expect("yaml value")
                }
            };
            out.insert(name.to_string(), v);
        }
        self.check_params(node, &out)?;
        Ok(out)
    }

    fn check_params(&self, node: &str, p: &Params) -> Result<(), GraphError> {
        let pos = |k: &str| -> Result<(), GraphError> {
            match p.get(k).and_then(|v| v.as_i64()) {
                Some(n) if n > 0 => Ok(()),
                _ => Err(bad(node, k, "must be a positive integer")),
            }
        };
        match self {
            Block::Linear | Block::Gcn => pos("out")?,
            Block::Lstm | Block::Gru => {
                pos("hidden")?;
                pos("layers")?;
            }
            Block::Conv1d | Block::Conv2d => {
                pos("out_channels")?;
                pos("kernel")?;
                pos("stride")?;
            }
            Block::MaxPool2d => pos("kernel")?,
            Block::Embedding => {
                pos("num_embeddings")?;
                pos("dim")?;
            }
            Block::Attention | Block::TransformerEncoder => pos("heads")?,
            Block::Activation => {
                let k = gets(p, "kind");
                if !["relu", "gelu", "tanh", "sigmoid", "silu", "softmax", "leaky_relu"].contains(&k.as_str()) {
                    return Err(bad(node, "kind", format!("unsupported activation '{k}'")));
                }
            }
            Block::Dropout => {
                let v = getf(p, "p");
                if !(0.0..1.0).contains(&v) {
                    return Err(bad(node, "p", "must be in [0, 1)"));
                }
            }
            Block::Reshape => {
                let ok = p
                    .get("shape")
                    .and_then(|v| v.as_sequence())
                    .map(|s| !s.is_empty() && s.iter().all(|x| x.as_i64().map_or(false, |n| n > 0)))
                    .unwrap_or(false);
                if !ok {
                    return Err(bad(node, "shape", "must be a non-empty list of positive integers"));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Infer output shapes from input shapes.
    pub fn infer(
        &self,
        node: &str,
        p: &Params,
        ins: &BTreeMap<String, Shape>,
    ) -> Result<BTreeMap<String, Shape>, GraphError> {
        let x = ins.get("in").or_else(|| ins.get("a")).or_else(|| ins.get("x"));
        match self {
            Block::Linear | Block::Gcn => {
                let port = if *self == Block::Gcn { "x" } else { "in" };
                let x = &ins[port];
                last_const(node, port, x)?;
                let mut o = x.clone();
                *o.last_mut().unwrap() = Dim::Const(geti(p, "out"));
                if *self == Block::Gcn {
                    rank(node, "x", x, 3)?;
                    let adj = &ins["adj"];
                    rank(node, "adj", adj, 3)?;
                    if adj[1] != adj[2] || adj[1] != x[1] {
                        return Err(shape_err(
                            node,
                            "adj",
                            format!("adjacency {} must be [B, N, N] matching node count of x {}", crate::graph::fmt_shape(adj), crate::graph::fmt_shape(x)),
                        ));
                    }
                }
                Ok(one("out", o))
            }
            Block::Activation | Block::Dropout => Ok(one("out", x.unwrap().clone())),
            Block::LayerNorm => {
                let x = x.unwrap();
                last_const(node, "in", x)?;
                Ok(one("out", x.clone()))
            }
            Block::BatchNorm => {
                let x = x.unwrap();
                if x.len() < 2 {
                    return Err(shape_err(node, "in", "needs at least [B, F]"));
                }
                dim_at(node, "in", x, 1)?;
                Ok(one("out", x.clone()))
            }
            Block::Add => {
                let (a, b) = (&ins["a"], &ins["b"]);
                if a != b {
                    return Err(shape_err(
                        node,
                        "b",
                        format!("cannot add {} and {}: shapes differ", crate::graph::fmt_shape(a), crate::graph::fmt_shape(b)),
                    ));
                }
                Ok(one("out", a.clone()))
            }
            Block::Concat => {
                let (a, b) = (&ins["a"], &ins["b"]);
                if a.len() != b.len() {
                    return Err(shape_err(node, "b", "rank mismatch between a and b"));
                }
                let axis = geti(p, "axis");
                let r = a.len() as i64;
                let ax = if axis < 0 { axis + r } else { axis };
                if ax < 0 || ax >= r {
                    return Err(bad(node, "axis", format!("out of range for rank {r}")));
                }
                let ax = ax as usize;
                for i in 0..a.len() {
                    if i != ax && a[i] != b[i] {
                        return Err(shape_err(
                            node,
                            "b",
                            format!("{} and {} differ at dimension {i}", crate::graph::fmt_shape(a), crate::graph::fmt_shape(b)),
                        ));
                    }
                }
                let mut o = a.clone();
                o[ax] = match (&a[ax], &b[ax]) {
                    (Dim::Const(x), Dim::Const(y)) => Dim::Const(x + y),
                    (x, y) => Dim::Sym(format!("{x}+{y}")),
                };
                Ok(one("out", o))
            }
            Block::Lstm | Block::Gru => {
                let x = x.unwrap();
                rank(node, "in", x, 3)?;
                last_const(node, "in", x)?;
                let h = geti(p, "hidden") * if getb(p, "bidirectional") { 2 } else { 1 };
                Ok(BTreeMap::from([
                    ("out".to_string(), vec![x[0].clone(), x[1].clone(), Dim::Const(h)]),
                    ("last".to_string(), vec![x[0].clone(), Dim::Const(h)]),
                ]))
            }
            Block::Conv1d => {
                let x = x.unwrap();
                rank(node, "in", x, 3)?;
                dim_at(node, "in", x, 1)?;
                let l = conv_len(node, "in", &x[2], geti(p, "kernel"), geti(p, "stride"), geti(p, "padding"))?;
                Ok(one("out", vec![x[0].clone(), Dim::Const(geti(p, "out_channels")), l]))
            }
            Block::Conv2d => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                dim_at(node, "in", x, 1)?;
                let (k, s, pad) = (geti(p, "kernel"), geti(p, "stride"), geti(p, "padding"));
                let h = conv_len(node, "in", &x[2], k, s, pad)?;
                let w = conv_len(node, "in", &x[3], k, s, pad)?;
                Ok(one("out", vec![x[0].clone(), Dim::Const(geti(p, "out_channels")), h, w]))
            }
            Block::MaxPool2d => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                let k = geti(p, "kernel");
                let s = if geti(p, "stride") == 0 { k } else { geti(p, "stride") };
                let h = conv_len(node, "in", &x[2], k, s, 0)?;
                let w = conv_len(node, "in", &x[3], k, s, 0)?;
                Ok(one("out", vec![x[0].clone(), x[1].clone(), h, w]))
            }
            Block::Flatten => {
                let x = x.unwrap();
                if x.len() < 2 {
                    return Err(shape_err(node, "in", "needs at least [B, ...]"));
                }
                let mut prod = 1i64;
                let mut sym = false;
                for d in &x[1..] {
                    match d {
                        Dim::Const(n) => prod *= n,
                        Dim::Sym(_) => sym = true,
                    }
                }
                let flat = if sym {
                    Dim::Sym(x[1..].iter().map(|d| d.to_string()).collect::<Vec<_>>().join("*"))
                } else {
                    Dim::Const(prod)
                };
                Ok(one("out", vec![x[0].clone(), flat]))
            }
            Block::Reshape => {
                let x = x.unwrap();
                let target: Vec<i64> = p["shape"].as_sequence().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
                if x.len() >= 2 && x[1..].iter().all(|d| matches!(d, Dim::Const(_))) {
                    let have: i64 = x[1..].iter().map(|d| if let Dim::Const(n) = d { *n } else { 1 }).product();
                    let want: i64 = target.iter().product();
                    if have != want {
                        return Err(shape_err(node, "in", format!("cannot reshape {have} elements per sample into {target:?} ({want})")));
                    }
                }
                let mut o = vec![x[0].clone()];
                o.extend(target.into_iter().map(Dim::Const));
                Ok(one("out", o))
            }
            Block::Embedding => {
                let x = x.unwrap();
                rank(node, "in", x, 2)?;
                let mut o = x.clone();
                o.push(Dim::Const(geti(p, "dim")));
                Ok(one("out", o))
            }
            Block::MeanPool => {
                let x = x.unwrap();
                let d = geti(p, "dim");
                if d < 1 || d as usize >= x.len() {
                    return Err(bad(node, "dim", format!("must be in 1..{} (batch dim 0 is kept)", x.len())));
                }
                let mut o = x.clone();
                o.remove(d as usize);
                Ok(one("out", o))
            }
            Block::LastStep => {
                let x = x.unwrap();
                rank(node, "in", x, 3)?;
                Ok(one("out", vec![x[0].clone(), x[2].clone()]))
            }
            Block::Attention | Block::TransformerEncoder => {
                let x = x.unwrap();
                rank(node, "in", x, 3)?;
                let e = last_const(node, "in", x)?;
                let h = geti(p, "heads");
                if e % h != 0 {
                    return Err(bad(node, "heads", format!("embedding size {e} is not divisible by {h} heads")));
                }
                Ok(one("out", x.clone()))
            }
        }
    }

    /// Python expression building the `nn.Module`, or None for parameter-free tensor ops.
    pub fn ctor(&self, _node: &str, p: &Params, ins: &BTreeMap<String, Shape>) -> Option<String> {
        let x = ins.get("in").or_else(|| ins.get("x"));
        let last = |s: &Shape| match s.last() {
            Some(Dim::Const(n)) => *n,
            _ => 0,
        };
        let dimn = |s: &Shape, i: usize| match s.get(i) {
            Some(Dim::Const(n)) => *n,
            _ => 0,
        };
        Some(match self {
            Block::Linear => format!(
                "nn.Linear({}, {}, bias={})",
                last(x?),
                geti(p, "out"),
                if getb(p, "bias") { "True" } else { "False" }
            ),
            Block::Gcn => format!("nn.Linear({}, {})", last(x?), geti(p, "out")),
            Block::Activation => match gets(p, "kind").as_str() {
                "relu" => "nn.ReLU()".into(),
                "gelu" => "nn.GELU()".into(),
                "tanh" => "nn.Tanh()".into(),
                "sigmoid" => "nn.Sigmoid()".into(),
                "silu" => "nn.SiLU()".into(),
                "softmax" => "nn.Softmax(dim=-1)".into(),
                _ => "nn.LeakyReLU()".into(),
            },
            Block::Dropout => format!("nn.Dropout({})", getf(p, "p")),
            Block::LayerNorm => format!("nn.LayerNorm({})", last(x?)),
            Block::BatchNorm => {
                let s = x?;
                let f = dimn(s, 1);
                match s.len() {
                    2 => format!("nn.BatchNorm1d({f})"),
                    3 => format!("nn.BatchNorm1d({f})"),
                    _ => format!("nn.BatchNorm2d({f})"),
                }
            }
            Block::Lstm | Block::Gru => {
                let kind = if *self == Block::Lstm { "LSTM" } else { "GRU" };
                let layers = geti(p, "layers");
                let drop = if layers > 1 { format!(", dropout={}", getf(p, "dropout")) } else { String::new() };
                format!(
                    "nn.{kind}({}, {}, num_layers={layers}, batch_first=True, bidirectional={}{drop})",
                    last(x?),
                    geti(p, "hidden"),
                    if getb(p, "bidirectional") { "True" } else { "False" }
                )
            }
            Block::Conv1d => format!(
                "nn.Conv1d({}, {}, kernel_size={}, stride={}, padding={})",
                dimn(x?, 1),
                geti(p, "out_channels"),
                geti(p, "kernel"),
                geti(p, "stride"),
                geti(p, "padding")
            ),
            Block::Conv2d => format!(
                "nn.Conv2d({}, {}, kernel_size={}, stride={}, padding={})",
                dimn(x?, 1),
                geti(p, "out_channels"),
                geti(p, "kernel"),
                geti(p, "stride"),
                geti(p, "padding")
            ),
            Block::MaxPool2d => {
                let k = geti(p, "kernel");
                let s = if geti(p, "stride") == 0 { k } else { geti(p, "stride") };
                format!("nn.MaxPool2d({k}, {s})")
            }
            Block::Flatten => "nn.Flatten()".into(),
            Block::Embedding => format!("nn.Embedding({}, {})", geti(p, "num_embeddings"), geti(p, "dim")),
            Block::Attention => format!("nn.MultiheadAttention({}, {}, batch_first=True)", last(x?), geti(p, "heads")),
            Block::TransformerEncoder => format!(
                "nn.TransformerEncoderLayer(d_model={}, nhead={}, dim_feedforward={}, dropout={}, batch_first=True)",
                last(x?),
                geti(p, "heads"),
                geti(p, "ff"),
                getf(p, "dropout")
            ),
            Block::Add | Block::Concat | Block::Reshape | Block::MeanPool | Block::LastStep => return None,
        })
    }

    /// Python statement(s) for the forward pass. `iv`/`ov` map port -> variable name.
    pub fn forward(&self, node: &str, p: &Params, iv: &BTreeMap<String, String>, ov: &BTreeMap<String, String>) -> String {
        let o = &ov["out"];
        match self {
            Block::Add => format!("{o} = {} + {}", iv["a"], iv["b"]),
            Block::Concat => format!("{o} = torch.cat([{}, {}], dim={})", iv["a"], iv["b"], geti(p, "axis")),
            Block::Reshape => {
                let dims: Vec<String> = p["shape"].as_sequence().unwrap().iter().map(|v| v.as_i64().unwrap().to_string()).collect();
                format!("{o} = {}.reshape({}.shape[0], {})", iv["in"], iv["in"], dims.join(", "))
            }
            Block::MeanPool => format!("{o} = {}.mean(dim={})", iv["in"], geti(p, "dim")),
            Block::LastStep => format!("{o} = {}[:, -1]", iv["in"]),
            Block::Lstm => {
                let last = if getb(p, "bidirectional") { "torch.cat([_h[-2], _h[-1]], dim=-1)" } else { "_h[-1]" };
                format!("{o}, (_h, _) = self.{node}({})\n        {} = {last}", iv["in"], ov["last"])
            }
            Block::Gru => {
                let last = if getb(p, "bidirectional") { "torch.cat([_h[-2], _h[-1]], dim=-1)" } else { "_h[-1]" };
                format!("{o}, _h = self.{node}({})\n        {} = {last}", iv["in"], ov["last"])
            }
            Block::Attention => format!("{o}, _ = self.{node}({0}, {0}, {0})", iv["in"]),
            Block::Gcn => format!("{o} = torch.bmm({}, self.{node}({}))", iv["adj"], iv["x"]),
            _ => format!("{o} = self.{node}({})", iv["in"]),
        }
    }
}
