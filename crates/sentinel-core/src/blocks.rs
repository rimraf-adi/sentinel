//! Block library: every neural-network building block (ANN, RNN, CNN, attention, GNN, ...)
//! with its ports, parameters, shape rule, and PyTorch emission.

use crate::graph::{Dim, GraphError, Params, Shape};
use serde_yaml::Value;
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
    AvgPool2d,
    GlobalAvgPool,
    ConvTranspose2d,
    Upsample,
    PatchEmbed,
    ClsToken,
    ClsSelect,
    PositionalEncoding,
    Sampling,
    Gat,
    Rnn,
    Multiply,
    Subtract,
    Divide,
    Scale,
    Clamp,
    Reduce,
    Argmax,
    Permute,
    Transpose,
    Squeeze,
    Unsqueeze,
    Split,
    Slice,
    Stack,
    Repeat,
    Pad,
    Matmul,
    GroupNorm,
    InstanceNorm,
    RmsNorm,
    L2Normalize,
    Dropout2d,
    AlphaDropout,
    GaussianNoise,
    MaxPool1d,
    AvgPool1d,
    AdaptiveAvgPool2d,
    GlobalMaxPool,
    Bilinear,
    Conv3d,
    ConvTranspose1d,
    PixelShuffle,
    DepthwiseConv2d,
    Rope,
    CrossAttention,
    SageConv,
    GinConv,
    EdgeDropout,
    NoiseLike,
    VectorQuantizer,
    GumbelSoftmax,
}

/// Every variant implemented in `more.rs` (the second batch of blocks).
macro_rules! more_blocks {
    () => {
        Block::Multiply | Block::Subtract | Block::Divide | Block::Scale | Block::Clamp | Block::Reduce | Block::Argmax | Block::Permute | Block::Transpose | Block::Squeeze | Block::Unsqueeze | Block::Split | Block::Slice | Block::Stack | Block::Repeat | Block::Pad | Block::Matmul | Block::GroupNorm | Block::InstanceNorm | Block::RmsNorm | Block::L2Normalize | Block::Dropout2d | Block::AlphaDropout | Block::GaussianNoise | Block::MaxPool1d | Block::AvgPool1d | Block::AdaptiveAvgPool2d | Block::GlobalMaxPool | Block::Bilinear | Block::Conv3d | Block::ConvTranspose1d | Block::PixelShuffle | Block::DepthwiseConv2d | Block::Rope | Block::CrossAttention | Block::SageConv | Block::GinConv | Block::EdgeDropout | Block::NoiseLike | Block::VectorQuantizer | Block::GumbelSoftmax
    };
}

mod more;

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
    Block::AvgPool2d,
    Block::GlobalAvgPool,
    Block::ConvTranspose2d,
    Block::Upsample,
    Block::PatchEmbed,
    Block::ClsToken,
    Block::ClsSelect,
    Block::PositionalEncoding,
    Block::Sampling,
    Block::Gat,
    Block::Rnn,
    Block::Multiply,
    Block::Subtract,
    Block::Divide,
    Block::Scale,
    Block::Clamp,
    Block::Reduce,
    Block::Argmax,
    Block::Permute,
    Block::Transpose,
    Block::Squeeze,
    Block::Unsqueeze,
    Block::Split,
    Block::Slice,
    Block::Stack,
    Block::Repeat,
    Block::Pad,
    Block::Matmul,
    Block::GroupNorm,
    Block::InstanceNorm,
    Block::RmsNorm,
    Block::L2Normalize,
    Block::Dropout2d,
    Block::AlphaDropout,
    Block::GaussianNoise,
    Block::MaxPool1d,
    Block::AvgPool1d,
    Block::AdaptiveAvgPool2d,
    Block::GlobalMaxPool,
    Block::Bilinear,
    Block::Conv3d,
    Block::ConvTranspose1d,
    Block::PixelShuffle,
    Block::DepthwiseConv2d,
    Block::Rope,
    Block::CrossAttention,
    Block::SageConv,
    Block::GinConv,
    Block::EdgeDropout,
    Block::NoiseLike,
    Block::VectorQuantizer,
    Block::GumbelSoftmax,
];

const HELPER_POSITIONAL: &str = r#"class PositionalEncoding(nn.Module):
    def __init__(self, max_len, dim, kind="learned"):
        super().__init__()
        if kind == "learned":
            self.pe = nn.Parameter(torch.zeros(1, max_len, dim))
            nn.init.normal_(self.pe, std=0.02)
        else:
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
            pe = torch.zeros(1, max_len, dim)
            pe[0, :, 0::2] = torch.sin(pos * div)
            pe[0, :, 1::2] = torch.cos(pos * div)[:, : dim // 2]
            self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]
"#;

const HELPER_CLS: &str = r#"class ClsToken(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x):
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
"#;

const HELPER_SAMPLING: &str = r#"class Sampling(nn.Module):
    def forward(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu
"#;

const HELPER_GAT: &str = r#"class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features, heads=4, concat=True):
        super().__init__()
        self.heads, self.out, self.concat = heads, out_features, concat
        self.proj = nn.Linear(in_features, heads * out_features, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, 1, heads, out_features))
        self.att_dst = nn.Parameter(torch.empty(1, 1, heads, out_features))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        B, N, _ = x.shape
        h = self.proj(x).view(B, N, self.heads, self.out)
        e = self.leaky((h * self.att_src).sum(-1).unsqueeze(2) + (h * self.att_dst).sum(-1).unsqueeze(1))
        mask = ((adj + torch.eye(N, device=adj.device)) > 0).unsqueeze(-1)
        a = torch.softmax(e.masked_fill(~mask, float("-inf")), dim=2)
        out = torch.einsum("bijh,bjhd->bihd", a, h)
        return out.reshape(B, N, -1) if self.concat else out.mean(dim=2)
"#;

/// Every activation the `activation` block can build (`alpha` tunes the ones that take a parameter).
pub const ACTIVATIONS: &[&str] = &[
    "relu", "leaky_relu", "prelu", "elu", "selu", "gelu", "gelu_tanh", "silu", "mish", "softplus", "softsign", "tanh", "sigmoid", "hardtanh",
    "hardswish", "hardsigmoid", "relu6", "logsigmoid", "softmax", "log_softmax", "tanhshrink", "glu", "identity",
];

/// (name, default as JSON — empty string means required, help)
type ParamDef = (&'static str, &'static str, &'static str);

#[derive(Serialize)]
pub struct ParamInfo {
    pub name: String,
    pub required: bool,
    pub default: Option<serde_json::Value>,
    pub help: String,
    /// For choice parameters: the allowed values (empty otherwise), so UIs need not hardcode them.
    pub options: Vec<String>,
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
            Block::AvgPool2d => "avgpool2d",
            Block::GlobalAvgPool => "global_avg_pool",
            Block::ConvTranspose2d => "conv_transpose2d",
            Block::Upsample => "upsample",
            Block::PatchEmbed => "patch_embed",
            Block::ClsToken => "cls_token",
            Block::ClsSelect => "cls_select",
            Block::PositionalEncoding => "positional_encoding",
            Block::Sampling => "sampling",
            Block::Gat => "gat",
            Block::Rnn => "rnn",
            more_blocks!() => self.more_name(),
        }
    }

    pub fn family(&self) -> &'static str {
        match self {
            Block::Linear | Block::Activation | Block::Dropout | Block::LayerNorm | Block::BatchNorm => "core",
            Block::Add | Block::Concat | Block::Flatten | Block::Reshape | Block::MeanPool | Block::LastStep => "shape",
            Block::Lstm | Block::Gru | Block::Rnn | Block::Embedding => "sequence",
            Block::Attention | Block::TransformerEncoder => "attention",
            Block::Conv1d | Block::Conv2d | Block::MaxPool2d | Block::AvgPool2d | Block::GlobalAvgPool | Block::ConvTranspose2d | Block::Upsample | Block::PatchEmbed => "vision",
            Block::PositionalEncoding | Block::ClsToken | Block::ClsSelect => "attention",
            Block::Sampling => "generative",
            Block::Gcn | Block::Gat => "graph",
            more_blocks!() => self.more_family(),
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Block::Linear => "Fully connected layer on the last dimension (the ANN building block)",
            Block::Activation => "Nonlinearity: relu, leaky_relu, prelu (learnable), elu, selu, gelu, silu, mish, softplus, tanh, sigmoid, hard variants, softmax, glu (halves the width), identity and more",
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
            Block::AvgPool2d => "2-D average pooling",
            Block::GlobalAvgPool => "Average over the spatial dimensions: [B, C, H, W] -> [B, C]",
            Block::ConvTranspose2d => "Transposed convolution that upsamples [B, C, H, W] (decoders, U-Net)",
            Block::Upsample => "Resize [B, C, H, W] by an integer factor (nearest or bilinear)",
            Block::PatchEmbed => "Cut an image into patches and embed each one: [B, C, H, W] -> [B, N, dim] (Vision Transformer)",
            Block::ClsToken => "Prepend a learned classification token: [B, T, E] -> [B, T+1, E]",
            Block::ClsSelect => "Take the first token (the classification token): [B, T, E] -> [B, E]",
            Block::PositionalEncoding => "Add position information to a sequence [B, T, E] (learned or sinusoidal)",
            Block::Sampling => "VAE reparameterization: z = mu + eps * exp(logvar / 2) while training, mu at inference",
            Block::Gat => "Graph attention over node features [B, N, F] and adjacency [B, N, N], with multiple heads",
            Block::Rnn => "Plain recurrent network (tanh or relu) over a sequence [B, T, F]",
            more_blocks!() => self.more_description(),
        }
    }

    pub fn inputs(&self) -> &'static [&'static str] {
        match self {
            Block::Add | Block::Concat | Block::Multiply | Block::Subtract | Block::Divide | Block::Stack | Block::Matmul | Block::Bilinear => &["a", "b"],
            Block::Gcn | Block::Gat | Block::SageConv | Block::GinConv => &["x", "adj"],
            Block::CrossAttention => &["q", "kv"],
            Block::Sampling => &["mu", "logvar"],
            _ => &["in"],
        }
    }

    pub fn outputs(&self) -> &'static [&'static str] {
        match self {
            Block::Lstm | Block::Gru | Block::Rnn => &["out", "last"],
            Block::Split => &["a", "b"],
            Block::VectorQuantizer => &["out", "indices"],
            _ => &["out"],
        }
    }

    fn param_defs(&self) -> &'static [ParamDef] {
        match self {
            Block::Linear => &[("out", "", "Output features"), ("bias", "true", "Learn an additive bias")],
            Block::Activation => &[
                ("kind", "\"relu\"", "which nonlinearity (see the options)"),
                ("alpha", "null", "leaky_relu: negative slope (0.01), elu: alpha (1.0), softplus: beta (1.0); ignored by the others"),
            ],
            Block::Dropout => &[("p", "0.1", "Drop probability")],
            Block::Concat => &[("axis", "-1", "Axis to concatenate along")],
            Block::Rnn => &[
                ("hidden", "", "Hidden units"),
                ("layers", "1", "Stacked layers"),
                ("bidirectional", "false", "Run in both directions"),
                ("dropout", "0.0", "Dropout between layers"),
                ("nonlinearity", "\"tanh\"", "tanh | relu"),
            ],
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
            Block::Attention => &[("heads", "4", "Attention heads"), ("causal", "false", "Mask future positions (autoregressive models)")],
            Block::TransformerEncoder => &[
                ("heads", "4", "Attention heads"),
                ("ff", "256", "Feed-forward width"),
                ("dropout", "0.1", "Dropout"),
                ("causal", "false", "Mask future positions (autoregressive models)"),
            ],
            Block::Gcn => &[("out", "", "Output features")],
            Block::AvgPool2d => &[("kernel", "2", "Window size"), ("stride", "0", "Stride (0 = kernel)")],
            Block::ConvTranspose2d => &[
                ("out_channels", "", "Output channels"),
                ("kernel", "2", "Kernel size"),
                ("stride", "2", "Stride (the upsampling factor)"),
                ("padding", "0", "Zero padding"),
            ],
            Block::Upsample => &[("scale", "2", "Integer scale factor"), ("mode", "\"nearest\"", "nearest | bilinear")],
            Block::PatchEmbed => &[("dim", "", "Embedding size"), ("patch", "", "Patch size (image sides must be divisible by it)")],
            Block::PositionalEncoding => &[("max_len", "512", "Longest supported sequence"), ("kind", "\"learned\"", "learned | sinusoidal")],
            Block::Gat => &[("out", "", "Output features per head"), ("heads", "4", "Attention heads"), ("concat", "true", "Concatenate heads (else average)")],
            Block::LayerNorm | Block::BatchNorm | Block::Add | Block::Flatten | Block::LastStep | Block::GlobalAvgPool | Block::ClsToken | Block::ClsSelect | Block::Sampling => &[],
            more_blocks!() => self.more_param_defs(),
        }
    }

    /// Allowed values of a choice parameter.
    fn param_options(&self, name: &str) -> Vec<String> {
        let list: &[&str] = match (self, name) {
            (Block::Activation, "kind") => ACTIVATIONS,
            (Block::Upsample, "mode") => &["nearest", "bilinear"],
            (Block::PositionalEncoding, "kind") => &["learned", "sinusoidal"],
            (Block::Rnn, "nonlinearity") => &["tanh", "relu"],
            (Block::Reduce, "op") => &["sum", "mean", "max", "min", "std", "prod"],
            _ => &[],
        };
        list.iter().map(|s| s.to_string()).collect()
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
                    options: self.param_options(n),
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
            Block::Gat => {
                pos("out")?;
                pos("heads")?;
            }
            Block::AvgPool2d => pos("kernel")?,
            Block::ConvTranspose2d => {
                pos("out_channels")?;
                pos("kernel")?;
                pos("stride")?;
            }
            Block::Upsample => {
                pos("scale")?;
                let m = gets(p, "mode");
                if m != "nearest" && m != "bilinear" {
                    return Err(bad(node, "mode", "must be nearest or bilinear"));
                }
            }
            Block::PatchEmbed => {
                pos("dim")?;
                pos("patch")?;
            }
            Block::PositionalEncoding => {
                pos("max_len")?;
                let k = gets(p, "kind");
                if k != "learned" && k != "sinusoidal" {
                    return Err(bad(node, "kind", "must be learned or sinusoidal"));
                }
            }
            Block::Lstm | Block::Gru | Block::Rnn => {
                pos("hidden")?;
                pos("layers")?;
                if *self == Block::Rnn && !matches!(gets(p, "nonlinearity").as_str(), "tanh" | "relu") {
                    return Err(bad(node, "nonlinearity", "must be tanh or relu"));
                }
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
                let k = if k == "swish" { "silu".to_string() } else { k };
                if !ACTIVATIONS.contains(&k.as_str()) {
                    return Err(bad(node, "kind", format!("unsupported activation '{k}' (options: {})", ACTIVATIONS.join(", "))));
                }
                match p.get("alpha") {
                    None | Some(Value::Null) => {}
                    Some(v) if v.as_f64().is_some_and(|a| a.is_finite() && a > 0.0) => {}
                    Some(_) => return Err(bad(node, "alpha", "must be a positive number (or null for the default)")),
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
            more_blocks!() => self.more_check_params(node, p)?,
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
            Block::Activation if gets(p, "kind") == "glu" => {
                let x = x.unwrap();
                let n = last_const(node, "in", x)?;
                if n % 2 != 0 {
                    return Err(shape_err(node, "in", format!("glu halves the last dimension, which must be even (found {n})")));
                }
                let mut o = x.clone();
                *o.last_mut().unwrap() = Dim::Const(n / 2);
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
            Block::Lstm | Block::Gru | Block::Rnn => {
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
            Block::AvgPool2d => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                let k = geti(p, "kernel");
                let st = if geti(p, "stride") == 0 { k } else { geti(p, "stride") };
                let h = conv_len(node, "in", &x[2], k, st, 0)?;
                let w = conv_len(node, "in", &x[3], k, st, 0)?;
                Ok(one("out", vec![x[0].clone(), x[1].clone(), h, w]))
            }
            Block::GlobalAvgPool => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                Ok(one("out", vec![x[0].clone(), x[1].clone()]))
            }
            Block::ConvTranspose2d => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                dim_at(node, "in", x, 1)?;
                let (k, st, pad) = (geti(p, "kernel"), geti(p, "stride"), geti(p, "padding"));
                let grow = |d: &Dim| match d {
                    Dim::Const(n) => {
                        let v = (n - 1) * st - 2 * pad + k;
                        if v < 1 { Err(shape_err(node, "in", format!("transposed convolution of size {n} would be empty"))) } else { Ok(Dim::Const(v)) }
                    }
                    Dim::Sym(s) => Ok(Dim::Sym(format!("{s}'"))),
                };
                Ok(one("out", vec![x[0].clone(), Dim::Const(geti(p, "out_channels")), grow(&x[2])?, grow(&x[3])?]))
            }
            Block::Upsample => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                let sc = geti(p, "scale");
                let up = |d: &Dim| match d {
                    Dim::Const(n) => Dim::Const(n * sc),
                    Dim::Sym(s) => Dim::Sym(format!("{s}*{sc}")),
                };
                Ok(one("out", vec![x[0].clone(), x[1].clone(), up(&x[2]), up(&x[3])]))
            }
            Block::PatchEmbed => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                dim_at(node, "in", x, 1)?;
                let (h, w, pt) = (dim_at(node, "in", x, 2)?, dim_at(node, "in", x, 3)?, geti(p, "patch"));
                if h % pt != 0 || w % pt != 0 {
                    return Err(shape_err(node, "in", format!("image {h}x{w} is not divisible by patch size {pt}")));
                }
                Ok(one("out", vec![x[0].clone(), Dim::Const((h / pt) * (w / pt)), Dim::Const(geti(p, "dim"))]))
            }
            Block::ClsToken => {
                let x = x.unwrap();
                rank(node, "in", x, 3)?;
                last_const(node, "in", x)?;
                let t = match &x[1] {
                    Dim::Const(n) => Dim::Const(n + 1),
                    Dim::Sym(s) => Dim::Sym(format!("{s}+1")),
                };
                Ok(one("out", vec![x[0].clone(), t, x[2].clone()]))
            }
            Block::ClsSelect => {
                let x = x.unwrap();
                rank(node, "in", x, 3)?;
                Ok(one("out", vec![x[0].clone(), x[2].clone()]))
            }
            Block::PositionalEncoding => {
                let x = x.unwrap();
                rank(node, "in", x, 3)?;
                last_const(node, "in", x)?;
                if let Dim::Const(t) = &x[1] {
                    if *t > geti(p, "max_len") {
                        return Err(shape_err(node, "in", format!("sequence length {t} exceeds max_len {}", geti(p, "max_len"))));
                    }
                }
                Ok(one("out", x.clone()))
            }
            Block::Sampling => {
                let (mu, lv) = (&ins["mu"], &ins["logvar"]);
                if mu != lv {
                    return Err(shape_err(node, "logvar", format!("mu {} and logvar {} must have the same shape", crate::graph::fmt_shape(mu), crate::graph::fmt_shape(lv))));
                }
                Ok(one("out", mu.clone()))
            }
            Block::Gat => {
                let x = &ins["x"];
                rank(node, "x", x, 3)?;
                last_const(node, "x", x)?;
                let adj = &ins["adj"];
                rank(node, "adj", adj, 3)?;
                if adj[1] != adj[2] || adj[1] != x[1] {
                    return Err(shape_err(
                        node,
                        "adj",
                        format!("adjacency {} must be [B, N, N] matching node count of x {}", crate::graph::fmt_shape(adj), crate::graph::fmt_shape(x)),
                    ));
                }
                let out = geti(p, "out") * if getb(p, "concat") { geti(p, "heads") } else { 1 };
                Ok(one("out", vec![x[0].clone(), x[1].clone(), Dim::Const(out)]))
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
            more_blocks!() => self.more_infer(node, p, ins),
        }
    }

    /// Learnable parameters of this block for the given resolved params and input shapes. Matches
    /// `sum(p.numel() for p in module.parameters())` of the generated PyTorch module (buffers such as batch-norm running
    /// statistics are not parameters).
    pub fn param_count(&self, p: &Params, ins: &BTreeMap<String, Shape>) -> u64 {
        let first = ins.get("in").or_else(|| ins.get("x"));
        let last = |s: Option<&Shape>| -> u64 {
            match s.and_then(|s| s.last()) {
                Some(Dim::Const(n)) => *n as u64,
                _ => 0,
            }
        };
        let chan = |s: Option<&Shape>| -> u64 {
            match s.and_then(|s| s.get(1)) {
                Some(Dim::Const(n)) => *n as u64,
                _ => 0,
            }
        };
        let (i, o) = (last(first), geti(p, "out") as u64);
        let h = geti(p, "hidden") as u64;
        let k = geti(p, "kernel") as u64;
        match self {
            Block::Linear => i * o + if getb(p, "bias") { o } else { 0 },
            Block::Gcn => i * o + o,
            Block::LayerNorm => 2 * i,
            Block::BatchNorm => 2 * chan(first),
            Block::Lstm | Block::Gru | Block::Rnn => {
                let gates = match self { Block::Lstm => 4, Block::Gru => 3, _ => 1 };
                let dirs = if getb(p, "bidirectional") { 2 } else { 1 };
                let mut total = 0;
                for layer in 0..geti(p, "layers") as u64 {
                    let inp = if layer == 0 { i } else { h * dirs };
                    total += dirs * (gates * h * inp + gates * h * h + 2 * gates * h);
                }
                total
            }
            Block::Conv1d => {
                let oc = geti(p, "out_channels") as u64;
                oc * chan(first) * k + oc
            }
            Block::Conv2d => {
                let oc = geti(p, "out_channels") as u64;
                oc * chan(first) * k * k + oc
            }
            Block::ConvTranspose2d => {
                let oc = geti(p, "out_channels") as u64;
                chan(first) * oc * k * k + oc
            }
            Block::Embedding => geti(p, "num_embeddings") as u64 * geti(p, "dim") as u64,
            Block::Attention => 4 * i * i + 4 * i,
            Block::TransformerEncoder => {
                let ff = geti(p, "ff") as u64;
                4 * i * i + 4 * i + 2 * i * ff + ff + i + 4 * i
            }
            Block::PatchEmbed => {
                let (d, pt) = (geti(p, "dim") as u64, geti(p, "patch") as u64);
                d * chan(first) * pt * pt + d
            }
            Block::ClsToken => i,
            Block::PositionalEncoding => if gets(p, "kind") == "learned" { geti(p, "max_len") as u64 * i } else { 0 },
            Block::Gat => {
                let (hd, od) = (geti(p, "heads") as u64, geti(p, "out") as u64);
                i * hd * od + 2 * hd * od
            }
            Block::Activation if gets(p, "kind") == "prelu" => 1,
            Block::Activation | Block::Dropout | Block::Add | Block::Concat | Block::MaxPool2d | Block::AvgPool2d | Block::GlobalAvgPool | Block::Flatten
            | Block::Reshape | Block::Upsample | Block::MeanPool | Block::LastStep | Block::ClsSelect | Block::Sampling => 0,
            more_blocks!() => self.more_param_count(p, ins),
        }
    }

    /// Multiply-accumulates per sample (the batch dimension is excluded) in the matrix and convolution layers, where
    /// nearly all compute lives; element-wise operations are not counted. `None` when a needed size is symbolic.
    pub fn macs(&self, p: &Params, ins: &BTreeMap<String, Shape>, outs: &BTreeMap<String, Shape>) -> Option<u64> {
        let first = ins.get("in").or_else(|| ins.get("x"));
        let c = |s: &Shape, i: usize| -> Option<u64> {
            match s.get(i) {
                Some(Dim::Const(n)) => Some(*n as u64),
                _ => None,
            }
        };
        // product of the dims after the batch dim up to (not including) the last
        let rows = |s: &Shape| -> Option<u64> { (1..s.len().saturating_sub(1)).try_fold(1u64, |a, i| Some(a * c(s, i)?)) };
        let out = outs.get("out");
        match self {
            Block::Linear | Block::Gcn => {
                let x = first?;
                let n = rows(x)?;
                let (i, o) = (c(x, x.len() - 1)?, geti(p, "out") as u64);
                let mut m = n * i * o;
                if *self == Block::Gcn {
                    m += n * n * o; // adj @ (x W)
                }
                Some(m)
            }
            Block::Conv1d => {
                let (x, y) = (first?, out?);
                Some(c(y, 2)? * c(y, 1)? * c(x, 1)? * geti(p, "kernel") as u64)
            }
            Block::Conv2d => {
                let (x, y) = (first?, out?);
                Some(c(y, 2)? * c(y, 3)? * c(y, 1)? * c(x, 1)? * (geti(p, "kernel") * geti(p, "kernel")) as u64)
            }
            Block::ConvTranspose2d => {
                let (x, y) = (first?, out?);
                Some(c(x, 2)? * c(x, 3)? * c(x, 1)? * c(y, 1)? * (geti(p, "kernel") * geti(p, "kernel")) as u64)
            }
            Block::PatchEmbed => {
                let (x, y) = (first?, out?);
                Some(c(y, 1)? * c(y, 2)? * c(x, 1)? * (geti(p, "patch") * geti(p, "patch")) as u64)
            }
            Block::Lstm | Block::Gru | Block::Rnn => {
                let x = first?;
                let (t, f) = (c(x, 1)?, c(x, 2)?);
                let gates = match self { Block::Lstm => 4, Block::Gru => 3, _ => 1 };
                let h = geti(p, "hidden") as u64;
                let dirs = if getb(p, "bidirectional") { 2 } else { 1 };
                let mut per_step = 0;
                for layer in 0..geti(p, "layers") as u64 {
                    let inp = if layer == 0 { f } else { h * dirs };
                    per_step += dirs * gates * h * (inp + h);
                }
                Some(t * per_step)
            }
            Block::Attention | Block::TransformerEncoder => {
                let x = first?;
                let (t, e) = (c(x, 1)?, c(x, 2)?);
                let mut m = 4 * t * e * e + 2 * t * t * e;
                if *self == Block::TransformerEncoder {
                    m += 2 * t * e * geti(p, "ff") as u64;
                }
                Some(m)
            }
            Block::Gat => {
                let x = first?;
                let (n, f) = (c(x, 1)?, c(x, 2)?);
                let (hd, od) = (geti(p, "heads") as u64, geti(p, "out") as u64);
                Some(n * f * hd * od + n * n * hd * od)
            }
            more_blocks!() => self.more_macs(p, ins, outs),
            _ => Some(0),
        }
    }

    /// Blocks that are not a stock `torch.nn` layer need a small helper class in the generated code:
    /// `(class name, Python source)`. Code generation emits each one once.
    pub fn helper(&self) -> Option<(&'static str, &'static str)> {
        match self {
            Block::PositionalEncoding => Some(("PositionalEncoding", HELPER_POSITIONAL)),
            Block::ClsToken => Some(("ClsToken", HELPER_CLS)),
            Block::Sampling => Some(("Sampling", HELPER_SAMPLING)),
            Block::Gat => Some(("GraphAttention", HELPER_GAT)),
            more_blocks!() => self.more_helper(),
            _ => None,
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
            Block::Activation => {
                let alpha = p.get("alpha").and_then(|v| v.as_f64());
                match gets(p, "kind").as_str() {
                    "relu" => "nn.ReLU()".into(),
                    "leaky_relu" => format!("nn.LeakyReLU({})", alpha.unwrap_or(0.01)),
                    "prelu" => "nn.PReLU()".into(),
                    "elu" => format!("nn.ELU({})", alpha.unwrap_or(1.0)),
                    "selu" => "nn.SELU()".into(),
                    "gelu" => "nn.GELU()".into(),
                    "gelu_tanh" => "nn.GELU(approximate='tanh')".into(),
                    "silu" | "swish" => "nn.SiLU()".into(),
                    "mish" => "nn.Mish()".into(),
                    "softplus" => format!("nn.Softplus(beta={})", alpha.unwrap_or(1.0)),
                    "softsign" => "nn.Softsign()".into(),
                    "tanh" => "nn.Tanh()".into(),
                    "sigmoid" => "nn.Sigmoid()".into(),
                    "hardtanh" => "nn.Hardtanh()".into(),
                    "hardswish" => "nn.Hardswish()".into(),
                    "hardsigmoid" => "nn.Hardsigmoid()".into(),
                    "relu6" => "nn.ReLU6()".into(),
                    "logsigmoid" => "nn.LogSigmoid()".into(),
                    "softmax" => "nn.Softmax(dim=-1)".into(),
                    "log_softmax" => "nn.LogSoftmax(dim=-1)".into(),
                    "tanhshrink" => "nn.Tanhshrink()".into(),
                    "glu" => "nn.GLU(dim=-1)".into(),
                    _ => "nn.Identity()".into(),
                }
            }
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
            Block::Lstm | Block::Gru | Block::Rnn => {
                let kind = match self { Block::Lstm => "LSTM", Block::Gru => "GRU", _ => "RNN" };
                let extra = if *self == Block::Rnn { format!(", nonlinearity='{}'", gets(p, "nonlinearity")) } else { String::new() };
                let layers = geti(p, "layers");
                let drop = if layers > 1 { format!(", dropout={}", getf(p, "dropout")) } else { String::new() };
                format!(
                    "nn.{kind}({}, {}, num_layers={layers}, batch_first=True, bidirectional={}{drop}{extra})",
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
            Block::AvgPool2d => {
                let k = geti(p, "kernel");
                let st = if geti(p, "stride") == 0 { k } else { geti(p, "stride") };
                format!("nn.AvgPool2d({k}, {st})")
            }
            Block::ConvTranspose2d => format!(
                "nn.ConvTranspose2d({}, {}, kernel_size={}, stride={}, padding={})",
                dimn(x?, 1),
                geti(p, "out_channels"),
                geti(p, "kernel"),
                geti(p, "stride"),
                geti(p, "padding")
            ),
            Block::Upsample => {
                let mode = gets(p, "mode");
                let align = if mode == "bilinear" { ", align_corners=False" } else { "" };
                format!("nn.Upsample(scale_factor={}, mode='{mode}'{align})", geti(p, "scale"))
            }
            Block::PatchEmbed => format!("nn.Conv2d({}, {}, kernel_size={}, stride={})", dimn(x?, 1), geti(p, "dim"), geti(p, "patch"), geti(p, "patch")),
            Block::ClsToken => format!("ClsToken({})", last(x?)),
            Block::PositionalEncoding => format!("PositionalEncoding({}, {}, \"{}\")", geti(p, "max_len"), last(x?), gets(p, "kind")),
            Block::Sampling => "Sampling()".into(),
            Block::Gat => format!("GraphAttention({}, {}, {}, {})", last(x?), geti(p, "out"), geti(p, "heads"), if getb(p, "concat") { "True" } else { "False" }),
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
            Block::Add | Block::Concat | Block::Reshape | Block::MeanPool | Block::LastStep | Block::GlobalAvgPool | Block::ClsSelect => return None,
            more_blocks!() => return self.more_ctor(p, ins),
        })
    }

    /// Python statement(s) for the forward pass. `iv`/`ov` map port -> variable name.
    pub fn forward(&self, node: &str, p: &Params, iv: &BTreeMap<String, String>, ov: &BTreeMap<String, String>) -> String {
        let o = ov.get("out").map(String::as_str).unwrap_or("");
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
            Block::Gru | Block::Rnn => {
                let last = if getb(p, "bidirectional") { "torch.cat([_h[-2], _h[-1]], dim=-1)" } else { "_h[-1]" };
                format!("{o}, _h = self.{node}({})\n        {} = {last}", iv["in"], ov["last"])
            }
            Block::Attention if getb(p, "causal") => format!(
                "{o}, _ = self.{node}({0}, {0}, {0}, attn_mask=torch.triu(torch.ones({0}.shape[1], {0}.shape[1], dtype=torch.bool, device={0}.device), diagonal=1))",
                iv["in"]
            ),
            Block::TransformerEncoder if getb(p, "causal") => format!(
                "{o} = self.{node}({0}, src_mask=torch.triu(torch.full(({0}.shape[1], {0}.shape[1]), float('-inf'), device={0}.device), diagonal=1))",
                iv["in"]
            ),
            Block::Attention => format!("{o}, _ = self.{node}({0}, {0}, {0})", iv["in"]),
            Block::Gcn => format!("{o} = torch.bmm({}, self.{node}({}))", iv["adj"], iv["x"]),
            Block::GlobalAvgPool => format!("{o} = {}.mean(dim=(2, 3))", iv["in"]),
            Block::ClsSelect => format!("{o} = {}[:, 0]", iv["in"]),
            Block::PatchEmbed => format!("{o} = self.{node}({}).flatten(2).transpose(1, 2)", iv["in"]),
            Block::Sampling => format!("{o} = self.{node}({}, {})", iv["mu"], iv["logvar"]),
            Block::Gat => format!("{o} = self.{node}({}, {})", iv["x"], iv["adj"]),
            more_blocks!() => self.more_forward(node, p, iv, ov),
            _ => format!("{o} = self.{node}({})", iv["in"]),
        }
    }
}
