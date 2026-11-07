//! The second batch of fundamental blocks: tensor operations, extra norms/dropouts/pools, 3-D and 1-D convolution
//! variants, rotary embeddings, cross-attention, more graph layers and a few generative pieces. `blocks.rs` routes
//! every variant listed in `more_blocks!` here.

use super::*;

const HELPER_NOISE: &str = r#"class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std if self.training else x
"#;

const HELPER_ROPE: &str = r#"class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device).float()
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
"#;

const HELPER_SAGE: &str = r#"class GraphSage(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.self_lin = nn.Linear(in_features, out_features)
        self.neigh_lin = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        deg = adj.sum(-1, keepdim=True).clamp(min=1)
        return self.self_lin(x) + self.neigh_lin(torch.bmm(adj / deg, x))
"#;

const HELPER_GIN: &str = r#"class GraphIsomorphism(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU(), nn.Linear(out_features, out_features))

    def forward(self, x, adj):
        return self.mlp((1 + self.eps) * x + torch.bmm(adj, x))
"#;

const HELPER_EDGE_DROPOUT: &str = r#"class EdgeDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, adj):
        if self.training and self.p > 0:
            return adj * (torch.rand_like(adj) >= self.p).to(adj.dtype)
        return adj
"#;

const HELPER_VQ: &str = r#"class VectorQuantizer(nn.Module):
    """Nearest-codebook quantizer with a straight-through gradient. The codebook is an exponential moving
    average of the inputs assigned to each code, so it has no learnable parameters of its own."""

    def __init__(self, num_embeddings, dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.k, self.decay, self.eps = num_embeddings, decay, eps
        self.register_buffer("embed", torch.randn(num_embeddings, dim))
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_sum", self.embed.clone())

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.t() + self.embed.pow(2).sum(1)
        idx = dist.argmin(1)
        if self.training:
            with torch.no_grad():
                onehot = torch.nn.functional.one_hot(idx, self.k).type(flat.dtype)
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.embed_sum.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                size = (self.cluster_size + self.eps) / (n + self.k * self.eps) * n
                self.embed.copy_(self.embed_sum / size.unsqueeze(1))
        q = self.embed[idx].view_as(x)
        return x + (q - x).detach(), idx.view(x.shape[:-1])
"#;

// ---- helpers -----------------------------------------------------------------------------------------------------

fn list_i64(p: &Params, k: &str) -> Vec<i64> {
    p.get(k).and_then(|v| v.as_sequence()).map(|s| s.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default()
}
fn opt_f(p: &Params, k: &str) -> Option<f64> {
    p.get(k).and_then(|v| v.as_f64())
}
fn opt_i(p: &Params, k: &str) -> Option<i64> {
    p.get(k).and_then(|v| v.as_i64())
}
fn pyf(v: f64) -> String {
    format!("{v:?}")
}
fn pyb(b: bool) -> &'static str {
    if b { "True" } else { "False" }
}
fn pyopt(v: Option<f64>) -> String {
    v.map(pyf).unwrap_or_else(|| "None".into())
}
fn py_list(v: &[i64]) -> String {
    v.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")
}

fn pos_int(node: &str, p: &Params, k: &str) -> Result<(), GraphError> {
    match p.get(k).and_then(|v| v.as_i64()) {
        Some(n) if n > 0 => Ok(()),
        _ => Err(bad(node, k, "must be a positive integer")),
    }
}

/// Resolve a (possibly negative) dimension index, refusing the batch dimension.
fn norm_dim(node: &str, param: &str, d: i64, rank: usize) -> Result<usize, GraphError> {
    let r = rank as i64;
    let a = if d < 0 { d + r } else { d };
    if a < 1 || a >= r {
        return Err(bad(node, param, format!("must address a non-batch dimension of a rank-{r} tensor (1..{} or negative)", r - 1)));
    }
    Ok(a as usize)
}

fn add_dim(d: &Dim, n: i64) -> Dim {
    match d {
        Dim::Const(x) => Dim::Const(x + n),
        Dim::Sym(s) if n == 0 => Dim::Sym(s.clone()),
        Dim::Sym(s) => Dim::Sym(format!("{s}{n:+}")),
    }
}

fn mul_dim(d: &Dim, n: i64) -> Dim {
    match d {
        Dim::Const(x) => Dim::Const(x * n),
        Dim::Sym(s) if n == 1 => Dim::Sym(s.clone()),
        Dim::Sym(s) => Dim::Sym(format!("{s}*{n}")),
    }
}

/// NumPy-style broadcasting of two shapes.
fn broadcast(node: &str, a: &Shape, b: &Shape) -> Result<Shape, GraphError> {
    let n = a.len().max(b.len());
    let at = |s: &Shape, i: usize| -> Option<Dim> {
        let off = n - s.len();
        (i >= off).then(|| s[i - off].clone())
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(match (at(a, i), at(b, i)) {
            (Some(x), None) | (None, Some(x)) => x,
            (Some(x), Some(y)) if x == y => x,
            (Some(Dim::Const(1)), Some(y)) => y,
            (Some(x), Some(Dim::Const(1))) => x,
            (Some(x), Some(y)) => {
                return Err(shape_err(node, "b", format!("cannot broadcast {} with {} (dimension {i}: {x} vs {y})", crate::graph::fmt_shape(a), crate::graph::fmt_shape(b))));
            }
            (None, None) => unreachable!(),
        });
    }
    Ok(out)
}

fn graph_inputs(node: &str, x: &Shape, adj: &Shape) -> Result<(), GraphError> {
    rank(node, "x", x, 3)?;
    last_const(node, "x", x)?;
    rank(node, "adj", adj, 3)?;
    if adj[1] != adj[2] || adj[1] != x[1] {
        return Err(shape_err(node, "adj", format!("adjacency {} must be [B, N, N] matching node count of x {}", crate::graph::fmt_shape(adj), crate::graph::fmt_shape(x))));
    }
    Ok(())
}

fn cu(s: &Shape, i: usize) -> Option<u64> {
    match s.get(i) {
        Some(Dim::Const(n)) => Some(*n as u64),
        _ => None,
    }
}
fn last_u(s: Option<&Shape>) -> u64 {
    s.and_then(|s| cu(s, s.len().saturating_sub(1))).unwrap_or(0)
}
fn chan_u(s: Option<&Shape>) -> u64 {
    s.and_then(|s| cu(s, 1)).unwrap_or(0)
}

impl Block {
    pub(super) fn more_name(&self) -> &'static str {
        match self {
            Block::Multiply => "multiply",
            Block::Subtract => "subtract",
            Block::Divide => "divide",
            Block::Scale => "scale",
            Block::Clamp => "clamp",
            Block::Reduce => "reduce",
            Block::Argmax => "argmax",
            Block::Permute => "permute",
            Block::Transpose => "transpose",
            Block::Squeeze => "squeeze",
            Block::Unsqueeze => "unsqueeze",
            Block::Split => "split",
            Block::Slice => "slice",
            Block::Stack => "stack",
            Block::Repeat => "repeat",
            Block::Pad => "pad",
            Block::Matmul => "matmul",
            Block::GroupNorm => "groupnorm",
            Block::InstanceNorm => "instancenorm",
            Block::RmsNorm => "rmsnorm",
            Block::L2Normalize => "l2_normalize",
            Block::Dropout2d => "dropout2d",
            Block::AlphaDropout => "alpha_dropout",
            Block::GaussianNoise => "gaussian_noise",
            Block::MaxPool1d => "maxpool1d",
            Block::AvgPool1d => "avgpool1d",
            Block::AdaptiveAvgPool2d => "adaptive_avgpool2d",
            Block::GlobalMaxPool => "global_max_pool",
            Block::Bilinear => "bilinear",
            Block::Conv3d => "conv3d",
            Block::ConvTranspose1d => "conv_transpose1d",
            Block::PixelShuffle => "pixel_shuffle",
            Block::DepthwiseConv2d => "depthwise_conv2d",
            Block::Rope => "rope",
            Block::CrossAttention => "cross_attention",
            Block::SageConv => "sage_conv",
            Block::GinConv => "gin_conv",
            Block::EdgeDropout => "edge_dropout",
            Block::NoiseLike => "noise_like",
            Block::VectorQuantizer => "vector_quantizer",
            Block::GumbelSoftmax => "gumbel_softmax",
            _ => unreachable!("not a 'more' block"),
        }
    }

    pub(super) fn more_family(&self) -> &'static str {
        match self {
            Block::Multiply | Block::Subtract | Block::Divide | Block::Scale | Block::Clamp | Block::Reduce | Block::Argmax | Block::Permute
            | Block::Transpose | Block::Squeeze | Block::Unsqueeze | Block::Split | Block::Slice | Block::Stack | Block::Repeat | Block::Pad
            | Block::Matmul => "shape",
            Block::GroupNorm | Block::InstanceNorm | Block::RmsNorm | Block::L2Normalize | Block::Dropout2d | Block::AlphaDropout
            | Block::GaussianNoise | Block::Bilinear => "core",
            Block::MaxPool1d | Block::AvgPool1d | Block::AdaptiveAvgPool2d | Block::GlobalMaxPool | Block::Conv3d | Block::ConvTranspose1d
            | Block::PixelShuffle | Block::DepthwiseConv2d => "vision",
            Block::Rope | Block::CrossAttention => "attention",
            Block::SageConv | Block::GinConv | Block::EdgeDropout => "graph",
            Block::NoiseLike | Block::VectorQuantizer | Block::GumbelSoftmax => "generative",
            _ => unreachable!("not a 'more' block"),
        }
    }

    pub(super) fn more_description(&self) -> &'static str {
        match self {
            Block::Multiply => "Element-wise product of two tensors, with broadcasting (gating, squeeze-and-excitation)",
            Block::Subtract => "Element-wise a - b, with broadcasting",
            Block::Divide => "Element-wise a / b, with broadcasting",
            Block::Scale => "Multiply by a constant and add a constant: x * factor + shift",
            Block::Clamp => "Limit values to the range [min, max] (either bound may be left open)",
            Block::Reduce => "Reduce one dimension with sum, mean, max, min, std or prod (global pooling over any axis)",
            Block::Argmax => "Index of the largest value along a dimension (integer output)",
            Block::Permute => "Reorder the per-sample dimensions: dims lists the old dimension indices in their new order, e.g. [2, 1]",
            Block::Transpose => "Swap two dimensions of the tensor, e.g. [B, T, F] -> [B, F, T] with dim0 = 1 and dim1 = 2",
            Block::Squeeze => "Remove a dimension of size 1",
            Block::Unsqueeze => "Insert a dimension of size 1",
            Block::Split => "Cut a tensor into two equal halves along a dimension: outputs a and b",
            Block::Slice => "Take the range [start, end) along one dimension",
            Block::Stack => "Join two same-shaped tensors along a new dimension",
            Block::Repeat => "Tile the tensor: times gives the repeat count for each per-sample dimension",
            Block::Pad => "Constant-pad the trailing dimensions: pad is [left, right, top, bottom, ...] starting from the last dimension",
            Block::Matmul => "Batched matrix product a @ b over the last two dimensions (optionally with b transposed)",
            Block::GroupNorm => "Group normalization over [B, C, ...]: statistics per group of channels, independent of batch size",
            Block::InstanceNorm => "Instance normalization over [B, C, L] / [B, C, H, W] / [B, C, D, H, W]: per-sample, per-channel statistics",
            Block::RmsNorm => "Root-mean-square normalization over the last dimension (as used in modern LLMs)",
            Block::L2Normalize => "Scale vectors to unit length along a dimension",
            Block::Dropout2d => "Zero entire channels of [B, C, H, W] during training",
            Block::AlphaDropout => "Dropout that keeps the mean and variance of its input (use with selu)",
            Block::GaussianNoise => "Add zero-mean Gaussian noise during training (regularizer)",
            Block::MaxPool1d => "1-D max pooling over [B, C, L]",
            Block::AvgPool1d => "1-D average pooling over [B, C, L]",
            Block::AdaptiveAvgPool2d => "Average-pool [B, C, H, W] to a fixed size x size output, whatever the input size",
            Block::GlobalMaxPool => "Maximum over the spatial dimensions: [B, C, H, W] -> [B, C]",
            Block::Bilinear => "Bilinear layer: out = a^T W b + bias, mixing two feature vectors",
            Block::Conv3d => "3-D convolution over [B, C, D, H, W] (video, volumes)",
            Block::ConvTranspose1d => "1-D transposed convolution that upsamples [B, C, L]",
            Block::PixelShuffle => "Rearrange channels into space: [B, C*r*r, H, W] -> [B, C, H*r, W*r] (sub-pixel upsampling)",
            Block::DepthwiseConv2d => "Depthwise convolution: one filter per input channel, [B, C, H, W] -> [B, C*multiplier, H', W']",
            Block::Rope => "Rotary position embedding: rotates feature pairs by a position-dependent angle, [B, T, E] (E even)",
            Block::CrossAttention => "Multi-head attention from queries q [B, Tq, E] onto a context kv [B, Tk, E]",
            Block::SageConv => "GraphSAGE layer: self features plus the mean of neighbour features, over x [B, N, F] and adj [B, N, N]",
            Block::GinConv => "Graph isomorphism layer: an MLP over (1 + eps) * x + the sum of neighbour features",
            Block::EdgeDropout => "Randomly remove edges from an adjacency matrix [B, N, N] during training",
            Block::NoiseLike => "A tensor of random normal noise with the same shape as the input (noise injection, latent codes)",
            Block::VectorQuantizer => "Snap each vector to its nearest codebook entry (VQ-VAE); outputs the quantized vectors and their indices",
            Block::GumbelSoftmax => "Differentiable sampling of a categorical distribution from logits (optionally one-hot)",
            _ => unreachable!("not a 'more' block"),
        }
    }

    pub(super) fn more_param_defs(&self) -> &'static [ParamDef] {
        match self {
            Block::Scale => &[("factor", "1.0", "Multiplier"), ("shift", "0.0", "Constant added after scaling")],
            Block::Clamp => &[("min", "null", "Lower bound (null = none)"), ("max", "null", "Upper bound (null = none)")],
            Block::Reduce => &[
                ("op", "\"sum\"", "sum | mean | max | min | std | prod"),
                ("dim", "1", "Dimension to reduce (negative counts from the end; the batch dimension is kept)"),
                ("keepdim", "false", "Keep the reduced dimension with size 1"),
            ],
            Block::Argmax => &[("dim", "-1", "Dimension to search"), ("keepdim", "false", "Keep the reduced dimension with size 1")],
            Block::Permute => &[("dims", "", "New order of dimensions 1..rank-1, e.g. [2, 1]")],
            Block::Transpose => &[("dim0", "1", "First dimension"), ("dim1", "2", "Second dimension")],
            Block::Squeeze => &[("dim", "1", "Dimension to remove (must have size 1)")],
            Block::Unsqueeze => &[("dim", "1", "Position of the new dimension")],
            Block::Split => &[("dim", "-1", "Dimension to cut (its size must be even)")],
            Block::Slice => &[("dim", "1", "Dimension to slice"), ("start", "0", "First index"), ("end", "null", "End index, exclusive (null = to the end)")],
            Block::Stack => &[("dim", "1", "Position of the new dimension")],
            Block::Repeat => &[("times", "", "Repeat count for each per-sample dimension, e.g. [1, 3]")],
            Block::Pad => &[("pad", "", "[left, right, top, bottom, ...] starting from the last dimension"), ("value", "0.0", "Fill value")],
            Block::Matmul => &[("transpose_b", "false", "Transpose the last two dimensions of b first")],
            Block::GroupNorm => &[("groups", "8", "Channel groups (must divide the channel count)")],
            Block::InstanceNorm => &[("affine", "true", "Learn a per-channel scale and shift")],
            Block::L2Normalize => &[("dim", "-1", "Dimension to normalize")],
            Block::Dropout2d | Block::AlphaDropout | Block::EdgeDropout => &[("p", "0.1", "Drop probability")],
            Block::GaussianNoise => &[("std", "0.1", "Standard deviation of the noise")],
            Block::MaxPool1d | Block::AvgPool1d => &[("kernel", "2", "Window size"), ("stride", "0", "Stride (0 = kernel)")],
            Block::AdaptiveAvgPool2d => &[("size", "1", "Output height and width")],
            Block::Bilinear => &[("out", "", "Output features"), ("bias", "true", "Learn an additive bias")],
            Block::Conv3d => &[
                ("out_channels", "", "Output channels"),
                ("kernel", "3", "Kernel size"),
                ("stride", "1", "Stride"),
                ("padding", "0", "Zero padding"),
            ],
            Block::ConvTranspose1d => &[
                ("out_channels", "", "Output channels"),
                ("kernel", "2", "Kernel size"),
                ("stride", "2", "Stride (the upsampling factor)"),
                ("padding", "0", "Zero padding"),
            ],
            Block::PixelShuffle => &[("factor", "2", "Upscale factor r (channels must be divisible by r*r)")],
            Block::DepthwiseConv2d => &[
                ("kernel", "3", "Kernel size"),
                ("stride", "1", "Stride"),
                ("padding", "0", "Zero padding"),
                ("multiplier", "1", "Output channels per input channel"),
            ],
            Block::Rope => &[("base", "10000.0", "Frequency base")],
            Block::CrossAttention => &[("heads", "4", "Attention heads")],
            Block::SageConv | Block::GinConv => &[("out", "", "Output features")],
            Block::NoiseLike => &[("std", "1.0", "Standard deviation of the noise")],
            Block::VectorQuantizer => &[("num_embeddings", "", "Codebook size"), ("decay", "0.99", "Moving-average decay of the codebook")],
            Block::GumbelSoftmax => &[("tau", "1.0", "Temperature"), ("hard", "false", "Return one-hot samples (straight-through gradient)")],
            Block::Multiply | Block::Subtract | Block::Divide | Block::RmsNorm | Block::GlobalMaxPool => &[],
            _ => unreachable!("not a 'more' block"),
        }
    }

    pub(super) fn more_check_params(&self, node: &str, p: &Params) -> Result<(), GraphError> {
        let pos = |k: &str| pos_int(node, p, k);
        let prob = |k: &str| {
            let v = getf(p, k);
            if (0.0..1.0).contains(&v) { Ok(()) } else { Err(bad(node, k, "must be in [0, 1)")) }
        };
        let finite = |k: &str| match opt_f(p, k) {
            Some(v) if v.is_finite() => Ok(()),
            _ => Err(bad(node, k, "must be a finite number")),
        };
        match self {
            Block::Scale => {
                finite("factor")?;
                finite("shift")?;
            }
            Block::Clamp => {
                for k in ["min", "max"] {
                    if !matches!(p.get(k), None | Some(Value::Null)) {
                        finite(k)?;
                    }
                }
                if let (Some(lo), Some(hi)) = (opt_f(p, "min"), opt_f(p, "max")) {
                    if lo > hi {
                        return Err(bad(node, "min", "must not exceed max"));
                    }
                }
            }
            Block::Reduce => {
                let op = gets(p, "op");
                if !["sum", "mean", "max", "min", "std", "prod"].contains(&op.as_str()) {
                    return Err(bad(node, "op", "must be sum, mean, max, min, std or prod"));
                }
            }
            Block::Permute => {
                let d = list_i64(p, "dims");
                if d.is_empty() || d.len() != p.get("dims").and_then(|v| v.as_sequence()).map_or(0, |s| s.len()) || d.iter().any(|n| *n < 1) {
                    return Err(bad(node, "dims", "must be a non-empty list of positive dimension indices"));
                }
            }
            Block::Slice => {
                pos("dim")?;
                if opt_i(p, "start").is_none_or(|s| s < 0) {
                    return Err(bad(node, "start", "must be a non-negative integer"));
                }
                if !matches!(p.get("end"), None | Some(Value::Null)) {
                    match opt_i(p, "end") {
                        Some(e) if e > opt_i(p, "start").unwrap_or(0) => {}
                        _ => return Err(bad(node, "end", "must be an integer greater than start (or null)")),
                    }
                }
            }
            Block::Stack => pos("dim")?,
            Block::Repeat => {
                let t = list_i64(p, "times");
                if t.is_empty() || t.len() != p.get("times").and_then(|v| v.as_sequence()).map_or(0, |s| s.len()) || t.iter().any(|n| *n < 1) {
                    return Err(bad(node, "times", "must be a non-empty list of positive integers"));
                }
            }
            Block::Pad => {
                let d = list_i64(p, "pad");
                let n = p.get("pad").and_then(|v| v.as_sequence()).map_or(0, |s| s.len());
                if d.is_empty() || d.len() != n || d.len() % 2 != 0 || d.iter().any(|v| *v < 0) {
                    return Err(bad(node, "pad", "must be a non-empty list of non-negative integers with an even length"));
                }
                finite("value")?;
            }
            Block::GroupNorm => pos("groups")?,
            Block::Dropout2d | Block::AlphaDropout | Block::EdgeDropout => prob("p")?,
            Block::GaussianNoise | Block::NoiseLike => {
                if !opt_f(p, "std").is_some_and(|s| s.is_finite() && s >= 0.0) {
                    return Err(bad(node, "std", "must be a number >= 0"));
                }
            }
            Block::MaxPool1d | Block::AvgPool1d => {
                pos("kernel")?;
                if opt_i(p, "stride").is_none_or(|s| s < 0) {
                    return Err(bad(node, "stride", "must be >= 0 (0 = kernel)"));
                }
            }
            Block::AdaptiveAvgPool2d => pos("size")?,
            Block::Bilinear | Block::SageConv | Block::GinConv => pos("out")?,
            Block::Conv3d | Block::ConvTranspose1d => {
                pos("out_channels")?;
                pos("kernel")?;
                pos("stride")?;
            }
            Block::PixelShuffle => pos("factor")?,
            Block::DepthwiseConv2d => {
                pos("kernel")?;
                pos("stride")?;
                pos("multiplier")?;
            }
            Block::Rope => {
                if !opt_f(p, "base").is_some_and(|b| b.is_finite() && b > 1.0) {
                    return Err(bad(node, "base", "must be a number > 1"));
                }
            }
            Block::CrossAttention => pos("heads")?,
            Block::VectorQuantizer => {
                pos("num_embeddings")?;
                if !opt_f(p, "decay").is_some_and(|d| d > 0.0 && d < 1.0) {
                    return Err(bad(node, "decay", "must be in (0, 1)"));
                }
            }
            Block::GumbelSoftmax => {
                if !opt_f(p, "tau").is_some_and(|t| t.is_finite() && t > 0.0) {
                    return Err(bad(node, "tau", "must be a number > 0"));
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn more_infer(&self, node: &str, p: &Params, ins: &BTreeMap<String, Shape>) -> Result<BTreeMap<String, Shape>, GraphError> {
        let x = ins.get("in");
        match self {
            Block::Multiply | Block::Subtract | Block::Divide => Ok(one("out", broadcast(node, &ins["a"], &ins["b"])?)),
            Block::Scale | Block::Clamp | Block::AlphaDropout | Block::GaussianNoise | Block::NoiseLike | Block::GumbelSoftmax => {
                Ok(one("out", x.unwrap().clone()))
            }
            Block::Reduce | Block::Argmax => {
                let x = x.unwrap();
                if x.len() < 2 {
                    return Err(shape_err(node, "in", "needs at least [B, F]"));
                }
                let d = norm_dim(node, "dim", geti(p, "dim"), x.len())?;
                let mut o = x.clone();
                if getb(p, "keepdim") {
                    o[d] = Dim::Const(1);
                } else {
                    o.remove(d);
                }
                Ok(one("out", o))
            }
            Block::Permute => {
                let x = x.unwrap();
                let dims = list_i64(p, "dims");
                let mut sorted = dims.clone();
                sorted.sort_unstable();
                if x.len() < 2 || sorted != (1..x.len() as i64).collect::<Vec<_>>() {
                    return Err(bad(node, "dims", format!("must be a permutation of 1..{} for input {}", x.len().saturating_sub(1), crate::graph::fmt_shape(x))));
                }
                let mut o = vec![x[0].clone()];
                o.extend(dims.iter().map(|d| x[*d as usize].clone()));
                Ok(one("out", o))
            }
            Block::Transpose => {
                let x = x.unwrap();
                let (a, b) = (norm_dim(node, "dim0", geti(p, "dim0"), x.len())?, norm_dim(node, "dim1", geti(p, "dim1"), x.len())?);
                let mut o = x.clone();
                o.swap(a, b);
                Ok(one("out", o))
            }
            Block::Squeeze => {
                let x = x.unwrap();
                let d = norm_dim(node, "dim", geti(p, "dim"), x.len())?;
                if x[d] != Dim::Const(1) {
                    return Err(shape_err(node, "in", format!("dimension {d} of {} has size {}, only size 1 can be squeezed", crate::graph::fmt_shape(x), x[d])));
                }
                let mut o = x.clone();
                o.remove(d);
                Ok(one("out", o))
            }
            Block::Unsqueeze => {
                let x = x.unwrap();
                let r = x.len() as i64;
                let d = geti(p, "dim");
                let at = if d < 0 { d + r + 1 } else { d };
                if at < 1 || at > r {
                    return Err(bad(node, "dim", format!("must be in 1..={r} (or negative) for a rank-{r} input")));
                }
                let mut o = x.clone();
                o.insert(at as usize, Dim::Const(1));
                Ok(one("out", o))
            }
            Block::Split => {
                let x = x.unwrap();
                let d = norm_dim(node, "dim", geti(p, "dim"), x.len())?;
                let n = dim_at(node, "in", x, d)?;
                if n % 2 != 0 {
                    return Err(shape_err(node, "in", format!("dimension {d} has size {n}, which cannot be cut into two equal halves")));
                }
                let mut o = x.clone();
                o[d] = Dim::Const(n / 2);
                Ok(BTreeMap::from([("a".to_string(), o.clone()), ("b".to_string(), o)]))
            }
            Block::Slice => {
                let x = x.unwrap();
                let d = geti(p, "dim") as usize;
                if d >= x.len() {
                    return Err(bad(node, "dim", format!("out of range for input {}", crate::graph::fmt_shape(x))));
                }
                let (start, end) = (geti(p, "start"), opt_i(p, "end"));
                let len = match (&x[d], end) {
                    (Dim::Const(n), e) => {
                        let e = e.unwrap_or(*n);
                        if e > *n || start >= *n {
                            return Err(shape_err(node, "in", format!("slice {start}..{e} is outside dimension {d} of size {n}")));
                        }
                        Dim::Const(e - start)
                    }
                    (_, Some(e)) => Dim::Const(e - start),
                    (s, None) => add_dim(s, -start),
                };
                let mut o = x.clone();
                o[d] = len;
                Ok(one("out", o))
            }
            Block::Stack => {
                let (a, b) = (&ins["a"], &ins["b"]);
                if a != b {
                    return Err(shape_err(node, "b", format!("cannot stack {} and {}: shapes differ", crate::graph::fmt_shape(a), crate::graph::fmt_shape(b))));
                }
                let d = geti(p, "dim") as usize;
                if d > a.len() {
                    return Err(bad(node, "dim", format!("must be in 1..={} for rank-{} inputs", a.len(), a.len())));
                }
                let mut o = a.clone();
                o.insert(d, Dim::Const(2));
                Ok(one("out", o))
            }
            Block::Repeat => {
                let x = x.unwrap();
                let t = list_i64(p, "times");
                if t.len() + 1 != x.len() {
                    return Err(bad(node, "times", format!("needs one count per per-sample dimension: {} for input {}", x.len() - 1, crate::graph::fmt_shape(x))));
                }
                let mut o = vec![x[0].clone()];
                o.extend(x[1..].iter().zip(&t).map(|(d, n)| mul_dim(d, *n)));
                Ok(one("out", o))
            }
            Block::Pad => {
                let x = x.unwrap();
                let pad = list_i64(p, "pad");
                if pad.len() / 2 > x.len() - 1 {
                    return Err(bad(node, "pad", format!("pads {} dimensions but the input has only {} besides the batch", pad.len() / 2, x.len() - 1)));
                }
                let mut o = x.clone();
                for (j, pair) in pad.chunks(2).enumerate() {
                    let i = x.len() - 1 - j;
                    o[i] = add_dim(&x[i], pair[0] + pair[1]);
                }
                Ok(one("out", o))
            }
            Block::Matmul => {
                let (a, b) = (&ins["a"], &ins["b"]);
                if a.len() < 2 || a.len() != b.len() {
                    return Err(shape_err(node, "b", format!("matmul needs two tensors of the same rank >= 2, got {} and {}", crate::graph::fmt_shape(a), crate::graph::fmt_shape(b))));
                }
                let r = a.len();
                if a[..r - 2] != b[..r - 2] {
                    return Err(shape_err(node, "b", format!("leading dimensions of {} and {} differ", crate::graph::fmt_shape(a), crate::graph::fmt_shape(b))));
                }
                let (k_b, n) = if getb(p, "transpose_b") { (&b[r - 1], &b[r - 2]) } else { (&b[r - 2], &b[r - 1]) };
                if &a[r - 1] != k_b {
                    return Err(shape_err(node, "b", format!("inner dimensions differ: a has {}, b has {} ({})", a[r - 1], k_b, if getb(p, "transpose_b") { "after transposing" } else { "set transpose_b to multiply by b^T" })));
                }
                let mut o = a.clone();
                o[r - 1] = n.clone();
                Ok(one("out", o))
            }
            Block::GroupNorm => {
                let x = x.unwrap();
                if x.len() < 2 {
                    return Err(shape_err(node, "in", "needs at least [B, C]"));
                }
                let c = dim_at(node, "in", x, 1)?;
                if c % geti(p, "groups") != 0 {
                    return Err(bad(node, "groups", format!("{} groups do not divide {c} channels", geti(p, "groups"))));
                }
                Ok(one("out", x.clone()))
            }
            Block::InstanceNorm => {
                let x = x.unwrap();
                if !(3..=5).contains(&x.len()) {
                    return Err(shape_err(node, "in", format!("expected rank 3, 4 or 5, got {}", crate::graph::fmt_shape(x))));
                }
                dim_at(node, "in", x, 1)?;
                Ok(one("out", x.clone()))
            }
            Block::RmsNorm => {
                let x = x.unwrap();
                last_const(node, "in", x)?;
                Ok(one("out", x.clone()))
            }
            Block::L2Normalize => {
                let x = x.unwrap();
                if x.len() < 2 {
                    return Err(shape_err(node, "in", "needs at least [B, F]"));
                }
                norm_dim(node, "dim", geti(p, "dim"), x.len())?;
                Ok(one("out", x.clone()))
            }
            Block::Dropout2d => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                Ok(one("out", x.clone()))
            }
            Block::MaxPool1d | Block::AvgPool1d => {
                let x = x.unwrap();
                rank(node, "in", x, 3)?;
                let k = geti(p, "kernel");
                let s = if geti(p, "stride") == 0 { k } else { geti(p, "stride") };
                Ok(one("out", vec![x[0].clone(), x[1].clone(), conv_len(node, "in", &x[2], k, s, 0)?]))
            }
            Block::AdaptiveAvgPool2d => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                let s = Dim::Const(geti(p, "size"));
                Ok(one("out", vec![x[0].clone(), x[1].clone(), s.clone(), s]))
            }
            Block::GlobalMaxPool => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                Ok(one("out", vec![x[0].clone(), x[1].clone()]))
            }
            Block::Bilinear => {
                let (a, b) = (&ins["a"], &ins["b"]);
                last_const(node, "a", a)?;
                last_const(node, "b", b)?;
                if a.len() != b.len() || a[..a.len() - 1] != b[..b.len() - 1] {
                    return Err(shape_err(node, "b", format!("{} and {} must agree on every dimension except the last", crate::graph::fmt_shape(a), crate::graph::fmt_shape(b))));
                }
                let mut o = a.clone();
                *o.last_mut().unwrap() = Dim::Const(geti(p, "out"));
                Ok(one("out", o))
            }
            Block::Conv3d => {
                let x = x.unwrap();
                rank(node, "in", x, 5)?;
                dim_at(node, "in", x, 1)?;
                let (k, s, pad) = (geti(p, "kernel"), geti(p, "stride"), geti(p, "padding"));
                Ok(one(
                    "out",
                    vec![
                        x[0].clone(),
                        Dim::Const(geti(p, "out_channels")),
                        conv_len(node, "in", &x[2], k, s, pad)?,
                        conv_len(node, "in", &x[3], k, s, pad)?,
                        conv_len(node, "in", &x[4], k, s, pad)?,
                    ],
                ))
            }
            Block::ConvTranspose1d => {
                let x = x.unwrap();
                rank(node, "in", x, 3)?;
                dim_at(node, "in", x, 1)?;
                let (k, st, pad) = (geti(p, "kernel"), geti(p, "stride"), geti(p, "padding"));
                let l = match &x[2] {
                    Dim::Const(n) => {
                        let v = (n - 1) * st - 2 * pad + k;
                        if v < 1 {
                            return Err(shape_err(node, "in", format!("transposed convolution of size {n} would be empty")));
                        }
                        Dim::Const(v)
                    }
                    Dim::Sym(s) => Dim::Sym(format!("{s}'")),
                };
                Ok(one("out", vec![x[0].clone(), Dim::Const(geti(p, "out_channels")), l]))
            }
            Block::PixelShuffle => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                let (c, r) = (dim_at(node, "in", x, 1)?, geti(p, "factor"));
                if c % (r * r) != 0 {
                    return Err(shape_err(node, "in", format!("{c} channels are not divisible by factor^2 = {}", r * r)));
                }
                Ok(one("out", vec![x[0].clone(), Dim::Const(c / (r * r)), mul_dim(&x[2], r), mul_dim(&x[3], r)]))
            }
            Block::DepthwiseConv2d => {
                let x = x.unwrap();
                rank(node, "in", x, 4)?;
                let c = dim_at(node, "in", x, 1)?;
                let (k, s, pad) = (geti(p, "kernel"), geti(p, "stride"), geti(p, "padding"));
                Ok(one(
                    "out",
                    vec![x[0].clone(), Dim::Const(c * geti(p, "multiplier")), conv_len(node, "in", &x[2], k, s, pad)?, conv_len(node, "in", &x[3], k, s, pad)?],
                ))
            }
            Block::Rope => {
                let x = x.unwrap();
                rank(node, "in", x, 3)?;
                let e = last_const(node, "in", x)?;
                if e % 2 != 0 {
                    return Err(shape_err(node, "in", format!("rotary embedding rotates feature pairs, so the size {e} must be even")));
                }
                Ok(one("out", x.clone()))
            }
            Block::CrossAttention => {
                let (q, kv) = (&ins["q"], &ins["kv"]);
                rank(node, "q", q, 3)?;
                rank(node, "kv", kv, 3)?;
                let (e, ek) = (last_const(node, "q", q)?, last_const(node, "kv", kv)?);
                if e != ek {
                    return Err(shape_err(node, "kv", format!("context feature size {ek} must equal the query feature size {e}")));
                }
                if q[0] != kv[0] {
                    return Err(shape_err(node, "kv", "batch dimension differs between q and kv"));
                }
                if e % geti(p, "heads") != 0 {
                    return Err(bad(node, "heads", format!("embedding size {e} is not divisible by {} heads", geti(p, "heads"))));
                }
                Ok(one("out", q.clone()))
            }
            Block::SageConv | Block::GinConv => {
                let (xs, adj) = (&ins["x"], &ins["adj"]);
                graph_inputs(node, xs, adj)?;
                Ok(one("out", vec![xs[0].clone(), xs[1].clone(), Dim::Const(geti(p, "out"))]))
            }
            Block::EdgeDropout => {
                let adj = x.unwrap();
                rank(node, "in", adj, 3)?;
                if adj[1] != adj[2] {
                    return Err(shape_err(node, "in", format!("adjacency {} must be square [B, N, N]", crate::graph::fmt_shape(adj))));
                }
                Ok(one("out", adj.clone()))
            }
            Block::VectorQuantizer => {
                let x = x.unwrap();
                last_const(node, "in", x)?;
                if x.len() < 2 {
                    return Err(shape_err(node, "in", "needs at least [B, D]"));
                }
                Ok(BTreeMap::from([("out".to_string(), x.clone()), ("indices".to_string(), x[..x.len() - 1].to_vec())]))
            }
            _ => unreachable!("not a 'more' block"),
        }
    }

    pub(super) fn more_param_count(&self, p: &Params, ins: &BTreeMap<String, Shape>) -> u64 {
        let first = ins.get("in").or_else(|| ins.get("x"));
        let (i, o) = (last_u(first), geti(p, "out") as u64);
        let k = geti(p, "kernel") as u64;
        let oc = geti(p, "out_channels") as u64;
        match self {
            Block::GroupNorm => 2 * chan_u(first),
            Block::InstanceNorm => if getb(p, "affine") { 2 * chan_u(first) } else { 0 },
            Block::RmsNorm => i,
            Block::Bilinear => last_u(ins.get("a")) * last_u(ins.get("b")) * o + if getb(p, "bias") { o } else { 0 },
            Block::Conv3d => oc * chan_u(first) * k * k * k + oc,
            Block::ConvTranspose1d => chan_u(first) * oc * k + oc,
            Block::DepthwiseConv2d => {
                let cm = chan_u(first) * geti(p, "multiplier") as u64;
                cm * k * k + cm
            }
            Block::CrossAttention => {
                let e = last_u(ins.get("q"));
                4 * e * e + 4 * e
            }
            Block::SageConv => 2 * i * o + o,
            Block::GinConv => i * o + o + o * o + o + 1,
            _ => 0,
        }
    }

    pub(super) fn more_macs(&self, p: &Params, ins: &BTreeMap<String, Shape>, outs: &BTreeMap<String, Shape>) -> Option<u64> {
        let first = ins.get("in").or_else(|| ins.get("x"));
        let out = outs.get("out");
        let spatial = |s: &Shape, from: usize| -> Option<u64> { (from..s.len()).try_fold(1u64, |a, i| Some(a * cu(s, i)?)) };
        match self {
            Block::Matmul => {
                let (a, y) = (ins.get("a")?, out?);
                Some(spatial(y, 1)? * cu(a, a.len() - 1)?)
            }
            Block::Bilinear => {
                let (a, b, y) = (ins.get("a")?, ins.get("b")?, out?);
                Some(spatial(y, 1)? * cu(a, a.len() - 1)? * cu(b, b.len() - 1)?)
            }
            Block::Conv3d => {
                let (x, y) = (first?, out?);
                Some(spatial(y, 1)? * cu(x, 1)? * (geti(p, "kernel") as u64).pow(3))
            }
            Block::ConvTranspose1d => {
                let (x, y) = (first?, out?);
                Some(cu(x, 2)? * cu(x, 1)? * cu(y, 1)? * geti(p, "kernel") as u64)
            }
            Block::DepthwiseConv2d => {
                let y = out?;
                Some(spatial(y, 1)? * (geti(p, "kernel") as u64).pow(2))
            }
            Block::CrossAttention => {
                let (q, kv) = (ins.get("q")?, ins.get("kv")?);
                let (tq, tk, e) = (cu(q, 1)?, cu(kv, 1)?, cu(q, 2)?);
                Some(2 * tq * e * e + 2 * tk * e * e + 2 * tq * tk * e)
            }
            Block::SageConv => {
                let x = first?;
                let (n, f, o) = (cu(x, 1)?, cu(x, 2)?, geti(p, "out") as u64);
                Some(2 * n * f * o + n * n * f)
            }
            Block::GinConv => {
                let x = first?;
                let (n, f, o) = (cu(x, 1)?, cu(x, 2)?, geti(p, "out") as u64);
                Some(n * n * f + n * f * o + n * o * o)
            }
            _ => Some(0),
        }
    }

    pub(super) fn more_helper(&self) -> Option<(&'static str, &'static str)> {
        match self {
            Block::GaussianNoise => Some(("GaussianNoise", HELPER_NOISE)),
            Block::Rope => Some(("RotaryEmbedding", HELPER_ROPE)),
            Block::SageConv => Some(("GraphSage", HELPER_SAGE)),
            Block::GinConv => Some(("GraphIsomorphism", HELPER_GIN)),
            Block::EdgeDropout => Some(("EdgeDropout", HELPER_EDGE_DROPOUT)),
            Block::VectorQuantizer => Some(("VectorQuantizer", HELPER_VQ)),
            _ => None,
        }
    }

    pub(super) fn more_ctor(&self, p: &Params, ins: &BTreeMap<String, Shape>) -> Option<String> {
        let first = ins.get("in").or_else(|| ins.get("x"));
        let (i, c) = (last_u(first), chan_u(first));
        Some(match self {
            Block::GroupNorm => format!("nn.GroupNorm({}, {c})", geti(p, "groups")),
            Block::InstanceNorm => {
                let n = first?.len() - 2;
                format!("nn.InstanceNorm{n}d({c}, affine={})", pyb(getb(p, "affine")))
            }
            Block::RmsNorm => format!("nn.RMSNorm({i})"),
            Block::Dropout2d => format!("nn.Dropout2d({})", getf(p, "p")),
            Block::AlphaDropout => format!("nn.AlphaDropout({})", getf(p, "p")),
            Block::GaussianNoise => format!("GaussianNoise({})", pyf(getf(p, "std"))),
            Block::MaxPool1d | Block::AvgPool1d => {
                let k = geti(p, "kernel");
                let s = if geti(p, "stride") == 0 { k } else { geti(p, "stride") };
                format!("nn.{}({k}, {s})", if *self == Block::MaxPool1d { "MaxPool1d" } else { "AvgPool1d" })
            }
            Block::AdaptiveAvgPool2d => format!("nn.AdaptiveAvgPool2d({})", geti(p, "size")),
            Block::Bilinear => format!(
                "nn.Bilinear({}, {}, {}, bias={})",
                last_u(ins.get("a")),
                last_u(ins.get("b")),
                geti(p, "out"),
                pyb(getb(p, "bias"))
            ),
            Block::Conv3d => format!(
                "nn.Conv3d({c}, {}, kernel_size={}, stride={}, padding={})",
                geti(p, "out_channels"),
                geti(p, "kernel"),
                geti(p, "stride"),
                geti(p, "padding")
            ),
            Block::ConvTranspose1d => format!(
                "nn.ConvTranspose1d({c}, {}, kernel_size={}, stride={}, padding={})",
                geti(p, "out_channels"),
                geti(p, "kernel"),
                geti(p, "stride"),
                geti(p, "padding")
            ),
            Block::PixelShuffle => format!("nn.PixelShuffle({})", geti(p, "factor")),
            Block::DepthwiseConv2d => format!(
                "nn.Conv2d({c}, {}, kernel_size={}, stride={}, padding={}, groups={c})",
                c as i64 * geti(p, "multiplier"),
                geti(p, "kernel"),
                geti(p, "stride"),
                geti(p, "padding")
            ),
            Block::Rope => format!("RotaryEmbedding({i}, {})", pyf(getf(p, "base"))),
            Block::CrossAttention => format!("nn.MultiheadAttention({}, {}, batch_first=True)", last_u(ins.get("q")), geti(p, "heads")),
            Block::SageConv => format!("GraphSage({i}, {})", geti(p, "out")),
            Block::GinConv => format!("GraphIsomorphism({i}, {})", geti(p, "out")),
            Block::EdgeDropout => format!("EdgeDropout({})", pyf(getf(p, "p"))),
            Block::VectorQuantizer => format!("VectorQuantizer({}, {i}, {})", geti(p, "num_embeddings"), pyf(getf(p, "decay"))),
            // parameter-free tensor operations
            _ => return None,
        })
    }

    pub(super) fn more_forward(&self, node: &str, p: &Params, iv: &BTreeMap<String, String>, ov: &BTreeMap<String, String>) -> String {
        let o = ov.get("out").map(String::as_str).unwrap_or("");
        let x = iv.get("in").map(String::as_str).unwrap_or("");
        match self {
            Block::Multiply => format!("{o} = {} * {}", iv["a"], iv["b"]),
            Block::Subtract => format!("{o} = {} - {}", iv["a"], iv["b"]),
            Block::Divide => format!("{o} = {} / {}", iv["a"], iv["b"]),
            Block::Scale => format!("{o} = {x} * {} + {}", pyf(getf(p, "factor")), pyf(getf(p, "shift"))),
            Block::Clamp => format!("{o} = torch.clamp({x}, min={}, max={})", pyopt(opt_f(p, "min")), pyopt(opt_f(p, "max"))),
            Block::Reduce => {
                let (d, k) = (geti(p, "dim"), pyb(getb(p, "keepdim")));
                let call = match gets(p, "op").as_str() {
                    "max" => "amax",
                    "min" => "amin",
                    "mean" => "mean",
                    "std" => "std",
                    "prod" => "prod",
                    _ => "sum",
                };
                format!("{o} = {x}.{call}(dim={d}, keepdim={k})")
            }
            Block::Argmax => format!("{o} = torch.argmax({x}, dim={}, keepdim={})", geti(p, "dim"), pyb(getb(p, "keepdim"))),
            Block::Permute => format!("{o} = {x}.permute(0, {})", py_list(&list_i64(p, "dims"))),
            Block::Transpose => format!("{o} = {x}.transpose({}, {})", geti(p, "dim0"), geti(p, "dim1")),
            Block::Squeeze => format!("{o} = {x}.squeeze({})", geti(p, "dim")),
            Block::Unsqueeze => format!("{o} = {x}.unsqueeze({})", geti(p, "dim")),
            Block::Split => format!("{}, {} = torch.chunk({x}, 2, dim={})", ov["a"], ov["b"], geti(p, "dim")),
            Block::Slice => {
                let end = opt_i(p, "end").map(|e| e.to_string()).unwrap_or_default();
                format!("{o} = {x}[{}{}:{end}]", ":, ".repeat(geti(p, "dim") as usize), geti(p, "start"))
            }
            Block::Stack => format!("{o} = torch.stack([{}, {}], dim={})", iv["a"], iv["b"], geti(p, "dim")),
            Block::Repeat => format!("{o} = {x}.repeat(1, {})", py_list(&list_i64(p, "times"))),
            Block::Pad => format!("{o} = torch.nn.functional.pad({x}, [{}], value={})", py_list(&list_i64(p, "pad")), pyf(getf(p, "value"))),
            Block::Matmul => {
                let b = if getb(p, "transpose_b") { format!("{}.transpose(-1, -2)", iv["b"]) } else { iv["b"].clone() };
                format!("{o} = torch.matmul({}, {b})", iv["a"])
            }
            Block::L2Normalize => format!("{o} = torch.nn.functional.normalize({x}, dim={})", geti(p, "dim")),
            Block::GlobalMaxPool => format!("{o} = {x}.amax(dim=(2, 3))"),
            Block::Bilinear => format!("{o} = self.{node}({}, {})", iv["a"], iv["b"]),
            Block::CrossAttention => format!("{o}, _ = self.{node}({}, {}, {})", iv["q"], iv["kv"], iv["kv"]),
            Block::SageConv | Block::GinConv => format!("{o} = self.{node}({}, {})", iv["x"], iv["adj"]),
            Block::NoiseLike => format!("{o} = torch.randn_like({x}) * {}", pyf(getf(p, "std"))),
            Block::VectorQuantizer => format!("{o}, {} = self.{node}({x})", ov["indices"]),
            Block::GumbelSoftmax => format!("{o} = torch.nn.functional.gumbel_softmax({x}, tau={}, hard={}, dim=-1)", pyf(getf(p, "tau")), pyb(getb(p, "hard"))),
            _ => format!("{o} = self.{node}({x})"),
        }
    }
}
